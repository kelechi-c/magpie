import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from magpie.kernels import Fp8Mlp, flash, max_free, mlp_fp8, qk_prep

Segment = tuple[int, int, bool]  # (start, end, is_text) run of the prefix
KV = tuple[Tensor, Tensor]  # one layer's prefix keys and values, each (B, P, heads, head_dim)
SLOT = 4  # one vision-language image slot stands for a 2x2 group of latent tokens


@dataclass(frozen=True)
class DitConfig:
    width: int = 4096
    heads: int = 32
    layers: int = 32
    latent_channels: int = 64
    context_dim: int = 4096
    mlp_ratio: int = 3
    rope_axes: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6


# everything about a request's token sequence that does not change between denoising steps.
# the joint sequence is [text and condition-image tokens ... , target-image tokens]; `prefix` is its
# step-independent head, the part the kv cache holds.
@dataclass(frozen=True)
class Layout:
    repeats: Tensor  # (T + N/4,) how many joint tokens each encoder position expands into (4 or 1)
    image: Tensor  # (S,) bool, true where a latent token sits
    prefix: int  # number of text + condition-image tokens
    segments: tuple[Segment, ...]  # runs of the prefix: each text run, each condition image
    rope: Tensor  # (S, head_dim/2) complex rotations


# complex rotations e^{i * index * theta^(-2k/dim)} for one rope axis.
def rope_table(index: Tensor, dim: int, theta: float = 10000.0) -> Tensor:
    freqs = torch.outer(index.float(), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


# positions for the three rope axes (frame, height, width). text tokens advance one shared counter
# on all axes; an image block freezes the frame axis at that counter and lays its tokens on a
# zero-centred height/width grid, so its spatial positions do not depend on where it sits.
def rope_positions(image: list[bool], shapes: list[tuple[int, int]]) -> tuple[list[int], list[int], list[int]]:
    frame: list[int] = []
    height: list[int] = []
    width: list[int] = []
    cursor = position = 0
    for h, w in shapes:
        start = image.index(True, cursor)
        text = list(range(position, position + start - cursor))
        frame, height, width = frame + text, height + text, width + text
        position += start - cursor
        cursor = start + h * w
        frame += [position] * (h * w)
        height += [y for y in range(-(h - h // 2), h // 2) for _ in range(w)]
        width += [x for _ in range(h) for x in range(-(w - w // 2), w // 2)]
        position += max(h, w)
    tail = list(range(position, position + len(image) - cursor))
    return frame + tail, height + tail, width + tail


# builds the per-request layout. `slots` is the encoder's image-slot mask (text length T, true at
# <|image_pad|> positions); `shapes` lists (h, w) in latent tokens, condition images first, target last.
def layout(slots: Tensor, shapes: list[tuple[int, int]], cfg: DitConfig) -> Layout:
    target = shapes[-1][0] * shapes[-1][1]
    slots = torch.cat([slots.bool(), slots.new_ones(target // SLOT, dtype=torch.bool)])
    repeats = torch.where(slots, SLOT, 1)
    image = torch.repeat_interleave(slots, repeats)
    prefix = image.numel() - target

    ids = torch.full_like(image, -1, dtype=torch.long)
    lengths = torch.tensor([h * w for h, w in shapes], device=image.device)
    ids[image] = torch.repeat_interleave(torch.arange(len(shapes), device=image.device), lengths)
    edges = [0] + (ids[1:prefix] != ids[: prefix - 1]).nonzero().flatten().add(1).tolist() + [prefix]
    runs = tuple((a, b, bool(ids[a] < 0)) for a, b in zip(edges, edges[1:]) if b > a)

    tables = [
        torch.cat([rope_table(torch.arange(8192), dim), rope_table(-torch.arange(1024).flip(0) - 1, dim)])
        for dim in cfg.rope_axes
    ]
    axes = rope_positions(image.tolist(), shapes)
    rope = torch.cat([table[torch.tensor(axis)] for table, axis in zip(tables, axes)], dim=-1)
    return Layout(repeats, image, prefix, runs, rope.to(image.device))


# rms norm over the last axis, statistics in fp32, as diffusers.RMSNorm does.
def rms(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x * scale).to(x.dtype) * weight


# rotates channel pairs of q or k (B, S, heads, head_dim) by the per-token complex phases.
def rotate(x: Tensor, rope: Tensor) -> Tensor:
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * rope[:, None]).flatten(3).type_as(x)


# qk-norm then rope. `rotate(rms(...))` defines it; bf16 tensors on the gpu take the fused kernel instead.
def prep(x: Tensor, weight: Tensor, rope: Tensor, eps: float) -> Tensor:
    fused = x.is_cuda and x.dtype == torch.bfloat16
    return qk_prep(x, weight, rope, eps) if fused else rotate(rms(x, weight, eps), rope)


# scaled dot-product attention over (B, S, heads, head_dim) tensors.
def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2)


# attention as the model runs it. unmasked bf16 calls on the gpu take the max-free flash kernel where `bound`
# (the block's largest possible logit) proves exp cannot overflow. everything else is `sdpa`: torch's flash
# kernel beats our own running-max variant, so the blocks that need the max stay on it.
def attend(q: Tensor, k: Tensor, v: Tensor, bound: float, mask: Tensor | None = None) -> Tensor:
    fused = mask is None and q.is_cuda and q.dtype == torch.bfloat16 and max_free(bound, k.shape[1])
    return flash(q, k, v, stable=False) if fused else sdpa(q, k, v, mask)


# block-causal attention over the whole joint sequence: a token sees every earlier token, plus all
# of its own image block. done as one flash call per prefix run (text runs get a causal triangle
# over their own keys) and one for the target image, which sees everything.
def attend_prefill(q: Tensor, k: Tensor, v: Tensor, lay: Layout, bound: float) -> Tensor:
    def run(start: int, end: int, is_text: bool) -> Tensor:
        size = end - start
        mask = (
            torch.cat([q.new_ones(size, start, dtype=torch.bool), q.new_ones(size, size, dtype=torch.bool).tril()], 1)
            if is_text
            else None
        )
        return attend(q[:, start:end], k[:, :end], v[:, :end], bound, mask)

    return torch.cat([run(*s) for s in lay.segments] + [attend(q[:, lay.prefix :], k, v, bound)], dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: DitConfig) -> None:
        super().__init__()
        self.heads, self.eps = cfg.heads, cfg.eps
        self.bound = math.inf  # largest possible logit; `pipeline.adopt` sets it from the loaded qk-norm weights
        self.to_q, self.to_k, self.to_v = (nn.Linear(cfg.width, cfg.width, bias=False) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(cfg.width, cfg.width, bias=False)])
        self.norm_q = nn.Module()
        self.norm_k = nn.Module()
        self.norm_q.weight = nn.Parameter(torch.ones(cfg.width // cfg.heads))
        self.norm_k.weight = nn.Parameter(torch.ones(cfg.width // cfg.heads))

    # projects to per-head q, k, v; q and k are rms-normed (bounds the logits) then rotated.
    def qkv(self, x: Tensor, rope: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = (proj(x).unflatten(-1, (self.heads, -1)) for proj in (self.to_q, self.to_k, self.to_v))
        return prep(q, self.norm_q.weight, rope, self.eps), prep(k, self.norm_k.weight, rope, self.eps), v

    def out(self, x: Tensor) -> Tensor:
        return self.to_out[0](x.flatten(2, 3))


class SwiGLU(nn.Module):
    def __init__(self, cfg: DitConfig) -> None:
        super().__init__()
        hidden = cfg.width * cfg.mlp_ratio
        self.proj = nn.Linear(cfg.width, hidden, bias=False)
        self.gate_layer = nn.Linear(cfg.width, hidden, bias=False)
        self.out = nn.Linear(hidden, cfg.width, bias=False)
        self.packed: Fp8Mlp | None = None

    def forward(self, x: Tensor) -> Tensor:
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


# the mlp branch of a block: `mlp(layer_norm(x) * mod)` defines it; bf16 tensors on the gpu take the fp8
# kernels when the mlp carries packed fp8 weights (`pipeline.adopt` packs them).
def feed(mlp: SwiGLU, x: Tensor, mod: Tensor, eps: float) -> Tensor:
    fused = mlp.packed is not None and x.is_cuda and x.dtype == torch.bfloat16
    return mlp_fp8(x, mod, mlp.packed, eps) if fused else mlp(F.layer_norm(x, x.shape[-1:], eps=eps) * mod)


# one single-stream block. it owns no modulation: `mod` = (scale1, gate1, scale2, gate2) is computed
# once per step by the model and shared by all 32 blocks, already as (1 + scale) and tanh(gate).
class Block(nn.Module):
    def __init__(self, cfg: DitConfig) -> None:
        super().__init__()
        self.width, self.eps = cfg.width, cfg.eps
        self.attn = Attention(cfg)
        self.img_mlp = SwiGLU(cfg)

    def norm(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (self.width,), eps=self.eps)

    # full-sequence pass; also returns this layer's prefix keys and values for the cache.
    def prefill(self, x: Tensor, mod: tuple[Tensor, ...], lay: Layout) -> tuple[Tensor, KV]:
        q, k, v = self.attn.qkv(self.norm(x) * mod[0], lay.rope)
        x = x + mod[1] * self.attn.out(attend_prefill(q, k, v, lay, self.attn.bound))
        x = x + mod[3] * feed(self.img_mlp, x, mod[2], self.eps)
        return x, (k[:, : lay.prefix].clone(), v[:, : lay.prefix].clone())

    # target-only pass: target queries attend to [cached prefix, target] with no mask at all.
    def decode(self, x: Tensor, mod: tuple[Tensor, ...], rope: Tensor, cache: KV) -> Tensor:
        q, k, v = self.attn.qkv(self.norm(x) * mod[0], rope)
        attended = attend(q, torch.cat([cache[0], k], dim=1), torch.cat([cache[1], v], dim=1), self.attn.bound)
        x = x + mod[1] * self.attn.out(attended)
        return x + mod[3] * feed(self.img_mlp, x, mod[2], self.eps)


# qwen-image 2.1 single-stream dit. parameter names match the diffusers checkpoint, so
# `load_state_dict` takes the released weights unchanged.
class Dit(nn.Module):
    def __init__(self, cfg: DitConfig = DitConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        width = cfg.width
        self.time_text_embed = nn.Module()
        self.time_text_embed.timestep_embedder = nn.Module()
        self.time_text_embed.timestep_embedder.linear_1 = nn.Linear(256, width, bias=False)
        self.time_text_embed.timestep_embedder.linear_2 = nn.Linear(width, width, bias=False)
        self.txt_in = nn.Module()
        self.txt_in.text_norm = nn.Module()
        self.txt_in.text_norm.weight = nn.Parameter(torch.zeros(cfg.context_dim))
        self.txt_in.in_layer = nn.Linear(cfg.context_dim, width, bias=False)
        self.txt_in.out_layer = nn.Linear(width, width, bias=False)
        self.img_in = nn.Linear(cfg.latent_channels, width, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width, bias=False))
        self.transformer_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(width, width, bias=False)
        self.proj_out = nn.Linear(width, cfg.latent_channels, bias=False)

    # sinusoidal embedding of sigma in [0, 1] (cos half then sin half) through a two-layer mlp.
    def time_embed(self, sigma: Tensor) -> Tensor:
        embedder = self.time_text_embed.timestep_embedder
        # frequencies are built on the cpu like the reference's buffer: gpu `exp` differs in the last bit,
        # which is enough to flip a bf16 rounding of the embedding on some steps.
        freqs = torch.exp(-math.log(10000) * torch.arange(128).float() / 128).to(sigma.device)
        args = 1000.0 * sigma.float()[:, None] * freqs[None]
        wave = torch.cat([args.cos(), args.sin()], dim=-1).to(embedder.linear_1.weight.dtype)
        return embedder.linear_2(F.silu(embedder.linear_1(wave)))

    # zero-centred rms norm (stored weight is scale - 1) then a two-layer mlp into the dit width.
    def text_embed(self, text: Tensor) -> Tensor:
        scale = torch.rsqrt(text.float().pow(2).mean(-1, keepdim=True) + self.cfg.eps)
        normed = (text.float() * scale * (self.txt_in.text_norm.weight.float() + 1)).to(text.dtype)
        return self.txt_in.out_layer(F.gelu(self.txt_in.in_layer(normed), approximate="tanh"))

    # per-token modulation for one step. `temb` rows are [sigma, 0]: target tokens read the sigma
    # row, prefix tokens the t=0 row (`causal_condition`), which is what makes the prefix cacheable.
    # `target` is the (S,) bool row selector, or None when every token is a target token.
    def modulate(self, temb: Tensor, target: Tensor | None) -> tuple[tuple[Tensor, ...], Tensor]:
        def rows(params: Tensor) -> Tensor:
            real, zero = params[:-1, None], params[-1:, None]
            return real if target is None else torch.where(target[None, :, None], real, zero)

        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        final = rows(self.norm_out.linear(F.silu(temb)))
        return (1 + rows(scale1), rows(gate1).tanh(), 1 + rows(scale2), rows(gate2).tanh()), 1 + final

    def head(self, x: Tensor, final: Tensor) -> Tensor:
        return self.proj_out(F.layer_norm(x, (self.cfg.width,), eps=self.cfg.eps) * final)

    # first step: runs the whole joint sequence and returns (velocity for the target tokens, cache).
    # latents: (B, R + N, 64) condition-image latents then the noisy target; text: (B, T, 4096).
    def prefill(self, latents: Tensor, text: Tensor, sigma: Tensor, lay: Layout) -> tuple[Tensor, list[KV]]:
        batch, target = latents.shape[0], lay.image.numel() - lay.prefix
        text = self.text_embed(text)
        x = torch.cat([text, text.new_zeros(batch, target // SLOT, text.shape[-1])], dim=1)
        x = x.repeat_interleave(lay.repeats, dim=1)
        x[:, lay.image] = self.img_in(latents)
        temb = self.time_embed(torch.cat([sigma, sigma.new_zeros(1)]))
        mod, final = self.modulate(temb, torch.arange(lay.image.numel(), device=x.device) >= lay.prefix)
        cache: list[KV] = []
        for block in self.transformer_blocks:
            x, kv = block.prefill(x, mod, lay)
            cache = cache + [kv]
        return self.head(x[:, lay.prefix :], final[:, lay.prefix :]), cache

    # later steps: only the target tokens are recomputed, attending to the cached prefix.
    def decode(self, target: Tensor, sigma: Tensor, lay: Layout, cache: list[KV]) -> Tensor:
        x = self.img_in(target)
        temb = self.time_embed(torch.cat([sigma, sigma.new_zeros(1)]))
        mod, final = self.modulate(temb, None)
        rope = lay.rope[lay.prefix :]
        for block, kv in zip(self.transformer_blocks, cache):
            x = block.decode(x, mod, rope, kv)
        return self.head(x, final)
