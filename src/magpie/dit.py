import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from magpie.kernels import (SAGE, SAGE_CACHE, Fp8Attention, Fp8Mlp, SageKV, flash, fp8_norm, fp8_project, fp8_rows, max_free, mlp_fp8, norm_mod, qk_prep,
                            sage, sage_cached, sage_pack, sage_refit, sage_tail, sage_unpack, sage_write)

Segment = tuple[int, int, bool]  # (start, end, is_text) run of the prefix
KV = tuple[Tensor, Tensor]  # one layer's keys and values for the whole sequence, each (B, S, heads, head_dim):
# the prefix rows are the cache, the target rows are scratch that every decode step overwrites
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
    head: int  # where the last condition image ends: the rows before it do not depend on the prompt, which comes after
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
    head = max([end for _, end, is_text in runs if not is_text], default=0)
    return Layout(repeats, image, prefix, head, runs, rope.to(image.device))


# row numbers of the text runs before the last condition image's end: the rows a kept cache holds in bf16.
def early_text(lay: Layout) -> Tensor:
    rows = [row for start, end, is_text in lay.segments if is_text and end <= lay.head for row in range(start, end)]
    return torch.tensor(rows, dtype=torch.long, device=lay.rope.device)


# rms norm over the last axis, statistics in fp32, as diffusers.RMSNorm does.
def rms(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x * scale).to(x.dtype) * weight


# rotates channel pairs of q or k (B, S, heads, head_dim) by the per-token complex phases.
def rotate(x: Tensor, rope: Tensor) -> Tensor:
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * rope[:, None]).flatten(3).type_as(x)


# qk-norm then rope, written into `out` when given. `rotate(rms(...))` defines it; bf16 tensors on the gpu
# take the fused kernel instead.
def prep(x: Tensor, weight: Tensor, rope: Tensor, eps: float, out: Tensor | None = None) -> Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16:
        return qk_prep(x, weight, rope, eps, out)
    plain = rotate(rms(x, weight, eps), rope)
    return plain if out is None else out.copy_(plain)


# true when a layer carries packed fp8 weights (`pipeline.adopt` packs them) and x is bf16 on the gpu.
def fp8_ready(packed: Fp8Attention | Fp8Mlp | None, x: Tensor) -> bool:
    return packed is not None and x.is_cuda and x.dtype == torch.bfloat16


# layer_norm(x) * mod, the input of both branches of a block; one kernel for bf16 tensors on the gpu.
def normed(x: Tensor, mod: Tensor, eps: float) -> Tensor:
    fused = x.is_cuda and x.dtype == torch.bfloat16
    return norm_mod(x, mod, eps) if fused else F.layer_norm(x, x.shape[-1:], eps=eps) * mod


# scaled dot-product attention over (B, S, heads, head_dim) tensors.
def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2)


# attention as the model runs it. unmasked bf16 calls on the gpu take sageattention 2's 8-bit kernel when the package
# is installed. without it they take the max-free flash kernel where `bound` (the block's largest possible logit)
# proves exp cannot overflow. everything else is `sdpa`: masked text runs, cpu, and the blocks that need the max.
def attend(q: Tensor, k: Tensor, v: Tensor, bound: float, mask: Tensor | None = None) -> Tensor:
    fused = mask is None and q.is_cuda and q.dtype == torch.bfloat16
    if fused and SAGE:
        return sage(q, k, v)
    return flash(q, k, v, stable=False) if fused and max_free(bound, k.shape[1]) else sdpa(q, k, v, mask)


# block-causal attention over the whole joint sequence: a token sees every earlier token, plus all
# of its own image block. done as one flash call per prefix run (text runs get a causal triangle
# over their own keys) and one for the target image, which sees everything; with `packed`, the whole sequence's
# keys and values already in sage's 8-bit form, the target call reads those.
def attend_prefill(q: Tensor, k: Tensor, v: Tensor, lay: Layout, bound: float, packed: SageKV | None = None) -> Tensor:
    def run(start: int, end: int, is_text: bool) -> Tensor:
        size = end - start
        mask = (
            torch.cat([q.new_ones(size, start, dtype=torch.bool), q.new_ones(size, size, dtype=torch.bool).tril()], 1)
            if is_text
            else None
        )
        return attend(q[:, start:end], k[:, :end], v[:, :end], bound, mask)

    target = attend(q[:, lay.prefix :], k, v, bound) if packed is None else sage_cached(q[:, lay.prefix :], packed)
    return torch.cat([run(*s) for s in lay.segments] + [target], dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: DitConfig) -> None:
        super().__init__()
        self.heads, self.eps = cfg.heads, cfg.eps
        self.bound = math.inf  # largest possible logit; `pipeline.adopt` sets it from the loaded qk-norm weights
        self.packed: Fp8Attention | None = None
        self.to_q, self.to_k, self.to_v = (nn.Linear(cfg.width, cfg.width, bias=False) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(cfg.width, cfg.width, bias=False)])
        self.norm_q = nn.Module()
        self.norm_k = nn.Module()
        self.norm_q.weight = nn.Parameter(torch.ones(cfg.width // cfg.heads))
        self.norm_k.weight = nn.Parameter(torch.ones(cfg.width // cfg.heads))

    # layer_norm(x) * mod projected to per-head q, k, v; q and k are rms-normed (bounds the logits) then rotated.
    # with `into` = (keys, values) views of a kv buffer, k and v are written straight there. a projection with
    # fp8 weights is an fp8 matmul on the norm's fp8 rows; the others read the norm's bf16 output.
    def qkv(self, x: Tensor, mod: Tensor, rope: Tensor, into: KV | None = None) -> tuple[Tensor, Tensor, Tensor]:
        packed = self.packed if fp8_ready(self.packed, x) else Fp8Attention(None, None, None, None)
        rows = fp8_norm(x.flatten(0, 1), mod.flatten(0, 1).contiguous(), self.eps) if any(packed[:3]) else None
        n = normed(x, mod, self.eps) if not all(packed[:3]) else None

        def project(linear: nn.Linear, fp8: tuple[Tensor, Tensor] | None, out: Tensor | None = None) -> Tensor:
            if fp8 is not None:
                return fp8_project(*rows, fp8, None if out is None else out.flatten(0, 1).flatten(1)).view(*x.shape[:2], self.heads, -1)
            flat = linear(n) if out is None else torch.matmul(n, linear.weight.t(), out=out.flatten(2))
            return flat.unflatten(-1, (self.heads, -1))

        q = prep(project(self.to_q, packed.q), self.norm_q.weight, rope, self.eps)
        k = prep(project(self.to_k, packed.k), self.norm_k.weight, rope, self.eps, None if into is None else into[0])
        return q, k, project(self.to_v, packed.v, None if into is None else into[1])

    # the attention branch's residual: skip + gate * to_out(attended).
    def out(self, attended: Tensor, skip: Tensor, gate: Tensor) -> Tensor:
        if fp8_ready(self.packed, skip) and self.packed.out is not None:
            rows = fp8_rows(attended.reshape(-1, skip.shape[-1]))
            return fp8_project(*rows, self.packed.out, skip=skip.flatten(0, 1), gate=gate.flatten(0, 1).contiguous()).view_as(skip)
        return torch.addcmul(skip, gate, self.to_out[0](attended.flatten(2, 3)))


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


# a block's mlp branch with its residual: `x + gate * mlp(layer_norm(x) * mod)` defines it; bf16 tensors on
# the gpu take the fp8 kernels when the mlp carries packed fp8 weights.
def feed(mlp: SwiGLU, x: Tensor, mod: Tensor, gate: Tensor, eps: float) -> Tensor:
    return mlp_fp8(x, mod, gate, mlp.packed, eps) if fp8_ready(mlp.packed, x) else x + gate * mlp(F.layer_norm(x, x.shape[-1:], eps=eps) * mod)


# one single-stream block. it owns no modulation: `mod` = (scale1, gate1, scale2, gate2) is computed
# once per step by the model and shared by all 32 blocks, already as (1 + scale) and tanh(gate).
class Block(nn.Module):
    def __init__(self, cfg: DitConfig) -> None:
        super().__init__()
        self.width, self.eps = cfg.width, cfg.eps
        self.attn = Attention(cfg)
        self.img_mlp = SwiGLU(cfg)

    # full-sequence pass; also returns this layer's keys and values, whose prefix rows are the cache. on the sage
    # path they are quantised once here and returned in that form, sharing `tail` (an earlier layer's) when given;
    # `keep` (row numbers) also keeps those rows in bf16.
    def prefill(self, x: Tensor, mod: tuple[Tensor, ...], lay: Layout, keep: Tensor | None = None, tail: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, KV | SageKV]:
        q, k, v = self.attn.qkv(x, mod[0], lay.rope)
        packed = sage_pack(k, v, lay.head, lay.prefix, keep, tail) if SAGE_CACHE and k.is_cuda and k.dtype == torch.bfloat16 and k.shape[0] == 1 else None
        x = self.attn.out(attend_prefill(q, k, v, lay, self.attn.bound, packed), x, mod[1])
        return feed(self.img_mlp, x, mod[2], mod[3], self.eps), packed or (k, v)

    # target-only pass: target queries attend to [cached prefix, target] with no mask at all. the new target
    # keys and values are written into the end of the cache's bf16 buffers, so nothing is concatenated; a SageKV
    # then requantises just its tail (held rows and target) next to its already quantised prefix.
    def decode(self, x: Tensor, mod: tuple[Tensor, ...], rope: Tensor, cache: KV | SageKV) -> Tensor:
        packed = isinstance(cache, SageKV)
        k, v = sage_tail(cache) if packed else cache
        q, _, _ = self.attn.qkv(x, mod[0], rope, (k[:, -x.shape[1] :], v[:, -x.shape[1] :]))
        attended = sage_cached(q, sage_write(cache, k, v, cache.start)) if packed else attend(q, k, v, self.attn.bound)
        x = self.attn.out(attended, x, mod[1])
        return feed(self.img_mlp, x, mod[2], mod[3], self.eps)

    # first step of a later request on the same condition images: only the rows after the kept head are computed,
    # the closing text run (`texts` rows) then the target. the text run reads every earlier row under its causal
    # triangle, as in `prefill`, but the condition images' rows come dequantised from the cache (only the text rows
    # are kept in bf16); the target reads the quantised cache, as in `decode`. returns the cache holding this
    # request's text rows.
    def resume(self, x: Tensor, mod: tuple[Tensor, ...], rope: Tensor, cache: SageKV, texts: int) -> tuple[Tensor, SageKV]:
        k, v = sage_tail(cache)
        rows = x.shape[1]
        q, _, _ = self.attn.qkv(x, mod[0], rope, (k[:, -rows:], v[:, -rows:]))
        keys, values = (torch.cat([old, new[:, : new.shape[1] - rows + texts]], dim=1) for old, new in zip(sage_unpack(cache), (k, v)))
        mask = torch.cat([q.new_ones(texts, keys.shape[1] - texts, dtype=torch.bool), q.new_ones(texts, texts, dtype=torch.bool).tril()], 1)
        attended = torch.cat([attend(q[:, :texts], keys, values, self.attn.bound, mask), sage_cached(q[:, texts:], sage_write(cache, k, v, cache.start))], dim=1)
        x = self.attn.out(attended, x, mod[1])
        held = tuple(rows[:, : rows.shape[1] - x.shape[1] + texts].clone() for rows in (k, v))
        return feed(self.img_mlp, x, mod[2], mod[3], self.eps), cache._replace(held=held)


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
    # `keep` also keeps each layer's bf16 keys and values of the text rows before `lay.head`, which `resume` needs.
    def prefill(self, latents: Tensor, text: Tensor, sigma: Tensor, lay: Layout, keep: bool = False) -> tuple[Tensor, list[KV | SageKV]]:
        batch, target = latents.shape[0], lay.image.numel() - lay.prefix
        text = self.text_embed(text)
        x = torch.cat([text, text.new_zeros(batch, target // SLOT, text.shape[-1])], dim=1)
        x = x.repeat_interleave(lay.repeats, dim=1)
        x[:, lay.image] = self.img_in(latents)
        temb = self.time_embed(torch.cat([sigma, sigma.new_zeros(1)]))
        mod, final = self.modulate(temb, torch.arange(lay.image.numel(), device=x.device) >= lay.prefix)
        cache: list[KV | SageKV] = []
        for block in self.transformer_blocks:
            x, kv = block.prefill(x, mod, lay, early_text(lay) if keep else None, getattr(cache[-1], "tail", None) if cache else None)
            cache = cache + [kv]
        return self.head(x[:, lay.prefix :], final[:, lay.prefix :]), cache

    # first step when `cache` was kept from an earlier request on the same condition images (`prefill` with `keep`):
    # the rows up to the last image's end are prompt-independent (the prompt follows the images, and both the text
    # encoder and this model are causal), so only the closing text run and the target are computed. the cache comes
    # back refitted to this request's sequence length.
    def resume(self, target: Tensor, text: Tensor, sigma: Tensor, lay: Layout, cache: list[SageKV]) -> tuple[Tensor, list[SageKV]]:
        texts = lay.prefix - lay.head
        x = torch.cat([self.text_embed(text[:, text.shape[1] - texts :]), self.img_in(target)], dim=1)
        temb = self.time_embed(torch.cat([sigma, sigma.new_zeros(1)]))
        mod, final = self.modulate(temb, torch.arange(x.shape[1], device=x.device) >= texts)
        resumed: list[SageKV] = []
        for block, kv in zip(self.transformer_blocks, cache):
            x, kv = block.resume(x, mod, lay.rope[lay.head :], sage_refit(kv, lay.image.numel(), lay.head, resumed[-1].tail if resumed else None), texts)
            resumed = resumed + [kv]
        return self.head(x[:, texts:], final[:, texts:]), resumed

    # later steps: only the target tokens are recomputed, attending to the cached prefix. batch 1 only: the
    # kernels write the new keys and values into the buffers' tails, which are contiguous for a single sequence.
    def decode(self, target: Tensor, sigma: Tensor, lay: Layout, cache: list[KV | SageKV]) -> Tensor:
        assert target.shape[0] == 1
        x = self.img_in(target)
        temb = self.time_embed(torch.cat([sigma, sigma.new_zeros(1)]))
        mod, final = self.modulate(temb, None)
        rope = lay.rope[lay.prefix :]
        for block, kv in zip(self.transformer_blocks, cache):
            x = block.decode(x, mod, rope, kv)
        return self.head(x, final)
