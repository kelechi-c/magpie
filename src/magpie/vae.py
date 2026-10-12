from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from magpie.kernels import grow_add, pixel_norm


# the released vae descends from a causal video vae; for one image every temporal op is a no-op,
# so this port is plain 2d: (B, C, H, W) in and out. `temporal[i]` only decides how the parameter-free
# shortcuts group channels (a missing frame is averaged in as zeros / dropped).
@dataclass(frozen=True)
class VaeConfig:
    encoder_dim: int = 96
    decoder_dim: int = 144
    dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8)
    temporal: tuple[bool, ...] = (False, True, True, True)
    z_dim: int = 64
    channels: int = 4  # rgba


# true for the tensors the triton kernels take (bf16 on the gpu); they read them channels-last.
def fusable(x: Tensor) -> bool:
    return x.is_cuda and x.dtype == torch.bfloat16


def channels_last(x: Tensor) -> Tensor:
    return x.contiguous(memory_format=torch.channels_last)


# l2-normalises over channels in fp32 and rescales by sqrt(C) * gamma, then applies silu if asked.
# bf16 tensors on the gpu take one fused kernel.
class Norm(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *[1] * rank))

    def forward(self, x: Tensor, silu: bool = False) -> Tensor:
        if fusable(x):
            return pixel_norm(channels_last(x), self.gamma, self.scale, silu)
        plain = F.normalize(x.float(), dim=1).to(x.dtype) * self.scale * self.gamma.view(1, -1, 1, 1)
        return F.silu(plain) if silu else plain


class Resnet(nn.Module):
    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        self.norm1, self.conv1 = Norm(inp, 3), nn.Conv2d(inp, out, 3, padding=1)
        self.norm2, self.conv2 = Norm(out, 3), nn.Conv2d(out, out, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.norm1(x, True))
        return self.conv2(self.norm2(h, True)) + self.conv_shortcut(x)


# single-head self-attention over all H*W positions, at the lowest resolution only.
class SpatialAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm, self.to_qkv, self.proj = Norm(dim, 2), nn.Conv2d(dim, dim * 3, 1), nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(batch, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
        out = F.scaled_dot_product_attention(*qkv.chunk(3, dim=-1))
        return self.proj(out.squeeze(1).permute(0, 2, 1).reshape(batch, channels, height, width)) + x


class Mid(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([Resnet(dim, dim), Resnet(dim, dim)])
        self.attentions = nn.ModuleList([SpatialAttention(dim)])

    def forward(self, x: Tensor) -> Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Resample(nn.Module):
    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.resample = nn.ModuleList([nn.Identity(), nn.Conv2d(dim, dim, 3, stride=stride, padding=2 - stride)])


# parameter-free downsampling shortcut: space-to-depth, then average channel groups down to `out`.
# on a temporal stage the (absent) previous frame contributes zeros to each average.
def avg_down(x: Tensor, out: int, temporal: bool) -> Tensor:
    folded = F.pixel_unshuffle(x, 2).unflatten(1, (x.shape[1], 4))
    folded = torch.cat([torch.zeros_like(folded), folded], dim=2) if temporal else folded
    return folded.reshape(x.shape[0], out, -1, *folded.shape[-2:]).mean(dim=2)


# parameter-free upsampling shortcut: repeat channels, then depth-to-space. on a temporal stage
# only the last of the two frames it would produce is kept.
def dup_up(x: Tensor, out: int, temporal: bool) -> Tensor:
    frames = 2 if temporal else 1
    repeated = x.repeat_interleave(out * frames * 4 // x.shape[1], dim=1)
    return F.pixel_shuffle(repeated.unflatten(1, (out, frames, 4))[:, :, -1].flatten(1, 2), 2)


class Down(nn.Module):
    def __init__(self, inp: int, out: int, down: bool, temporal: bool) -> None:
        super().__init__()
        self.out, self.temporal = out, temporal
        self.resnets = nn.ModuleList([Resnet(inp, out), Resnet(out, out)])
        self.downsampler = Resample(out, 2) if down else None

    def forward(self, x: Tensor) -> Tensor:
        h = self.resnets[1](self.resnets[0](x))
        if self.downsampler is None:
            return h + x
        return self.downsampler.resample[1](F.pad(h, (0, 1, 0, 1))) + avg_down(x, self.out, self.temporal)


class Up(nn.Module):
    def __init__(self, inp: int, out: int, up: bool, temporal: bool) -> None:
        super().__init__()
        self.out, self.temporal = out, temporal
        self.resnets = nn.ModuleList([Resnet(inp, out), Resnet(out, out), Resnet(out, out)])
        self.upsampler = Resample(out, 1) if up else None

    def forward(self, x: Tensor) -> Tensor:
        h = self.resnets[2](self.resnets[1](self.resnets[0](x)))
        if self.upsampler is None:
            return h
        grown = self.upsampler.resample[1](F.interpolate(h, scale_factor=2.0, mode="nearest-exact"))
        return grow_add(channels_last(grown), channels_last(x), self.temporal) if fusable(grown) else grown + dup_up(x, self.out, self.temporal)


class Encoder(nn.Module):
    def __init__(self, cfg: VaeConfig) -> None:
        super().__init__()
        dims = [cfg.encoder_dim * m for m in (1, *cfg.dim_mult)]
        last = len(cfg.dim_mult) - 1
        self.conv_in = nn.Conv2d(cfg.channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            [Down(a, b, i != last, i != last and cfg.temporal[i]) for i, (a, b) in enumerate(zip(dims, dims[1:]))]
        )
        self.mid_block = Mid(dims[-1])
        self.norm_out, self.conv_out = Norm(dims[-1], 3), nn.Conv2d(dims[-1], cfg.z_dim * 2, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(self.norm_out(self.mid_block(x), True))


class Decoder(nn.Module):
    def __init__(self, cfg: VaeConfig) -> None:
        super().__init__()
        mults = cfg.dim_mult[::-1]
        dims = [cfg.decoder_dim * m for m in (mults[0], *mults)]
        temporal, last = cfg.temporal[::-1], len(mults) - 1
        self.conv_in = nn.Conv2d(cfg.z_dim, dims[0], 3, padding=1)
        self.mid_block = Mid(dims[0])
        self.up_blocks = nn.ModuleList(
            [Up(a, b, i != last, i != last and temporal[i]) for i, (a, b) in enumerate(zip(dims, dims[1:]))]
        )
        self.norm_out, self.conv_out = Norm(dims[-1], 3), nn.Conv2d(dims[-1], cfg.channels, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(z))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(self.norm_out(x, True))


# qwen-image 2.1 autoencoder for single images: 16x spatial compression, 64 latent channels, rgba.
# parameter names match the diffusers checkpoint minus the unused `time_conv` weights.
class Vae(nn.Module):
    def __init__(self, cfg: VaeConfig, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.encoder, self.decoder = Encoder(cfg), Decoder(cfg)
        self.quant_conv = nn.Conv2d(cfg.z_dim * 2, cfg.z_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(cfg.z_dim, cfg.z_dim, 1)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    # pixels (B, 4, H, W) in [-1, 1] -> normalised latents (B, 64, H/16, W/16); takes the posterior mean.
    def encode(self, pixels: Tensor) -> Tensor:
        moments = self.quant_conv(self.encoder(channels_last(pixels)))
        return (moments.chunk(2, dim=1)[0] - self.mean) / self.std

    # normalised latents -> pixels (B, 4, H, W) in [-1, 1].
    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(self.post_quant_conv(channels_last(latents * self.std + self.mean))).clamp(-1.0, 1.0)


# drops the temporal convolutions a single image never runs, leaving exactly this port's keys.
def image_weights(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: weight for name, weight in state.items() if ".time_conv." not in name}
