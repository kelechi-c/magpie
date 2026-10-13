import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from magpie.dit import Dit, layout
from magpie.kernels import Fp8Attention, MxMlp, fp8_columns, logit_bound, pack_mlp
from magpie.sampling import SamplerConfig, euler, schedule
from magpie.text import encode, reslot
from magpie.vae import Vae, VaeConfig, image_weights

Shape = tuple[int, int]


@dataclass(frozen=True)
class Models:
    encoder: Any  # transformers Qwen3VLForConditionalGeneration
    processor: Any
    dit: Dit
    vae: Vae


# builds the port's dit and vae around already-loaded weights (e.g. a diffusers pipeline's state
# dicts) without copying them: the dit is created on the meta device so 7b parameters are never
# initialised, then `assign=True` swaps the real tensors in. each block's logit bound is read off its weights,
# and bf16 weights on the gpu also get fp8 copies of the mlp weights and of the attention linears named in
# `attention` (any of q, k, v, out; all four by default); `slim` then drops the bf16 originals (the caller's state dicts must be released too), which leaves
# those layers no bf16 path.
def adopt(dit_state: dict[str, Tensor], vae_state: dict[str, Tensor], mean: list[float], std: list[float], slim: bool = False,
          attention: tuple[str, ...] = ("q", "k", "v", "out")) -> tuple[Dit, Vae]:
    with torch.device("meta"):
        dit = Dit()
    dit.load_state_dict(dit_state, assign=True)
    for block in dit.transformer_blocks:
        block.attn.bound = logit_bound(block.attn.norm_q.weight, block.attn.norm_k.weight)
        mlp, attn = block.img_mlp, block.attn
        if mlp.out.weight.is_cuda and mlp.out.weight.dtype == torch.bfloat16:
            mlp.packed = pack_mlp(mlp.gate_layer.weight, mlp.proj.weight, mlp.out.weight)
            layers = {name: layer for name, layer in zip(Fp8Attention._fields, (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0])) if name in attention}
            attn.packed = Fp8Attention(*(fp8_columns(layers[name].weight) if name in layers else None for name in Fp8Attention._fields))
            layers = tuple(layers.values())
            for layer in (mlp.gate_layer, mlp.proj, mlp.out, *layers) if slim else ():
                layer.weight = None
    return dit.eval(), adopt_vae(vae_state, mean, std)


# the port's vae around already-loaded weights, channels-last.
def adopt_vae(vae_state: dict[str, Tensor], mean: list[float], std: list[float]) -> Vae:
    reference = next(iter(vae_state.values()))
    vae = Vae(VaeConfig(), mean, std).to(reference.device, reference.dtype)
    vae.load_state_dict(image_weights(vae_state), assign=True)
    vae.to(memory_format=torch.channels_last)
    return vae.eval()


PACKED = ("img_mlp.hidden", "img_mlp.out", "attn.q", "attn.k", "attn.v", "attn.out")  # a block's block-scaled linears


# everything a slim, block-scaled dit holds as one flat dict: its bf16 parameters, and per block the values and
# scales of each packed linear under "packed.<block>.<name>.<values|scales>". a hybrid bf16 + fp8 + nvfp4 checkpoint.
def packed_state(dit: Dit) -> dict[str, Tensor]:
    state = dict(dit.state_dict())
    for index, block in enumerate(dit.transformer_blocks):
        pairs = zip(PACKED, (*block.img_mlp.packed, *block.attn.packed))
        state |= {f"packed.{index}.{name}.{part}": tensor for name, pair in pairs if pair is not None for part, tensor in zip(("values", "scales"), pair)}
    return state


# `adopt` from a `packed_state` dict: the bf16 originals of the packed linears never exist, so loading peaks at the
# size of the packed weights. needs the native matmul the state was packed for.
def adopt_packed(state: dict[str, Tensor], vae_state: dict[str, Tensor], mean: list[float], std: list[float]) -> tuple[Dit, Vae]:
    with torch.device("meta"):
        dit = Dit()
    dit.load_state_dict({name: weight for name, weight in state.items() if not name.startswith("packed.")}, strict=False, assign=True)
    for index, block in enumerate(dit.transformer_blocks):
        pairs = [tuple(state[f"packed.{index}.{name}.{part}"] for part in ("values", "scales")) if f"packed.{index}.{name}.values" in state else None for name in PACKED]
        mlp, attn = block.img_mlp, block.attn
        attn.bound = logit_bound(attn.norm_q.weight, attn.norm_k.weight)
        mlp.packed, attn.packed = MxMlp(*pairs[:2]), Fp8Attention(*pairs[2:])
        for layer, pair in zip((mlp.gate_layer, mlp.proj, mlp.out, attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]), (pairs[0], pairs[0], *pairs[1:])):
            layer.weight = layer.weight if pair is None else None
    return dit.eval(), adopt_vae(vae_state, mean, std)


# (width, height) with the requested pixel area and aspect ratio, both multiples of 32.
def fit(area: int, ratio: float) -> Shape:
    width = math.sqrt(area * ratio)
    return round(width / 32) * 32, round(width / ratio / 32) * 32


# pil -> (1, 4, H, W) in [-1, 1].
# the bytes are moved to `device` first, so the arithmetic runs there.
def to_pixels(image: Image.Image, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
    raw = torch.from_numpy(np.array(image.convert("RGBA"))).to(device)
    return (raw.float().div(255.0).permute(2, 0, 1)[None] * 2.0 - 1.0).to(dtype)


# (1, 4, H, W) in [-1, 1] -> rgba pil.
def to_image(pixels: Tensor) -> Image.Image:
    array = ((pixels[0].float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
    return Image.fromarray(array.cpu().numpy())


# (B, 64, h, w) <-> (B, h*w, 64): the dit reads one token per latent pixel, no patching.
def pack(latents: Tensor) -> Tensor:
    return latents.flatten(2).transpose(1, 2)


def unpack(tokens: Tensor, shape: Shape) -> Tensor:
    return tokens.transpose(1, 2).unflatten(2, shape)


# the denoising loop on precomputed conditioning. `text`/`slots` come from `text.encode`,
# `references` are packed condition-image latents (B, R, 64) or None, `noise` is (B, N, 64).
# step 0 runs the full joint sequence and fills the kv cache; later steps recompute target tokens only.
@torch.no_grad()
def denoise(
    dit: Dit, text: Tensor, slots: Tensor, references: Tensor | None, noise: Tensor, shapes: list[Shape],
    sampler: SamplerConfig,
) -> Tensor:
    lay = layout(slots, shapes, dit.cfg)
    sigmas = schedule(noise.shape[1], sampler).to(noise.device)
    levels = (sigmas * 1000).to(noise.dtype) / 1000  # the dit sees sigma rounded like the reference does
    joint = noise if references is None else torch.cat([references, noise], dim=1)
    velocity, cache = dit.prefill(joint, text, levels[:1], lay)
    latents = euler(noise, velocity, sigmas[0], sigmas[1])
    for i in range(1, len(sigmas) - 1):
        velocity = dit.decode(latents, levels[i : i + 1], lay, cache)
        latents = euler(latents, velocity, sigmas[i], sigmas[i + 1])
    return latents


# condition images at the working size: `resolution`^2 pixels, their own aspect ratio, sides multiples of 32.
def resized(images: tuple[Image.Image, ...], resolution: int) -> list[Image.Image]:
    return [image.convert("RGBA").resize(fit(resolution**2, image.width / image.height), Image.LANCZOS) for image in images]


# conditioning (+ already resized condition images) -> rgba image, with no text encoder in sight:
# vae-encode the images, denoise, decode. with images and no size, the output takes the last image's size.
@torch.no_grad()
def render(
    dit: Dit, vae: Vae, text: Tensor, slots: Tensor, sampler: SamplerConfig, images: list[Image.Image] | None = None,
    size: Shape | None = None, resolution: int = 1024, seed: int = 0,
) -> Image.Image:
    device, dtype = dit.img_in.weight.device, dit.img_in.weight.dtype
    images = images or []
    width, height = size or (images[-1].size if images else (resolution,) * 2)
    encoded = [vae.encode(to_pixels(image, device, dtype)) for image in images]
    shapes = [tuple(latent.shape[-2:]) for latent in encoded] + [(height // 16, width // 16)]
    references = torch.cat([pack(latent) for latent in encoded], dim=1) if encoded else None
    generator = torch.Generator("cpu").manual_seed(seed)
    noise = torch.randn((1, 1, 64, *shapes[-1]), generator=generator, dtype=dtype).to(device)
    latents = denoise(dit, text.to(device, dtype), slots.to(device), references, pack(noise[:, 0]), shapes, sampler)
    return to_image(vae.decode(unpack(latents, shapes[-1])))


# prompt (+ optional condition images) -> rgba image: the text encoder, then `render`. with `look`, the encoder reads
# look^2-pixel copies of the condition images while the vae still reads them at `resolution` (a 2048 edit with
# look=1024 encodes three times faster); the slots are then resized to what the real images need.
@torch.no_grad()
def generate(
    models: Models, prompt: str, sampler: SamplerConfig, images: tuple[Image.Image, ...] = (),
    size: Shape | None = None, resolution: int = 1024, seed: int = 0, look: int | None = None,
) -> Image.Image:
    sized = resized(images, resolution)
    text, slots = encode(models.encoder, models.processor, prompt, sized if look is None else resized(images, look))
    if look is not None:
        text, slots = reslot(text, slots, [image.width * image.height // 1024 for image in sized])
    return render(models.dit, models.vae, text, slots, sampler, sized, size, resolution, seed)
