import os
import pathlib as pl
from typing import Any, Literal

import torch
from PIL import Image
from torch import Tensor, nn

from magpie.kernels import NV_WEIGHT, Fp8Linear, MxMlp, mx_columns, mx_project, mx_rows, nv_rows, nv_slots, pack_mlp_nv, swiglu_block

Format = Literal["bf16", "mxfp8", "nvfp4"]

SYSTEM = "Comprehend and analyze the provided prompt."
SLOT = "<image{}><|vision_start|><|image_pad|><|vision_end|>"


# the raw chat string the checkpoint was trained on; each condition image gets one "<imageN>" slot.
def template(prompt: str, images: int) -> str:
    vision = " ".join(SLOT.format(i + 1) for i in range(images))
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{vision}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"


# the vision tower saw rgba composited over white during training.
def flatten_alpha(image: Image.Image) -> Image.Image:
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


# runs the qwen3-vl model as an encoder: returns the last decoder layer's output *before* the final
# rmsnorm with the system-prompt tokens dropped, (1, T, 4096), and the (T,) mask of image slots.
# calls the bare model, so neither the lm head nor the stack of per-layer hidden states is computed, and without
# the kv cache a single forward would only write.
@torch.no_grad()
def encode(encoder: Any, processor: Any, prompt: str, images: list[Image.Image]) -> tuple[Tensor, Tensor]:
    system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    drop = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    pad = processor.tokenizer.encode("<|image_pad|>")[0]
    vision = {"images": [flatten_alpha(image) for image in images]} if images else {}
    inputs = processor(text=[template(prompt, len(images))], return_tensors="pt", **vision).to(encoder.device)
    norm = getattr(encoder.model, "language_model", encoder.model).norm
    hook = norm.register_forward_hook(lambda module, args, output: args[0])
    hidden = encoder.model(**inputs, use_cache=False).last_hidden_state
    hook.remove()
    return hidden[:, drop:], inputs.input_ids[0, drop:] == pad


# `encode`'s output when the encoder was shown smaller copies of the condition images: every run of image slots is
# resized to the count its real image needs (latent tokens / 4), so the dit's layout is unchanged. slot states are
# never read by the dit, only counted, so the inserted rows are zeros.
def reslot(text: Tensor, slots: Tensor, counts: list[int]) -> tuple[Tensor, Tensor]:
    edges = [0] + (slots[1:] != slots[:-1]).nonzero().flatten().add(1).tolist() + [len(slots)]
    needed = iter(counts)
    pieces = [(text.new_zeros(1, next(needed), text.shape[-1]), slots.new_ones(1)) if slots[a] else (text[:, a:b], slots[a:b]) for a, b in zip(edges, edges[1:])]
    return torch.cat([piece for piece, _ in pieces], dim=1), torch.cat([mask.expand(piece.shape[1]) if mask.numel() == 1 else mask for piece, mask in pieces])


# the language model inside the vision-language model: 36 decoder layers of width 4096.
def language(encoder: Any) -> Any:
    return getattr(encoder.model, "language_model", encoder.model)


# a bias-free linear on block-scaled weights (mxfp8, or nvfp4 when `nv`): each call writes its input rows in the
# same format and runs the native matmul. `alpha` undoes the weights' scaling; `scale` multiplies the input before
# it is rounded (a calibrated nvfp4 checkpoint brings one per layer) and is undone by `alpha` too.
class PackedLinear(nn.Module):
    def __init__(self, packed: Fp8Linear, nv: bool, alpha: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.packed, self.nv, self.alpha, self.scale = packed, nv, alpha, scale

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        rows = nv_rows(flat, scale=self.scale) if self.nv else mx_rows(flat)
        return mx_project(*rows, self.packed, alpha=self.alpha).view(*x.shape[:-1], -1)


# a linear's weight rounded here, with no calibration.
def rounded(weight: Tensor, nv: bool) -> PackedLinear:
    return PackedLinear(nv_rows(weight.contiguous(), scale=NV_WEIGHT), True, 1 / NV_WEIGHT) if nv else PackedLinear(mx_columns(weight), False)


# one linear of a calibrated nvfp4 checkpoint in llm-compressor's layout (auto-round, nvidia's and red hat's
# releases): `weight_packed` holds two e2m1 codes per byte, `weight_scale` one fp8 scale per 16 weights, and the
# weight is code * scale / weight_global_scale. inputs were calibrated as round(x * input_global_scale).
def calibrated(state: dict[str, Tensor], name: str) -> PackedLinear:
    weights, inputs = float(state[f"{name}.weight_global_scale"]), float(state[f"{name}.input_global_scale"])
    return PackedLinear((state[f"{name}.weight_packed"].cuda(), nv_slots(state[f"{name}.weight_scale"].cuda())), True, 1 / (weights * inputs), inputs)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# a calibrated linear's weight back in fp32: the check that the layout above is read correctly, and that the
# checkpoint was made from the same weights as ours.
def unpacked(state: dict[str, Tensor], name: str) -> Tensor:
    packed = state[f"{name}.weight_packed"].long()
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(1)
    values = torch.tensor(E2M1, device=packed.device)[codes & 7] * torch.where(codes > 7, -1.0, 1.0)
    return values * state[f"{name}.weight_scale"].float().repeat_interleave(16, dim=1) / float(state[f"{name}.weight_global_scale"])


CALIBRATED = "kaitchup/Qwen3-VL-8B-Instruct-NVFP4"  # auto-round w4a4 of the stock qwen3-vl-8b-instruct language model, apache-2.0


# the calibrated nvfp4 encoder's folder when MAGPIE_ENCODER=nvfp4 asks for it (downloaded once), else None. it holds
# 5.9 gib instead of 15.2, which lets a 32 gb card keep encoder, dit and vae loaded for 1024 edits; it fails the
# gate on one image in sixteen (a different composition, not damage), so it is not the default.
def calibrated_folder() -> pl.Path | None:
    from huggingface_hub import snapshot_download

    return pl.Path(snapshot_download(CALIBRATED, allow_patterns=["*.safetensors", "config.json"])) if os.environ.get("MAGPIE_ENCODER") == "nvfp4" else None


# a decoder layer's swiglu mlp on block-scaled weights; `nv` names its matmuls that run in nvfp4.
class PackedMlp(nn.Module):
    def __init__(self, mlp: Any, nv: set[str]) -> None:
        super().__init__()
        self.nv = nv
        self.packed: MxMlp = pack_mlp_nv(mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, nv)

    def forward(self, x: Tensor) -> Tensor:
        return swiglu_block(x.reshape(-1, x.shape[-1]).contiguous(), self.packed, self.nv).view_as(x)


# the checkpoint's qwen3-vl as the pipeline's text encoder, on the gpu in bf16, and its processor. the lm head is
# dropped before the move (the encoder never calls it, 1.2 gib). `mlp` and `attention` pick the format of the
# language model's mlps and of its q, k, v, o projections; the bf16 originals are released. the vision tower stays bf16.
# `nvfp4_from` is the folder of a calibrated nvfp4 checkpoint of the same model: its seven linears per layer replace
# ours (the mlp then runs as three matmuls, since its gate and up layers carry different scales).
def load_encoder(snapshot: pl.Path, mlp: Format = "bf16", attention: Format = "bf16", nvfp4_from: pl.Path | None = None) -> tuple[Any, Any]:
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(snapshot / "text_encoder", dtype=torch.bfloat16)
    encoder.lm_head = nn.Identity()
    nvfp4_from = nvfp4_from or calibrated_folder()
    # the calibrated linears replace ours while the model is still in host memory, so their bf16 originals never reach the gpu
    encoder = (encoder if nvfp4_from else encoder.to("cuda")).eval()
    state = {name: tensor for file in sorted(nvfp4_from.glob("*.safetensors")) for name, tensor in load_file(file).items()} if nvfp4_from else {}
    for index, layer in enumerate(language(encoder).layers):
        prefix = f"model.language_model.layers.{index}"
        if index == 0 and state:
            ours, theirs = layer.self_attn.q_proj.weight.float(), unpacked(state, f"{prefix}.self_attn.q_proj")
            print(f"calibrated checkpoint, layer 0 q_proj against our bf16 weight: {float((theirs - ours).norm() / ours.norm()):.3f} relative error", flush=True)
        for owner, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))) if state else ():
            for name in names:
                setattr(owner, name, calibrated(state, f"{prefix}.{'self_attn' if owner is layer.self_attn else 'mlp'}.{name}"))
        if mlp != "bf16" and not state:
            layer.mlp = PackedMlp(layer.mlp, {"hidden", "out"} if mlp == "nvfp4" else set())
        for name in ("q_proj", "k_proj", "v_proj", "o_proj") if attention != "bf16" and not state else ():
            setattr(layer.self_attn, name, rounded(getattr(layer.self_attn, name).weight, attention == "nvfp4"))
    return encoder.to("cuda"), AutoProcessor.from_pretrained(snapshot / "processor")
