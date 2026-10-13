import pathlib as pl
from typing import Any, Literal

import torch
from PIL import Image
from torch import Tensor, nn

from magpie.kernels import NV_WEIGHT, MxMlp, mx_columns, mx_project, mx_rows, nv_rows, pack_mlp_nv, swiglu_block

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
# same format and runs the native matmul.
class PackedLinear(nn.Module):
    def __init__(self, weight: Tensor, nv: bool) -> None:
        super().__init__()
        self.nv = nv
        self.packed = nv_rows(weight.contiguous(), scale=NV_WEIGHT) if nv else mx_columns(weight)

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        rows = nv_rows(flat) if self.nv else mx_rows(flat)
        return mx_project(*rows, self.packed, alpha=1 / NV_WEIGHT if self.nv else 1.0).view(*x.shape[:-1], -1)


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
def load_encoder(snapshot: pl.Path, mlp: Format = "bf16", attention: Format = "bf16") -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(snapshot / "text_encoder", dtype=torch.bfloat16)
    encoder.lm_head = nn.Identity()
    encoder = encoder.to("cuda").eval()
    for layer in language(encoder).layers:
        if mlp != "bf16":
            layer.mlp = PackedMlp(layer.mlp, {"hidden", "out"} if mlp == "nvfp4" else set())
        for name in ("q_proj", "k_proj", "v_proj", "o_proj") if attention != "bf16" else ():
            setattr(layer.self_attn, name, PackedLinear(getattr(layer.self_attn, name).weight, attention == "nvfp4"))
    return encoder, AutoProcessor.from_pretrained(snapshot / "processor")
