import os
import pathlib as pl
from typing import Any, Literal

import torch
from torch import Tensor, nn

from magpie.kernels import MX, Fp8Linear, mx_project, nv_rows, nv_slots

Encoder = Literal["nvfp4", "bf16"]  # the text encoder a run loads: the calibrated 4-bit checkpoint, or the original

SYSTEM = "Comprehend and analyze the provided prompt."
SLOT = "<image{}><|vision_start|><|image_pad|><|vision_end|>"


# the raw chat string the checkpoint was trained on; each condition image gets one "<imageN>" slot.
def template(prompt: str, images: int) -> str:
    vision = " ".join(SLOT.format(i + 1) for i in range(images))
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{vision}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"


# the vision tower saw rgba composited over white during training: uint8 (4, H, W) -> uint8 (3, H, W), where it is.
def flatten_alpha(image: Tensor) -> Tensor:
    rgb, alpha = image[:3].float(), image[3:].float() / 255
    return (rgb * alpha + 255 * (1 - alpha)).round().to(torch.uint8)


# runs the qwen3-vl model as an encoder: returns the last decoder layer's output *before* the final
# rmsnorm with the system-prompt tokens dropped, (1, T, 4096), and the (T,) mask of image slots.
# calls the bare model, so neither the lm head nor the stack of per-layer hidden states is computed, and without
# the kv cache a single forward would only write.
@torch.no_grad()
def encode(encoder: Any, processor: Any, prompt: str, images: list[Tensor]) -> tuple[Tensor, Tensor]:
    system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    drop = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    pad = processor.tokenizer.encode("<|image_pad|>")[0]
    vision = {"images": [flatten_alpha(image.to(encoder.device)) for image in images], "device": encoder.device} if images else {}
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


# a bias-free linear on nvfp4 weights: each call writes its input rows times `scale` as nvfp4 (a calibrated
# checkpoint brings one per layer) and runs the native matmul; `alpha` undoes both the weights' and the input's scaling.
# the packed weight is two buffers, so `state_dict` (and the server's parking of the encoder) carries it.
class PackedLinear(nn.Module):
    def __init__(self, packed: Fp8Linear, alpha: float, scale: float) -> None:
        super().__init__()
        self.register_buffer("values", packed[0])
        self.register_buffer("scales", packed[1])
        self.alpha, self.scale = alpha, scale

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        return mx_project(*nv_rows(flat, scale=self.scale), (self.values, self.scales), alpha=self.alpha).view(*x.shape[:-1], -1)


# one linear of a calibrated nvfp4 checkpoint in llm-compressor's layout (auto-round, nvidia's and red hat's
# releases): `weight_packed` holds two e2m1 codes per byte, `weight_scale` one fp8 scale per 16 weights, and the
# weight is code * scale / weight_global_scale. inputs were calibrated as round(x * input_global_scale).
def calibrated(state: dict[str, Tensor], name: str) -> PackedLinear:
    weights, inputs = float(state[f"{name}.weight_global_scale"]), float(state[f"{name}.input_global_scale"])
    return PackedLinear((state[f"{name}.weight_packed"].cuda(), nv_slots(state[f"{name}.weight_scale"].cuda())), 1 / (weights * inputs), inputs)


CALIBRATED = "kaitchup/Qwen3-VL-8B-Instruct-NVFP4"  # auto-round w4a4 of the stock qwen3-vl-8b-instruct language model, apache-2.0
ENCODER: Encoder = os.environ.get("MAGPIE_ENCODER") or ("nvfp4" if MX else "bf16")  # nvfp4 needs the native matmul (blackwell); MAGPIE_ENCODER overrides


# the calibrated nvfp4 encoder's folder for "nvfp4" (downloaded once), None for "bf16". it holds 5.9 gib instead of
# 15.2, so a 32 gb card keeps encoder, dit and vae loaded even for a 2048 edit. against the bf16 encoder it moves one
# edit in sixteen past the gate by composition, not damage (the vlm judge ties or flips with the order); the default
# where the native matmul is built, since its linears run on it.
def calibrated_folder(encoder: Encoder = ENCODER) -> pl.Path | None:
    from huggingface_hub import snapshot_download

    if encoder == "nvfp4" and not MX:
        raise ValueError("the nvfp4 text encoder runs on magpie_gemm, which needs a blackwell (sm_120) gpu and tools/build_native.sh; use --encoder bf16")
    return pl.Path(snapshot_download(CALIBRATED, allow_patterns=["*.safetensors", "config.json"])) if encoder == "nvfp4" else None


# the checkpoint's qwen3-vl as the pipeline's text encoder, on the gpu in bf16, and its processor. the lm head is
# dropped before the move (the encoder never calls it, 1.2 gib). `nvfp4_from` is the folder of a calibrated nvfp4
# checkpoint of the same model (`calibrated_folder()`): its seven linears per layer replace ours while the model is
# still in host memory, so their bf16 originals never reach the gpu; the vision tower stays bf16. None keeps bf16.
def load_encoder(snapshot: pl.Path, nvfp4_from: pl.Path | None = None) -> tuple[Any, Any]:
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(snapshot / "text_encoder", dtype=torch.bfloat16)
    encoder.lm_head = nn.Identity()
    state = {name: tensor for file in sorted(nvfp4_from.glob("*.safetensors")) for name, tensor in load_file(file).items()} if nvfp4_from else {}
    for index, layer in enumerate(language(encoder).layers if state else ()):
        for owner, part, names in ((layer.self_attn, "self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, "mlp", ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                setattr(owner, name, calibrated(state, f"model.language_model.layers.{index}.{part}.{name}"))
    return encoder.to("cuda").eval(), AutoProcessor.from_pretrained(snapshot / "processor")
