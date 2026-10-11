from typing import Any

import torch
from PIL import Image
from torch import Tensor

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
# calls the bare model, so neither the lm head nor the stack of per-layer hidden states is computed.
@torch.no_grad()
def encode(encoder: Any, processor: Any, prompt: str, images: list[Image.Image]) -> tuple[Tensor, Tensor]:
    system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    drop = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    pad = processor.tokenizer.encode("<|image_pad|>")[0]
    vision = {"images": [flatten_alpha(image) for image in images]} if images else {}
    inputs = processor(text=[template(prompt, len(images))], return_tensors="pt", **vision).to(encoder.device)
    norm = getattr(encoder.model, "language_model", encoder.model).norm
    hook = norm.register_forward_hook(lambda module, args, output: args[0])
    hidden = encoder.model(**inputs).last_hidden_state
    hook.remove()
    return hidden[:, drop:], inputs.input_ids[0, drop:] == pad
