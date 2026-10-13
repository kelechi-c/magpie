# a second opinion on the items the gate flags, from the checkpoint's own qwen3-vl (8.8 b, the pipeline's text
# encoder with its lm head): it sees the source image, the instruction and both edits, and says which edit follows
# the instruction better and whether either is damaged. each pair is asked in both orders; a verdict that flips
# with the order is reported as "order-dependent". only for flagged items: the gate's pass rule stays numeric.
import json
import pathlib as pl
import re
from typing import Any

QUESTION = (
    "The first image is the source. The instruction was: \"{instruction}\"\n"
    "The second image is edit A and the third image is edit B, two attempts at that instruction.\n"
    "Judge only what you can see. Answer with one JSON object and nothing else:\n"
    "{{\"follows_instruction\": \"A\" | \"B\" | \"both\" | \"neither\", \"better\": \"A\" | \"B\" | \"tie\", "
    "\"artifacts\": \"A\" | \"B\" | \"both\" | \"none\", \"difference\": \"<one sentence on how A and B differ>\"}}"
)
SIDE = 768  # the judge reads every image at this size


# the checkpoint's qwen3-vl as a generator, and its processor.
def load_judge(snapshot: pl.Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(snapshot / "text_encoder", dtype=torch.bfloat16).to("cuda").eval()
    return model, AutoProcessor.from_pretrained(snapshot / "processor")


# one question about (source, edit a, edit b) -> the model's answer, parsed when it is json.
def ask(model: Any, processor: Any, instruction: str, source: Any, a: Any, b: Any) -> dict[str, Any]:
    import torch

    images = [image.convert("RGB").resize((SIDE, SIDE)) for image in (source, a, b)]
    content = [*({"type": "image", "image": image} for image in images), {"type": "text", "text": QUESTION.format(instruction=instruction)}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt").to("cuda")
    with torch.no_grad():
        tokens = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = processor.decode(tokens[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    found = re.search(r"\{.*\}", text, re.S)
    return json.loads(found.group(0)) if found else {"raw": text}


# the pair judged in both orders, in baseline / candidate terms.
def judge(model: Any, processor: Any, instruction: str, source: Any, baseline: Any, candidate: Any) -> dict[str, Any]:
    first = ask(model, processor, instruction, source, baseline, candidate)
    second = ask(model, processor, instruction, source, candidate, baseline)
    named = [{key: {"A": names[0], "B": names[1]}.get(value, value) for key, value in answer.items()}
             for answer, names in ((first, ("baseline", "candidate")), (second, ("candidate", "baseline")))]
    agree = named[0].get("better") == named[1].get("better")
    return {"better": named[0].get("better") if agree else "order-dependent", "answers": named}
