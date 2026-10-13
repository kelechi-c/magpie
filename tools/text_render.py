# does the model still letter words correctly? eight requests whose image must carry a given string (six
# text-to-image, two edits), rendered by one build, then read back by the checkpoint's own qwen3-vl.
#   python tools/text_render.py render <tag> [--mlp F] [--attention F] [--baseline]   -> /out/lettering/<tag>/NN.png
#   python tools/text_render.py read                                                 -> /out/lettering/read.json
# `--baseline` is the port with every kernel off and bf16 weights. `read` transcribes every folder's images and
# scores each against its string (difflib ratio on lowercased letters and digits; 1.0 is a perfect reading).
import argparse
import difflib
import json
import pathlib as pl
import re
from typing import Any

import torch
from PIL import Image

OUT = pl.Path("/out/lettering")
INPUTS = "/root/inputs"
ITEMS: list[tuple[str, str, str | None]] = [  # (string the image must carry, prompt, condition image or None)
    ("MAGPIE LANE 3 km", 'A pencil drawing of a weathered wooden signpost at a country crossroads. The sign reads "MAGPIE LANE 3 km" in carved capitals.', None),
    ("OPEN UNTIL MIDNIGHT", 'An ink-on-paper drawing of a narrow bookshop front at dusk, with the words "OPEN UNTIL MIDNIGHT" lettered across the window glass.', None),
    ("Soup of the day: leek & potato 4.50", 'A pencil sketch of a cafe chalkboard on an easel. It says "Soup of the day: leek & potato 4.50" in neat handwriting.', None),
    ("PLATFORM 9 07:42 SEAT 14A", 'An ink drawing of a paper train ticket lying on a table, printed with "PLATFORM 9  07:42  SEAT 14A".', None),
    ("Measure twice, cut once.", 'A pencil drawing of an open spiral notebook on a workbench. One line is handwritten on the page: "Measure twice, cut once."', None),
    ("THE QUIET HOUR a play in three acts", 'An ink-on-paper theatre poster with the large title "THE QUIET HOUR" and the smaller subtitle "a play in three acts" beneath it.', None),
    ("Plate VII", 'Add a small hand-lettered caption centred along the bottom edge that reads "Plate VII". Keep the drawing unchanged.', "01.png"),
    ("NOT FOR SALE", 'Add a rectangular ink stamp in the top right corner with the words "NOT FOR SALE". Keep everything else unchanged.', "06.png"),
]


def letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def render(tag: str, mlp: str, attention: str, baseline: bool) -> None:
    from huggingface_hub import hf_hub_download

    import magpie.dit as dit_module
    import magpie.pipeline as lean
    import magpie.vae as vae_module
    from gate import REPOS, load_port, unfused
    from magpie.sampling import TURBO
    from magpie.text import encode, load_encoder

    snapshot = pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent
    dit, vae = load_port("turbo", slim=not baseline)
    if baseline:
        unfused(dit_module, vae_module)
    encoder, processor = load_encoder(snapshot, mlp, attention)
    (OUT / tag).mkdir(parents=True, exist_ok=True)
    for index, (_, prompt, source) in enumerate(ITEMS):
        sized = lean.resized((Image.open(f"{INPUTS}/{source}"),), 1024) if source else []
        text, slots = encode(encoder, processor, prompt, sized)
        lean.render(dit, vae, text, slots, TURBO, sized, None if sized else (1024, 1024), 1024, index).convert("RGB").save(OUT / tag / f"{index:02d}.png")
    print("rendered", tag, flush=True)


# the judge's transcription of one image.
def transcribe(model: Any, processor: Any, image: Image.Image) -> str:
    content = [{"type": "image", "image": image.convert("RGB")}, {"type": "text", "text": "Transcribe every word and number written in this image, exactly as written. Reply with the text only."}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return processor.decode(tokens[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def read() -> None:
    from huggingface_hub import hf_hub_download

    from gate import REPOS
    from vlm_judge import load_judge

    model, processor = load_judge(pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent)
    report: dict[str, Any] = {}
    for folder in sorted(path for path in OUT.iterdir() if path.is_dir()):
        seen = [transcribe(model, processor, Image.open(folder / f"{index:02d}.png")) for index in range(len(ITEMS))]
        scores = [round(difflib.SequenceMatcher(None, letters(want), letters(got)).ratio(), 3) for (want, _, _), got in zip(ITEMS, seen)]
        report[folder.name] = {"mean": round(sum(scores) / len(scores), 3), "worst": min(scores), "scores": scores, "read": seen}
        print(folder.name, json.dumps({"mean": report[folder.name]["mean"], "worst": min(scores), "scores": scores}), flush=True)
    (OUT / "read.json").write_text(json.dumps({"wanted": [item[0] for item in ITEMS], **report}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("render", "read"))
    parser.add_argument("tag", nargs="?", default="")
    parser.add_argument("--mlp", default="bf16")
    parser.add_argument("--attention", default="bf16")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    read() if args.mode == "read" else render(args.tag, args.mlp, args.attention, args.baseline)
