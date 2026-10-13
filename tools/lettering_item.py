# reads the lettering item (gate.WIDE[15]) back with the checkpoint's qwen3-vl: its input, the frozen baseline edit and
# a gate run's edit, at 1024 and 2048, each scored line by line against gate.LETTERING.
#   python tools/lettering_item.py <gate tag>  -> /out/gate/<tag>-lettering.json
import difflib
import json
import pathlib as pl
import sys

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from gate import LETTERING, OUT_ROOT, REPOS, WIDE_INPUTS
from text_render import letters, transcribe
from vlm_judge import load_judge


# the share of each wanted line found in the transcription (best matching window), and the transcription.
def scored(wanted: list[str], seen: str) -> dict[str, object]:
    got = letters(seen)
    found = [round(max((difflib.SequenceMatcher(None, letters(line), got[start : start + len(letters(line))]).ratio()
                        for start in range(max(1, len(got) - len(letters(line)) + 1))), default=0.0), 3) for line in wanted]
    return {"lines": dict(zip(wanted, found)), "worst": min(found), "read": seen}


if __name__ == "__main__":
    tag = sys.argv[1]
    torch.set_grad_enabled(False)
    model, processor = load_judge(pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent)
    root = pl.Path(OUT_ROOT)
    report = {"input": scored(LETTERING[0], transcribe(model, processor, Image.open(f"{WIDE_INPUTS}/15.png"), 200))}
    for case in ("edit16_1024", "edit16_2048"):
        for name, path in (("baseline", root / "baseline" / "turbo" / "magpie" / case / "15.png"), ("candidate", root / "gate" / tag / "turbo" / case / "15.png")):
            report[f"{case} {name}"] = scored(LETTERING[1], transcribe(model, processor, Image.open(path), 200))
    for name, row in report.items():
        print(name, row["worst"], json.dumps(row["lines"]), flush=True)
    (root / "gate" / f"{tag}-lettering.json").write_text(json.dumps(report, indent=1))
