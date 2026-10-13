# the vlm judge on chosen items of two text_gate runs of one case: `python tools/pair_judge.py <case> <reference tag>
# <candidate tag> <index,index,...>` -> /out/textgate/<case>/judge-<candidate>.json. each pair is shown in both orders.
import json
import pathlib as pl
import sys

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from gate import OUT_ROOT, REPOS, gate_input
from vlm_judge import judge, load_judge

if __name__ == "__main__":
    case, reference, candidate, items = sys.argv[1], sys.argv[2], sys.argv[3], [int(item) for item in sys.argv[4].split(",")]
    torch.set_grad_enabled(False)
    folder = pl.Path(OUT_ROOT) / "textgate" / case
    model, processor = load_judge(pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent)
    opinions = {index: judge(model, processor, gate_input(case, index)[0], gate_input(case, index)[1][0],
                             Image.open(folder / reference / f"{index:02d}.png"), Image.open(folder / candidate / f"{index:02d}.png")) for index in items}
    for index, opinion in opinions.items():
        print(candidate, index, opinion["better"], json.dumps([answer.get("difference", "")[:140] for answer in opinion["answers"]]), flush=True)
    (folder / f"judge-{candidate}.json").write_text(json.dumps(opinions, indent=1))
