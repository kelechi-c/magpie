#!/usr/bin/env bash
# on a fresh colab vm with the code and the hf token in place: bring-up, how many condition images fit (resident and
# staged), then the calibrated nvfp4 encoder against bf16 on the sixteen edits and the lettering strings, judged and read back.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
CAL=kaitchup/Qwen3-VL-8B-Instruct-NVFP4
mkdir -p /root/inputs /out
pip install -q -U torch==2.14.1 torchvision dreamsim ninja && pip uninstall -q -y torchao
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tensorkelechi/magpie-gate', repo_type='dataset', local_dir='/data')
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'vae/*', 'text_encoder/*', 'processor/*', 'tokenizer/*'])
snapshot_download('facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415', allow_patterns=['*.json', '*.safetensors'])
snapshot_download('$CAL', allow_patterns=['*.safetensors', 'config.json'])"
cp -r /data/baseline /out/ && cp /data/inputs/*.png /root/inputs/
bash tools/build_native.sh
python tools/full_request.py refs resident
python tools/full_request.py refs staged
echo finished > /out/refs.done
python tools/text_gate.py bf16
python tools/text_gate.py nv-cal --calibrated $CAL
python tools/pair_judge.py edit16_1024 bf16 nv-cal 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
rm -rf /out/lettering
python tools/text_render.py render enc-bf16
python tools/text_render.py render enc-cal --calibrated $CAL
python tools/text_render.py read
python - <<'PY'
import json, pathlib as pl
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from gate import LETTERING, REPOS
from lettering_item import scored
from text_render import transcribe
from vlm_judge import load_judge
torch.set_grad_enabled(False)
model, processor = load_judge(pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent)
report = {tag: scored(LETTERING[1], transcribe(model, processor, Image.open(f"/out/textgate/edit16_1024/{tag}/15.png"), 200)) for tag in ("bf16", "nv-cal")}
for tag, row in report.items():
    print("poster", tag, row["worst"], json.dumps(row["lines"]), flush=True)
pl.Path("/out/textgate/edit16_1024/poster-read.json").write_text(json.dumps(report, indent=1))
PY
echo finished > /out/cal.done
