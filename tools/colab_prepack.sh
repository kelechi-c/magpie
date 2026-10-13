#!/usr/bin/env bash
# on a fresh colab vm: pack the dit for the default build, publish it to the private repo, then prove a second load
# fetches it from the hub with no transformer/ folder and no local packed file, and renders the same image.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
pip install -q -U torch==2.14.1 torchvision ninja && pip uninstall -q -y torchao
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'transformer/*', 'vae/*'])"
bash tools/build_native.sh
CHECK="
import hashlib, time, torch
from gate import load_port
import magpie.pipeline as lean
from magpie.sampling import TURBO
start = time.perf_counter(); dit, vae = load_port('turbo'); took = time.perf_counter() - start
g = torch.Generator('cpu').manual_seed(0)
text = torch.randn((1, 40, 4096), generator=g).to('cuda', torch.bfloat16)
image = lean.render(dit, vae, text, torch.zeros(40, dtype=torch.bool), TURBO, [], (512, 512), 512, 0)
print('load', round(took, 1), 's, resident', round(torch.cuda.memory_allocated() / 2**30, 2), 'gib, image', hashlib.sha256(image.tobytes()).hexdigest()[:16], flush=True)"
python -c "$CHECK"
python tools/publish_packed.py
rm -rf /hf/magpie /hf/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/*/transformer
python -c "$CHECK"
ls -la /hf/magpie
echo finished > /out/prepack.done
