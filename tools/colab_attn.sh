#!/usr/bin/env bash
# on a fresh colab vm with the code and the hf token in place: bring-up, loads from disk, the calibrated encoder
# and the reference counts under a 32 gb cap, then the sage tile sweep.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
mkdir -p /root/inputs /out
pip install -q -U torch==2.14.1 torchvision dreamsim ninja && pip uninstall -q -y torchao
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tensorkelechi/magpie-gate', repo_type='dataset', local_dir='/data')
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'vae/*', 'text_encoder/*', 'processor/*', 'tokenizer/*'])"
cp -r /data/baseline /out/ && cp /data/inputs/*.png /root/inputs/
bash tools/build_native.sh
python tools/full_request.py cold warmup
python tools/full_request.py cold disk
python tools/full_request.py cold cached
MAGPIE_ENCODER=nvfp4 python tools/full_request.py cold warmup
MAGPIE_ENCODER=nvfp4 python tools/full_request.py cold disk
export MAGPIE_VRAM_GIB=31.36
MAGPIE_ENCODER=nvfp4 python tools/full_request.py resident
MAGPIE_ENCODER=nvfp4 python tools/full_request.py refs resident
python tools/full_request.py refs resident
python tools/full_request.py refs staged
unset MAGPIE_VRAM_GIB
echo finished > /out/cap.done
bash tools/sage_tiles.sh
echo finished > /out/attn.done
