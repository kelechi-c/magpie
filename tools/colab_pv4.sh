#!/usr/bin/env bash
# on a colab vm with the code, the hf token, the gate files and the weights in place: the native attention kernel with
# 4-bit weights x values (native/attn.cu) against the 8-bit cache it replaces. microbench, the whole request both ways
# (MAGPIE_PV4_KEYS above any sequence turns it off, 0 forces it at every size), then the 16-image gate at both sizes.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
pip install -q dreamsim
python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415', allow_patterns=['*.json', '*.safetensors'])"
bash tools/build_native.sh
python tools/tune_attn.py
for keys in 999999999 0 999999999 0; do MAGPIE_PV4_KEYS=$keys python tools/full_request.py resident pv4-$keys; done
python tools/gate.py --variant turbo --tag pv4 --cases edit16_2048
MAGPIE_PV4_KEYS=0 python tools/gate.py --variant turbo --tag pv4-all --cases edit16_1024
echo finished > /out/pv4.done
