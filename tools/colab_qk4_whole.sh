#!/usr/bin/env bash
# on a colab vm set up by tools/colab_pv4.sh, after the gates: the whole request (text encoder included) with the
# 8-bit attention kernel and with the 4-bit one, alternating, in one session.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
until test -f /out/final4.done; do sleep 20; done
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'vae/*', 'text_encoder/*', 'processor/*', 'tokenizer/*'])"
for keys in 999999999 16384 999999999 16384; do MAGPIE_PV4_KEYS=$keys python tools/full_request.py resident attn-$keys; done
echo finished > /out/whole4.done
