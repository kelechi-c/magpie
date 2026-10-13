#!/usr/bin/env bash
# on a colab vm already set up by tools/colab_pv4.sh: both attention matmuls on the 4-bit instruction (magpie_attn4)
# through the model's kv cache. step error at 1024 (forced), then the 16-image gate at 2048.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
bash tools/build_native.sh
MAGPIE_PV4_KEYS=0 MAGPIE_QK4_KEYS=0 python tools/gate.py --variant turbo --tag qk4-1024 --cases edit16_1024
MAGPIE_QK4_KEYS=16384 python tools/gate.py --variant turbo --tag qk4 --cases edit16_2048
echo finished > /out/qk4.done
