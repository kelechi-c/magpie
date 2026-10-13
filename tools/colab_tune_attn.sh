#!/usr/bin/env bash
# on a fresh colab vm with the code in place: torch, the native builds, then the attention kernel search. no weights.
set -uo pipefail
cd /content/magpie && export PYTHONPATH=src:tools
mkdir -p /out
pip install -q -U torch==2.14.1 torchvision ninja && pip uninstall -q -y torchao
bash tools/build_native.sh
python tools/tune_attn.py
echo finished > /out/tune_attn.done
