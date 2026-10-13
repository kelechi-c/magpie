#!/usr/bin/env bash
# on the colab vm, after `tools/colab.sh up`: native build, then nvfp4 on both mlp matmuls through the gate and the
# vlm judge, then a trace of the default build. logs in /out/followup.log, done marker /out/followup.done.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
bash tools/build_native.sh
pip install -q modal
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['text_encoder/*', 'processor/*', 'tokenizer/*'])"
MAGPIE_NVFP4=hidden,out python tools/gate.py --tag nv-both --cases edit16_1024
python tools/colab_extras.py trace final
python tools/colab_extras.py judge nv-both edit16_1024
echo finished > /out/followup.done
