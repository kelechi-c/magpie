#!/usr/bin/env bash
# on the colab vm, after `tools/colab.sh up`: native build, the text encoder's files, then tools/text_gate.py for
# bf16 (twice: the first load packs the dit, the second reads the packed file) and each low-precision format.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
bash tools/build_native.sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['text_encoder/*', 'processor/*', 'tokenizer/*'])"
python tools/text_gate.py bf16-first
python tools/text_gate.py bf16
python tools/text_gate.py nv-all --mlp nvfp4 --attention nvfp4
python tools/text_gate.py nv-mlp --mlp nvfp4
python tools/text_gate.py mx-all --mlp mxfp8 --attention mxfp8
echo finished > /out/textgate.done
