#!/usr/bin/env bash
# on the colab vm, after the bring-up: native build, the text encoder's files and a calibrated nvfp4 checkpoint, then
# tools/text_gate.py per encoder format, the vlm judge on the items each run flags, and the lettering test.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
CAL=kaitchup/Qwen3-VL-8B-Instruct-NVFP4
bash tools/build_native.sh
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['text_encoder/*', 'processor/*', 'tokenizer/*'])
snapshot_download('$CAL', allow_patterns=['*.safetensors', 'config.json'])"
python tools/text_gate.py bf16
python tools/text_gate.py nv-all --mlp nvfp4 --attention nvfp4
python tools/text_gate.py nv-mlp --mlp nvfp4
python tools/text_gate.py mx-all --mlp mxfp8 --attention mxfp8
python tools/text_gate.py nv-cal --calibrated $CAL
for tag in nv-all nv-mlp mx-all nv-cal; do python tools/pair_judge.py edit16_1024 bf16 $tag 0,3,6,9,12; done
python tools/text_render.py render current
python tools/text_render.py render enc-mxfp8 --mlp mxfp8 --attention mxfp8
python tools/text_render.py read
echo finished > /out/textgate.done
