#!/usr/bin/env bash
# on a gpu machine after its bring-up (code, gate files, native kernels): the encoder's files, then the whole request
# cold (page cache dropped, then warm), everything resident, and with the encoder staged. results in /out/full/.
set -uo pipefail
export HF_HOME=/hf PYTHONPATH=src:tools
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['text_encoder/*', 'processor/*', 'tokenizer/*'])"
python tools/full_request.py cold warmup
sync; echo 3 > /proc/sys/vm/drop_caches && echo "page cache dropped" || echo "page cache not dropped"
python tools/full_request.py cold disk
python tools/full_request.py cold cached
python tools/full_request.py resident
python tools/full_request.py staged
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo finished > /out/whole.done
