#!/usr/bin/env bash
# on a fresh colab vm with the code and the hf token in place: bring-up, then the quantised kv cache against the
# per-call requantisation it replaces (MAGPIE_SAGE_CACHE=0): kernel microbench, the whole request both ways, the gate
# with the poster's lettering, and the reference counts both ways under a 32 gb cap and on the whole card.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
mkdir -p /root/inputs /out
pip install -q -U torch==2.14.1 torchvision dreamsim ninja && pip uninstall -q -y torchao
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tensorkelechi/magpie-gate', repo_type='dataset', local_dir='/data')
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'vae/*', 'text_encoder/*', 'processor/*', 'tokenizer/*'])
snapshot_download('facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415', allow_patterns=['*.json', '*.safetensors'])"
cp -r /data/baseline /out/ && cp /data/inputs/*.png /root/inputs/
bash tools/build_native.sh
python tools/sage_cache_bench.py
for cache in 0 1 0 1; do MAGPIE_SAGE_CACHE=$cache python tools/full_request.py resident cache$cache; done
python tools/gate.py --variant turbo --tag cache --cases edit16_1024,edit16_2048
python tools/lettering_item.py cache
for cache in 0 1; do MAGPIE_SAGE_CACHE=$cache MAGPIE_VRAM_GIB=31.36 python tools/full_request.py refs staged cache$cache; done
for cache in 0 1; do MAGPIE_SAGE_CACHE=$cache python tools/full_request.py refs staged full$cache; done
echo finished > /out/cache.done
