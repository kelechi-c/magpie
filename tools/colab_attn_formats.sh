#!/usr/bin/env bash
# on a colab vm with the code and the hf token in place and the native kernels built: the gate files and weights,
# then the step error of attention in each number format (tools/quant_probe.py formats).
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
mkdir -p /root/inputs /out
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tensorkelechi/magpie-gate', repo_type='dataset', local_dir='/data')
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'vae/*'])"
cp -r /data/baseline /out/ && cp /data/inputs/*.png /root/inputs/
python tools/quant_probe.py formats
echo finished > /out/formats.done
