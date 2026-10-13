#!/usr/bin/env bash
# on a fresh colab vm with the code and the hf token in place: the bring-up `tools/colab.sh up` does, then tools/whole_request.sh.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
mkdir -p /root/inputs /out
pip install -q -U torch==2.14.1 torchvision ninja && pip uninstall -q -y torchao
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tensorkelechi/magpie-gate', repo_type='dataset', local_dir='/data')
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'vae/*'])"
cp -r /data/baseline /out/ && cp /data/inputs/*.png /root/inputs/
bash tools/build_native.sh
bash tools/whole_request.sh
