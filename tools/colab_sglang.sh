#!/usr/bin/env bash
# on a fresh colab vm with the code and the hf token in place: the edit input, then tools/bench_sglang.sh.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf
mkdir -p /root/inputs /out
pip install -q huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tensorkelechi/magpie-gate', repo_type='dataset', local_dir='/data', allow_patterns=['inputs/*'])"
cp /data/inputs/*.png /root/inputs/
VARIANTS="${VARIANTS:-sage fp8sage}" bash tools/bench_sglang.sh
grep -a -i -m2 "sage" /out/sglang/server-sage.log | cut -c1-200
echo finished > /out/sglang.done
