#!/usr/bin/env bash
# drives the gate and the kernel benchmarks on a colab g4 (rtx pro 6000 blackwell), which costs a third of modal's.
#   tools/colab.sh up                      start a session, install the baseline's torch, fetch weights, push code + baseline files
#   tools/colab.sh push                    re-upload src/ and tools/ after an edit
#   tools/colab.sh run "<shell command>"   run in /content/magpie on the vm, e.g. "python tools/gate.py --tag x --compare"
#   tools/colab.sh get <remote> <local>    download a result file
#   tools/colab.sh down                    stop the session (an idle session keeps billing)
set -euo pipefail
ROOT=/home/tensor/Shared/code/ml/magpie
WORK=$(mktemp -d)
SESSION=magpie

# runs a shell command on the vm from /content/magpie and prints the end of its output.
remote() {
    python3 -c 'import sys; print("import subprocess\nr = subprocess.run(%r, shell=True, capture_output=True, text=True)\nprint(r.returncode, (r.stdout + r.stderr)[-3000:])" % sys.argv[1])' \
        "cd /content/magpie 2>/dev/null; export HF_HOME=/hf PYTHONPATH=src; $1" > "$WORK/remote.py"
    colab exec -s $SESSION -f "$WORK/remote.py" --timeout 3000
}

# the upload api rejects large files, so a tarball goes up in 12 mb parts and is rejoined on the vm.
send() {
    split -b 12M "$1" "$WORK/part_"
    for part in "$WORK"/part_*; do colab upload -s $SESSION "$part" "/content/$(basename "$part")"; done
    remote "cd /content && cat part_* > up.tar.gz && rm part_* && tar xzf up.tar.gz -C $2 && rm up.tar.gz"
    rm "$WORK"/part_*
}

push() {
    tar czf "$WORK/code.tar.gz" -C "$ROOT" --exclude __pycache__ src tools baseline
    remote "mkdir -p /content/magpie /out/gate"
    send "$WORK/code.tar.gz" /content/magpie
}

case "$1" in
up)
    colab new -s $SESSION --gpu G4
    mkdir -p "$WORK/data/out/baseline/turbo" "$WORK/data/root/inputs"
    for part in reference conditioning magpie steps; do
        mkdir -p "$WORK/data/out/baseline/turbo/$part"
        modal volume get magpie-out "baseline/turbo/$part/edit_1024" "$WORK/data/out/baseline/turbo/$part/"
    done
    cp /home/tensor/Shared/code/ml/flicker/out/article/images/{01,03,06,07,08}.png "$WORK/data/root/inputs/"
    tar czf "$WORK/data.tar.gz" -C "$WORK/data" out root
    push
    send "$WORK/data.tar.gz" /
    remote "pip install -q -U torch==2.14.1 torchvision && python -c \"
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'transformer/*', 'vae/*'])
snapshot_download('facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415', allow_patterns=['*.json', '*.safetensors'])\""
    ;;
push) push ;;
run) remote "$2" ;;
get) colab download -s $SESSION "$2" "$3" ;;
down) colab stop -s $SESSION ;;
esac
