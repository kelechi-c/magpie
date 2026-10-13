#!/usr/bin/env bash
# drives the gate on a runpod rtx 5090 (32 gb, sm_120) over ssh; the pod id is kept in out/runpod.pod.
#   tools/runpod.sh up                      create the pod, install torch, push code, pull weights and the gate files, build the native kernels
#   tools/runpod.sh push                    re-upload src/, tools/ and native/ after an edit
#   tools/runpod.sh run "<shell command>"   run in /root/magpie on the pod
#   tools/runpod.sh get <remote> <local>    download a result file
#   tools/runpod.sh down                    delete the pod (a running pod bills by the second)
set -euo pipefail
ROOT=/home/tensor/Shared/code/ml/magpie
IMAGE=runpod/pytorch:1.4.1-rc.177-cu1300-torch2130-ubuntu2404
GPU="NVIDIA GeForce RTX 5090"
POD_FILE=$ROOT/out/runpod.pod
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key

pod() { cat "$POD_FILE"; }

# "host port" of the pod's ssh endpoint.
address() {
    runpodctl ssh info "$(pod)" | python3 -c 'import json, sys; d = json.load(sys.stdin); print(d.get("ip") or d["host"], d["port"])'
}

remote() {
    read -r host port < <(address)
    ssh -q -i "$KEY" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$host" \
        "cd /root/magpie 2>/dev/null; export HF_HOME=/hf PYTHONPATH=src:tools; $1"
}

push() {
    read -r host port < <(address)
    tar czf - -C "$ROOT" --exclude __pycache__ src tools baseline native |
        ssh -q -i "$KEY" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$host" "mkdir -p /root/magpie /out/gate /hf /root/inputs && tar xzf - -C /root/magpie"
}

case "$1" in
up)
    runpodctl pod create --name magpie-5090 --image $IMAGE --gpu-id "$GPU" --container-disk-in-gb 80 --ports 22/tcp --wait --wait-timeout 15m |
        python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])' > "$POD_FILE"
    push
    read -r host port < <(address)
    scp -q -i "$KEY" -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOME/.cache/huggingface/token" "root@$host:/hf/token"
    remote "pip install -q -U torch==2.14.1 torchvision dreamsim ninja diffusers transformers accelerate safetensors huggingface_hub pillow && python -c \"
from huggingface_hub import snapshot_download
snapshot_download('tensorkelechi/magpie-gate', repo_type='dataset', local_dir='/data')
snapshot_download('Qwen/Qwen-Image-2.1-Turbo', allow_patterns=['model_index.json', 'transformer/*', 'vae/*'])
snapshot_download('facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415', allow_patterns=['*.json', '*.safetensors'])\" && cp -r /data/baseline /out/ && cp /data/inputs/*.png /root/inputs/ && bash tools/build_native.sh"
    ;;
push) push ;;
run) remote "$2" ;;
get)
    read -r host port < <(address)
    scp -q -i "$KEY" -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$host:$2" "$3"
    ;;
down) runpodctl pod delete "$(pod)" && rm "$POD_FILE" ;;
esac
