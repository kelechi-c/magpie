# benchmarks qwen-image 2.1 turbo served by sglang diffusion (sglang[diffusion]==0.5.21, the v0.5.21 tag of sgl-project/sglang; repo main 27b68303 checked for layout 2026-10-13)
# per case: one warm-up request, then 5 timed http requests (curl total time, includes png encode and transfer); peak gpu mem = max nvidia-smi memory.used during the timed requests
# variants: default (no speed flags), compile, speed, sage, fp8sage (flags in main); a variant whose server fails to start is reported and skipped; run with bash
set -uo pipefail

MODEL=Qwen/Qwen-Image-2.1-Turbo
PORT=30010
URL=http://127.0.0.1:$PORT
REF=/root/inputs/01.png
OUT=/out/sglang
EDIT_PROMPT="Add a wicker basket of yellow flowers to the handlebars"
T2I_PROMPT="A pencil drawing of a lighthouse on a cliff"

install_sglang() {
  echo "installing sglang[diffusion] into /sgl"
  # UNVERIFIED: sglang[diffusion]==0.5.21 on PyPI (the v0.5.21 git tag and the diffusion extra in pyproject are verified; the PyPI release is not checked)
  # uv makes the venv: colab's python has no ensurepip
  python3 -m pip install -q uv
  python3 -m uv venv /sgl --python 3.12
  python3 -m uv pip install --python /sgl/bin/python 'sglang[diffusion]==0.5.21' --prerelease=allow
}

wait_ready() {  # $1 server pid; returns 1 if the server exits or does not answer in 30 min
  local pid=$1 i
  # UNVERIFIED: /v1/models as the readiness probe (not confirmed in sglang docs)
  for i in $(seq 1 180); do
    curl -sf "$URL/v1/models" >/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

gen_request() {  # $1 size, $2 prompt, $3 response file; prints curl total time in seconds
  # UNVERIFIED: "seed" in the json body (the cookbook documents n, size, background, generator_device, output_format, response_format; the default seed is 42)
  curl -sS --fail-with-body -o "$3" -w '%{time_total}\n' "$URL/v1/images/generations" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"$2\", \"size\": \"$1\", \"seed\": 0, \"num_inference_steps\": 8, \"generator_device\": \"cpu\", \"output_format\": \"png\", \"response_format\": \"b64_json\"}"
}

edit_request() {  # $1 size, $2 prompt, $3 response file; prints curl total time in seconds
  # UNVERIFIED: "size" and "seed" as form fields on /v1/images/edits (the cookbook's edit example sends prompt, generator_device, output_format, response_format and image[] only)
  curl -sS --fail-with-body -o "$3" -w '%{time_total}\n' "$URL/v1/images/edits" \
    --form-string "prompt=$2" --form-string "size=$1" --form-string seed=0 --form-string num_inference_steps=8 \
    --form-string generator_device=cpu --form-string output_format=png \
    --form-string response_format=b64_json -F "image[]=@$REF;type=image/png"
}

save_png() {  # $1 response json, $2 png path
  python3 - "$1" "$2" <<'PY'
import base64, json, pathlib, sys
src, dst = sys.argv[1:3]
# UNVERIFIED: data[0].b64_json as the response key path (response_format=b64_json is documented; the key path is not)
image = json.loads(pathlib.Path(src).read_text())["data"][0]["b64_json"]
pathlib.Path(dst).write_bytes(base64.b64decode(image))
PY
}

summarise() {  # $1 case, $2 variant, $3 work dir; prints the result line
  python3 - "$1" "$2" "$3" <<'PY'
import json, statistics, sys
case, variant, work = sys.argv[1:4]
times = [float(t) for t in open(f"{work}/times.txt").read().split()]
mib = [float(m) for m in open(f"{work}/mem.txt").read().split()]
result = {"median_ms": round(statistics.median(times) * 1000, 1), "peak_gib": round(max(mib) / 1024, 2)}
print(f"sglang {case} {variant}", json.dumps(result))
PY
}

run_case() {  # $1 case, $2 variant, $3 kind (edit|gen), $4 size, $5 prompt
  local case=$1 variant=$2 kind=$3 size=$4 prompt=$5 work i
  work=$(mktemp -d)
  "${kind}_request" "$size" "$prompt" "$work/warm.json" >/dev/null
  # UNVERIFIED: nvidia-smi flags --id, -lms (sample interval in ms) and the --query-gpu field names
  nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 >"$work/mem.txt" &
  local sampler=$!
  for i in 1 2 3 4 5; do
    "${kind}_request" "$size" "$prompt" "$work/run-$i.json" >>"$work/times.txt"
  done
  kill "$sampler"
  wait "$sampler" 2>/dev/null || true
  save_png "$work/run-5.json" "$OUT/$case-$variant.png"
  summarise "$case" "$variant" "$work"
}

run_variant() {  # $1 variant name, $2... extra sglang serve flags
  local variant=$1
  shift
  echo "starting sglang serve, variant $variant"
  # UNVERIFIED: --enable-torch-compile and --warmup-mode on "sglang serve" (documented on the generate path and the performance page); --performance-mode speed is documented on serve, its exact effect is not
  /sgl/bin/sglang serve --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" "$@" >"$OUT/server-$variant.log" 2>&1 &
  local pid=$!
  wait_ready "$pid" || { echo "sglang $variant server did not start: $(grep -a -i -m3 error "$OUT/server-$variant.log" | cut -c1-200)"; /sgl/bin/killall_sglang >/dev/null 2>&1 || true; return 0; }
  run_case edit-1024 "$variant" edit 1024x1024 "$EDIT_PROMPT"
  run_case t2i-1024 "$variant" gen 1024x1024 "$T2I_PROMPT"
  run_case edit-2048 "$variant" edit 2048x2048 "$EDIT_PROMPT"
  /sgl/bin/killall_sglang
  wait "$pid" 2>/dev/null || true
  echo "stopped sglang serve, variant $variant"
}

main() {
  export HF_HOME=${HF_HOME:-/hf}
  export HF_TOKEN
  HF_TOKEN=$(cat "$HF_HOME/token")
  mkdir -p "$OUT"
  [[ -d /sgl ]] || install_sglang
  trap '/sgl/bin/killall_sglang >/dev/null 2>&1 || true' EXIT
  run_variant default
  run_variant compile --enable-torch-compile
  run_variant speed --performance-mode speed
  run_variant sage --attention-backend sage_attn
  run_variant fp8sage --component-quantizations.transformer fp8 --attention-backend sage_attn
}

main "$@"
