#!/usr/bin/env bash
# sweeps the tile sizes of sageattention 2's sm89 fp8 kernel (the one an sm_120 card runs) with an in-place build
# under /sage: "CTA_Q CTA_K WARP_Q WARP_K" per setting, the first being upstream's. results: lines starting "tiles".
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FILE=/sage/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu
export CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0
for setting in "128 64 32 64" "128 128 32 128" "128 128 32 64" "64 64 32 64" "64 64 16 64" "128 64 16 64" "256 128 32 128" "256 128 32 64"; do
    read -r cq ck wq wk <<< "$setting"
    sed -i -E "s/constexpr int CTA_Q = [0-9]+;/constexpr int CTA_Q = $cq;/; s/constexpr int CTA_K = [0-9]+;/constexpr int CTA_K = $ck;/; s/constexpr int WARP_Q = [0-9]+;/constexpr int WARP_Q = $wq;/; s/constexpr int WARP_K = [0-9]+;/constexpr int WARP_K = $wk;/" $FILE
    (cd /sage && python setup.py build_ext --inplace > /tmp/sage_build.log 2>&1) || { echo "tiles build failed $setting: $(grep -m1 -i error /tmp/sage_build.log | cut -c1-200)"; continue; }
    (cd "$ROOT" && PYTHONPATH=/sage:src:tools python tools/sage_tiles.py $cq $ck $wq $wk 2>&1 | grep -E "^tiles|Error|error" | cut -c1-300) || echo "tiles run failed $setting"
done
sed -i -E "s/constexpr int CTA_Q = [0-9]+;/constexpr int CTA_Q = 128;/; s/constexpr int CTA_K = [0-9]+;/constexpr int CTA_K = 64;/; s/constexpr int WARP_Q = [0-9]+;/constexpr int WARP_Q = 32;/; s/constexpr int WARP_K = [0-9]+;/constexpr int WARP_K = 64;/" $FILE
