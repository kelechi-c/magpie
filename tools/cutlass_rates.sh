#!/usr/bin/env bash
# what cutlass's own sm_120 matmul kernels reach on this card at the dit's linear shapes, before building on them:
# nvfp4 x nvfp4, mxfp8 x mxfp8 (example 79c with its mxfp6 operand swapped for mxfp8) and fp8 with blockwise scales,
# all writing bf16. prints each example's runtime and gflops per (m, n, k).
#   bash tools/cutlass_rates.sh
set -euo pipefail
CUTLASS=ee98dbd  # NVIDIA/cutlass
test -d /cutlass || git clone -q https://github.com/NVIDIA/cutlass.git /cutlass
cd /cutlass && git checkout -q $CUTLASS && mkdir -p /cutlass/bin
E=examples
sed 's/mx_float6_t<cutlass::float_e3m2_t>/mx_float8_t<cutlass::float_e4m3_t>/' $E/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu > $E/79_blackwell_geforce_gemm/mxfp8.cu
for pair in nvfp4:79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm mxfp8:79_blackwell_geforce_gemm/mxfp8 \
            fp8_blockwise:87_blackwell_geforce_gemm_blockwise/87a_blackwell_geforce_fp8_bf16_gemm_blockwise; do
    name=${pair%%:*}; source=$E/${pair#*:}.cu
    test -x bin/$name || nvcc -std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a \
        -Iinclude -Itools/util/include -I$E/common -I$(dirname $source) $source -o bin/$name &
done
wait
for name in nvfp4 mxfp8 fp8_blockwise; do
    for shape in "4096 12288 4096" "4096 4096 12288" "4096 4096 4096" "8239 12288 4096"; do
        set -- $shape
        echo "$name m=$1 n=$2 k=$3 $(bin/$name --m=$1 --n=$2 --k=$3 --iterations=20 | grep -E 'Avg runtime|GFLOPS|Disposition' | tr '\n' ' ')"
    done
done
