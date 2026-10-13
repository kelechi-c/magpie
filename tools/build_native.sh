#!/usr/bin/env bash
# builds the native kernels on the machine that will run them (needs nvcc >= 12.8 and the torch they link against):
# sageattention 2 (pinned, sm_120) and magpie_gemm (native/gemm.cu on cutlass). torch 2.14 headers need c++20.
#   bash tools/build_native.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SAGE=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5  # thu-ml/SageAttention, apache-2.0
CUTLASS=ee98dbd  # NVIDIA/cutlass, bsd-3-clause
export CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda
if ! (cd / && python -c "import sageattention" 2>/dev/null); then
    test -d /sage || git clone -q https://github.com/thu-ml/SageAttention /sage
    cd /sage && git checkout -q $SAGE && sed -i 's/-std=c++17/-std=c++20/g' setup.py
    TORCH_CUDA_ARCH_LIST=12.0 pip install -q . --no-build-isolation
fi
test -d /cutlass || git clone -q https://github.com/NVIDIA/cutlass.git /cutlass
git -C /cutlass checkout -q $CUTLASS
cd "$ROOT" && python - <<'PY'
import shutil, sysconfig
from torch.utils.cpp_extension import load
module = load("magpie_gemm", ["native/gemm.cu"], extra_include_paths=["/cutlass/include", "/cutlass/tools/util/include"],
              extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "--expt-relaxed-constexpr", "-gencode", "arch=compute_120a,code=sm_120a"],
              build_directory="/tmp", verbose=False)
shutil.copy(module.__file__, sysconfig.get_paths()["purelib"])
print("magpie_gemm built", module.mxfp8_slots(4096, 4096), module.nvfp4_slots(4096, 4096))
PY
