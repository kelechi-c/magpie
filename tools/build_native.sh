#!/usr/bin/env bash
# builds the native kernels on the machine that will run them (needs nvcc >= 12.8 and the torch they link against):
# sageattention 2 (pinned, sm_120), magpie_gemm (native/gemm.cu on cutlass) and magpie_attn (native/attn.cu, the
# attention kernel with 4-bit weights x values, and magpie_attn4 with 4-bit q x k too, on sageattention's headers). torch 2.14 headers need c++20.
#   bash tools/build_native.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SAGE=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5  # thu-ml/SageAttention, apache-2.0
CUTLASS=ee98dbd  # NVIDIA/cutlass, bsd-3-clause
export CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda
test -d /sage || git clone -q https://github.com/thu-ml/SageAttention /sage
git -C /sage checkout -q -f $SAGE && sed -i 's/-std=c++17/-std=c++20/g' /sage/setup.py
(cd / && python -c "import sageattention" 2>/dev/null) || (cd /sage && TORCH_CUDA_ARCH_LIST=12.0 pip install -q . --no-build-isolation)
test -d /cutlass || git clone -q https://github.com/NVIDIA/cutlass.git /cutlass
git -C /cutlass checkout -q $CUTLASS
cd "$ROOT" && python - <<'PY'
import os, shutil, sysconfig
for name in ('magpie_attn', 'magpie_attn4'): os.makedirs(f'/tmp/{name}', exist_ok=True)
from torch.utils.cpp_extension import load
module = load("magpie_gemm", ["native/gemm.cu"], extra_include_paths=["/cutlass/include", "/cutlass/tools/util/include"],
              extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "--expt-relaxed-constexpr", "-gencode", "arch=compute_120a,code=sm_120a"],
              build_directory="/tmp", verbose=False)
shutil.copy(module.__file__, sysconfig.get_paths()["purelib"])
print("magpie_gemm built", module.mxfp8_slots(4096, 4096), module.nvfp4_slots(4096, 4096))
for name, flags in (("magpie_attn", ["-DATTN_PV4=1"]), ("magpie_attn4", ["-DATTN_PV4=1", "-DATTN_QK4=1"])):  # 4-bit weights x values; both matmuls 4-bit
    module = load(name, ["native/attn.cu"], extra_include_paths=["/sage/csrc", "/sage/csrc/qattn"],
                  extra_cuda_cflags=["-O3", "-std=c++20", "-DENABLE_BF16", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "--use_fast_math",
                                     "-diag-suppress=174", *flags, "-gencode", "arch=compute_120a,code=sm_120a"],
                  build_directory=f"/tmp/{name}", verbose=False)
    shutil.copy(module.__file__, sysconfig.get_paths()["purelib"])
    print(name, "built")
PY
