#!/usr/bin/env bash
# builds magpie's native kernels on the machine that will run them, into the python environment that runs this script:
#   sageattention 2 (pinned commit, sm_120 only)
#   magpie_gemm  - native/gemm.cu on cutlass: the block-scaled mxfp8 and nvfp4 matmuls
#   magpie_attn  - native/attn.cu on sageattention's headers, 4-bit weights x values (2048-class outputs)
#   magpie_attn4 - the same with 4-bit q x k too
#   magpie_attn8 - 8-bit, weights x values on the block-scaled instruction (everything else)
# needs a blackwell sm_120 card's toolchain: nvcc from cuda 13 (CUDA_HOME, default /usr/local/cuda), gcc with c++20, and the
# torch magpie runs on. sources and build files go under MAGPIE_DEPS (default ~/.cache/magpie). about 10 minutes.
#   bash tools/build_native.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEPS=${MAGPIE_DEPS:-$HOME/.cache/magpie}
SAGE=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5  # thu-ml/SageAttention, apache-2.0
CUTLASS=ee98dbd  # NVIDIA/cutlass, bsd-3-clause
export CC=gcc CXX=g++ CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} DEPS
mkdir -p "$DEPS"
test -d "$DEPS/sage" || git clone -q https://github.com/thu-ml/SageAttention "$DEPS/sage"
git -C "$DEPS/sage" checkout -q -f $SAGE && sed -i 's/-std=c++17/-std=c++20/g' "$DEPS/sage/setup.py"  # torch 2.14 headers need c++20
(cd / && python -c "import sageattention" 2>/dev/null) || (cd "$DEPS/sage" && TORCH_CUDA_ARCH_LIST=12.0 python -m pip install -q . --no-build-isolation)
test -d "$DEPS/cutlass" || git clone -q https://github.com/NVIDIA/cutlass.git "$DEPS/cutlass"
git -C "$DEPS/cutlass" checkout -q $CUTLASS
cd "$ROOT" && python - <<'PY'
import os, shutil, sysconfig
from torch.utils.cpp_extension import load
deps = os.environ["DEPS"]
arch = ["-gencode", "arch=compute_120a,code=sm_120a"]
build = lambda name: os.makedirs(f"{deps}/build/{name}", exist_ok=True) or f"{deps}/build/{name}"
module = load("magpie_gemm", ["native/gemm.cu"], extra_include_paths=[f"{deps}/cutlass/include", f"{deps}/cutlass/tools/util/include"],
              extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "--expt-relaxed-constexpr", *arch], build_directory=build("magpie_gemm"), verbose=False)
shutil.copy(module.__file__, sysconfig.get_paths()["purelib"])
print("magpie_gemm built", module.mxfp8_slots(4096, 4096), module.nvfp4_slots(4096, 4096))
for name, flags in (("magpie_attn", ["-DATTN_PV4=1"]), ("magpie_attn4", ["-DATTN_PV4=1", "-DATTN_QK4=1"]), ("magpie_attn8", ["-DATTN_PVX=1"])):
    module = load(name, ["native/attn.cu"], extra_include_paths=[f"{deps}/sage/csrc", f"{deps}/sage/csrc/qattn"],
                  extra_cuda_cflags=["-O3", "-std=c++20", "-DENABLE_BF16", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "--use_fast_math",
                                     "-diag-suppress=174", *flags, *arch], build_directory=build(name), verbose=False)
    shutil.copy(module.__file__, sysconfig.get_paths()["purelib"])
    print(name, "built")
PY
