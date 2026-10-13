#!/usr/bin/env bash
# builds the native kernels on the machine that will run them (needs nvcc >= 12.8 and the torch they link against).
#   bash tools/build_native.sh        sageattention 2, pinned, for sm_120 (torch 2.14 headers need c++20)
set -euo pipefail
SAGE=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5  # thu-ml/SageAttention, apache-2.0
test -d /sage || git clone -q https://github.com/thu-ml/SageAttention /sage
cd /sage && git checkout -q $SAGE && sed -i 's/-std=c++17/-std=c++20/g' setup.py
CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 pip install -q . --no-build-isolation
cd / && python -c "import sageattention; print('sageattention built')"
