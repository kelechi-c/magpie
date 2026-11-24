# magpie

Lean PyTorch runtime for `Qwen/Qwen-Image-2.1-Turbo` (8 steps, 7.1B DiT for text-to-image and editing) on Blackwell GPUs (`sm_120`: RTX 5090, RTX PRO 6000) using block-scaled MXFP8/NVFP4 matmuls, 4-bit attention, and a quantised prefix cache.

| RTX 5090, request to PNG reply, text encoder included | edit 1024² | edit 2048² |
|---|---|---|
| **magpie, `fast`** | **1.26 s** | **5.00 s** |
| magpie, 8 steps | 1.49 s | 6.19 s |
| sglang 0.5.21, FP8 + SageAttention | 2.54 s | 12.07 s |
| LightX2V, text encoder offloaded | 4.54 s | 20.07 s |
| vLLM-Omni, text encoder offloaded | out of memory | out of memory |
| official diffusers pipeline, CPU offload (in-process) | 28.0 s | out of memory |

Median of 5 warm runs on an RTX 5090 (resident 4-bit encoder; 1448² reference at 2048²). On an RTX PRO 6000, magpie takes 1.02 s and 4.76 s at 8 steps (prompt remembered). DiT and VAE pass a 16-image gate vs bf16 at both sizes (DINOv2 ≥ 0.98, DreamSim ≤ 0.05); see `baseline/rtx5090.json` and `baseline/compare.json`.

## setup

Requires a Blackwell GPU (`sm_120`), driver ≥ 580, CUDA 13, PyTorch 2.14, and ~31 GiB for `Qwen/Qwen-Image-2.1-Turbo` (downloaded on first load; prepacked DiT at [tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16](https://huggingface.co/tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16)). Falls back to bf16 + PyTorch SDPA without native kernels or off `sm_120`; see the [`ada` branch](https://github.com/kelechi-c/magpie/tree/ada) for `sm_89`.

```sh
git clone https://github.com/kelechi-c/magpie && cd magpie
pip install -e .
bash tools/build_native.sh
```

`tools/build_native.sh` (~7 min on an RTX 5090; needs CUDA 13 `nvcc` and C++20 `gcc`) caches SageAttention 2 and CUTLASS under `MAGPIE_DEPS` (`~/.cache/magpie`) and builds `magpie_gemm`, `magpie_attn8` (< 2048), and `magpie_attn4` (2048-class). Verify with `python -c "import torch, magpie_gemm, magpie_attn4, magpie_attn8, sageattention"`.

## usage

```python
from PIL import Image
from magpie.pipeline import generate
from magpie.sampling import TURBO, TURBO_FAST
from magpie.server import load

models = load().models  # or load("bf16") for exact lettering

# text-to-image
drawn = generate(models, "Ink drawing of a magpie on a fence post", TURBO, size=(1024, 1024), seed=1)
drawn.save("drawn.png")

# image editing (single or multiple reference images)
photo = Image.open("input.png")
edited = generate(models, "Add falling snow", TURBO_FAST, images=(photo,), resolution=2048, seed=2)
edited.save("edited.png")
```

`TURBO_FAST` runs 6 of 8 DiT steps (reusing velocity on steps 3 and 6). When editing a generated image, pass a different `seed`—reusing the same seed yields harsh results.

## text encoder

| | `nvfp4` (Blackwell default) | `bf16` (default elsewhere) |
|---|---|---|
| weights | 5.9 GiB (calibrated 4-bit) | 15.2 GiB |
| RTX 5090 1024² / 2048² (1448² ref) | 1.31 s (14.7 GiB) / 5.63 s (18.2 GiB) | 1.32 s (24.0 GiB) / 5.71 s (27.5 GiB) |
| max resident refs on 5090 (1024² / 2048²) | 11 / 6 | 4 / 2 |
| tradeoffs | 1/16 edit composition shift (DINOv2 0.94, VLM tie); occasional lettering errors (2/6 cases) | exact |

Pass `load("bf16")` or set `MAGPIE_ENCODER=bf16` when exact text rendering matters and VRAM allows.

## limits

- **References:** ~0.4 s / 1.4 GiB per 1024² ref and ~1.2 s / 1.9 GiB per 1448² ref at 2048² on RTX 5090 (RTX PRO 6000 fits 10+ at both sizes).
- **Attention:** `magpie_attn4` runs only at 2048-class outputs; smaller outputs use 8-bit (`magpie_attn8`).
- **Text-to-image:** preserves subject, style, and detail, though NVFP4 MLPs can shift composition vs bf16 for the same seed (DINOv2 min 0.90).
- **Base model:** tuned for Turbo; the 40-step base checkpoint passes the gate (DiT + VAE 4.76 s at 1024², 27.90 s at 2048² on RTX 5090).

## layout

- `src/magpie/` — `dit.py`, `vae.py`, `text.py`, `sampling.py`, `pipeline.py`, `kernels.py`, `server.py`
- `native/` — CUDA sources for `magpie_gemm`, `magpie_attn8`, and `magpie_attn4`
- `baseline/` — measured benchmark and gate numbers

## license

No weights are in this repository. Qwen-Image 2.1 weights are under the Qwen Research License (non-commercial); do not redistribute them, and keep that notice with any derived checkpoint.
