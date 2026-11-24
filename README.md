# magpie

Lean PyTorch runtime for `Qwen/Qwen-Image-2.1-Turbo` (8 steps, 7.1B DiT for text-to-image and editing) on Blackwell GPUs (`sm_120`: RTX 5090, RTX PRO 6000), with block-scaled MXFP8/NVFP4 matmuls, a 4-bit attention kernel, and a quantised prefix cache.

| RTX 5090, request to PNG reply, text encoder included | edit 1024² | edit 2048² |
|---|---|---|
| **magpie, `fast`** | **1.26 s** | **5.00 s** |
| magpie, 8 steps | 1.49 s | 6.19 s |
| sglang 0.5.21, FP8 + SageAttention | 2.54 s | 12.07 s |
| LightX2V, text encoder offloaded | 4.54 s | 20.07 s |
| vLLM-Omni, text encoder offloaded | out of memory | out of memory |
| official diffusers pipeline, CPU offload (in-process) | 28.0 s | out of memory |

Median of five warm requests on one RTX 5090 (resident, default 4-bit text encoder, 1448² reference at 2048²). On an RTX PRO 6000, magpie takes 1.02 s and 4.76 s at 8 steps (prompt remembered). The transformer and VAE pass a 16-image quality gate against bf16 at both sizes (DINOv2 ≥ 0.98, DreamSim ≤ 0.05); the 4-bit encoder shifts one edit's composition past that line (DINOv2 0.94, VLM judge tie) and occasionally misspells requested lettering (`--encoder bf16` keeps the original). Measurements live in `baseline/rtx5090.json` and `baseline/compare.json`.

## setup

Requires a Blackwell GPU (`sm_120`), driver ≥ 580, CUDA 13, PyTorch 2.14, and ~31 GiB for `Qwen/Qwen-Image-2.1-Turbo` weights (downloaded on first start; transformer prepacked at [tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16](https://huggingface.co/tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16)). Without native kernels or off `sm_120`, magpie falls back to bf16 with PyTorch attention; for Ada (`sm_89`), see the [`ada` branch](https://github.com/kelechi-c/magpie/tree/ada).

```sh
git clone https://github.com/kelechi-c/magpie && cd magpie
pip install -e .
bash tools/build_native.sh
```

`tools/build_native.sh` runs once per machine (~7 min on an RTX 5090, needs CUDA 13 `nvcc` and C++20 `gcc`), caches SageAttention 2 and CUTLASS in `MAGPIE_DEPS` (`~/.cache/magpie`), and builds `magpie_gemm` (MXFP8/NVFP4 linears), `magpie_attn8` (< 2048 outputs), and `magpie_attn4` (2048-class outputs). Verify with:

```sh
python -c "import torch, magpie_gemm, magpie_attn4, magpie_attn8, sageattention"
```

## server

```sh
MAGPIE_TOKEN=<secret> python -m magpie.server --port 8188 [--encoder nvfp4|bf16] [--max-memory GIB]
```

Loads once and serves one request at a time:

```sh
curl -s localhost:8188/generate -H "Authorization: Bearer <secret>" -H "Content-Type: application/json" \
  -d '{"prompt": "Ink drawing of a magpie on a fence post", "seed": 1}' | python -c \
  "import sys, json, base64; open('out.png', 'wb').write(base64.b64decode(json.load(sys.stdin)['image']))"
```

`POST /generate` fields:

| field | default | notes |
|---|---|---|
| `prompt` | required | generation prompt or edit instruction |
| `images` | none | base64 images or data URLs to edit/combine |
| `resolution` | `1024` | `1024` or `2048` (side of equivalent-area square) |
| `width`, `height` | auto | exact output dimensions (rounded to 32); defaults to last image's shape |
| `reference` | `min(resolution, 1448)` | input side entering the model; `1448` passes the 2048² gate ~16% faster |
| `seed` | `0` | RNG seed |
| `format` | `png` | `png`, `jpeg`, or `webp` |
| `fast` | `false` | run 6 of 8 DiT steps (reusing velocity on steps 3 and 6); DiT + VAE 1024² 1,158 → 935 ms, 2048² 5,474 → 4,306 ms on RTX 5090 |
| `keep` | `false` | keep the images' KV cache on GPU for the next same-size edit (+4.6 / +11 GiB; DiT + VAE 1,158 → 1,023 ms / 5,474 → 5,247 ms on RTX 5090; DINOv2 median 0.998, worst 0.92 vs fresh) |

Replies return `image` (base64), `format`, `width`, `height`, `seed`, `peak_gib`, `mode`, `predicted_gib`, `remembered`, `kept`, `resumed`, and `ms` (`fetch`, `text`, `dit_vae`, `encode_image`). `GET /health` reports GPU, encoder, and reference limits.

On startup the server runs six quick probe edits to model VRAM usage and plans each request as `resident` (fits beside the encoder), `staged` (parks the encoder in host RAM during DiT/VAE when < 10 resident 2048² refs fit or `MAGPIE_PARK=1`), or HTTP `507` with card limits.

| env | default | purpose |
|---|---|---|
| `MAGPIE_TOKEN` | unset | require `Authorization: Bearer <token>` |
| `MAGPIE_ENCODER` | `nvfp4` on Blackwell | `nvfp4` or `bf16` (same as `--encoder`) |
| `MAGPIE_PARK` | unset | set `1` to allow encoder parking on large cards |
| `MAGPIE_PROMPTS` | `8` | cached `(prompt, images)` encoder outputs (`0` disables) |
| `HF_HOME` | HF default | Hugging Face weight cache directory |

## text encoder

Chosen at startup (not per request):

| | `nvfp4` (Blackwell default) | `bf16` (default elsewhere) |
|---|---|---|
| weights | calibrated 4-bit, 5.9 GiB | original, 15.2 GiB |
| RTX 5090 1024² / 2048² (1448² ref) | 1.31 s (14.7 GiB) / 5.63 s (18.2 GiB) | 1.32 s (24.0 GiB) / 5.71 s (27.5 GiB) |
| max resident refs on RTX 5090 (1024² / 2048²) | 11 / 6 | 4 / 2 |
| lettering | occasionally misspells or keeps old titles (2/6 test cases) | exact |

Select via `--encoder bf16`, `MAGPIE_ENCODER=bf16`, or `load("bf16")` when exact text rendering matters and VRAM allows.

## python

```python
from magpie.pipeline import generate
from magpie.sampling import TURBO
from magpie.server import load

models = load().models
drawn = generate(models, "Ink drawing of a magpie on a fence post", TURBO, size=(1024, 1024), seed=1)
edited = generate(models, "Add falling snow", TURBO, images=(drawn,), seed=2)
```

Use a different seed when editing a generated image; reusing the same seed yields harsh results.

## limits

- Per-reference cost on RTX 5090: ~0.4 s / 1.4 GiB per 1024² reference (up to 15 staged with `nvfp4`), ~1.2 s / 1.9 GiB per 1448² reference at 2048² (up to 8 staged). RTX PRO 6000 fits 10+ references at both sizes with either encoder.
- `magpie_attn4` runs only for 2048-class outputs; smaller outputs use 8-bit attention (`magpie_attn8`).
- Text-to-image preserves subject, style, and detail, though NVFP4 MLP weights can shift composition vs bf16 at the same seed (DINOv2 min 0.90).
- Tuned for Turbo; the 40-step base checkpoint runs through the same build and passes the gate (DiT + VAE 4.76 s at 1024², 27.90 s at 2048² on RTX 5090).

## layout

- `src/magpie/` — `dit.py`, `vae.py`, `text.py`, `sampling.py`, `pipeline.py`, `kernels.py`, `server.py`
- `native/` — CUDA sources for `magpie_gemm`, `magpie_attn8`, and `magpie_attn4`
- `baseline/` — measured benchmark and gate numbers

## license

No weights are in this repository. Qwen-Image 2.1 weights are under the Qwen Research License (non-commercial); do not redistribute them, and keep that notice with any derived checkpoint.
