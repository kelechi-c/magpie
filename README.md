# magpie

custom runtime for Qwen-Image 2.1 Turbo on Blackwell/sm_120 GPUs (like the 5090). 
Qwen-Image 2.1 Turbo (8 steps, one 7.1 B transformer for text-to-image and image editing) made fast on
Blackwell GPUs (sm_120: RTX 5090, RTX PRO 6000): a lean PyTorch port with block-scaled FP8 and NVFP4
matmuls, a 4-bit attention kernel written for the chip, and a prefix cache that stays quantised.

| whole request, text encoder included | edit 1024² | edit 2048² |
|---|---|---|
| **magpie, RTX 5090** | **1.31 s** | **7.14 s** |
| official diffusers pipeline, RTX 5090 (CPU offload) | 27.1 s | out of memory |
| magpie, RTX PRO 6000 | 1.10 s | 5.80 s |

Median of five warm requests. Both sizes pass a 16-image gate against the bf16 baseline (DINOv2 ≥ 0.98,
DreamSim ≤ 0.05). Numbers are in `baseline/rtx5090.json` and `baseline/compare.json`.

## requirements

- A Blackwell GPU with a driver ≥ 580, CUDA 13, PyTorch 2.14.
- About 31 GiB of weights from `Qwen/Qwen-Image-2.1-Turbo`, downloaded on first start. The transformer comes
  prepacked from [tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16](https://huggingface.co/tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16)
  (under the same license); without the native matmul kernels it loads the original bf16 transformer instead.
- The fast paths are native extensions built from `native/` on the machine that runs them (see below).
  Without them magpie falls back to its Triton and PyTorch paths: same images, slower.

## install

```
git clone https://github.com/kelechi-c/magpie && cd magpie
pip install -e .
```

## build the kernels

```
bash tools/build_native.sh
```

Run it once per machine, inside the Python environment magpie runs in. It needs `nvcc` from CUDA 13
(`CUDA_HOME`, default `/usr/local/cuda`) and gcc with C++20, and takes about ten minutes. It fetches
SageAttention 2 and CUTLASS at pinned commits into `MAGPIE_DEPS` (default `~/.cache/magpie`), installs
`sageattention`, and builds four extensions into the environment:

| extension | from | used for |
|---|---|---|
| `magpie_gemm` | `native/gemm.cu` on CUTLASS | block-scaled MXFP8 and NVFP4 matmuls in the transformer |
| `magpie_attn8` | `native/attn.cu` | 8-bit attention, everything below 2048-class outputs |
| `magpie_attn4` | `native/attn.cu` | 4-bit attention (both matmuls) for 2048-class outputs |
| `magpie_attn` | `native/attn.cu` | 4-bit weights × values only; switches the 4-bit path on, and runs it when `MAGPIE_QK4_KEYS` is raised |

Check (torch first: the extensions link against its libraries): `python -c "import torch, magpie_gemm, magpie_attn, magpie_attn4, magpie_attn8, sageattention"`.

## the server

```
MAGPIE_TOKEN=<secret> python -m magpie.server --port 8188
```

It loads the model once and takes one request at a time.

```
curl -s localhost:8188/generate -H "Authorization: Bearer <secret>" -H "Content-Type: application/json" \
  -d '{"prompt": "Ink drawing of a magpie on a fence post", "seed": 1}' | python -c \
  "import sys, json, base64; open('out.png', 'wb').write(base64.b64decode(json.load(sys.stdin)['image']))"
```

`POST /generate` takes JSON:

| field | | |
|---|---|---|
| `prompt` | required | the description, or the edit instruction |
| `images` | optional | base64 images (or data URLs) to edit or combine; none means text-to-image |
| `resolution` | 1024 | 1024 or 2048: the side of a square with the output's area |
| `width`, `height` | optional | exact output size, rounded to multiples of 32. Without them an edit takes the last image's shape |
| `seed` | 0 | |
| `format` | `png` | `png`, `jpeg` or `webp` |

The reply holds `image` (base64), `width`, `height`, `seed`, `peak_gib` and `ms` (`fetch`, `text`,
`dit_vae`, `encode_image`). Out of memory comes back as status 507. `GET /health` reports the card.

On a card under 40 GiB the text encoder steps aside to host memory when a request is too large to share
the card with it (any 2048² request, or more than three 1024² references), which costs about 0.4 s on
that request and the next. It needs about 16 GiB of free host memory.

| environment | |
|---|---|
| `MAGPIE_TOKEN` | when set, requests must carry `Authorization: Bearer <token>` |
| `MAGPIE_ENCODER=nvfp4` | a calibrated 4-bit text encoder: 9 GiB less on the card, slightly different images |
| `MAGPIE_PARK=1` | let the encoder step aside on a large card too |
| `HF_HOME` | where weights are cached |

## from python

```python
from PIL import Image
from magpie.pipeline import generate
from magpie.sampling import TURBO
from magpie.server import load

models = load().models
drawn = generate(models, "Ink drawing of a magpie on a fence post", TURBO, size=(1024, 1024), seed=1)
edited = generate(models, "Add falling snow", TURBO, images=(drawn,), seed=2)
```

Edit a generated image with a different seed from the one that drew it; the same noise gives a harsh result.

## limits

- An RTX 5090 fits one reference at 2048² and up to eleven at 1024² (three with the encoder resident).
- 4-bit attention is used only for 2048-class outputs; everything else runs 8-bit.
- Turbo only. The 40-step base checkpoint runs in the port but is not tuned or gated.

## layout

- `src/magpie/` — `dit.py`, `vae.py`, `text.py`, `sampling.py`, `pipeline.py`, `kernels.py`, `server.py`
- `native/` — CUDA sources of the matmul and attention kernels
- `baseline/` — measured numbers behind the charts in `assets/charts/`

## license

No weights are in this repository. Qwen-Image 2.1 weights are under the Qwen Research License
(non-commercial); do not redistribute them, and keep that notice with any derived checkpoint.
