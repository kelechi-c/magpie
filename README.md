# magpie

custom runtime for Qwen-Image 2.1 Turbo on Blackwell/sm_120 GPUs (like the 5090). 
Qwen-Image 2.1 Turbo (8 steps, one 7.1 B transformer for text-to-image and image editing) made fast on
Blackwell GPUs (sm_120: RTX 5090, RTX PRO 6000): a lean PyTorch port with block-scaled FP8 and NVFP4
matmuls, a 4-bit attention kernel written for the chip, and a prefix cache that stays quantised.

| whole request, text encoder included | edit 1024² | edit 2048² |
|---|---|---|
| **magpie, RTX 5090** | **1.27 s** | **5.35 s** |
| official diffusers pipeline, RTX 5090 (CPU offload) | 27.1 s | out of memory |
| magpie, RTX PRO 6000 (bf16 text encoder) | 1.10 s | 5.80 s |

Median of five warm requests, everything resident, with the default calibrated 4-bit text encoder and, for 2048², a
1448² reference. The transformer and VAE pass a 16-image gate against the bf16 baseline at both sizes (DINOv2 ≥ 0.98,
DreamSim ≤ 0.05). The 4-bit encoder moves one of the sixteen edits' composition past that line (DINOv2 0.94; a VLM
judge calls it a tie) and now and then gets requested lettering wrong; `--encoder bf16` keeps the original (see
[the text encoder](#the-text-encoder)). Numbers are in `baseline/rtx5090.json` and `baseline/compare.json`.

## requirements

- A Blackwell GPU (sm_120) with a driver ≥ 580, CUDA 13, PyTorch 2.14.
- About 31 GiB of weights from `Qwen/Qwen-Image-2.1-Turbo`, downloaded on first start. The transformer comes
  prepacked from [tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16](https://huggingface.co/tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16)
  (under the same license); without the native matmul kernels it loads the original bf16 transformer instead.
- The fast paths are native extensions built from `native/` on the machine that runs them (see below).

**This is the `ada` branch:** it keeps the FP8 (Triton) linears for Ada cards (sm_89: RTX 4090, L40S, L4) and the paths below. `master` is Blackwell-only.

magpie reads the GPU's architecture when it starts and picks what that card can run:

| GPU | transformer linears | attention | text encoder |
|---|---|---|---|
| Blackwell, sm_120, kernels built | MXFP8 and NVFP4 (CUTLASS) | 8-bit, 4-bit for 2048² | calibrated NVFP4 |
| Blackwell, kernels not built | FP8 (Triton) | SageAttention 2 if installed | bf16 |
| Ada, sm_89 (RTX 4090, L40S, L4) | FP8 (Triton) | SageAttention 2 if installed | bf16 |
| Ampere and older (RTX 3090, A100) | bf16 | Triton / PyTorch | bf16 |

Off Blackwell the images are not the gated ones (other kernels, other rounding) and nothing there is tuned.

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
`sageattention`, and builds three extensions into the environment:

| extension | from | used for |
|---|---|---|
| `magpie_gemm` | `native/gemm.cu` on CUTLASS | block-scaled MXFP8 and NVFP4 matmuls in the transformer |
| `magpie_attn8` | `native/attn.cu` | 8-bit attention, everything below 2048-class outputs |
| `magpie_attn4` | `native/attn.cu` | 4-bit attention (both matmuls) for 2048-class outputs |

Check (torch first: the extensions link against its libraries): `python -c "import torch, magpie_gemm, magpie_attn4, magpie_attn8, sageattention"`.

## the server

```
MAGPIE_TOKEN=<secret> python -m magpie.server --port 8188 [--encoder nvfp4|bf16] [--max-memory GIB]
```

It loads the model once and takes one request at a time. `--encoder nvfp4|bf16` picks the text encoder (see below);
`--max-memory` caps the GiB of the card it plans with and allocates.

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
| `reference` | `resolution`, at most 1448 | the side the images enter the model at. 1448 for a 2048² edit passes the quality gate, 16 % sooner than a full-size reference |
| `seed` | 0 | |
| `format` | `png` | `png`, `jpeg` or `webp` |

The reply holds `image` (base64), `width`, `height`, `seed`, `peak_gib`, `mode`, `predicted_gib`, `remembered` and `ms`
(`fetch`, `text`, `dit_vae`, `encode_image`). The server keeps the text encoder's output for its last eight prompt and
image pairs, so a repeat with another seed or size skips the encoder (`remembered` is then true). `GET /health` reports
the card, the text encoder and the reference limits.

At start the server measures the card, as vLLM does before it sizes its cache: six small edits (1024² and 2048², with
one, two and four references) give it the memory each request needs. It then prints what the card takes:

```
probe 1024² edit, 1024² references: dit + vae 8.78, 10.18, 12.97 gib with 1, 2, 4
probe 2048² edit, 1448² references: dit + vae 12.27, 13.74, 17.54 gib with 1, 2, 4
probed in 31.2 s: 30.2 gib usable, encoder 5.9 gib, parks when a request needs the room
most references, 1024² edit: 12 resident, 16 staged
most references, 2048² edit: 6 resident, 8 staged
```

Every request is planned from that: resident when it fits beside the text encoder; staged when it fits only without
it, with the encoder parked in host memory while the transformer runs (about 0.1–0.6 s, on that request and the next);
refused with status 507 and the limits when it fits neither way. The encoder can park only on a card that cannot hold
ten 2048² references beside it (or with `MAGPIE_PARK=1`), and parking needs free host memory the size of the encoder.

| environment | |
|---|---|
| `MAGPIE_TOKEN` | when set, requests must carry `Authorization: Bearer <token>` |
| `MAGPIE_ENCODER` | `nvfp4` (default where `magpie_gemm` is built) or `bf16`: the text encoder, as `--encoder` |
| `MAGPIE_PARK=1` | let the encoder park on a large card too |
| `MAGPIE_PROMPTS` | how many prompt and image pairs to remember (8; 0 turns it off) |
| `HF_HOME` | where weights are cached |

## the text encoder

Two text encoders, chosen when the model loads (not per request):

| | `nvfp4` (default on Blackwell) | `bf16` (default elsewhere) |
|---|---|---|
| what | Qwen's encoder with calibrated 4-bit weights, 5.9 GiB | the original, 15.2 GiB |
| edit 1024², RTX 5090 | 1.31 s, 14.7 GiB | 1.32 s, 24.0 GiB |
| edit 2048² (1448² reference), RTX 5090 | 5.63 s, 18.2 GiB | 5.71 s, 27.5 GiB |
| most references, RTX 5090 (1024² / 2048²) | 11 / 6 | 4 / 2 |
| lettering | now and then misspells a requested title or leaves the old one (2 of 6 test renders that bf16 got right) | |

Whole request, median of five warm runs, both encoders on the same 575 W RTX 5090 (the headline's card ran a few percent
faster).

Pick it with any of:

| where | |
|---|---|
| server | `python -m magpie.server --encoder bf16` |
| environment | `MAGPIE_ENCODER=bf16` (also the default for `load()` and the server) |
| Python | `load("bf16")` |

Use `bf16` when an edit must render exact text and the card has the room; pictures without lettering look the same
with either.

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

- A 2048² edit on an RTX 5090 stays fully resident: 5.35 s and 18.2 GiB (1448² reference), 6.45 s and 19.7 GiB
  with a 2048² reference; with the bf16 encoder, 5.41 s and 27.5 GiB.
- References per edit, as Qwen suggests up to ten: an RTX PRO 6000 takes 10+ at both sizes with either encoder. An
  RTX 5090 takes eleven at 1024² and six at 2048² (1448² references) with the 4-bit encoder resident, four and two
  with the bf16 encoder, thirteen and seven with the encoder off the card. Each 1024² reference adds about 0.5 s and
  1.4 GiB to a 1024² edit; each 1448² reference about 1.2 s and 1.9 GiB to a 2048² edit.
- 4-bit attention is used only for 2048-class outputs; everything else runs 8-bit.
- Text-to-image keeps the subject, style and detail, but its composition can differ from the bf16 picture for the
  same seed (DINOv2 min 0.90 at 1024² and 2048²; the NVFP4 MLP weights shift it). It is judged on quality, not on
  matching that picture.
- Turbo only. The 40-step base checkpoint runs in the port but is not tuned or gated.

## layout

- `src/magpie/` — `dit.py`, `vae.py`, `text.py`, `sampling.py`, `pipeline.py`, `kernels.py`, `server.py`
- `native/` — CUDA sources of the matmul and attention kernels
- `baseline/` — measured numbers behind the article's charts

## license

No weights are in this repository. Qwen-Image 2.1 weights are under the Qwen Research License
(non-commercial); do not redistribute them, and keep that notice with any derived checkpoint.
