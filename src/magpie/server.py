"""http api for magpie: one qwen-image 2.1 turbo model on one gpu, one request at a time.

    python -m magpie.server [--host 0.0.0.0] [--port 8188]

    GET  /health    -> {"gpu", "vram_gib", "allocated_gib", "busy", "parks"}
    POST /generate  <- {"prompt", "images": [base64, ...], "width", "height", "resolution", "reference", "seed", "format"}
                    -> {"image": base64, "format", "width", "height", "seed", "ms": {...}, "peak_gib"}

only `prompt` is required. with images and no width/height the output takes the last image's shape at `resolution`
(1024 or 2048, the side of a square of the same area); with neither it is a square. `reference` (default: `resolution`)
is the side the images enter the model at: 1024 for a 2048 output shortens its sequence by a third. MAGPIE_TOKEN, when set, must
arrive as `Authorization: Bearer <token>`. weights come from the hugging face cache (downloaded on first start).
"""

import argparse
import base64
import io
import json
import os
import pathlib as pl
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Literal

import torch
from PIL import Image

from magpie import pipeline
from magpie.kernels import MX, NV
from magpie.sampling import TURBO
from magpie.text import encode, load_encoder, reslot

REPO = "Qwen/Qwen-Image-2.1-Turbo"
PACKED_REPO = "tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16"  # the transformer prepacked; qwen research license, non-commercial
GIB = 2**30
LOOK = 1024  # the encoder reads condition images no larger than this
SMALL_CARD_GIB = 40  # below this the encoder cannot stay beside the dit for every request
RESIDENT_PIXELS = 4 * 1024**2  # output + reference pixels a 32 gb card holds with the encoder resident (measured: 3 references)
Format = Literal["png", "jpeg", "webp"]
Job = dict[str, Any]


# the loaded models and how to move the encoder: `fetch` brings it onto the gpu, `park` leaves only its
# pinned host copy. on a large card both do nothing.
@dataclass(frozen=True)
class Runtime:
    models: pipeline.Models
    fetch: Callable[[], None]
    park: Callable[[], None]
    parks: bool


# a `packed_state` file as written by the packer: fp8 tensors are stored as bytes under "<name>@f8".
def from_disk(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name.removesuffix("@f8"): (tensor.view(torch.float8_e4m3fn) if name.endswith("@f8") else tensor) for name, tensor in state.items()}


# the other direction, for a dit packed here.
def to_disk(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {(f"{name}@f8" if tensor.dtype == torch.float8_e4m3fn else name): (tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor).cpu()
            for name, tensor in state.items()}


# where the prepacked hybrid dit for this build of the kernels is kept.
def packed_path() -> pl.Path:
    return pl.Path(os.environ.get("HF_HOME", pl.Path.home() / ".cache/huggingface")) / "magpie" / f"turbo-nv_{'_'.join(sorted(NV))}-fp8_q_k_v_out.safetensors"


# whether that file is there, downloading it from PACKED_REPO when it is not. false (no native matmul, or the
# download failed) means the original transformer has to be packed here.
def prepacked(local: pl.Path) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    if local.exists() or not MX:
        return MX
    try:
        hf_hub_download(PACKED_REPO, local.name, local_dir=local.parent)
    except (HfHubHTTPError, OSError):
        return False
    return True


# (fetch, park) for an encoder kept in pinned host memory: fetch copies it to the gpu, park drops that copy.
def parking(encoder: Any) -> tuple[Callable[[], None], Callable[[], None]]:
    host = {name: weight.cpu().pin_memory() for name, weight in encoder.state_dict().items()}

    def park() -> None:
        encoder.load_state_dict(host, assign=True)
        torch.cuda.empty_cache()

    def fetch() -> None:
        encoder.load_state_dict({name: weight.to("cuda", non_blocking=True) for name, weight in host.items()}, assign=True)

    return fetch, park


# downloads what is missing and puts dit, vae and encoder on the gpu. the dit comes prepacked when it can (and is
# packed and saved here when it cannot);
# the encoder gets a pinned host copy on cards under SMALL_CARD_GIB (or with MAGPIE_PARK=1) so it can step aside.
def load() -> Runtime:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    torch.set_grad_enabled(False)
    local = packed_path()
    ready = prepacked(local)
    parts = ["model_index.json", "vae/*", "text_encoder/*", "processor/*"] + ([] if ready else ["transformer/*"])
    snapshot = pl.Path(snapshot_download(REPO, allow_patterns=parts))

    def read(part: str) -> dict[str, torch.Tensor]:
        return {name: weight.to("cuda", torch.bfloat16) for file in sorted((snapshot / part).glob("*.safetensors")) for name, weight in load_file(file).items()}

    config = json.loads((snapshot / "vae" / "config.json").read_text())
    mean, std = config["latents_mean"], config["latents_std"]
    dit, vae = pipeline.adopt_packed(from_disk(load_file(local, device="cuda")), read("vae"), mean, std) if ready else pipeline.adopt(read("transformer"), read("vae"), mean, std, slim=MX)
    if MX and not ready:  # packed here: keep it, so the next start reads one file and skips the bf16 transformer
        local.parent.mkdir(parents=True, exist_ok=True)
        save_file(to_disk(pipeline.packed_state(dit)), local)
    encoder, processor = load_encoder(snapshot)
    parks = os.environ.get("MAGPIE_PARK") == "1" or torch.cuda.get_device_properties(0).total_memory < SMALL_CARD_GIB * GIB
    fetch, park = parking(encoder) if parks else (lambda: None, lambda: None)
    return Runtime(pipeline.Models(encoder, processor, dit, vae), fetch, park, parks)


def decoded(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data.split(",")[-1])))


def encoded(image: Image.Image, fmt: Format) -> str:
    buffer = io.BytesIO()
    (image if fmt == "png" else image.convert("RGB")).save(buffer, fmt, **({} if fmt == "png" else {"quality": 95}))
    return base64.b64encode(buffer.getvalue()).decode()


def elapsed(start: float) -> float:
    torch.cuda.synchronize()
    return round((time.perf_counter() - start) * 1e3, 1)


# one request, as `pipeline.generate` runs it but in two timed stages, so the encoder can leave the gpu
# between them when the image is too large to share the card with it. returns the json reply.
@torch.no_grad()
def generate(runtime: Runtime, job: Job, parked: bool) -> tuple[Job, bool]:
    models, resolution = runtime.models, int(job.get("resolution", 1024))
    reference = int(job.get("reference") or resolution)
    images = tuple(decoded(data) for data in job.get("images", []))
    sized = pipeline.resized(images, reference)
    size = (round(job["width"] / 32) * 32, round(job["height"] / 32) * 32) if job.get("width") and job.get("height") else None
    width, height = size or (pipeline.fit(resolution**2, images[-1].width / images[-1].height) if images else (resolution,) * 2)
    crowded = runtime.parks and width * height + sum(image.width * image.height for image in sized) > RESIDENT_PIXELS
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if parked:
        runtime.fetch()
    fetch_ms, start = elapsed(start), time.perf_counter()
    small = reference > LOOK
    text, slots = encode(models.encoder, models.processor, job["prompt"], pipeline.resized(images, LOOK) if small else sized)
    if small:
        text, slots = reslot(text, slots, [image.width * image.height // 1024 for image in sized])
    text_ms, start = elapsed(start), time.perf_counter()
    if crowded:
        runtime.park()
    image = pipeline.render(models.dit, models.vae, text, slots, TURBO, sized, (width, height), resolution, int(job.get("seed", 0)))
    render_ms, start = elapsed(start), time.perf_counter()
    fmt: Format = job.get("format", "png")
    reply = {"image": encoded(image, fmt), "format": fmt, "width": image.width, "height": image.height, "seed": int(job.get("seed", 0)),
             "peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 2)}
    return {**reply, "ms": {"fetch": fetch_ms, "text": text_ms, "dit_vae": render_ms, "encode_image": round((time.perf_counter() - start) * 1e3, 1)}}, crowded


# the request handler for one runtime. the gpu takes one job at a time (`lock`); `parked` remembers whether
# the last job left the encoder off the card.
def handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    lock, parked, token = threading.Lock(), [False], os.environ.get("MAGPIE_TOKEN")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: a proxy in front reuses its connection, and http/1.0 would close it under it

        def reply(self, status: int, body: Job) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            for name, value in (("Content-Type", "application/json"), ("Content-Length", str(len(data))), ("Access-Control-Allow-Origin", "*"),
                                ("Access-Control-Allow-Headers", "Authorization, Content-Type"), ("Access-Control-Allow-Methods", "GET, POST, OPTIONS")):
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def allowed(self) -> bool:
            return not token or self.headers.get("Authorization") == f"Bearer {token}"

        def do_OPTIONS(self) -> None:
            self.reply(200, {})

        def do_GET(self) -> None:
            props = torch.cuda.get_device_properties(0)
            health = {"gpu": props.name, "vram_gib": round(props.total_memory / GIB, 1), "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 2),
                      "busy": lock.locked(), "parks": runtime.parks}
            self.reply(*((200, health) if self.path == "/health" and self.allowed() else (401 if self.path == "/health" else 404, {"error": "not found or not allowed"})))

        def do_POST(self) -> None:
            if self.path != "/generate" or not self.allowed():
                return self.reply(404 if self.allowed() else 401, {"error": "not found or not allowed"})
            job = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not isinstance(job.get("prompt"), str) or not job["prompt"].strip():
                return self.reply(400, {"error": "prompt is required"})
            with lock:
                try:
                    body, parked[0] = generate(runtime, job, parked[0])
                except torch.OutOfMemoryError as error:  # the one failure a caller can act on: ask for less
                    message = str(error)[:200]
                    del error
                    runtime.park()
                    parked[0] = runtime.parks
                    return self.reply(507, {"error": f"out of gpu memory: {message}"})
            print(f"{body['width']}x{body['height']} {len(job.get('images', []))} images {body['ms']} peak {body['peak_gib']} gib", flush=True)
            self.reply(200, body)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()
    start = time.perf_counter()
    runtime = load()
    print(f"loaded in {time.perf_counter() - start:.1f} s, encoder {'parks when crowded' if runtime.parks else 'resident'}; listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler(runtime)).serve_forever()


if __name__ == "__main__":
    main()
