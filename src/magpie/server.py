"""http api for magpie: one qwen-image 2.1 turbo model on one gpu, one request at a time.

    python -m magpie.server [--host 0.0.0.0] [--port 8188] [--encoder nvfp4|bf16] [--max-memory GIB]

    GET  /health    -> {"gpu", "vram_gib", "allocated_gib", "busy", "parks", "encoder", "limits"}
    POST /generate  <- {"prompt", "images": [base64, ...], "width", "height", "resolution", "reference", "seed", "format", "fast", "keep"}
                    -> {"image": base64, "format", "width", "height", "seed", "ms": {...}, "peak_gib", "mode", "remembered", "kept", "resumed"}

only `prompt` is required. with images and no width/height the output takes the last image's shape at `resolution`
(1024 or 2048, the side of a square of the same area); with neither it is a square. `reference` (default: `resolution`,
at most 1448) is the side the images enter the model at. `fast` runs 6 of the 8 transformer steps (steps 3 and 6
reuse the previous step's velocity). `keep` holds the images' kv cache on the card after the edit (~4.6 gib more at
1024 with one reference, ~11 at 2048), so the next `keep` edit of the same images at the same size skips their encode
and their rows of the first step (~115 ms at 1024, ~250 at 2048); any other request drops it. MAGPIE_TOKEN, when set,
must arrive as `Authorization: Bearer <token>`. weights come from the hugging face cache (downloaded on first start). the text
encoder's output for the last MAGPIE_PROMPTS (prompt, images) pairs is kept, so a repeat (another seed or size) skips it.
at start the server runs six small probe edits to learn this card's memory (see `probe`); every request is then planned
from them: resident, with the encoder parked in host memory while the dit runs (staged), or refused with the limit.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # set before torch loads: without it reserved memory runs ~1.2 gib past what is allocated, and the plan budgets allocated

import argparse
import base64
import hashlib
import io
import json
import math
import pathlib as pl
import threading
import time
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Literal

import torch
from PIL import Image

from magpie import pipeline
from magpie.kernels import MX, NV
from magpie.sampling import TURBO, TURBO_FAST
from magpie.text import ENCODER, Encoder, calibrated_folder, encode, load_encoder, reslot

REPO = "Qwen/Qwen-Image-2.1-Turbo"
PACKED_REPO = "tensorkelechi/magpie-qwen-image-2.1-turbo-hybrid-nvfp4-mxfp8-bf16"  # the transformer prepacked; qwen research license, non-commercial
GIB = 2**30
LOOK = 1024  # the encoder reads condition images no larger than this
PROBES = ((1024, 1024), (2048, pipeline.REFERENCE))  # (output side, reference side) of the startup probes, one and two references each
REFERENCE_CAP = 10  # a card that cannot hold this many references at 2048 resident keeps a pinned host copy of the encoder, to park it
PROMPTS = int(os.environ.get("MAGPIE_PROMPTS", "8"))  # conditionings kept for repeated requests
# what a resumed `keep` request adds over a plain one, gib per reference token and per output token: the kept bf16 keys
# and values of the reference rows, and the refitted cache built while the old one is still held. fitted to the pro
# 6000's resumed peaks: +4.58 gib at 1024 with one 1024 reference, +11.28 at 2048 with one 1448.
KEPT_GIB = (0.87 / 1024, 0.28 / 1024)  # rounded up: plans 4.60 and 11.36 gib
Format = Literal["png", "jpeg", "webp"]
Job = dict[str, Any]
Remembered = dict[tuple, tuple[torch.Tensor, torch.Tensor]]  # (prompt, image sizes and digests) -> the encoder's (text, slots), oldest first


# what the server plans memory with, in gib: what the allocator may hold, what the encoder holds on the gpu, and per probe
# output size (output pixels, reference pixels, and the dit + vae peak without the encoder as a quadratic a + b n + c n^2
# in n, the reference pixels in units of that size's reference).
@dataclass(frozen=True)
class Budget:
    capacity: float = math.inf
    encoder: float = 0.0
    fits: tuple[tuple[int, int, tuple[float, float, float]], ...] = ()


# the loaded models, how to move the encoder (`fetch` brings it onto the gpu, `park` leaves only its pinned host copy;
# both do nothing when `parks` is false) and the card's budget.
@dataclass(frozen=True)
class Runtime:
    models: pipeline.Models
    fetch: Callable[[], None]
    park: Callable[[], None]
    parks: bool
    encoder: Encoder
    budget: Budget = Budget()


# the dit + vae peak (gib, encoder not counted) of an edit with `output` pixels and `references` reference pixels in all:
# each probe size's quadratic, its coefficients interpolated in output pixels and extended past 2048 the same way. the
# peak grows faster than the reference count (2048 at 1448 references, measured: +1.5 gib from one to two, +2.0 from five
# to six), so a straight line would plan edits that do not fit. with no probes, nothing is predicted.
def predict(budget: Budget, output: int, references: int) -> float:
    if not budget.fits:
        return 0.0
    (low, low_side, low_fit), (high, high_side, high_fit) = budget.fits
    t = max((output - low) / (high - low), 0.0)
    mix = lambda a, b: a + t * (b - a) if t else a
    a, b, c = (mix(x, y) for x, y in zip(low_fit, high_fit))
    n = references / mix(low_side, high_side)
    return a + b * n + c * n * n


COUNTS = (1, 2, 4)  # references in each size's probes


# a + b n + c n^2 through the peaks at n = COUNTS (divided differences; inf where a probe ran out of memory).
def quadratic(peaks: list[float]) -> tuple[float, float, float]:
    if not all(map(math.isfinite, peaks)):
        return (peaks[0], math.inf, math.inf) if math.isfinite(peaks[0]) else (math.inf, 0.0, 0.0)
    (n1, n2, n3), (p1, p2, p3) = COUNTS, peaks
    first, second = (p2 - p1) / (n2 - n1), (p3 - p2) / (n3 - n2)
    c = max((second - first) / (n3 - n1), 0.0)
    b = first - c * (n1 + n2)
    return p1 - b * n1 - c * n1 * n1, b, c


# how a request runs: "resident" (the encoder stays), "staged" (it is parked after the text stage) or "refused".
Mode = Literal["resident", "staged", "refused"]


# resident when dit + vae fit beside the encoder, staged when they fit alone and the encoder can park; returns the
# mode and the predicted peak of the whole request in it.
def plan(runtime: Runtime, output: int, references: int) -> tuple[Mode, float]:
    budget = runtime.budget
    need = predict(budget, output, references)
    if need + budget.encoder <= budget.capacity:
        return "resident", need + budget.encoder
    return ("staged" if runtime.parks and need <= budget.capacity else "refused"), need


# the most square references of `side` a square edit of `output` side takes, resident and staged (staged only where it parks).
def limits(runtime: Runtime, output: int, side: int, cap: int = 99) -> tuple[int, int]:
    budget = runtime.budget
    fits = lambda n, room: predict(budget, output**2, n * side**2) <= room
    return (max([n for n in range(1, cap + 1) if fits(n, budget.capacity - budget.encoder)], default=0),
            max([n for n in range(1, cap + 1) if fits(n, budget.capacity)], default=0) if runtime.parks else 0)


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


# downloads what is missing and puts dit, vae and encoder (`encoder_kind`: the calibrated nvfp4 checkpoint or the bf16
# original) on the gpu. the dit comes prepacked when it can (and is packed and saved here when it cannot). nothing parks
# yet: `probe` decides that.
def load(encoder_kind: Encoder = ENCODER) -> Runtime:
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
    dit, vae = pipeline.adopt_packed(from_disk(load_file(local, device="cuda")), read("vae"), mean, std) if ready else pipeline.adopt(read("transformer"), read("vae"), mean, std)
    if MX and not ready:  # packed here: keep it, so the next start reads one file and skips the bf16 transformer
        local.parent.mkdir(parents=True, exist_ok=True)
        save_file(to_disk(pipeline.packed_state(dit)), local)
    encoder, processor = load_encoder(snapshot, nvfp4_from=calibrated_folder(encoder_kind))
    return Runtime(pipeline.Models(encoder, processor, dit, vae), lambda: None, lambda: None, False, encoder_kind)


# a gray square of `side` pixels as a request carries it.
def blank(side: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (side, side), (128, 128, 128)).save(buffer, "png")
    return base64.b64encode(buffer.getvalue()).decode()


# the card's budget, measured the way vllm sizes its cache at start: six staged edits (each probe size with COUNTS
# references; the encoder parked while dit and vae run), whose dit + vae peaks give `predict` its quadratics. a probe that
# runs out of memory makes its size's peak infinite, so those requests are refused. the allocator may hold the card's
# memory less what is not torch's (context, other processes) and a margin for fragmentation. the probes also warm up
# every kernel. the encoder keeps its pinned host copy (and parks when a request needs the room) where a 2048 edit with
# REFERENCE_CAP references would not fit beside it, or with MAGPIE_PARK=1. `most` (gib) caps what it plans with, as
# `--max-memory` sets it (torch's allocator is held to it too).
def probe(runtime: Runtime, most: float | None = None) -> Runtime:
    free, total = torch.cuda.mem_get_info()
    room = min((free + torch.cuda.memory_reserved()) / GIB, most or math.inf)
    capacity = 0.97 * room - 0.25
    fetch, park = parking(runtime.models.encoder)
    held = torch.cuda.memory_allocated()
    park()
    encoder = (held - torch.cuda.memory_allocated()) / GIB
    trial = replace(runtime, fetch=fetch, park=park, parks=True)

    def peak(output: int, side: int, count: int) -> float:
        try:
            return generate(trial, {"prompt": "probe", "images": [blank(side)] * count, "resolution": output}, True, {}, None, "staged")[0]["dit_peak_gib"]
        except torch.OutOfMemoryError:
            park()
            return math.inf

    fits = []
    for output, side in PROBES:
        peaks = [peak(output, side, count) for count in COUNTS]
        fits.append((output**2, side**2, quadratic(peaks)))
        print(f"probe {output}² edit, {side}² references: dit + vae {', '.join(f'{p:.2f}' for p in peaks)} gib with {', '.join(map(str, COUNTS))}", flush=True)
    runtime = replace(runtime, budget=Budget(round(capacity, 2), round(encoder, 2), tuple(fits)))
    parks = os.environ.get("MAGPIE_PARK") == "1" or limits(runtime, 2048, pipeline.REFERENCE)[0] < REFERENCE_CAP
    fetch()
    return replace(runtime, fetch=fetch, park=park, parks=True) if parks else runtime


def decoded(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data.split(",")[-1])))


# the reply's image as base64. png at zlib level 1, not pil's 6: the same pixels, files 6-15 % larger, about 4.7 times
# sooner (a 2048 reply: ~430 -> ~90 ms of cpu on the colab vm).
def encoded(image: Image.Image, fmt: Format) -> str:
    buffer = io.BytesIO()
    (image if fmt == "png" else image.convert("RGB")).save(buffer, fmt, **({"compress_level": 1} if fmt == "png" else {"quality": 95}))
    return base64.b64encode(buffer.getvalue()).decode()


def elapsed(start: float) -> float:
    torch.cuda.synchronize()
    return round((time.perf_counter() - start) * 1e3, 1)


# `remembered` with (key, conditioning) as its newest entry, the oldest dropped past PROMPTS.
def remember(remembered: Remembered, key: tuple, conditioning: tuple[torch.Tensor, torch.Tensor]) -> Remembered:
    kept = [(name, value) for name, value in remembered.items() if name != key] + [(key, conditioning)]
    return dict(kept[max(len(kept) - PROMPTS, 0) :]) if PROMPTS else {}


# the condition images a job carries (pil opens them lazily: sizes are read, pixels not yet decoded), at the side they
# enter the dit at, and the output size.
def shapes(job: Job) -> tuple[tuple[Image.Image, ...], int, tuple[int, int]]:
    resolution = int(job.get("resolution", 1024))
    reference = int(job.get("reference") or pipeline.reference_side(resolution))
    images = tuple(decoded(data) for data in job.get("images", []))
    size = (round(job["width"] / 32) * 32, round(job["height"] / 32) * 32) if job.get("width") and job.get("height") else None
    return images, reference, size or (pipeline.fit(resolution**2, images[-1].width / images[-1].height) if images else (resolution,) * 2)


# what a `keep` cache is filed under: the reference side, the output size and a digest of each image as sent.
def pictures(job: Job) -> tuple:
    _, reference, size = shapes(job)
    return reference, *size, tuple(hashlib.blake2b(data.encode(), digest_size=16).hexdigest() for data in job.get("images", []))


# a job's plan from its sizes alone: (mode, predicted peak gib, whether it keeps its references' kv cache). a job asking
# to `keep` keeps it only when it runs resident and the cache fits beside it too, and its peak then includes the cache.
def planned(runtime: Runtime, job: Job) -> tuple[Mode, float, bool]:
    images, reference, (width, height) = shapes(job)
    references = sum(math.prod(pipeline.fit(reference**2, image.width / image.height)) for image in images)
    mode, need = plan(runtime, width * height, references)
    extra = references / 256 * KEPT_GIB[0] + width * height / 256 * KEPT_GIB[1]
    keep = bool(job.get("keep")) and bool(images) and mode == "resident" and need + extra <= runtime.budget.capacity
    return mode, need + keep * extra, keep


# one request, as `pipeline.generate` runs it but in two timed stages, so the encoder can leave the gpu between them
# when the plan says the request does not fit beside it. a (prompt, images) pair in `remembered` skips the encoder (and
# bringing it back). `kept` is the references' kv cache an earlier `keep` request left: a `keep` request on the same
# images resumes from it (no vae encode, no reference rows in the first dit step), any other request drops it first.
# returns the json reply, whether the encoder is parked, the conditionings kept and the cache kept.
@torch.no_grad()
def generate(
    runtime: Runtime, job: Job, parked: bool, remembered: Remembered, kept: pipeline.Kept | None = None, force: Mode | None = None,
) -> tuple[Job, bool, Remembered, pipeline.Kept | None]:
    models, resolution = runtime.models, int(job.get("resolution", 1024))
    images, reference, (width, height) = shapes(job)
    mode, need, keep = (force, 0.0, False) if force else planned(runtime, job)
    crowded = mode == "staged"
    filed = pictures(job)
    kept = kept if keep and kept is not None and kept[0] == filed else None
    resumed = kept is not None
    raw = pipeline.uploaded(images)
    sized = pipeline.resized(raw, reference)
    key = (job["prompt"], reference, filed[-1])
    hit = key in remembered
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if parked and not hit:
        runtime.fetch()
    fetch_ms, start = elapsed(start), time.perf_counter()
    small = reference > LOOK
    text, slots = remembered[key] if hit else encode(models.encoder, models.processor, job["prompt"], pipeline.resized(raw, LOOK) if small else sized)
    if small and not hit:
        text, slots = reslot(text, slots, [image[0].numel() // 1024 for image in sized])
    remembered = remember(remembered, key, (text, slots))
    text_ms, start = elapsed(start), time.perf_counter()
    if crowded and not (parked and hit):
        runtime.park()
    text_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    sampler, seed = TURBO_FAST if job.get("fast") else TURBO, int(job.get("seed", 0))
    if keep:
        image, kept = pipeline.render_kept(models.dit, models.vae, text, slots, sampler, sized, filed, (width, height), seed, kept)
    else:
        image = pipeline.render(models.dit, models.vae, text, slots, sampler, sized, (width, height), resolution, seed)
    render_ms, start = elapsed(start), time.perf_counter()
    fmt: Format = job.get("format", "png")
    reply = {"image": encoded(image, fmt), "format": fmt, "width": image.width, "height": image.height, "seed": int(job.get("seed", 0)),
             "peak_gib": round(max(text_peak, torch.cuda.max_memory_allocated()) / GIB, 2), "dit_peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
             "predicted_gib": round(need, 2), "mode": mode, "remembered": hit, "kept": keep, "resumed": resumed}
    ms = {"fetch": fetch_ms, "text": text_ms, "dit_vae": render_ms, "encode_image": round((time.perf_counter() - start) * 1e3, 1)}
    return {**reply, "ms": ms}, crowded or (parked and hit), remembered, kept


# the request handler for one runtime. the gpu takes one job at a time (`lock`), always on the same thread (`gpu`): the
# first gpu work on a new thread costs ~1.6 s (measured on the 5090: 2.7 s instead of 1.1 s for a 1024 edit), and the
# server gives every connection a new one. `state` holds whether the last job left the encoder off the card, the cache
# a `keep` job left (taken out of `state`, and dropped before the job unless it resumes from it) and the
# conditionings kept, as the last `generate` returned them.
def handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    lock, gpu, state, token = threading.Lock(), ThreadPoolExecutor(1), [False, {}, None], os.environ.get("MAGPIE_TOKEN")

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
                      "busy": lock.locked(), "parks": runtime.parks, "encoder": runtime.encoder, "limits": card_limits(runtime)}
            self.reply(*((200, health) if self.path == "/health" and self.allowed() else (401 if self.path == "/health" else 404, {"error": "not found or not allowed"})))

        def do_POST(self) -> None:
            if self.path != "/generate" or not self.allowed():
                return self.reply(404 if self.allowed() else 401, {"error": "not found or not allowed"})
            job = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not isinstance(job.get("prompt"), str) or not job["prompt"].strip():
                return self.reply(400, {"error": "prompt is required"})
            mode, need, keep = planned(runtime, job)
            if mode == "refused":
                return self.reply(507, {"error": f"this request needs about {need:.1f} gib, the card holds {runtime.budget.capacity:.1f}", "limits": card_limits(runtime)})
            with lock:
                try:
                    held = state.pop()
                    held = held if keep and held is not None and held[0] == pictures(job) else None  # the executor holds its arguments until the job ends
                    body, *state[:] = gpu.submit(generate, runtime, job, state[0], state[1], held).result()
                except torch.OutOfMemoryError as error:  # the one failure a caller can act on: ask for less
                    message = str(error)[:200]
                    del error
                    gpu.submit(runtime.park).result()
                    state[:] = [runtime.parks, state[1], None]
                    return self.reply(507, {"error": f"out of gpu memory: {message}"})
            print(f"{body['width']}x{body['height']} {len(job.get('images', []))} images, {body['mode']}: {body['ms']} peak {body['peak_gib']} gib (predicted {body['predicted_gib']})", flush=True)
            self.reply(200, body)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


# most references per square edit, as /health and a refusal report them: {"1024": {"resident": n, "staged": n}, "2048": ...}.
def card_limits(runtime: Runtime) -> dict[str, dict[str, int]]:
    return {str(output): dict(zip(("resident", "staged"), limits(runtime, output, side))) for output, side in PROBES}


# what this process runs on and with, before anything loads.
def describe() -> str:
    from magpie.kernels import ATTN4, ATTN8, CAPABILITY, MX, SAGE

    props = torch.cuda.get_device_properties(0)
    linears = "mxfp8 + nvfp4 (cutlass)" if MX else "bf16 (no native matmul: build it with tools/build_native.sh on an sm_120 card)"
    attention = "magpie_attn8" if ATTN8 else "sageattention 2" if SAGE else "torch sdpa"
    return (f"gpu: {props.name}, sm_{CAPABILITY[0]}{CAPABILITY[1]}, {props.total_memory / GIB:.1f} gib\n"
            f"paths: linears {linears}; attention {attention}{', magpie_attn4 for 2048-class outputs' if ATTN4 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--encoder", choices=("nvfp4", "bf16"), default=ENCODER, help="text encoder: calibrated 4-bit (default on blackwell) or the bf16 original")
    parser.add_argument("--max-memory", type=float, help="gib of the card this server may use (default: all of it)")
    args = parser.parse_args()
    print(describe(), flush=True)
    if args.max_memory:
        torch.cuda.set_per_process_memory_fraction(min(args.max_memory * GIB / torch.cuda.get_device_properties(0).total_memory, 1.0))
    start = time.perf_counter()
    runtime = load(args.encoder)
    print(f"loaded in {time.perf_counter() - start:.1f} s: {torch.cuda.memory_allocated() / GIB:.1f} gib, {runtime.encoder} encoder", flush=True)
    start = time.perf_counter()
    runtime = probe(runtime, args.max_memory)
    budget = runtime.budget
    print(f"probed in {time.perf_counter() - start:.1f} s: {budget.capacity:.1f} gib usable, encoder {budget.encoder:.1f} gib, "
          f"{'parks when a request needs the room' if runtime.parks else 'stays resident'}", flush=True)
    for output, counts in card_limits(runtime).items():
        print(f"most references, {output}² edit: {counts['resident']} resident" + (f", {counts['staged']} staged" if runtime.parks else ""), flush=True)
    print(f"listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler(runtime)).serve_forever()


if __name__ == "__main__":
    main()
