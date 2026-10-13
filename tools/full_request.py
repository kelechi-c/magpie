# whole requests, text encoder included, on one card (meant for a 32 gb rtx 5090; needs text_encoder/ and processor/
# in the cached snapshot). writes /out/full/<mode>.json and the images beside it.
#   python tools/full_request.py resident   encoder, dit and vae all on the gpu: edit and text-to-image at 1024, then
#                                           an edit at 2048 to see whether it fits
#   python tools/full_request.py staged     2048 edits with the dit and vae resident and the encoder parked in pinned
#                                           host memory: copied to the gpu for the encode, dropped before the denoise
#   python tools/full_request.py refs <resident|staged>   edits with 1, 2, 3, ... condition images at 1024 and 2048
#                                           until one does not fit: time and peak memory per count
#   python tools/full_request.py cold <tag> a new process up to its first and second 1024 edit: imports, both loads,
#                                           the first request. tag "disk" first evicts the weight files from the page cache
# MAGPIE_VRAM_GIB=31.36 caps torch's gpu memory (a 32 gb card on a larger one); MAGPIE_ENCODER=nvfp4 as in magpie.text.
import json
import os
import pathlib as pl
import sys
import time
from typing import Any, Callable

import torch
from PIL import Image

import magpie.pipeline as lean
from gate import REPOS, WIDE_INPUTS, gate_input, load_port
from magpie.sampling import TURBO
from magpie.text import encode, load_encoder as port_encoder, reslot

OUT = pl.Path("/out/full")
RUNS = 5
GIB = 2**30
MOST = 8  # condition images tried at most
LOOK = 1024  # the encoder reads condition images no larger than this, as pipeline.generate does
TFLOP = {1024: 685.8, 2048: 4537.5}  # arithmetic in one edit's dit + vae (tools/roofline.py), for the achieved rate


def snapshot() -> pl.Path:
    from huggingface_hub import hf_hub_download

    return pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent


# the checkpoint's qwen3-vl in bf16 on the gpu (language head dropped), and its processor.
def load_encoder() -> tuple[Any, Any]:
    return port_encoder(snapshot())


# `encode` as pipeline.generate does it: above LOOK the encoder reads a LOOK^2 copy and the slots are resized.
def conditioning(encoder: Any, processor: Any, prompt: str, images: tuple[Any, ...], sized: list[Any], size: int) -> tuple[Any, Any]:
    if size <= LOOK or not sized:
        return encode(encoder, processor, prompt, sized)
    return reslot(*encode(encoder, processor, prompt, lean.resized(images, LOOK)), [image.width * image.height // 1024 for image in sized])


# fn() with the gpu drained before and after -> (its result, ms, peak allocated gib during it).
def stage(fn: Callable[[], Any]) -> tuple[Any, float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1e3, torch.cuda.max_memory_allocated() / GIB


def median(values: list[float]) -> float:
    return round(sorted(values)[len(values) // 2], 1)


# one request in two stages: the encoder, then vae encode + denoise + vae decode. `between` runs after the encode
# (the staged mode frees the encoder there); `before` runs first (it brings the encoder back).
def request(encoder: Any, processor: Any, dit: Any, vae: Any, case: str, size: int, before: Callable[[], None] = lambda: None,
            between: Callable[[], None] = lambda: None, given: tuple[str, tuple[Any, ...]] | None = None) -> tuple[Any, dict[str, float]]:
    prompt, images = given or gate_input(case, 0)
    sized = lean.resized(images, size)
    _, load_ms, _ = stage(before)
    (text, slots), text_ms, text_peak = stage(lambda: conditioning(encoder, processor, prompt, images, sized, size))
    _, free_ms, _ = stage(between)
    image, render_ms, render_peak = stage(lambda: lean.render(dit, vae, text, slots, TURBO, sized, None if sized else (size, size), size, 0))
    return image, {"load_encoder_ms": load_ms, "text_ms": text_ms, "free_encoder_ms": free_ms, "dit_vae_ms": render_ms,
                   "total_ms": load_ms + text_ms + free_ms + render_ms, "dit_vae_tflops": TFLOP[size] / render_ms * 1e3,
                   "text_peak_gib": text_peak, "dit_vae_peak_gib": render_peak}


# one warm-up request, then RUNS timed ones -> medians of every time, the highest peaks, and the image saved.
def timed(name: str, run: Callable[[], tuple[Any, dict[str, float]]]) -> dict[str, Any]:
    run()
    rows = [run() for _ in range(RUNS)]
    rows[-1][0].convert("RGB").save(OUT / f"{name}.png")
    keys = rows[0][1]
    report = {key: (round(max(row[1][key] for row in rows), 2) if key.endswith("gib") else median([row[1][key] for row in rows])) for key in keys}
    print(name, json.dumps(report), flush=True)
    return report


# drops the weight files (prepacked dit, vae, text encoder) from the page cache, so the next load reads the disk.
def evict() -> None:
    home = pl.Path(os.environ.get("HF_HOME", pl.Path.home() / ".cache/huggingface"))
    for file in [*(home / "magpie").glob("*.safetensors"), *snapshot().glob("vae/*.safetensors"), *snapshot().glob("text_encoder/*.safetensors")]:
        with open(file, "rb") as handle:
            os.fsync(handle.fileno())
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


# seconds since this process started.
def alive() -> float:
    return time.time() - pl.Path("/proc/self").stat().st_ctime


# a new process up to its first image: seconds at each milestone, then the first and second 1024 edit.
def cold() -> dict[str, Any]:
    if sys.argv[2:] == ["disk"]:
        evict()
    marks = {"imports_s": alive()}
    dit, vae = load_port("turbo")
    torch.cuda.synchronize()
    marks["dit_vae_loaded_s"] = alive()
    encoder, processor = load_encoder()
    torch.cuda.synchronize()
    marks["encoder_loaded_s"] = alive()
    first = request(encoder, processor, dit, vae, "edit_1024", 1024)[1]["total_ms"]
    marks["first_image_s"] = alive()
    second = request(encoder, processor, dit, vae, "edit_1024", 1024)[1]["total_ms"]
    report = {**{name: round(value, 1) for name, value in marks.items()}, "first_request_ms": round(first), "second_request_ms": round(second),
              "resident_gib": round(torch.cuda.memory_allocated() / GIB, 2)}
    print("cold", json.dumps(report), flush=True)
    return report


def resident() -> dict[str, Any]:
    dit, vae = load_port("turbo")  # first: its bf16 weights are on the card until they are packed, 19 gib at the peak
    encoder, processor = load_encoder()
    report: dict[str, Any] = {"resident_gib": round(torch.cuda.memory_allocated() / GIB, 2), "load_peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 2)}
    print("resident", json.dumps(report), flush=True)
    for name, case, size in (("edit_1024", "edit_1024", 1024), ("t2i_1024", "t2i_1024", 1024), ("edit_2048", "edit16_2048", 2048)):
        try:
            report[name] = timed(f"resident-{name}", lambda: request(encoder, processor, dit, vae, case, size))
        except torch.OutOfMemoryError as error:  # the point of the run: which sizes fit 32 gb with everything loaded
            report[name] = {"out_of_memory": str(error)[:160]}
            print(name, "out of memory", flush=True)
            torch.cuda.empty_cache()
    return report


# (fetch, park) for an encoder kept in pinned host memory: fetch copies it to the gpu, park drops that copy.
def parking(encoder: Any) -> tuple[Callable[[], None], Callable[[], None]]:
    host = {name: weight.cpu().pin_memory() for name, weight in encoder.state_dict().items()}

    def park() -> None:
        encoder.load_state_dict(host, assign=True)
        torch.cuda.empty_cache()

    def fetch() -> None:
        encoder.load_state_dict({name: weight.to("cuda", non_blocking=True) for name, weight in host.items()}, assign=True)

    park()
    return fetch, park


# how many condition images fit: one edit per count at each size, up to MOST, stopping at the first that fails.
def references(mode: str) -> dict[str, Any]:
    dit, vae = load_port("turbo")
    encoder, processor = load_encoder()
    fetch, park = parking(encoder) if mode == "staged" else (lambda: None, lambda: None)
    pictures = [Image.open(f"{WIDE_INPUTS}/{index:02d}.png") for index in range(MOST)]
    report: dict[str, Any] = {"mode": mode}
    for size in (1024, 2048):
        for count in range(1, MOST + 1):
            given = ("Combine the subjects of all the images into one ink-and-watercolour drawing on cream paper.", tuple(pictures[:count]))
            try:
                image, row = request(encoder, processor, dit, vae, "", size, fetch, park, given)
            except (torch.OutOfMemoryError, RuntimeError) as error:  # the point of the run: the first count that does not fit
                report[f"{size} x{count}"] = {"failed": str(error)[:200]}
                (OUT / f"refs-{mode}-partial.json").write_text(json.dumps(report, indent=1))
                print(size, count, "failed", str(error)[:200].replace("\n", " "), flush=True)
                del error
                park()
                torch.cuda.empty_cache()
                break
            report[f"{size} x{count}"] = {key: round(value, 2 if key.endswith("gib") else 0) for key, value in row.items() if key != "dit_vae_tflops"}
            print(size, count, json.dumps(report[f"{size} x{count}"]), flush=True)
    return report


def staged() -> dict[str, Any]:
    start = time.perf_counter()
    dit, vae = load_port("turbo")
    torch.cuda.synchronize()
    port_s = time.perf_counter() - start
    start = time.perf_counter()
    encoder, processor = load_encoder()
    encoder_s = time.perf_counter() - start
    fetch, park = parking(encoder)
    report: dict[str, Any] = {"load_encoder_from_disk_s": round(encoder_s, 1), "load_dit_vae_from_disk_s": round(port_s, 1),
                              "resident_gib_encoder_parked": round(torch.cuda.memory_allocated() / GIB, 2)}
    for name, case, size in (("edit_2048", "edit16_2048", 2048), ("edit_1024", "edit_1024", 1024)):
        report[name] = timed(f"staged-{name}", lambda: request(encoder, processor, dit, vae, case, size, fetch, park))
    return report


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    if os.environ.get("MAGPIE_VRAM_GIB"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["MAGPIE_VRAM_GIB"]) * GIB / torch.cuda.get_device_properties(0).total_memory)
    result = {"device": torch.cuda.get_device_name(0), "total_gib": round(torch.cuda.get_device_properties(0).total_memory / GIB, 2),
              **(references(sys.argv[2]) if mode == "refs" else {"resident": resident, "staged": staged, "cold": cold}[mode]())}
    (OUT / f"{'-'.join([*sys.argv[1:], os.environ.get('MAGPIE_ENCODER', 'bf16'), os.environ.get('MAGPIE_VRAM_GIB', 'full')])}.json").write_text(json.dumps(result, indent=1))
