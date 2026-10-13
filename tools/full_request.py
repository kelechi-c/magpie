# whole requests, text encoder included, on one card (meant for a 32 gb rtx 5090; needs text_encoder/ and processor/
# in the cached snapshot). writes /out/full/<mode>.json and the images beside it.
#   python tools/full_request.py resident   encoder, dit and vae all on the gpu: edit and text-to-image at 1024, then
#                                           an edit at 2048 to see whether it fits
#   python tools/full_request.py staged     2048 edits with the dit and vae resident and the encoder parked in pinned
#                                           host memory: copied to the gpu for the encode, dropped before the denoise
import json
import pathlib as pl
import sys
import time
from typing import Any, Callable

import torch

import magpie.pipeline as lean
from gate import REPOS, gate_input, load_port
from magpie.sampling import TURBO
from magpie.text import encode

OUT = pl.Path("/out/full")
RUNS = 5
GIB = 2**30


def snapshot() -> pl.Path:
    from huggingface_hub import hf_hub_download

    return pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent


# the checkpoint's qwen3-vl in bf16 on the gpu, and its processor.
def load_encoder() -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(snapshot() / "text_encoder", dtype=torch.bfloat16).to("cuda").eval()
    return model, AutoProcessor.from_pretrained(snapshot() / "processor")


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
            between: Callable[[], None] = lambda: None) -> tuple[Any, dict[str, float]]:
    prompt, images = gate_input(case, 0)
    sized = lean.resized(images, size)
    _, load_ms, _ = stage(before)
    (text, slots), text_ms, text_peak = stage(lambda: encode(encoder, processor, prompt, sized))
    _, free_ms, _ = stage(between)
    image, render_ms, render_peak = stage(lambda: lean.render(dit, vae, text, slots, TURBO, sized, None if sized else (size, size), size, 0))
    return image, {"load_encoder_ms": load_ms, "text_ms": text_ms, "free_encoder_ms": free_ms, "dit_vae_ms": render_ms,
                   "total_ms": load_ms + text_ms + free_ms + render_ms, "text_peak_gib": text_peak, "dit_vae_peak_gib": render_peak}


# one warm-up request, then RUNS timed ones -> medians of every time, the highest peaks, and the image saved.
def timed(name: str, run: Callable[[], tuple[Any, dict[str, float]]]) -> dict[str, Any]:
    run()
    rows = [run() for _ in range(RUNS)]
    rows[-1][0].convert("RGB").save(OUT / f"{name}.png")
    keys = rows[0][1]
    report = {key: (round(max(row[1][key] for row in rows), 2) if key.endswith("gib") else median([row[1][key] for row in rows])) for key in keys}
    print(name, json.dumps(report), flush=True)
    return report


def resident() -> dict[str, Any]:
    dit, vae = load_port("turbo")  # first: its bf16 weights are on the card until they are packed, 19 gib at the peak
    encoder, processor = load_encoder()
    report: dict[str, Any] = {"resident_gib": round(torch.cuda.memory_allocated() / GIB, 2)}
    for name, case, size in (("edit_1024", "edit_1024", 1024), ("t2i_1024", "t2i_1024", 1024), ("edit_2048", "edit16_2048", 2048)):
        try:
            report[name] = timed(f"resident-{name}", lambda: request(encoder, processor, dit, vae, case, size))
        except torch.OutOfMemoryError as error:  # the point of the run: which sizes fit 32 gb with everything loaded
            report[name] = {"out_of_memory": str(error)[:160]}
            print(name, "out of memory", flush=True)
            torch.cuda.empty_cache()
    return report


def staged() -> dict[str, Any]:
    start = time.perf_counter()
    dit, vae = load_port("turbo")
    torch.cuda.synchronize()
    port_s = time.perf_counter() - start
    start = time.perf_counter()
    encoder, processor = load_encoder()
    encoder_s = time.perf_counter() - start
    host = {name: weight.cpu().pin_memory() for name, weight in encoder.state_dict().items()}

    def park() -> None:
        encoder.load_state_dict(host, assign=True)
        torch.cuda.empty_cache()

    def fetch() -> None:
        encoder.load_state_dict({name: weight.to("cuda", non_blocking=True) for name, weight in host.items()}, assign=True)

    park()
    report: dict[str, Any] = {"load_encoder_from_disk_s": round(encoder_s, 1), "load_dit_vae_from_disk_s": round(port_s, 1),
                              "resident_gib_encoder_parked": round(torch.cuda.memory_allocated() / GIB, 2)}
    for name, case, size in (("edit_2048", "edit16_2048", 2048), ("edit_1024", "edit_1024", 1024)):
        report[name] = timed(f"staged-{name}", lambda: request(encoder, processor, dit, vae, case, size, fetch, park))
    return report


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    result = {"device": torch.cuda.get_device_name(0), "total_gib": round(torch.cuda.get_device_properties(0).total_memory / GIB, 2),
              **(resident() if mode == "resident" else staged())}
    (OUT / f"{mode}.json").write_text(json.dumps(result, indent=1))
