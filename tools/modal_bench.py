import json
import pathlib as pl
import sys
import time
from typing import Any, Callable

import modal

sys.path.insert(0, str(pl.Path(__file__).resolve().parent))
from gate import (CASES, GATE_CASES, INPUTS, OUT_ROOT, REPOS, SUITE, TIMED_CASES, Timings, collect, device_info, gate_input, gate_items, gate_pass, load_port, psnr, reduce, staged, summary, unfused, verdict)
from modal_common import ROOT, VOLUMES, image, out_volume

app = modal.App("magpie-bench", image=image)


# wraps a callable so each call appends its gpu-synchronised wall time to timings[name].
def timed(function: Callable, name: str, timings: Timings) -> Callable:
    import torch

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        torch.cuda.synchronize()
        timings.setdefault(name, []).append(time.perf_counter() - start)
        return result

    return wrapper


def save(image_out: Any, folder: pl.Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    image_out.convert("RGB").save(folder / f"{name}.jpg", quality=93)


# times the official diffusers pipeline: `reps` warm requests per case after one untimed warm-up,
# each split into text encoder / vae encode / dit (per step) / vae decode / everything else.
def bench_diffusers(pipe: Any, cases: list[str], reps: int, folder: pl.Path, steps: int | None, kept: dict) -> dict[str, Any]:
    import torch
    from PIL import Image

    timings: Timings = {}
    pipe.text_encoder.forward = timed(pipe.text_encoder.forward, "text", timings)
    pipe.transformer.forward = timed(pipe.transformer.forward, "dit", timings)
    pipe.vae.encode = timed(pipe.vae.encode, "vae_encode", timings)
    pipe.vae.decode = timed(pipe.vae.decode, "vae_decode", timings)
    results: dict[str, Any] = {}
    for case in cases:
        spec = CASES[case]

        def request(index: int) -> Any:
            prompt, source, instruction = SUITE[index % len(SUITE)]
            kwargs: dict[str, Any] = dict(use_kv_cache=spec.get("use_kv_cache", True),
                                          generator=torch.Generator("cpu").manual_seed(index))
            if steps is not None:
                kwargs["num_inference_steps"] = steps
            if spec["edit"]:
                source_image = Image.open(f"{INPUTS}/{source}").convert("RGBA")
                return pipe(prompt=instruction, image=source_image, output_resolution=spec["size"], **kwargs).images[0]
            return pipe(prompt=prompt, width=spec["size"], height=spec["size"], **kwargs).images[0]

        request(0)
        timings.clear()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for index in range(reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = request(index)
            torch.cuda.synchronize()
            rows.append(collect(timings, time.perf_counter() - start))
            save(output, folder / case, f"{index:02d}")
            kept[(case, index)] = output
        results[case] = {**reduce(rows), "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        print(f"{case}: total p50 {results[case]['total']['p50']:.3f} s", flush=True)
    return results


# the same requests through the lean port (`src/magpie`), sharing the pipeline's weights.
def bench_magpie(pipe: Any, sampler: Any, cases: list[str], reps: int, folder: pl.Path, kept: dict) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    import magpie.pipeline as lean

    dit, vae = lean.adopt(pipe.transformer.state_dict(), pipe.vae.state_dict(), pipe.vae.config.latents_mean, pipe.vae.config.latents_std)
    timings: Timings = {}
    encode = lean.encode
    lean.encode = timed(encode, "text", timings)
    dit.prefill, dit.decode = timed(dit.prefill, "dit", timings), timed(dit.decode, "dit", timings)
    vae.encode, vae.decode = timed(vae.encode, "vae_encode", timings), timed(vae.decode, "vae_decode", timings)
    models = lean.Models(pipe.text_encoder, pipe.processor, dit, vae)
    results: dict[str, Any] = {}
    for case in cases:
        spec = CASES[case]

        def request(index: int) -> Any:
            prompt, source, instruction = SUITE[index % len(SUITE)]
            if spec["edit"]:
                source_image = Image.open(f"{INPUTS}/{source}")
                return lean.generate(models, instruction, sampler, (source_image,), resolution=spec["size"], seed=index)
            return lean.generate(models, prompt, sampler, size=(spec["size"],) * 2, seed=index)

        request(0)
        timings.clear()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for index in range(reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = request(index)
            torch.cuda.synchronize()
            rows.append(collect(timings, time.perf_counter() - start))
            save(output, folder / case, f"{index:02d}")
            ours, theirs = (np.asarray(image_.convert("RGB"), dtype=np.float64) for image_ in (output, kept[(case, index)]))
            rows[-1]["psnr_db"] = float(10 * np.log10(255.0**2 / max(np.mean((ours - theirs) ** 2), 1e-12)))
        results[case] = {**reduce(rows), "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(f"magpie {case}: total p50 {results[case]['total']['p50']:.3f} s", flush=True)
    lean.encode = encode
    return results


# op-level view of one warm request: the 30 kernels with the most gpu time.
def profile_ops(pipe: Any, case: str) -> list[dict[str, Any]]:
    import torch
    from PIL import Image
    from torch.profiler import ProfilerActivity, profile

    spec = CASES[case]
    prompt, source, instruction = SUITE[0]
    generator = torch.Generator("cpu").manual_seed(0)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        if spec["edit"]:
            pipe(prompt=instruction, image=Image.open(f"{INPUTS}/{source}").convert("RGBA"),
                 output_resolution=spec["size"], generator=generator)
        else:
            pipe(prompt=prompt, width=spec["size"], height=spec["size"], generator=generator)
        torch.cuda.synchronize()
    events = sorted(prof.key_averages(), key=lambda event: event.self_device_time_total, reverse=True)[:30]
    return [{"op": event.key, "calls": event.count, "self_gpu_ms": event.self_device_time_total / 1e3,
             "self_cpu_ms": event.self_cpu_time_total / 1e3} for event in events]


# what the card can do on the shapes this model uses, so utilisation can be stated against a
# measured ceiling instead of a datasheet: dense matmul, attention, conv, and memory bandwidth.
def roofline() -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    def clock(function: Callable[[], Any], reps: int = 10) -> float:
        function()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            function()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / reps

    out: dict[str, float] = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        a, b = (torch.randn(8192, 8192, device="cuda", dtype=dtype) for _ in range(2))
        out[f"gemm_8192_{name}_tflops"] = 2 * 8192**3 / clock(lambda: a @ b) / 1e12
        tokens, weight = torch.randn(4096, 4096, device="cuda", dtype=dtype), torch.randn(12288, 4096, device="cuda", dtype=dtype)
        out[f"mlp_4096x4096x12288_{name}_tflops"] = 2 * 4096 * 4096 * 12288 / clock(lambda: F.linear(tokens, weight)) / 1e12
    for length in (4096, 16384):
        q, k, v = (torch.randn(1, 32, length, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        seconds = clock(lambda: F.scaled_dot_product_attention(q, k, v), reps=5)
        out[f"sdpa_{length}_bf16_ms"] = seconds * 1e3
        out[f"sdpa_{length}_bf16_tflops"] = 4 * 32 * length * length * 128 / seconds / 1e12
    x = torch.randn(1, 288, 1024, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(288, 288, 3, 3, device="cuda", dtype=torch.bfloat16)
    out["conv3x3_288ch_1024_bf16_tflops"] = 2 * 288 * 288 * 9 * 1024 * 1024 / clock(lambda: F.conv2d(x, w, padding=1)) / 1e12
    big = torch.empty(8 * 2**30, dtype=torch.uint8, device="cuda")
    out["copy_gb_per_s"] = 2 * 8 * 2**30 / clock(lambda: big.clone(), reps=5) / 1e9
    return out


def load(variant: str, offload: bool) -> tuple[Any, float]:
    import torch
    from diffusers import QwenImage21Pipeline

    start = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(REPOS[variant], torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
        torch.cuda.synchronize()
    return pipe, time.perf_counter() - start


# benchmark of one checkpoint: one container, one load, every case. `offload` keeps one model on the
# card at a time (for cards the pipeline does not fit) and skips the lean port, which needs all three.
def run_bench(variant: str, reps: int, cases: list[str], extras: bool, offload: bool) -> str:
    import torch

    from magpie.sampling import BASE, TURBO

    torch.set_grad_enabled(False)
    folder = pl.Path(OUT_ROOT) / ("smoke" if offload else variant)
    pipe, load_seconds = load(variant, offload=offload)
    result: dict[str, Any] = {"variant": variant, "device": device_info(), "load_seconds": load_seconds,
                              "resident_gib": round(torch.cuda.memory_allocated() / 2**30, 2), "reps": reps}
    kept: dict = {}
    result["diffusers"] = bench_diffusers(pipe, cases, reps, folder / "diffusers", None, kept)
    if extras:
        result["ops"] = {case: profile_ops(pipe, case) for case in cases if "nocache" not in case and "2048" not in case}
    lean_cases = [] if offload else [case for case in cases if "nocache" not in case]
    result["magpie"] = bench_magpie(pipe, TURBO if variant == "turbo" else BASE, lean_cases, reps, folder / "magpie", kept)
    if extras:
        del pipe
        torch.cuda.empty_cache()
        result["roofline"] = roofline()
    payload = json.dumps(result)
    (folder / "bench.json").write_text(payload)
    out_volume.commit()
    return payload


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def bench(variant: str, reps: int, cases: list[str], extras: bool) -> str:
    return run_bench(variant, reps, cases, extras, offload=False)


# cheap rehearsal of the benchmark harness itself on an l4, at 512 px with cpu offload.
@app.function(gpu="L4", cpu=8, memory=73728, timeout=3600, retries=0, volumes=VOLUMES)
def smoke() -> str:
    return run_bench("turbo", 1, ["t2i_512", "edit_512"], True, offload=True)


# l4 check that the lean port reproduces the official pipeline on the real turbo weights, 512 px.
# the full pipeline does not fit 24 gb, so the reference runs with cpu offload and the port is
# exercised stage by stage, one model on the card at a time.
@app.function(gpu="L4", cpu=8, memory=73728, timeout=3600, retries=0, volumes=VOLUMES)
def parity() -> str:
    import torch
    from PIL import Image

    from magpie.dit import layout
    from magpie.pipeline import adopt, denoise, fit, pack, to_image, to_pixels, unpack
    from magpie.sampling import TURBO, schedule
    from magpie.text import encode

    def gap(ours: Any, theirs: Any) -> float:
        return float((ours.double().cpu() - theirs.double().cpu()).abs().max() / theirs.double().abs().max())

    torch.set_grad_enabled(False)
    folder = pl.Path(OUT_ROOT) / "parity"
    pipe, _ = load("turbo", offload=True)
    report: dict[str, Any] = {"device": device_info()}
    prompt, source, instruction = SUITE[0]
    source_image = Image.open(f"{INPUTS}/{source}")
    jobs = {"t2i": dict(prompt=prompt, width=512, height=512), "edit": dict(prompt=instruction, image=source_image.convert("RGBA"), output_resolution=512)}

    captured: dict[str, dict[str, Any]] = {}
    forward, decode, vae_encode = pipe.transformer.forward, pipe.vae.decode, pipe.vae.encode
    for name, kwargs in jobs.items():
        record: dict[str, Any] = {"velocity": []}

        def spy_dit(*args: Any, _record: dict = record, **kw: Any) -> Any:
            output = forward(*args, **kw)
            _record.setdefault("inputs", {k: kw[k].cpu() for k in ("hidden_states", "encoder_hidden_states", "img_mask")})
            _record["velocity"].append(output[0].cpu())
            _record.setdefault("latents", []).append(kw["hidden_states"].cpu())
            _record.setdefault("timestep", []).append(kw["timestep"].float().cpu())
            return output

        def spy_decode(z: Any, *args: Any, _record: dict = record, **kw: Any) -> Any:
            output = decode(z, *args, **kw)
            _record["decode"] = (z.cpu(), output[0].cpu())
            return output

        def spy_encode(x: Any, *args: Any, _record: dict = record, **kw: Any) -> Any:
            _record["pixels"] = x.cpu()
            return vae_encode(x, *args, **kw)

        pipe.transformer.forward, pipe.vae.decode, pipe.vae.encode = spy_dit, spy_decode, spy_encode
        record["image"] = pipe(generator=torch.Generator("cpu").manual_seed(0), **kwargs).images[0]
        save(record["image"], folder, f"{name}_reference")
        captured[name] = record
        pipe.transformer.forward, pipe.vae.decode, pipe.vae.encode = forward, decode, vae_encode
        uncached = pipe(generator=torch.Generator("cpu").manual_seed(0), use_kv_cache=False, **kwargs).images[0]
        report[f"{name}_reference_cache_vs_nocache_psnr_db"] = psnr(uncached, record["image"])
    pipe.transformer.forward, pipe.vae.decode, pipe.vae.encode = forward, decode, vae_encode
    pipe.remove_all_hooks()

    target = 32 * 32
    encoder = pipe.text_encoder.to("cuda")
    conditioning = {}
    for name in jobs:
        images = [source_image.convert("RGBA").resize(fit(512**2, 1.0), Image.LANCZOS)] if name == "edit" else []
        text, slots = encode(encoder, pipe.processor, jobs[name]["prompt"], images)
        reference = captured[name]["inputs"]
        report[f"{name}_text"] = gap(text, reference["encoder_hidden_states"])
        report[f"{name}_slots_match"] = bool((slots.cpu() == reference["img_mask"][0, : slots.numel()]).all())
        report[f"{name}_text_tokens"] = int(slots.numel())
        conditioning[name] = (text, slots, images)
    encoder.to("cpu")
    torch.cuda.empty_cache()

    dit, vae = adopt(pipe.transformer.state_dict(), pipe.vae.state_dict(), pipe.vae.config.latents_mean, pipe.vae.config.latents_std)
    dit.to("cuda")
    vae.to("cuda")
    for name in jobs:
        text, slots, images = conditioning[name]
        record = captured[name]
        joint = record["inputs"]["hidden_states"].to("cuda")
        noise, references, shapes = joint[:, -target:], None, [(32, 32)]
        if images:
            pixels = to_pixels(images[0]).to("cuda", torch.bfloat16)
            report["edit_pixels"] = gap(pixels, record["pixels"][:, :, 0])
            latents = vae.encode(pixels)
            references, shapes = pack(latents), [tuple(latents.shape[-2:]), (32, 32)]
            report["edit_vae_encode"] = gap(references, joint[:, :-target])
        lay = layout(slots, shapes, dit.cfg)
        levels = (schedule(target, TURBO).to("cuda") * 1000).to(torch.bfloat16) / 1000
        first, cache = dit.prefill(joint, text, levels[:1], lay)
        report[f"{name}_dit_step0"] = gap(first, record["velocity"][0][:, -target:])
        # every later step on the reference's own latents: per-step error with accumulation removed
        report[f"{name}_dit_steps_same_input"] = [
            gap(dit.decode(record["latents"][i][:, -target:].to("cuda"), levels[i : i + 1], lay, cache), record["velocity"][i][:, -target:])
            for i in range(1, len(record["velocity"]))]
        report[f"{name}_sigma_reference"] = [float(t[0]) for t in record["timestep"]]
        report[f"{name}_sigma_magpie"] = [float(level) for level in levels[:-1].float()]
        final = denoise(dit, text, slots, references, noise, shapes, TURBO)
        z, reference_pixels = record["decode"]
        report[f"{name}_final_latents"] = gap(unpack(final, (32, 32)).float() * vae.std.float() + vae.mean.float(), z[:, :, 0].float().to("cuda"))
        raw = vae.decoder(vae.post_quant_conv(z[:, :, 0].to("cuda"))).clamp(-1.0, 1.0)
        report[f"{name}_vae_decode_same_input"] = gap(raw, reference_pixels[:, :, 0])
        report[f"{name}_vae_decode_mean_abs"] = float((raw.float().cpu() - reference_pixels[:, :, 0].float()).abs().mean())
        ours = to_image(vae.decode(unpack(final, (32, 32))))
        save(ours, folder, f"{name}_magpie")
        report[f"{name}_image_psnr_db"] = psnr(ours, record["image"])
    out_volume.commit()
    return json.dumps(report)


# phase a of the freeze: the official pipeline on every gate item. writes the reference image (png)
# and the port's text conditioning to the volume, so later gate runs need neither diffusers' pipeline
# nor the text encoder. returns per-stage peaks, the text stage's time, and (if `capture`) every
# step's dit inputs and outputs for the step-level diagnosis.
def reference_pass(pipe: Any, variant: str, cases: list[str], folder: pl.Path, capture: bool) -> tuple[dict[str, Any], dict]:
    import torch

    import magpie.pipeline as lean
    from magpie.text import encode

    forward, decode, vae_encode, text_forward = pipe.transformer.forward, pipe.vae.decode, pipe.vae.encode, pipe.text_encoder.forward
    report: dict[str, Any] = {}
    captured: dict = {}
    for case in cases:
        spec, peaks, text_seconds = CASES[case], {}, []
        target = (spec["size"] // 16) ** 2
        for index in range(gate_items(variant, case)):
            prompt, images = gate_input(case, index)
            record: dict[str, Any] = {"latents": [], "velocity": []}

            def spy_dit(*args: Any, _record: dict = record, **kw: Any) -> Any:
                output = forward(*args, **kw)
                _record.setdefault("joint", kw["hidden_states"].cpu())
                _record.setdefault("text", kw["encoder_hidden_states"].cpu())
                _record["latents"].append(kw["hidden_states"][:, -target:].cpu())
                _record["velocity"].append(output[0][:, -target:].cpu())
                return output

            def spy_decode(z: Any, *args: Any, _record: dict = record, **kw: Any) -> Any:
                output = decode(z, *args, **kw)
                _record["decode"] = (z.cpu(), output[0].cpu())
                return output

            # re-wrapped per request: cpu-offload hooks replace `forward` after every call
            pipe.text_encoder.forward = staged(text_forward, "text", {}, peaks)
            pipe.transformer.forward = staged(spy_dit if capture else forward, "dit", {}, peaks)
            pipe.vae.encode = staged(vae_encode, "vae_encode", {}, peaks)
            pipe.vae.decode = staged(spy_decode if capture else decode, "vae_decode", {}, peaks)
            kwargs: dict[str, Any] = dict(generator=torch.Generator("cpu").manual_seed(index))
            if spec["edit"]:
                output = pipe(prompt=prompt, image=images[0].convert("RGBA"), output_resolution=spec["size"], **kwargs).images[0]
            else:
                output = pipe(prompt=prompt, width=spec["size"], height=spec["size"], **kwargs).images[0]
            pipe.text_encoder.forward, pipe.transformer.forward, pipe.vae.encode, pipe.vae.decode = text_forward, forward, vae_encode, decode
            (folder / "reference" / case).mkdir(parents=True, exist_ok=True)
            output.convert("RGB").save(folder / "reference" / case / f"{index:02d}.png")
            for attempt in range(2):  # the second call is the warm one
                torch.cuda.synchronize()
                start = time.perf_counter()
                text, slots = encode(pipe.text_encoder, pipe.processor, prompt, lean.resized(images, spec["size"]))
                torch.cuda.synchronize()
            text_seconds.append(time.perf_counter() - start)
            (folder / "conditioning" / case).mkdir(parents=True, exist_ok=True)
            torch.save({"text": text.cpu(), "slots": slots.cpu()}, folder / "conditioning" / case / f"{index:02d}.pt")
            if capture:
                captured[(case, index)] = record
        report[case] = {"stage_peak_gib": {k: round(v, 2) for k, v in peaks.items()}, "text_ms": summary([t * 1e3 for t in text_seconds])}
        print(f"reference {case}: done", flush=True)
    return report, captured


# freezes the baseline for one checkpoint: official references, then the port's dit + vae alone on the card.
def run_freeze(variant: str, cases: list[str], offload: bool) -> str:
    import torch

    import magpie.pipeline as lean

    torch.set_grad_enabled(False)
    folder = pl.Path(OUT_ROOT) / ("baseline-rehearsal" if offload else "baseline") / variant
    pipe, _ = load(variant, offload=offload)
    result: dict[str, Any] = {"variant": variant, "device": device_info(), "resident_all_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
    result["reference"], captured = reference_pass(pipe, variant, cases, folder, capture=variant == "turbo")
    if offload:
        pipe.remove_all_hooks()
    pipe.text_encoder.to("cpu")
    dit, vae = lean.adopt(pipe.transformer.state_dict(), pipe.vae.state_dict(), pipe.vae.config.latents_mean, pipe.vae.config.latents_std)
    dit.to("cuda")
    vae.to("cuda")
    torch.cuda.empty_cache()
    result["resident_dit_vae_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    result["magpie"] = gate_pass(dit, vae, variant, cases, folder, folder / "magpie", captured)
    out_volume.commit()
    return json.dumps(result)


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def freeze(variant: str) -> str:
    return run_freeze(variant, GATE_CASES, offload=False)


# cheap rehearsal of the freeze on an l4: 512 px, the reference with cpu offload.
@app.function(gpu="L4", cpu=8, memory=73728, timeout=3600, retries=0, volumes=VOLUMES)
def freeze_rehearsal() -> str:
    return run_freeze("turbo", ["t2i_512", "edit_512"], offload=True)


# records what the baseline port feeds each dit step and what comes out, for every gate item, into
# /out/baseline/<variant>/steps. kernels are switched off so this is the frozen baseline's arithmetic;
# `repeat_psnr_db` against the frozen baseline images shows that it is (168 = bit-identical).
@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def steps(variant: str) -> str:
    import torch
    from PIL import Image

    import magpie.dit as dit_module
    import magpie.pipeline as lean
    from magpie.sampling import BASE, TURBO

    dit, vae = load_port(variant, slim=False)
    unfused(dit_module)
    sampler = TURBO if variant == "turbo" else BASE
    folder = pl.Path(OUT_ROOT) / "baseline" / variant
    prefill, decode = dit.prefill, dit.decode
    record: dict[str, Any] = {}

    def spy_prefill(latents: Any, *args: Any) -> Any:
        velocity, cache = prefill(latents, *args)
        record.update(joint=latents.cpu(), latents=[], velocity=[velocity.cpu()])
        return velocity, cache

    def spy_decode(latents: Any, *args: Any) -> Any:
        velocity = decode(latents, *args)
        record["latents"].append(latents.cpu())
        record["velocity"].append(velocity.cpu())
        return velocity

    dit.prefill, dit.decode = spy_prefill, spy_decode
    repeat: dict[str, list[float]] = {}
    for case in GATE_CASES:
        spec = CASES[case]
        (folder / "steps" / case).mkdir(parents=True, exist_ok=True)
        for index in range(gate_items(variant, case)):
            saved = torch.load(folder / "conditioning" / case / f"{index:02d}.pt")
            images = lean.resized(gate_input(case, index)[1], spec["size"])
            output = lean.render(dit, vae, saved["text"], saved["slots"], sampler, images, None if images else (spec["size"],) * 2, spec["size"], index)
            torch.save(dict(record), folder / "steps" / case / f"{index:02d}.pt")
            repeat.setdefault(case, []).append(psnr(output, Image.open(folder / "magpie" / case / f"{index:02d}.png")))
        print(f"steps {case}: repeat psnr min {min(repeat[case]):.1f} dB", flush=True)
    out_volume.commit()
    return json.dumps({"variant": variant, "repeat_psnr_db": repeat})


# the gate for a candidate build of `src/magpie`: only the dit and vae are loaded. results go to
# /out/gate/<tag>/<variant>; `verdict` judges them against STEP_ERROR_MAX and DINO_MIN.
# tag "baseline" instead re-freezes the port half of the baseline against the existing references.
@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def gate(variant: str, tag: str, cases: list[str], slim: bool = True, attention: str = "k,v") -> str:
    import torch

    dit, vae = load_port(variant, slim, tuple(filter(None, attention.split(","))))
    folder = pl.Path(OUT_ROOT) / "baseline" / variant
    out = folder / "magpie" if tag == "baseline" else pl.Path(OUT_ROOT) / "gate" / tag / variant
    result = {"variant": variant, "tag": tag, "device": device_info(), "resident_dit_vae_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
              "magpie": gate_pass(dit, vae, variant, cases, folder, out, {})}
    out_volume.commit()
    return json.dumps(result)


# wraps `owner.attr` so every call shows up in the profiler trace as a named span.
def label(owner: Any, attr: str, name: str) -> None:
    from torch.profiler import record_function

    function = getattr(owner, attr)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with record_function(name):
            return function(*args, **kwargs)

    setattr(owner, attr, wrapper)


# names every stage of the port for the trace: steps, blocks, the pieces of a block, and the vae's levels.
def label_port(dit: Any, vae: Any) -> None:
    import magpie.dit as dit_module

    label(dit, "prefill", "dit step 1 (prefill)")
    label(dit, "decode", "dit step (decode)")
    for attr, name in (("time_embed", "time embed"), ("text_embed", "text embed"), ("modulate", "modulation"), ("head", "head")):
        label(dit, attr, name)
    label(dit_module, "attend", "attention (sdpa)")
    for index, block in enumerate(dit.transformer_blocks):
        label(block, "prefill", f"block {index:02d}")
        label(block, "decode", f"block {index:02d}")
        label(block, "norm", "layer norm")
        label(block.attn, "qkv", "qkv + qk-norm + rope")
        label(block.attn, "out", "attention out")
        label(block.img_mlp, "forward", "mlp (swiglu)")
    label(vae, "encode", "vae encode")
    label(vae, "decode", "vae decode")
    label(vae.encoder.mid_block, "forward", "vae encoder mid")
    label(vae.decoder.mid_block, "forward", "vae decoder mid")
    for index, block in enumerate(vae.encoder.down_blocks):
        label(block, "forward", f"vae down {index}")
    for index, block in enumerate(vae.decoder.up_blocks):
        label(block, "forward", f"vae up {index}")


# one warm, profiled turbo request per case through the port's dit + vae. writes a chrome trace per case
# to /out/trace/<variant>/ (open in perfetto) and returns gpu time per aten op. span times come from the
# trace files (tools/trace_kernels.py): key_averages() double-counts nested labels.
@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def trace(variant: str, cases: list[str]) -> str:
    import gzip

    import torch
    from torch.profiler import ProfilerActivity, profile

    import magpie.pipeline as lean
    from magpie.sampling import BASE, TURBO

    dit, vae = load_port(variant)
    label_port(dit, vae)
    sampler = TURBO if variant == "turbo" else BASE
    folder, out = pl.Path(OUT_ROOT) / "baseline" / variant, pl.Path(OUT_ROOT) / "trace" / variant
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"device": device_info()}
    for case in cases:
        spec = CASES[case]
        saved = torch.load(folder / "conditioning" / case / "00.pt")
        images = lean.resized(gate_input(case, 0)[1], spec["size"])

        def request() -> Any:
            return lean.render(dit, vae, saved["text"], saved["slots"], sampler, images, None if images else (spec["size"],) * 2, spec["size"], 0)

        request()
        torch.cuda.synchronize()
        start = time.perf_counter()
        request()
        torch.cuda.synchronize()
        unprofiled = time.perf_counter() - start
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            start = time.perf_counter()
            request()
            torch.cuda.synchronize()
            profiled = time.perf_counter() - start
        prof.export_chrome_trace(str(out / f"{case}.json"))
        (out / f"{case}.json.gz").write_bytes(gzip.compress((out / f"{case}.json").read_bytes()))
        (out / f"{case}.json").unlink()
        ops = {event.key: {"calls": event.count, "self_gpu_ms": event.self_device_time_total / 1e3}
               for event in prof.key_averages() if event.key.startswith("aten::") and event.self_device_time_total > 0}
        report[case] = {"unprofiled_ms": unprofiled * 1e3, "profiled_ms": profiled * 1e3, "ops": ops}
        print(f"trace {case}: {unprofiled:.3f} s plain, {profiled:.3f} s profiled", flush=True)
    out_volume.commit()
    return json.dumps(report)


# the attention logits the model really reaches, per block, over the gate set: decides whether a flash
# kernel without the running max is safe (no exp may overflow, and no row's sum of exps may overflow or
# vanish). every sdpa call is shadowed: the scores are rebuilt in fp32, a chunk of queries at a time,
# and the largest logit and the largest / smallest row log-sum-exp are kept. `bound` is the worst case
# the qk-norm weights allow, sqrt(d) * max|s_q| * max|s_k|; a measured logit above it means the spy is wrong.
@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def logits(variant: str, cases: list[str]) -> str:
    import math

    import torch

    import magpie.dit as dit_module
    import magpie.pipeline as lean
    from magpie.sampling import BASE, TURBO

    dit, vae = load_port(variant, slim=False)
    sampler = TURBO if variant == "turbo" else BASE
    folder = pl.Path(OUT_ROOT) / "baseline" / variant
    blocks = dit.transformer_blocks
    unfused(dit_module)
    plain = dit_module.sdpa
    state: dict[str, Any] = {"block": 0, "rows": [], "keys": 0}

    def spy(q: Any, k: Any, v: Any, mask: Any = None) -> Any:
        queries = q.transpose(1, 2).float() / math.sqrt(q.shape[-1])
        keys = k.transpose(1, 2).float().transpose(-1, -2)
        chunks = queries.split(max(1, 2**31 // (q.shape[2] * k.shape[1])), dim=2)
        masks = mask.split(chunks[0].shape[2]) if mask is not None else [None] * len(chunks)
        row = state["rows"][state["block"]]
        state["keys"] = max(state["keys"], k.shape[1])
        for chunk, visible in zip(chunks, masks):
            scores = chunk @ keys
            if visible is not None:
                scores.masked_fill_(~visible, -math.inf)
            top = scores.amax(-1, keepdim=True)
            sums = scores.sub_(top).exp_().sum(-1).log_() + top[..., 0]  # log-sum-exp, in place
            row[:] = max(row[0], float(top.max())), max(row[1], float(sums.max())), min(row[2], float(sums.min()))
        return plain(q, k, v, mask)

    def tag(block: Any, attr: str, index: int) -> None:
        function = getattr(block, attr)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state["block"] = index
            return function(*args, **kwargs)

        setattr(block, attr, wrapper)

    dit_module.sdpa = spy
    for index, block in enumerate(blocks):
        tag(block, "prefill", index)
        tag(block, "decode", index)
    width = blocks[0].attn.norm_q.weight.numel()
    report: dict[str, Any] = {"device": device_info(), "variant": variant, "cases": {}, "bound": [
        math.sqrt(width) * float(b.attn.norm_q.weight.float().abs().max()) * float(b.attn.norm_k.weight.float().abs().max())
        for b in blocks]}
    for case in cases:
        spec = CASES[case]
        state.update(rows=[[-math.inf, -math.inf, math.inf] for _ in blocks], keys=0)
        for index in range(gate_items(variant, case)):
            saved = torch.load(folder / "conditioning" / case / f"{index:02d}.pt")
            images = lean.resized(gate_input(case, index)[1], spec["size"])
            lean.render(dit, vae, saved["text"], saved["slots"], sampler, images, None if images else (spec["size"],) * 2, spec["size"], index)
        names = ("max_logit", "max_row_logsumexp", "min_row_logsumexp")
        report["cases"][case] = {"keys": state["keys"], "items": gate_items(variant, case),
                                 **{name: [row[i] for row in state["rows"]] for i, name in enumerate(names)}}
        print(f"logits {case}: max {max(report['cases'][case]['max_logit']):.2f}, "
              f"row sum max {max(report['cases'][case]['max_row_logsumexp']):.2f}", flush=True)
    return json.dumps(report)


@app.local_entrypoint()
def main(mode: str = "rehearse", reps: int = 0, tag: str = "candidate", variants: str = "turbo,base", cases: str = TIMED_CASES, slim: bool = True,
         attention: str = "k,v") -> None:
    out = ROOT / "out" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "baseline"
    if mode == "freeze-rehearsal":
        print(json.dumps(json.loads(freeze_rehearsal.remote()), indent=1))
        return
    if mode == "freeze":
        frozen.mkdir(exist_ok=True)
        calls = [freeze.spawn("turbo"), freeze.spawn("base"), parity.spawn()]
        for name, payload in zip(("turbo", "base", "parity"), modal.FunctionCall.gather(*calls)):
            target = out / "parity.json" if name == "parity" else frozen / f"{name}.json"
            target.write_text(json.dumps(json.loads(payload), indent=1))
        return
    if mode == "trace":
        (ROOT / "out" / "trace").mkdir(parents=True, exist_ok=True)
        (ROOT / "out" / "trace" / "ops.json").write_text(json.dumps(json.loads(trace.remote("turbo", ["edit_1024", "edit_2048"])), indent=1))
        return
    if mode == "steps":
        print(steps.remote("turbo"))
        return
    if mode == "logits":
        (ROOT / "out" / "logits").mkdir(parents=True, exist_ok=True)
        for variant in variants.split(","):
            (ROOT / "out" / "logits" / f"{variant}.json").write_text(json.dumps(json.loads(logits.remote(variant, GATE_CASES)), indent=1))
        return
    if mode == "gate":
        (ROOT / "out" / "gate").mkdir(parents=True, exist_ok=True)
        chosen = GATE_CASES if tag == "baseline" else cases.split(",")
        for payload in gate.starmap([(variant, tag, chosen, slim, attention) for variant in variants.split(",")]):
            result = json.loads(payload)
            baseline = json.loads((frozen / f"{result['variant']}.json").read_text())
            if tag == "baseline":
                merged = {**baseline, "resident_dit_vae_gib": result["resident_dit_vae_gib"], "magpie": result["magpie"]}
                (frozen / f"{result['variant']}.json").write_text(json.dumps(merged, indent=1))
                continue
            result["verdict"] = verdict(result, baseline)
            (ROOT / "out" / "gate" / f"{tag}-{result['variant']}.json").write_text(json.dumps(result, indent=1))
            print(result["variant"], json.dumps(result["verdict"], indent=1))
        return
    if mode == "parity":
        (out / "parity.json").write_text(json.dumps(json.loads(parity.remote()), indent=2))
        print((out / "parity.json").read_text())
        return
    if mode == "rehearse":
        for name, payload in zip(("parity", "smoke"), modal.FunctionCall.gather(parity.spawn(), smoke.spawn())):
            (out / f"{name}.json").write_text(json.dumps(json.loads(payload), indent=2))
        print((out / "parity.json").read_text())
        return
    heavy = [case for case in CASES if "512" not in case]
    jobs = [("turbo", reps or 5, heavy, True), ("base", reps or 3, heavy, False)]
    for payload in bench.starmap(jobs):
        result = json.loads(payload)
        (out / f"{result['variant']}.json").write_text(json.dumps(result, indent=2))
        for engine in ("diffusers", "magpie"):
            for case, row in result[engine].items():
                print(f"{result['variant']} {engine} {case}: {row['total']['p50']:.3f} s")
