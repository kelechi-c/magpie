"""the kernels against their torch references, with no provider in it.

    python tools/kernel_bench.py prep|attention|mlp|vae check|sweep

`check` is small enough for any gpu; `sweep` uses the model's real shapes and every tuning setting.
"""

import itertools
import json
import sys

SWEEP = list(itertools.product((8, 16, 32), (1, 2, 4, 8)))  # (rows per program, warps)


# the fused qk-prep kernel against `rotate(rms(...))` on synthetic bf16 projections shaped like the model's:
# how many outputs differ and by how much, then milliseconds per call for the reference and for every
# (rows, warps) setting, beside the time one read and one write of the tensor take at copy speed.
def run(tokens: list[int]) -> str:
    import torch

    from magpie.dit import DitConfig, rms, rope_table, rotate
    from magpie.kernels import qk_prep

    def bench(fn, iters: int = 30) -> float:
        for _ in range(3):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = DitConfig()
    dim = cfg.width // cfg.heads
    report = {"gpu": torch.cuda.get_device_name(0), "cases": {}}
    for count in tokens:
        x = (torch.randn(1, count, cfg.heads, dim, device="cuda") * torch.rand(1, count, cfg.heads, 1, device="cuda") * 8).bfloat16()
        weight = (torch.rand(dim, device="cuda") * 4 + 0.2).bfloat16()
        rope = torch.cat([rope_table(torch.arange(count) - count // 2, d) for d in cfg.rope_axes], dim=-1).to("cuda")
        want, got = rotate(rms(x, weight, cfg.eps), rope), qk_prep(x, weight, rope, cfg.eps)
        apart = (want.view(torch.int16).int() - got.view(torch.int16).int()).abs()
        copy = torch.empty_like(x)
        report["cases"][count] = {
            "differing_fraction": float((apart > 0).float().mean()), "max_bf16_steps_apart": int(apart.max()),
            "max_abs_gap": float((want.float() - got.float()).abs().max()), "max_abs_value": float(want.float().abs().max()),
            "reference_ms": bench(lambda: rotate(rms(x, weight, cfg.eps), rope)), "copy_ms": bench(lambda: copy.copy_(x)),
            "fused_ms": {f"{rows}x{warps}": bench(lambda: qk_prep(x, weight, rope, cfg.eps, rows, warps)) for rows, warps in SWEEP}}
        print(count, json.dumps(report["cases"][count]), flush=True)
    return json.dumps(report)


ATTENTION_SWEEP = list(itertools.product((128, 256), (16, 32, 48, 64), (4, 8, 16), (1, 2, 3, 4)))  # block_m, block_n, warps, stages


# the flash kernels against torch's sdpa on q, k that went through the real qk-prep (so logits are bounded like
# the model's) for (queries, keys) shapes: largest gap relative to the largest output, and ms per call for
# sdpa and for each setting of both variants. tflop/s counts 4 * queries * keys * width flops per call.
def run_attention(shapes: list[tuple[int, int]], sweep: list[tuple[int, int, int, int]]) -> str:
    import torch
    import torch.nn.functional as F

    from magpie.dit import DitConfig, rope_table
    from magpie.kernels import flash, logit_bound, max_free, qk_prep
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    def bench(fn, iters: int = 10) -> float:
        for _ in range(2):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = DitConfig()
    dim = cfg.width // cfg.heads
    report = {"gpu": torch.cuda.get_device_name(0), "cases": {}}
    for queries, keys in shapes:
        raw = lambda count: (torch.randn(1, count, cfg.heads, dim, device="cuda") * 3).bfloat16()
        weight = (torch.rand(dim, device="cuda") * 1.5 + 0.5).bfloat16()
        rope = torch.cat([rope_table(torch.arange(keys) - keys // 2, d) for d in cfg.rope_axes], dim=-1).to("cuda")
        k, v = qk_prep(raw(keys), weight, rope, cfg.eps), raw(keys)
        q = qk_prep(raw(queries), weight, rope[-queries:], cfg.eps)
        sdpa = lambda: F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        want = sdpa().float()
        gap = lambda got: float((got.float() - want).abs().max() / want.abs().max())
        tflops = lambda ms: 4 * queries * keys * cfg.width / ms / 1e9
        sdpa_ms = bench(sdpa)
        row = {"bound": logit_bound(weight, weight), "max_free_allowed": max_free(logit_bound(weight, weight), keys),
               "gap_stable": gap(flash(q, k, v, True)), "gap_max_free": gap(flash(q, k, v, False)),
               "sdpa_ms": sdpa_ms, "sdpa_tflops": tflops(sdpa_ms), "stable": {}, "max_free": {}}
        for m, n, warps, stages in sweep:
            for name, stable in (("stable", True), ("max_free", False)):
                try:  # a tile setting can need more shared memory than an sm has; that setting is skipped
                    ms = bench(lambda: flash(q, k, v, stable, m, n, warps, stages))
                except (OutOfResources, CompilationError):
                    continue
                row[name][f"{m}x{n}_w{warps}_s{stages}"] = {"ms": ms, "tflops": tflops(ms)}
        report["cases"][f"{queries}x{keys}"] = row
        print(queries, keys, json.dumps(row), flush=True)
    return json.dumps(report)


MLP_SWEEP = list(itertools.product(((64, 128, 64), (64, 128, 128), (64, 128, 32), (32, 128, 64), (128, 64, 64), (128, 64, 128), (256, 64, 64), (64, 64, 128), (128, 128, 128), (128, 32, 64)), (2, 4)))


# the fp8 mlp branch against `out(silu(gate(n)) * proj(n))`, n = layer_norm(x) * mod, in bf16, on random weights of
# the model's shapes: relative l2 gap, ms for the reference and for each of the three kernels, and tflop/s of the
# two matmul kernels per (tile, warps). also whether torch's own fp8 cast on the gpu agrees with the cpu's.
def run_mlp(tokens: list[int], sweep: list[tuple[tuple[int, int, int], int]]) -> str:
    import torch
    import torch.nn.functional as F

    from magpie.dit import DitConfig
    from magpie.kernels import fp8_linear, fp8_norm, fp8_swiglu, mlp_fp8, pack_mlp
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    def bench(fn, iters: int = 10) -> float:
        for _ in range(2):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = DitConfig()
    width, hidden = cfg.width, cfg.width * cfg.mlp_ratio
    gate, proj = ((torch.randn(hidden, width, device="cuda") / width**0.5).bfloat16() for _ in range(2))
    out = (torch.randn(width, hidden, device="cuda") / hidden**0.5).bfloat16()
    sample = torch.randn(4096, device="cuda") * 3
    report = {"gpu": torch.cuda.get_device_name(0), "cases": {},
              "gpu_cast_matches_cpu": bool((sample.to(torch.float8_e4m3fn).cpu().view(torch.uint8) == sample.cpu().to(torch.float8_e4m3fn).view(torch.uint8)).all())}
    for count in tokens:
        x = (torch.randn(1, count, width, device="cuda") * 5).bfloat16()
        mod = (1 + torch.randn(1, 1, width, device="cuda") * 0.3).bfloat16()
        normed = lambda: F.layer_norm(x, (width,), eps=cfg.eps) * mod
        reference = lambda: F.linear(F.silu(F.linear(normed(), gate)) * F.linear(normed(), proj), out)
        packed = pack_mlp(gate, proj, out)
        want = reference().float()
        rows, step = fp8_norm(x[0], mod[0], cfg.eps)
        middle, middle_step = fp8_swiglu(rows, step, packed)
        row = {"gap": float((mlp_fp8(x, mod, packed, cfg.eps).float() - want).norm() / want.norm()),
               "norm_gap": float((rows.float() * step[:, None] - normed()[0].float()).norm() / normed().float().norm()),
               "reference_ms": bench(reference), "fused_ms": bench(lambda: mlp_fp8(x, mod, packed, cfg.eps)),
               "norm_ms": bench(lambda: fp8_norm(x[0], mod[0], cfg.eps)), "swiglu": {}, "linear": {}}
        for tile, warps in sweep:
            for name, fn, flops in (("swiglu", lambda: fp8_swiglu(rows, step, packed, tile, warps), 4 * count * width * hidden),
                                    ("linear", lambda: fp8_linear(middle, middle_step, packed.out, packed.out_step, tile, warps), 2 * count * width * hidden)):
                try:  # a tile can need more shared memory than an sm has; that setting is skipped
                    ms = bench(fn)
                except (OutOfResources, CompilationError):
                    continue
                row[name][f"{tile[0]}x{tile[1]}x{tile[2]}_w{warps}"] = {"ms": ms, "tflops": flops / ms / 1e9}
        report["cases"][count] = row
        print(count, json.dumps(row), flush=True)
    return json.dumps(report)


# where a warm vae pass and the request's work outside the dit and vae go: ms for encode and decode in both
# memory layouts, the gpu kernels of each ranked by time, and the cpu-side steps of `render` one by one.
def run_vae() -> str:
    import time

    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    import magpie.pipeline as lean
    from gate import OUT_ROOT, gate_input, load_port
    from magpie.dit import layout

    def clock(fn, iters: int = 5) -> float:
        fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e3

    def kernels(fn) -> dict[str, float]:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        rows = sorted(((e.self_device_time_total / 1e3, e.count, e.key[:70]) for e in prof.key_averages()), reverse=True)
        return {f"{name} x{count}": round(ms, 2) for ms, count, name in rows[:14]}

    dit, vae = load_port("turbo")
    image = lean.resized(gate_input("edit_1024", 0)[1], 1024)[0]
    saved = torch.load(f"{OUT_ROOT}/baseline/turbo/conditioning/edit_1024/00.pt")
    pixels = lean.to_pixels(image).to("cuda", torch.bfloat16)
    latents = vae.encode(pixels)
    shapes = [tuple(latents.shape[-2:])] * 2
    decoded = vae.decode(latents)
    small = torch.randn(1, 8, 4, 4, device="cuda", dtype=torch.bfloat16)
    report = {"image": image.size, "encode_ms": clock(lambda: vae.encode(pixels)), "decode_ms": clock(lambda: vae.decode(latents)),
              "encode_kernels": kernels(lambda: vae.encode(pixels)), "decode_kernels": kernels(lambda: vae.decode(latents)),
              "bf16_nearest_matches_fp32": bool((F.interpolate(small, scale_factor=2.0, mode="nearest-exact") ==
                                                 F.interpolate(small.float(), scale_factor=2.0, mode="nearest-exact").bfloat16()).all()),
              "cpu_steps_ms": {"to_pixels + upload": clock(lambda: lean.to_pixels(image).to("cuda", torch.bfloat16)),
                               "layout": clock(lambda: layout(saved["slots"].to("cuda"), shapes, dit.cfg)),
                               "noise": clock(lambda: torch.randn((1, 1, 64, *shapes[-1]), generator=torch.Generator("cpu").manual_seed(0), dtype=torch.bfloat16).to("cuda")),
                               "to_image": clock(lambda: lean.to_image(decoded))}}
    vae.to(memory_format=torch.channels_last)
    last = pixels.contiguous(memory_format=torch.channels_last)
    report["channels_last"] = {"encode_ms": clock(lambda: vae.encode(last)), "decode_ms": clock(lambda: vae.decode(latents)),
                               "decode_gap": float((vae.decode(latents).float() - decoded.float()).abs().max()),
                               "decode_kernels": kernels(lambda: vae.decode(latents)), "encode_kernels": kernels(lambda: vae.encode(last))}
    print(json.dumps(report, indent=1), flush=True)
    return json.dumps(report)


def bench_kernel(kernel: str, mode: str) -> str:
    if kernel == "vae":
        return run_vae()
    if kernel == "mlp":
        return run_mlp([1024] if mode == "check" else [4096], MLP_SWEEP[:1] if mode == "check" else MLP_SWEEP)
    if kernel == "prep":
        return run([1024, 4096] if mode == "check" else [4096, 8239, 16384, 32815])
    return run_attention(*(([(1024, 2087)], [(64, 64, 4, 2), (128, 64, 8, 2)]) if mode == "check" else ([(4096, 8239)], ATTENTION_SWEEP)))


if __name__ == "__main__":
    print(bench_kernel(*sys.argv[1:3]))
