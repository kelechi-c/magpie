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


SWIGLU_TILES = [(64, 128, 32), (64, 128, 64), (32, 128, 64), (64, 64, 64), (64, 256, 64), (128, 64, 64), (64, 128, 128), (32, 256, 64)]
PROJECT_TILES = list(itertools.product((64, 128, 256), (64, 128, 256), (32, 64, 128)))


# the fp8 mlp branch against `x + gate * out(silu(gate_layer(n)) * proj(n))`, n = layer_norm(x) * mod, in bf16, on
# random weights of the model's shapes: relative l2 gap of the branch, ms for the reference and for each kernel,
# and tflop/s of the two matmul kernels over (tile, warps), then the best few again over pipeline stages.
# `check` runs one setting of each.
def run_mlp(tokens: int, wide: bool) -> str:
    import torch
    import torch.nn.functional as F

    from magpie.dit import DitConfig
    from magpie.kernels import PROJECT_TILE, SWIGLU_TILE, fp8_align, fp8_norm, fp8_project, fp8_swiglu, mlp_fp8, pack_mlp
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

    # tflop/s of `run(tile, warps, stages)` per setting: every (tile, warps) at 3 stages, then the best four at 2 and 4.
    # a tile can need more shared memory than an sm has; that setting is skipped.
    def sweep(run, tiles: list, flops: float) -> dict[str, float]:
        def rate(tile: tuple, warps: int, stages: int) -> float:
            try:
                return flops / bench(lambda: run(tile, warps, stages)) / 1e9
            except (OutOfResources, CompilationError):
                return 0.0

        first = {(tile, warps, 3): rate(tile, warps, 3) for tile in tiles for warps in (4, 8)}
        top = sorted(first, key=first.get, reverse=True)[:4] if wide else []
        rates = {**first, **{(tile, warps, stages): rate(tile, warps, stages) for tile, warps, _ in top for stages in (2, 4)}}
        return {f"{t[0]}x{t[1]}x{t[2]}_w{w}_s{st}": round(v, 1) for (t, w, st), v in sorted(rates.items(), key=lambda kv: -kv[1]) if v}

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = DitConfig()
    width, hidden = cfg.width, cfg.width * cfg.mlp_ratio
    gate, proj = ((torch.randn(hidden, width, device="cuda") / width**0.5).bfloat16() for _ in range(2))
    out = (torch.randn(width, hidden, device="cuda") / hidden**0.5).bfloat16()
    x = (torch.randn(1, tokens, width, device="cuda") * 5).bfloat16()
    mod = (1 + torch.randn(1, 1, width, device="cuda") * 0.3).bfloat16()
    normed = lambda: F.layer_norm(x, (width,), eps=cfg.eps) * mod
    branch = lambda: F.linear(F.silu(F.linear(normed(), gate)) * F.linear(normed(), proj), out)
    packed = pack_mlp(gate, proj, out)
    want = branch().float()
    rows, step = fp8_norm(x[0], mod[0], cfg.eps)
    middle, steps = fp8_swiglu(rows, step, packed)
    blockwise = (middle.float().unflatten(1, (-1, hidden // steps.shape[1])) * steps[:, :, None]).flatten(1)
    row_step = fp8_align(middle, steps)
    last = (packed.out, packed.out_step)
    report = {"gpu": torch.cuda.get_device_name(0), "tokens": tokens,
              "gap": float((mlp_fp8(x, mod, torch.ones_like(mod), packed, cfg.eps).float() - x.float() - want).norm() / want.norm()),
              "align_gap": float((middle.float() * row_step[:, None] - blockwise).norm() / blockwise.norm()),
              "reference_ms": bench(lambda: x + mod * branch()), "fused_ms": bench(lambda: mlp_fp8(x, mod, mod, packed, cfg.eps)),
              "norm_ms": bench(lambda: fp8_norm(x[0], mod[0], cfg.eps)), "align_ms": bench(lambda: fp8_align(middle, steps)),
              "swiglu_ms": bench(lambda: fp8_swiglu(rows, step, packed)),
              "project_ms": bench(lambda: fp8_project(middle, row_step, last, skip=x[0], gate=mod[0])),
              "swiglu_tflops": sweep(lambda tile, warps, stages: fp8_swiglu(rows, step, packed, tile, warps, stages),
                                     SWIGLU_TILES if wide else [SWIGLU_TILE], 4 * tokens * width * hidden),
              "project_tflops": sweep(lambda tile, warps, stages: fp8_project(middle, row_step, last, None, x[0], mod[0], tile, warps, stages),
                                      PROJECT_TILES if wide else [PROJECT_TILE], 2 * tokens * width * hidden)}
    square = fp8_swiglu(rows, step, packed)[0][:, :width].contiguous()
    best = [tuple(int(v) for v in key.split("_")[0].split("x")) for key in list(report["project_tflops"])[:6]]
    report["project_4096_tflops"] = sweep(lambda tile, warps, stages: fp8_project(square, row_step, (packed.out[:width].contiguous(), packed.out_step), None, None, None, tile, warps, stages),
                                          list(dict.fromkeys(best)), 2 * tokens * width * width)
    print(json.dumps(report, indent=1), flush=True)
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
        return run_mlp(1024 if mode == "check" else 4096, mode == "sweep")
    if kernel == "prep":
        return run([1024, 4096] if mode == "check" else [4096, 8239, 16384, 32815])
    return run_attention(*(([(1024, 2087)], [(64, 64, 4, 2), (128, 64, 8, 2)]) if mode == "check" else ([(4096, 8239)], ATTENTION_SWEEP)))


if __name__ == "__main__":
    print(bench_kernel(*sys.argv[1:3]))
