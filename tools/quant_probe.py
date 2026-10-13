"""what low-precision linear layers would buy and cost, before any are built. no provider in it.

    python tools/quant_probe.py speed|quality|ranges|ablate|inside

`speed`: tflop/s of a matmul at the dit's three linear shapes, for torch (cublas) and a plain triton gemm
in bf16, fp16, fp8 (e4m3) and int8. `quality`: the dit's step error on the gate items when its block
linears see fp8- or nvfp4-rounded weights and activations (arithmetic still bf16, so only rounding is measured).
`ranges`: the largest hidden value of every block's mlp over the gate items, which decides how its fp8 scale must be chosen.
`ablate`: the current build through the gate (images, step error, dinov2), then again with the attention linears fp8-rounded.
`inside`: the same gate with changes inside attention: fp8 q and k, fp8 v, reference keys merged 2x2.
`formats`: step error with attention's operands rounded to int8 / fp8 / nvfp4, one half at a time.
"""

import itertools
import json
import pathlib as pl
import sys
from typing import Any, Callable

import triton
import triton.language as tl

SHAPES = [(4096, 4096, 4096), (4096, 4096, 12288), (4096, 12288, 4096)]  # (tokens, in, out): q/k/v/out, mlp in, mlp out
TILES = list(itertools.product(((128, 128, 32), (128, 128, 64), (128, 256, 32), (128, 256, 64), (256, 128, 32), (64, 128, 64), (128, 128, 128)), (4, 8)))
GROUPS = {"all": ("",), "qkv": ("to_q", "to_k", "to_v"), "attn_out": ("to_out.0",), "mlp_in": ("proj", "gate_layer"), "mlp_out": ("img_mlp.out",),
          "q": ("to_q",), "k": ("to_k",), "v": ("to_v",)}


# out = x @ w for one (BM, BN) tile of the output: x is (M, K), w is (K, N), walked BK columns of x at a time.
# the pointer types pick the arithmetic; BITS reads uint8 as fp8 bits, CAST converts x to w's type in registers.
@triton.jit
def gemm_kernel(x_ptr, w_ptr, out_ptr, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                INT: tl.constexpr, BITS: tl.constexpr, CAST: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    if INT:
        acc = tl.zeros((BM, BN), dtype=tl.int32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        ks = start + tl.arange(0, BK)
        a = tl.load(x_ptr + rows[:, None] * K + ks[None, :])
        b = tl.load(w_ptr + ks[:, None] * N + cols[None, :])
        if BITS:
            a, b = a.to(tl.float8e4nv, bitcast=True), b.to(tl.float8e4nv, bitcast=True)
        if CAST:
            a = a.to(b.dtype)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc.to(out_ptr.dtype.element_ty))


def gemm(x: Any, w: Any, out: Any, tile: tuple[int, int, int], warps: int, bits: bool = False, cast: bool = False) -> Any:
    import torch

    (m, k), n = x.shape, w.shape[1]
    gemm_kernel[(m // tile[0], n // tile[1])](x, w, out, n, k, BM=tile[0], BN=tile[1], BK=tile[2], INT=out.dtype == torch.int32,
                                              BITS=bits, CAST=cast, num_warps=warps, num_stages=3)
    return out


def speed() -> str:
    import torch

    def bench(fn: Callable, iters: int = 10) -> float:
        for _ in range(2):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    # a probe of what this stack supports: an unsupported type or tile is an answer, so it is recorded, not raised.
    def attempt(fn: Callable) -> Any:
        try:
            return fn()
        except Exception as error:  # noqa: BLE001
            return f"{type(error).__name__}: {str(error)[:160]}"

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    fp8 = torch.float8_e4m3fn
    report: dict[str, Any] = {"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
                              "torch": torch.__version__, "triton": triton.__version__, "shapes": {}}
    best: dict[str, tuple[tuple[int, int, int], int]] = {}
    for index, (m, k, n) in enumerate(SHAPES):
        x, w = torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda") / k**0.5
        rate = lambda fn: (lambda ms: {"ms": ms, "tflops": 2 * m * k * n / ms / 1e9})(bench(fn))
        empty = lambda dtype: torch.empty(m, n, device="cuda", dtype=dtype)
        xb, wb, xh, wh = x.bfloat16(), w.bfloat16(), x.half(), w.half()
        x8, w8 = x.cpu().to(fp8).cuda(), w.cpu().to(fp8).cuda()
        xi, wi = (x * 32).round().clamp(-127, 127).to(torch.int8), (w * 2048).round().clamp(-127, 127).to(torch.int8)
        row: dict[str, Any] = {f"torch_{name}": attempt(lambda: rate(lambda: a @ b))
                               for name, (a, b) in (("bf16", (xb, wb)), ("fp16", (xh, wh)), ("fp8", (x8, w8)), ("int8", (xi, wi)))}
        variants = {
            "triton_bf16": (xb, wb, empty(torch.bfloat16), {}),
            "triton_fp16": (xh, wh, empty(torch.float16), {}),
            "triton_fp8": (x8, w8, empty(torch.bfloat16), {}),
            "triton_fp8_bits": (x8.view(torch.uint8), w8.view(torch.uint8), empty(torch.bfloat16), {"bits": True}),
            "triton_fp8_weight_bf16_x": (xb, w8, empty(torch.bfloat16), {"cast": True}),
            "triton_int8": (xi, wi, empty(torch.int32), {}),
        }
        for name, (a, b, out, flags) in variants.items():
            settings = {f"{tile[0]}x{tile[1]}x{tile[2]}_w{warps}": (tile, warps) for tile, warps in (TILES if index == 0 else [best[name]] if name in best else [])}
            runs = {key: attempt(lambda: rate(lambda: gemm(a, b, out, *setting, **flags))) for key, setting in settings.items()}
            good = {key: got["tflops"] for key, got in runs.items() if not isinstance(got, str)}
            if index == 0 and good:
                best[name] = settings[max(good, key=good.get)]
                want = (a.view(fp8) if flags.get("bits") else a).float() @ (b.view(fp8) if flags.get("bits") else b).float()
                runs["gap"] = float((gemm(a, b, out, *best[name], **flags).float() - want).norm() / want.norm())
            row[name] = runs
            print(m, k, n, name, json.dumps(good)[:500], json.dumps([v for v in runs.values() if isinstance(v, str)][:1]), flush=True)
        report["shapes"][f"{m}x{k}->{n}"] = row
    return json.dumps(report)


# x snapped to a binary floating-point grid with `bits` mantissa bits, smallest exponent `low`, largest value `top`.
def grid(x: Any, bits: int, low: int, top: float) -> Any:
    import torch

    step = torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(2.0**low))) - bits)
    return (torch.round(x / step) * step).clamp(-top, top)


# fp8 e4m3 (3 mantissa bits, up to 448) with one scale per row of the last axis.
def fp8_rounded(x: Any) -> Any:
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    return (grid(x.float() / scale, 3, -6, 448.0) * scale).to(x.dtype)


# nvfp4: e2m1 values (0, 0.5, 1, 1.5, 2, 3, 4, 6) in blocks of 16 along the last axis, one fp8 scale per block.
def nvfp4_rounded(x: Any) -> Any:
    blocks = x.float().unflatten(-1, (-1, 16))
    scale = grid(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6, 3, -100, float("inf"))
    return (grid(blocks / scale, 1, 0, 6.0) * scale).flatten(-2).to(x.dtype)


# mxfp8: e4m3 values in blocks of 32 along the last axis, one power-of-two scale per block (what the block-scaled
# tensor-core instruction applies itself).
def mxfp8_rounded(x: Any) -> Any:
    import torch

    blocks = x.float().unflatten(-1, (-1, 32))
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0)))
    return (grid(blocks / scale, 3, -6, 448.0) * scale).flatten(-2).to(x.dtype)


FORMATS = {"fp8": fp8_rounded, "nvfp4": nvfp4_rounded, "mxfp8": mxfp8_rounded}


# step errors of `dit` on the first `items` gate items of `case`, each step fed the baseline's own inputs.
def step_pass(dit: Any, case: str, items: int) -> list[list[float]]:
    import torch

    import magpie.pipeline as lean
    from gate import OUT_ROOT, gate_input, step_errors
    from magpie.sampling import TURBO

    folder = pl.Path(OUT_ROOT) / "baseline" / "turbo"
    rows = []
    for index in range(items):
        saved = torch.load(folder / "conditioning" / case / f"{index:02d}.pt")
        image = lean.resized(gate_input(case, index)[1], 1024)[0]
        shapes = [(image.height // 16, image.width // 16)] * 2
        rows.append(step_errors(dit, torch.load(folder / "steps" / case / f"{index:02d}.pt"), saved["text"], saved["slots"], shapes, TURBO))
    return rows


# step error of the baseline arithmetic with the chosen block linears rounded: weights only, or weights and
# the activations entering them. every format on all linears over the five gate items, then one group of
# linears at a time over two items to see which layers carry the error.
def quality(case: str = "edit_1024") -> str:
    from torch import nn

    import magpie.dit as dit_module
    from gate import SUITE, load_port, unfused

    dit, _ = load_port("turbo", slim=False)
    unfused(dit_module)
    linears = {name: module for name, module in dit.transformer_blocks.named_modules() if isinstance(module, nn.Linear)}
    plain = {name: module.weight.data for name, module in linears.items()}

    report: dict[str, Any] = {"case": case, "unrounded": step_pass(dit, case, 1), "runs": {}}
    plans = [(fmt, acts, "all", len(SUITE)) for fmt in FORMATS for acts in (False, True)] + \
            [(fmt, True, group, 2) for fmt in FORMATS for group in list(GROUPS)[1:]]
    for fmt, acts, group, items in plans:
        chosen = [name for name in linears if any(name.endswith(end) for end in GROUPS[group])]
        hooks = []
        for name in chosen:
            linears[name].weight.data = FORMATS[fmt](plain[name])
            if acts:
                hooks.append(linears[name].register_forward_pre_hook(lambda _, args, fmt=fmt: (FORMATS[fmt](args[0]),)))
        rows = step_pass(dit, case, items)
        for name in chosen:
            linears[name].weight.data = plain[name]
        for hook in hooks:
            hook.remove()
        flat = [e for row in rows for e in row]
        key = f"{fmt}_{'weights+activations' if acts else 'weights'}_{group}"
        report["runs"][key] = {"layers": len(chosen), "worst": max(flat), "mean": sum(flat) / len(flat), "step_error": rows}
        print(key, len(chosen), f"worst {max(flat):.4f} mean {sum(flat) / len(flat):.4f}", flush=True)
    return json.dumps(report)


# per block, over every step of the five gate items in baseline arithmetic: the largest |silu(gate) * proj|
# entering the mlp's output layer, and the smallest per-token maximum of it (how far below the top a row can sit).
def ranges(case: str = "edit_1024") -> str:
    import magpie.dit as dit_module
    from gate import SUITE, load_port, unfused

    dit, _ = load_port("turbo", slim=False)
    unfused(dit_module)
    top, low = [0.0] * len(dit.transformer_blocks), [float("inf")] * len(dit.transformer_blocks)

    def watch(index: int) -> Callable:
        def hook(_: Any, args: tuple) -> None:
            peaks = args[0].float().abs().amax(-1)
            top[index], low[index] = max(top[index], float(peaks.max())), min(low[index], float(peaks.min()))
        return hook

    for index, block in enumerate(dit.transformer_blocks):
        block.img_mlp.out.register_forward_pre_hook(watch(index))
    step_pass(dit, case, len(SUITE))
    print("top", [round(v, 1) for v in top], "lowest row peak", [round(v, 2) for v in low], flush=True)
    return json.dumps({"case": case, "top": top, "lowest_row_peak": low})


# the gate on the build as it is, then with fp8-rounded weights and inputs on the attention linears (the layers
# still in bf16), one projection or pair at a time. each row is a full gate verdict against the frozen baseline.
def ablate(case: str = "edit_1024") -> str:
    from torch import nn

    from gate import OUT_ROOT, gate_pass, load_port

    dit, vae = load_port("turbo", slim=False, attention=())
    folder = pl.Path(OUT_ROOT) / "baseline" / "turbo"
    linears = {name: module for name, module in dit.transformer_blocks.named_modules() if isinstance(module, nn.Linear)}
    report: dict[str, Any] = {}
    for label, groups in (("+ q", ("q",)), ("+ k", ("k",)), ("+ v", ("v",)), ("+ q + k", ("q", "k")), ("+ v + attention out", ("v", "attn_out"))):
        chosen = [name for name in linears if any(name.endswith(end) for group in groups for end in GROUPS[group])]
        plain = {name: linears[name].weight.data for name in chosen}
        hooks = [linears[name].register_forward_pre_hook(lambda _, args: (fp8_rounded(args[0]),)) for name in chosen]
        for name in chosen:
            linears[name].weight.data = fp8_rounded(plain[name])
        row = gate_pass(dit, vae, "turbo", [case], folder, pl.Path(OUT_ROOT) / "gate" / f"ablate-{len(report)}" / "turbo", {})[case]
        for name in chosen:
            linears[name].weight.data = plain[name]
        for hook in hooks:
            hook.remove()
        steps = [e for item in row["step_error"] for e in item]
        report[label] = {"layers": len(chosen), "worst_step_error": max(steps), "mean_step_error": sum(steps) / len(steps),
                         "worst_dino": min(row["dino_vs_baseline"]), "dino": row["dino_vs_baseline"],
                         "psnr_vs_baseline_db": row["psnr_vs_baseline_db"], "dit_vae_ms": row["total"]["p50"] * 1e3, "dit_ms": row["dit"]["p50"] * 1e3}
        print(label, json.dumps(report[label]), flush=True)
    return json.dumps(report)


# the gate (dinov2, dreamsim, step error) with block linears emulated in a block-scaled format, weights and inputs
# both rounded, everything else as the current build runs it: nvfp4 or mxfp8 on the mlp alone, the attention
# linears alone, and all of them. the emulated layers run as bf16 matmuls, so only the quality means anything.
def lowbits(case: str = "edit16_1024") -> str:
    from torch import nn

    import magpie.dit as dit_module
    from gate import OUT_ROOT, gate_pass, load_port

    dit, vae = load_port("turbo", slim=False, attention=())
    dit_module.fp8_ready = lambda packed, x: False
    folder = pl.Path(OUT_ROOT) / "baseline" / "turbo"
    linears = {name: module for name, module in dit.transformer_blocks.named_modules() if isinstance(module, nn.Linear)}
    report: dict[str, Any] = {}
    for form, label, groups in ((form, label, groups) for form in ("nvfp4", "mxfp8")
                                for label, groups in (("mlp", ("mlp_in", "mlp_out")), ("attention linears", ("qkv", "attn_out")), ("all linears", ("all",)))):
        rounded = FORMATS[form]
        chosen = [name for name in linears if any(name.endswith(end) for group in groups for end in GROUPS[group])]
        plain = {name: linears[name].weight.data for name in chosen}
        hooks = [linears[name].register_forward_pre_hook(lambda _, args: (rounded(args[0]),)) for name in chosen]
        for name in chosen:
            linears[name].weight.data = rounded(plain[name])
        row = gate_pass(dit, vae, "turbo", [case], folder, pl.Path(OUT_ROOT) / "gate" / f"lowbits-{len(report)}" / "turbo", {})[case]
        for name in chosen:
            linears[name].weight.data = plain[name]
        for hook in hooks:
            hook.remove()
        steps = [e for item in row["step_error"] for e in item]
        report[f"{form}, {label}"] = {"layers": len(chosen), "worst_step_error": max(steps), "mean_step_error": sum(steps) / len(steps),
                                      "dino": row["dino_vs_baseline"], "dreamsim": row["dreamsim_vs_baseline"]}
        print(form, label, json.dumps(report[f"{form}, {label}"]), flush=True)
    (pl.Path(OUT_ROOT) / "lowbits.json").write_text(json.dumps(report, indent=1))
    return json.dumps(report)


# the gate on the current build with one change inside attention at a time, all emulated in torch so only the
# quality is measured: q and k rounded to fp8 per (token, head) (what an fp8 score matmul would see), v rounded
# the same way, and the reference image's keys and values averaged over 2x2 token groups with ln 4 added to
# their logits (so a merged key keeps the weight of the four it replaces; 8,239 keys become 5,167 at 1024).
# keys are averaged either as they are (rope already applied) or with rope undone first and reapplied at the
# group's centre. `pick` limits the run to the probes whose label contains it.
def inside(pick: str = "", case: str = "edit_1024") -> str:
    import math

    import torch

    import magpie.dit as dit_module
    import magpie.pipeline as lean
    from gate import OUT_ROOT, gate_pass, load_port

    dit, vae = load_port("turbo")
    folder = pl.Path(OUT_ROOT) / "baseline" / "turbo"
    plain, build = dit_module.attend, dit_module.layout
    seen: dict[str, Any] = {}

    def layout(slots: Any, shapes: list, cfg: Any) -> Any:
        seen.update(lay=build(slots, shapes, cfg), shape=shapes[0])
        return seen["lay"]

    def merged(q: Any, k: Any, v: Any, bound: float, mask: Any = None, centred: bool = False) -> Any:
        lay, (height, width) = seen["lay"], seen["shape"]
        if mask is not None or k.shape[1] != lay.image.numel() or q.shape[1] != k.shape[1] - lay.prefix:
            return plain(q, k, v, bound, mask)
        image = lay.image[: lay.prefix]
        grid = (height // 2, 2, width // 2, 2)

        def fold(x: Any, rotated: bool) -> Any:
            rows = x[:, : lay.prefix][:, image].float()
            if rotated:  # undo each token's rope, average, then rotate by the phase of the group's centre
                phase = lay.rope[: lay.prefix][image].unflatten(0, grid)
                corner = phase[:, 0, :, 0]
                centre = (corner * torch.sqrt(phase[:, 0, :, 1] / corner) * torch.sqrt(phase[:, 1, :, 0] / corner)).flatten(0, 1)
                pairs = torch.view_as_complex(rows.unflatten(-1, (-1, 2)).contiguous()) * lay.rope[: lay.prefix][image].conj()[None, :, None]
                groups = torch.view_as_real(pairs.unflatten(1, grid).mean((2, 4)).flatten(1, 2) * centre[None, :, None]).flatten(-2)
            else:
                groups = rows.unflatten(1, grid).mean((2, 4)).flatten(1, 2)
            return torch.cat([x[:, : lay.prefix][:, ~image], groups.to(x.dtype), x[:, lay.prefix :]], dim=1)

        texts, groups = int((~image).sum()), height * width // 4
        bias = q.new_zeros(texts + groups + q.shape[1])
        bias[texts : texts + groups] = math.log(4)
        return dit_module.sdpa(q, fold(k, centred), fold(v, False), bias[None])

    probes = {"current build": plain,
              "fp8 q and k": lambda q, k, v, bound, mask=None: plain(fp8_rounded(q), fp8_rounded(k), v, bound, mask),
              "fp8 v": lambda q, k, v, bound, mask=None: plain(q, k, fp8_rounded(v), bound, mask),
              "reference keys merged 2x2": merged,
              "reference keys merged 2x2, rope at the centre": lambda q, k, v, bound, mask=None: merged(q, k, v, bound, mask, True)}
    dit_module.layout = lean.layout = layout
    report: dict[str, Any] = {}
    for label, attend in probes.items():
        if pick not in label:
            continue
        dit_module.attend = attend
        row = gate_pass(dit, vae, "turbo", [case], folder, pl.Path(OUT_ROOT) / "gate" / f"inside-{list(probes).index(label)}" / "turbo", {})[case]
        steps = [e for item in row["step_error"] for e in item]
        report[label] = {"worst_step_error": max(steps), "mean_step_error": sum(steps) / len(steps), "dino": row["dino_vs_baseline"],
                         "psnr_vs_baseline_db": row["psnr_vs_baseline_db"]}
        print(label, json.dumps(report[label]), flush=True)
    return json.dumps(report)


# int8 with one scale per `block` rows of axis 1 (tokens), as sageattention's score matmul sees q and k.
def int8_rounded(x: Any, block: int) -> Any:
    import torch
    import torch.nn.functional as F

    padded = F.pad(x.float(), (0, 0, 0, 0, 0, -x.shape[1] % block)).unflatten(1, (-1, block))
    scale = padded.abs().amax((2, 4), keepdim=True).clamp_min(1e-12) / 127
    return (torch.round(padded / scale) * scale).flatten(1, 2)[:, : x.shape[1]]


# nvfp4 along the last axis where each block of 16 picks, among its largest value over 6 shrunk by `shrinks`, the
# scale with the least squared error (clipping the block's outliers can beat covering them).
def nvfp4_fitted(x: Any, shrinks: tuple[float, ...] = (1.0, 0.85, 0.7, 0.55)) -> Any:
    import torch

    blocks = x.float().unflatten(-1, (-1, 16))
    top = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6
    tries = torch.stack([grid(blocks / scale, 1, 0, 6.0) * scale for scale in (grid(top * shrink, 3, -100, float("inf")) for shrink in shrinks)])
    best = (tries - blocks).square().sum(-1, keepdim=True).argmin(0, keepdim=True)
    return tries.gather(0, best.expand(1, *blocks.shape))[0].flatten(-2).to(x.dtype)


# attention with exact fp32 arithmetic on rounded operands, to price a number format before a kernel exists.
# q, k, v are (1, tokens, heads, dim). `qk`: "int8" (q per 32 tokens, k per 64, mean key subtracted: sageattention 2)
# or "nvfp4" (blocks of 16 along the head dimension, mean key subtracted). `pv` is a set of flags: "p4" / "v4" put
# the weights / the values in nvfp4 in blocks of 16 keys (what blackwell's 4-bit instruction contracts over), fp8
# otherwise (weights on the e4m3 grid below 1, values with one scale per channel); "centre" subtracts each channel's
# mean value first and adds it back to the output (exact, the weights sum to 1); "fit" picks each block's scale by
# squared error; "renorm" divides the rounded weights by their sum, as a kernel that sums what it multiplied does.
def rounded_attention(q: Any, k: Any, v: Any, qk: str, pv: tuple[str, ...]) -> Any:
    import torch

    centred = k.float() - k.float().mean(1, keepdim=True)
    q, centred = (int8_rounded(q, 32), int8_rounded(centred, 64)) if qk == "int8" else (nvfp4_rounded(q), nvfp4_rounded(centred))
    weights = (q.transpose(1, 2).float() @ centred.permute(0, 2, 3, 1) * q.shape[-1] ** -0.5).softmax(-1)  # (1, heads, queries, keys)
    values = v.float().permute(0, 2, 3, 1)  # (1, heads, dim, keys)
    mean = values.mean(-1, keepdim=True) if "centre" in pv else 0.0
    pad = -weights.shape[-1] % 16
    block4 = lambda x, rounded: rounded(torch.nn.functional.pad(x, (0, pad)))[..., : x.shape[-1]]
    weights = block4(weights, nvfp4_rounded) if "p4" in pv else grid(weights * 448, 3, -6, 448.0) / 448
    weights = weights / weights.sum(-1, keepdim=True) if "renorm" in pv else weights
    values = block4(values - mean, nvfp4_fitted if "fit" in pv else nvfp4_rounded) if "v4" in pv else fp8_rounded(values - mean)
    return (weights @ (values + mean).transpose(-1, -2)).transpose(1, 2).to(v.dtype)


# step error of the current build with every unmasked attention call replaced by `rounded_attention` in one format
# at a time: which half of attention can take blackwell's 4-bit instruction. 5 gate items, same inputs per step.
def attention_formats(case: str = "edit_1024", items: str = "5") -> str:
    import magpie.dit as dit_module
    from gate import load_port

    dit, _ = load_port("turbo")
    plain = dit_module.attend
    dit_module.SAGE_CACHE = False
    probes = {"sageattention 2 (the real kernel)": None, "int8 qk, fp8 pv": ("int8", ()), "int8 qk, nvfp4 p": ("int8", ("p4",)),
              "int8 qk, nvfp4 pv": ("int8", ("p4", "v4")), "int8 qk, nvfp4 pv, v centred and fitted": ("int8", ("p4", "v4", "centre", "fit")),
              "int8 qk, nvfp4 pv, v centred and fitted, renormalised": ("int8", ("p4", "v4", "centre", "fit", "renorm")),
              "nvfp4 qk, fp8 pv": ("nvfp4", ())}
    report: dict[str, Any] = {}
    for label, formats in probes.items():
        dit_module.attend = plain if formats is None else (
            lambda q, k, v, bound, mask=None, formats=formats: plain(q, k, v, bound, mask) if mask is not None else rounded_attention(q, k, v, *formats))
        steps = [e for item in step_pass(dit, case, int(items)) for e in item]
        report[label] = {"mean_step_error": round(sum(steps) / len(steps), 4), "worst_step_error": round(max(steps), 4)}
        print("formats", label, json.dumps(report[label]), flush=True)
    return json.dumps(report)


FIT = (1.0, 0.85, 0.7, 0.55)


# a stand-in for `dit.attend` that sends unmasked calls to the native sm_120 kernel (native/attn.cu). `values` is None
# for the 8-bit build, else (centre, shrinks, rotate) for the build whose weights x values matmul is 4-bit: see
# `tune_attn.nvfp4_values`; rotated values are rounded in a hadamard basis of each head's channels and the output is
# turned back, which a build would fold into the to_v and to_out weights. operands are rounded per call in torch:
# this prices the arithmetic, not the speed.
def native_attend(values: tuple | None) -> Any:
    import torch

    import magpie.dit as dit_module
    from tune_attn import BASE, compiled, nvfp4_values, quantised

    plain, config = dit_module.attend, {**BASE, "PV4": int(values is not None)}
    rotate = values is not None and values[2]
    hadamard = torch.ones(1, 1, device="cuda")
    for _ in range(7):
        hadamard = torch.cat([torch.cat([hadamard, hadamard], 1), torch.cat([hadamard, -hadamard], 1)], 0)
    hadamard = hadamard / 128**0.5  # symmetric and orthogonal: its own inverse

    def attend(q: Any, k: Any, v: Any, bound: float, mask: Any = None) -> Any:
        if mask is not None:
            return plain(q, k, v, bound, mask)
        v = (v.float() @ hadamard).to(v.dtype) if rotate else v
        q8, k8, v8, q_scale, k_scale, v_scale = quantised(q.contiguous(), k.contiguous(), v, config)
        v8, v_scale = nvfp4_values(v, *values[:2]) if values else (v8, v_scale)
        out = torch.empty_like(q)
        compiled(config).attend(q8, k8, v8, out, q_scale, k_scale, v_scale, q.shape[-1] ** -0.5)
        return (out.float() @ hadamard).to(out.dtype) if rotate else out

    return attend


# step error of the current build on the native kernel, for the 8-bit build and each way of rounding the values in
# the 4-bit one. 5 gate items, each step fed the baseline's own inputs.
def native_formats(case: str = "edit_1024", items: str = "5") -> str:
    import magpie.dit as dit_module
    from gate import load_port

    dit, _ = load_port("turbo")
    plain = dit_module.attend
    dit_module.SAGE_CACHE = False
    probes = {"8-bit fork": None, "4-bit pv": (False, (1.0,), False), "4-bit pv, centred": (True, (1.0,), False), "4-bit pv, centred, fitted": (True, FIT, False),
              "4-bit pv, centred, rotated": (True, (1.0,), True), "4-bit pv, centred, fitted, rotated": (True, FIT, True)}
    report: dict[str, Any] = {}
    for label, values in probes.items():
        dit_module.attend = plain
        dit_module.attend = native_attend(values)
        steps = [e for item in step_pass(dit, case, int(items)) for e in item]
        report[label] = {"mean_step_error": round(sum(steps) / len(steps), 4), "worst_step_error": round(max(steps), 4)}
        print("native", label, json.dumps(report[label]), flush=True)
    return json.dumps(report)


# the image gate with attention on the native kernel's 4-bit weights x values matmul (values centred, fitted, rotated).
def native_gate(cases: str = "edit16_2048,edit16_1024", tag: str = "pv4") -> str:
    import magpie.dit as dit_module
    from gate import OUT_ROOT, gate_pass, load_port, verdict

    dit, vae = load_port("turbo")
    dit_module.SAGE_CACHE = False
    dit_module.attend = native_attend((True, FIT, True))
    folder = pl.Path(OUT_ROOT) / "baseline" / "turbo"
    result = {"magpie": gate_pass(dit, vae, "turbo", cases.split(","), folder, pl.Path(OUT_ROOT) / "gate" / tag / "turbo", {})}
    result["verdict"] = verdict(result, json.loads((pl.Path(__file__).resolve().parent.parent / "baseline" / "turbo.json").read_text()))
    (pl.Path(OUT_ROOT) / "gate" / f"{tag}-turbo.json").write_text(json.dumps(result, indent=1))
    return json.dumps(result["verdict"], indent=1)


if __name__ == "__main__":
    print({"speed": speed, "quality": quality, "ranges": ranges, "ablate": ablate, "inside": inside, "lowbits": lowbits, "formats": attention_formats, "native": native_formats, "native-gate": native_gate}[sys.argv[1]](*sys.argv[2:]))
