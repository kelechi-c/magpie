"""Results charts for magpie: Qwen-Image 2.1 on one RTX PRO 6000, official pipeline against the lean port.

    cd /home/tensor/Shared/code/ml/magpie && uv run --no-project --with matplotlib python tools/make_charts.py [chart ...]

Numbers come from out/bench/summary.json, out/trace/kernels.json, out/roofline.json, baseline/turbo.json and
baseline/ladder.json. Writes
qwen21_<chart>[_mobile][_dark].png and sources.json into assets/charts at 200 dpi.
"""

import datetime as dt
import hashlib
import json
import pathlib as pl
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, "/home/tensor/.claude/skills/chart-craft")
from chartkit import DARK, LIGHT, TALL, WIDE, Shape, Theme, bar, canvas, clean, fit, ms, row_label, style, tag  # noqa: E402

ROOT = pl.Path("/home/tensor/Shared/code/ml/magpie")
SUMMARY = ROOT / "out/bench/summary.json"
KERNELS = ROOT / "out/trace/kernels.json"
BASELINE = ROOT / "baseline/turbo.json"
LADDER = ROOT / "baseline/ladder.json"
FLOORS = ROOT / "out/roofline.json"
OUT = ROOT / "assets/charts"
PREFIX = "qwen21"
TURBO, BASE, OFFICIAL, LEAN = "turbo", "base", "diffusers", "magpie"
RTX_5090_GB = 32.0
ROOFLINE: tuple[str, ...] = ("gemm_8192_bf16_tflops", "mlp_4096x4096x12288_bf16_tflops", "sdpa_16384_bf16_tflops")
STAGE_FIELDS: tuple[str, ...] = ("total_ms", "text_ms", "dit_ms", "vae_encode_ms", "vae_decode_ms")
CaseRow = tuple[str, str, str]  # (variant, case, label)
StageRow = tuple[str, float, float]  # (stage, port ms, now ms)
CASE_ROWS: tuple[CaseRow, ...] = (  # edits only: the project follows image editing
    (TURBO, "edit_1024", "Turbo (8 steps)\nedit 1024"), (TURBO, "edit_2048", "Turbo (8 steps)\nedit 2048"),
    (BASE, "edit_1024", "base (40 steps)\nedit 1024"), (BASE, "edit_2048", "base (40 steps)\nedit 2048"),
)
SIZE_ROWS: tuple[tuple[str, str], ...] = (("edit_1024", "edit\n1024"), ("edit_2048", "edit\n2048"))
SCALE_PAIRS: tuple[tuple[str, str], ...] = (("edit_2048", "edit_1024"),)
OP_LABELS: dict[str, str] = {
    "matmul (linear layers)": "matmul\n(linear layers)", "attention": "attention",
    "convolution (vae)": "convolution\n(VAE)", "elementwise, casts and copies": "elementwise, casts\nand copies",
}
SpanNames = tuple[str, ...]
SPAN_SEGMENTS: tuple[tuple[str, SpanNames], ...] = (  # kernel-time segments in stacking order; VAE decode + encode summed
    ("MLP", ("mlp (swiglu)",)), ("QKV + QK-norm + RoPE", ("qkv + qk-norm + rope",)),
    ("attention", ("attention (sdpa)",)), ("attention output", ("attention out",)),
    ("VAE decode + encode", ("vae decode", "vae encode")),
)
KERNEL_SPANS: tuple[tuple[str, str], ...] = (
    ("qkv + qk-norm + rope", "qkv + qk-norm +\nrope"), ("mlp (swiglu)", "mlp (swiglu)"), ("vae decode", "vae decode"),
)
FLOOR_SPANS: tuple[tuple[str, str], ...] = (
    ("mlp (swiglu)", "MLP"), ("qkv + qk-norm + rope", "QKV + QK-norm\n+ RoPE"), ("attention (sdpa)", "attention"),
    ("attention out", "attention\noutput"), ("block glue", "norms, modulation,\nresiduals"), ("vae decode", "VAE decode"),
    ("vae encode", "VAE encode"),
)
MATMUL_KINDS: SpanNames = ("gemm", "conv", "attention")
SMALL_KINDS: SpanNames = ("elementwise", "copy", "layout")
LABEL_PCT = 6.0  # narrowest stacked segment, in percent, that still gets its label

Summary = dict[str, Any]
Plot = Callable[[Theme, Shape, Summary], tuple[Figure, tuple[str, ...]]]


class Pt(NamedTuple):
    """A plotted number and the summary.json path it was read from."""
    value: float
    src: str


def get(d: Summary, variant: str, engine: str, case: str, field: str) -> Pt:
    """One field of one benchmark row, with the path it came from."""
    row = next(r for r in d["rows"] if (r["variant"], r["engine"], r["case"]) == (variant, engine, case))
    return Pt(row[field], f"rows[variant={variant},engine={engine},case={case}].{field}")


def pct(part: float, whole: float) -> float:
    """Share of a whole, in percent."""
    return 100 * part / whole


def gib(v: float) -> str:
    """Memory value with its unit."""
    return f"{v:.1f} GiB"


def ref_line(ax: Axes, th: Theme, x: float, dotted: bool = False) -> None:
    """Vertical reference line across the plot: dashed for ceilings and limits, dotted for weights."""
    ax.axvline(x, color=th.muted, linewidth=1.1, linestyle=(0, (1, 2)) if dotted else (0, (4, 3)), zorder=2)


def ref_text(ax: Axes, th: Theme, sh: Shape, x: float, y: float, text: str, ha: str) -> None:
    """Boxed label placed beside a reference line, in the muted colour."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=10.5 * sh.k, color=th.muted, zorder=6,
            bbox={"boxstyle": "square,pad=0.12", "fc": th.bg, "ec": "none"})


def span_sum(d: Summary, case: str, names: SpanNames) -> Pt:
    """GPU milliseconds of one or more profiled spans added together, with the paths they came from."""
    pts = [Pt(d["kernels"][case]["spans"][n]["gpu_ms"], f"kernels[{case}].spans[{n}].gpu_ms") for n in names]
    return Pt(sum(p.value for p in pts), ", ".join(p.src for p in pts))


def kind_sum(d: Summary, case: str, name: str, kinds: SpanNames) -> Pt:
    """Kernel milliseconds of the given kernel classes inside one profiled span, with the path they came from."""
    by_kind = d["kernels"][case]["spans"][name]["kernels_ms"]
    return Pt(sum(by_kind.get(k, 0.0) for k in kinds), f"kernels[{case}].spans[{name}].kernels_ms[{'+'.join(kinds)}]")


def stage_split(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Share of each official request in text encoder, DiT and VAE; the unfilled tail is other overhead."""
    rows = [[get(d, v, OFFICIAL, c, f) for f in STAGE_FIELDS] for v, c, _ in CASE_ROWS]
    srcs = tuple(p.src for row in rows for p in row)
    shares: list[tuple[float, float, float, float]] = []
    for row in rows:
        total, text, dit, enc, dec = (p.value for p in row)
        shares.append((total, pct(text, total), pct(dit, total), pct(enc + dec, total)))
    dit_share = [s[2] for s in shares]
    fig, ax = canvas(th, sh, sh.pick(5.2, 6.4),
                     f"The DiT takes {min(dit_share):.0f} to {max(dit_share):.0f} percent of every edit request",
                     "Percent of each request's total time, official pipeline, bf16, batch 1. Total seconds at right.",
                     "One NVIDIA RTX PRO 6000 Blackwell (95 GiB), PyTorch 2.14.1, CUDA 13.0. Warm p50 of 5 Turbo and 3 "
                     "base requests. The unfilled end of each bar is other overhead, under 3 %. VAE encode and decode "
                     "are summed.", left=sh.pick(2.5, 2.1), right=sh.pick(1.0, 0.9),
                     key=[(th.peer, "text encoder"), (th.hero, "DiT, the diffusion transformer"),
                          (th.other, "VAE encode and decode")])
    n = len(shares)
    for i, ((total, text, dit, vae), (_, _, label)) in enumerate(zip(shares, CASE_ROWS)):
        y = n - 1 - i
        bar(ax, th, y, 0, text, th.peer, 0.56)
        bar(ax, th, y, text, text + dit, th.hero, 0.56)
        bar(ax, th, y, text + dit, text + dit + vae, th.other, 0.56)
        tag(ax, text + dit / 2, y, f"{dit:.0f}%", sh, th, ha="center", dx=0, size=12)
        tag(ax, 100, y, f"{total / 1000:.2f} s", sh, th)
        row_label(ax, th, sh, y, label)
    clean(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    return fig, srcs


def request_latency(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """End-to-end seconds, Turbo against base, for an edit at 1024 and 2048."""
    tt = {c: get(d, TURBO, OFFICIAL, c, "total_ms") for c, _ in SIZE_ROWS}
    bb = {c: get(d, BASE, OFFICIAL, c, "total_ms") for c, _ in SIZE_ROWS}
    srcs = tuple(p.src for p in (*tt.values(), *bb.values()))
    speed = [bb[c].value / tt[c].value for c, _ in SIZE_ROWS]
    cost = [row[big].value / row[small].value for row in (tt, bb) for big, small in SCALE_PAIRS]
    fig, ax = canvas(th, sh, sh.pick(4.2, 5.4),
                     f"Turbo is {min(speed):.1f} to {max(speed):.1f} x faster than base, end to end",
                     "Seconds per edit request, warm p50, official pipeline, batch 1. Lower is better.",
                     f"At 2048 an edit costs {min(cost):.1f} to {max(cost):.1f} times its 1024 time. Each request "
                     "includes the VAE encode of the input image. One RTX PRO 6000 Blackwell, bf16.",
                     left=sh.pick(2.5, 2.1), right=sh.pick(2.9, 1.0),
                     key=[(th.hero, "Turbo (8 steps)"), (th.other, "base (40 steps)")])
    n = len(SIZE_ROWS)
    for i, (case, label) in enumerate(SIZE_ROWS):
        y = n - 1 - i
        t, b = tt[case].value, bb[case].value
        bar(ax, th, y + 0.2, 0, t, th.hero, 0.36)
        bar(ax, th, y - 0.2, 0, b, th.other, 0.36)
        text = f"{t / 1000:.2f} s, {b / t:.1f} x faster than base" if sh.wide else f"{t / 1000:.2f} s"
        tag(ax, t, y + 0.2, text, sh, th, size=12)
        tag(ax, b, y - 0.2, f"{b / 1000:.2f} s", sh, th, color=th.muted, size=11)
        row_label(ax, th, sh, y, label)
    clean(ax)
    ax.set_xlim(0, 90000)
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_xticks([0, 20000, 40000, 60000, 80000], ["0", "20 s", "40 s", "60 s", "80 s"])
    return fig, srcs


def prefix_cache(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """DiT milliseconds per step for edit 1024, Turbo, with the prefix cache on against off."""
    on = [get(d, TURBO, OFFICIAL, "edit_1024", f) for f in ("step0_ms", "later_step_ms")]
    off = [get(d, TURBO, OFFICIAL, "edit_1024_nocache", f) for f in ("step0_ms", "later_step_ms")]
    totals = [get(d, TURBO, OFFICIAL, c, "total_ms") for c in ("edit_1024", "edit_1024_nocache")]
    srcs = tuple(p.src for p in (*on, *off, *totals))
    fig, ax = canvas(th, sh, sh.pick(4.4, 5.6),
                     f"After the first step, the prefix cache halves each DiT step: "
                     f"{off[1].value:.0f} ms to {on[1].value:.0f} ms",
                     "DiT milliseconds per denoising step, edit at 1024, Turbo (8 steps), official pipeline. "
                     "Lower is better.",
                     f"Step 1 computes the prefix either way. Steps 2 to 8 are averaged. Whole request: "
                     f"{totals[0].value / 1e3:.2f} s with the cache, {totals[1].value / 1e3:.2f} s without.",
                     left=sh.pick(2.0, 1.7), right=sh.pick(1.6, 0.9),
                     key=[(th.hero, "prefix cache on"), (th.other, "prefix cache off")])
    for y, label, a, b in ((1, "step 1", on[0], off[0]), (0, "each of\nsteps 2 to 8", on[1], off[1])):
        bar(ax, th, y + 0.2, 0, a.value, th.hero, 0.36)
        bar(ax, th, y - 0.2, 0, b.value, th.other, 0.36)
        tag(ax, a.value, y + 0.2, ms(a.value), sh, th, size=11)
        tag(ax, b.value, y - 0.2, ms(b.value), sh, th, color=th.muted, size=11)
        row_label(ax, th, sh, y, label)
    clean(ax)
    ax.set_xlim(0, 640)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xticks([0, 200, 400, 600], ["0", "200 ms", "400 ms", "600 ms"])
    return fig, srcs


def utilisation(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Achieved TFLOP/s per later DiT step for the four official cases, against measured matmul and attention ceilings."""
    util = [get(d, v, OFFICIAL, c, "later_step_tflops") for v, c, _ in CASE_ROWS]
    roof = [Pt(d["roofline"][k], f"roofline.{k}") for k in ROOFLINE]
    srcs = tuple(p.src for p in (*util, *roof))
    square, model, attn = (p.value for p in roof)
    lo, hi = min(u.value for u in util), max(u.value for u in util)
    fig, ax = canvas(th, sh, sh.pick(5.2, 6.8),
                     f"DiT steps in an edit run at {pct(lo, square):.0f} to {pct(hi, square):.0f} "
                     "percent of the card's measured matmul speed",
                     "Achieved TFLOP/s per denoising step after the first, official pipeline, bf16. Higher is better.",
                     f"Against the model-shaped matmul ({model:.1f} TFLOP/s) the same steps reach {pct(lo, model):.0f} "
                     f"to {pct(hi, model):.0f} percent. One RTX PRO 6000 Blackwell.",
                     left=sh.pick(2.5, 2.1), right=sh.pick(1.0, 0.9))
    n = len(util)
    for i, ((_, _, label), u) in enumerate(zip(CASE_ROWS, util)):
        y = n - 1 - i
        bar(ax, th, y, 0, u.value, th.hero, 0.52)
        tag(ax, u.value, y, f"{u.value:.1f}", sh, th)
        row_label(ax, th, sh, y, label)
    top = n - 0.4
    for i, (value, name) in enumerate(((square, "square matmul"), (model, "model matmul"), (attn, "attention"))):
        ref_line(ax, th, value)
        ref_text(ax, th, sh, value - 4, top + 0.3 + i * 0.45, f"{name} {value:.1f}", "right")
    clean(ax)
    ax.set_xlim(0, 460)
    ax.set_ylim(-0.6, top + 1.5)
    ax.set_xticks([0, 200, 400], ["0", "200", "400 TFLOP/s"])
    return fig, srcs


def peak_memory(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Peak GiB of the official pipeline against the lean port, Turbo, with the weights and the two card sizes."""
    off = [get(d, TURBO, OFFICIAL, c, "peak_gib") for c, _ in SIZE_ROWS]
    lean = [get(d, TURBO, LEAN, c, "peak_gib") for c, _ in SIZE_ROWS]
    weights = Pt(d["resident_gib"], "resident_gib")
    card = Pt(d["device"]["vram_gib"], "device.vram_gib")
    saved = [o.value - l.value for o, l in zip(off, lean)]
    srcs = tuple(p.src for p in (*off, *lean, weights, card))
    fig, ax = canvas(th, sh, sh.pick(4.4, 5.6),
                     f"The lean port needs {min(saved):.1f} to {max(saved):.1f} GiB less at peak",
                     "Peak GPU memory in GiB during one Turbo (8 steps) edit, batch 1, text encoder loaded. Lower is better.",
                     f"Dotted line: resident weights, {gib(weights.value)}. Dashed: the 32 GB RTX 5090, plotted as 32 "
                     f"GiB, and this card at {gib(card.value)}. With the text encoder on the card the lowest lean-port "
                     f"peak, {gib(min(l.value for l in lean))}, still misses 32 GB.",
                     left=sh.pick(2.5, 2.1), right=sh.pick(1.0, 0.9),
                     key=[(th.other, "official pipeline"), (th.hero, "lean port")])
    n = len(SIZE_ROWS)
    for i, ((_, label), o, l) in enumerate(zip(SIZE_ROWS, off, lean)):
        y = n - 1 - i
        bar(ax, th, y + 0.2, 0, o.value, th.other, 0.36)
        bar(ax, th, y - 0.2, 0, l.value, th.hero, 0.36)
        tag(ax, o.value, y + 0.2, gib(o.value), sh, th, color=th.muted, size=11)
        tag(ax, l.value, y - 0.2, f"{gib(l.value)}, {o.value - l.value:.1f} GiB less", sh, th, size=11)
        row_label(ax, th, sh, y, label)
    top = n - 0.4
    ref_line(ax, th, RTX_5090_GB)
    ref_line(ax, th, card.value)
    ref_line(ax, th, weights.value, dotted=True)
    ref_text(ax, th, sh, RTX_5090_GB + 1, top + 0.3, "RTX 5090, 32 GB", "left")
    ref_text(ax, th, sh, card.value - 1, top + 0.75, f"this card, {gib(card.value)}", "right")
    ref_text(ax, th, sh, weights.value - 1, top + 0.75, f"weights {gib(weights.value)}", "right")
    clean(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, top + 1.2)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100 GiB"])
    return fig, srcs


def op_groups(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """GPU time per operation group for one edit request at 1024, from the profile in summary.json."""
    ops = {k: Pt(v, f"op_groups_ms.edit_1024.{k}") for k, v in d["op_groups_ms"]["edit_1024"].items()}
    total = sum(p.value for p in ops.values())
    srcs = tuple(p.src for p in ops.values())
    order = sorted(ops, key=lambda k: -ops[k].value)
    matmul = ops["matmul (linear layers)"].value
    elem = ops["elementwise, casts and copies"].value
    attn = ops["attention"].value
    fig, ax = canvas(th, sh, sh.pick(4.6, 5.6),
                     f"Matmul is {pct(matmul, total):.0f} percent of op time; elementwise, casts and copies cost "
                     f"{elem / attn:.1f} times attention",
                     "Milliseconds of GPU time in one Turbo edit request at 1024, by operation group.",
                     f"Turbo, official pipeline, one profiled request (torch profiler, top 30 ops). Shares are of "
                     f"the four groups' {total:.0f} ms sum.",
                     left=sh.pick(2.6, 2.1), right=sh.pick(1.0, 0.9))
    n = len(order)
    for i, k in enumerate(order):
        y = n - 1 - i
        v = ops[k].value
        bar(ax, th, y, 0, v, th.hero if k.startswith("matmul") else th.other, 0.52)
        tag(ax, v, y, f"{v:.0f} ms, {pct(v, total):.0f}%" if sh.wide else f"{v:.0f} ms", sh, th)
        row_label(ax, th, sh, y, OP_LABELS[k])
    clean(ax)
    ax.set_xlim(0, 1650)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 400, 800, 1200], ["0", "400 ms", "800 ms", "1200 ms"])
    return fig, srcs


def span_split(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Share of GPU kernel time per labelled span in one Turbo edit, 1024 against 2048; the tail is everything else."""
    colors = (th.peer, th.soft, th.hero, th.other, th.muted)
    totals = {c: Pt(d["kernels"][c]["kernel_ms"], f"kernels[{c}].kernel_ms") for c, _ in SIZE_ROWS}
    parts = {c: [span_sum(d, c, names) for _, names in SPAN_SEGMENTS] for c, _ in SIZE_ROWS}
    srcs = tuple(p.src for c, _ in SIZE_ROWS for p in (totals[c], *parts[c]))
    attn = [pct(parts[c][2].value, totals[c].value) for c, _ in SIZE_ROWS]
    fig, ax = canvas(th, sh, sh.pick(4.0, 7.2),
                     f"Attention grows from {attn[0]:.0f} to {attn[1]:.0f} percent of an edit's GPU time at 2048",
                     "Percent of GPU kernel time in one Turbo (8 steps) edit, by span. Total kernel seconds at right.",
                     "Lean port, DiT and VAE only, one profiled request per size on an RTX PRO 6000 Blackwell. The "
                     "unfilled end of each bar is all other kernels. The profiler slowed the 2048 request by 8 %.",
                     left=sh.pick(1.3, 1.2), right=sh.pick(1.0, 0.9),
                     key=[(color, name) for color, (name, _) in zip(colors, SPAN_SEGMENTS)])
    n = len(SIZE_ROWS)
    for i, (case, label) in enumerate(SIZE_ROWS):
        y, x = n - 1 - i, 0.0
        for color, part in zip(colors, parts[case]):
            share = pct(part.value, totals[case].value)
            bar(ax, th, y, x, x + share, color, 0.56)
            if share >= LABEL_PCT:
                tag(ax, x + share / 2, y, f"{share:.0f}%", sh, th, ha="center", dx=0, size=11)
            x += share
        tag(ax, 100, y, f"{totals[case].value / 1000:.2f} s", sh, th)
        row_label(ax, th, sh, y, label)
    clean(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    return fig, srcs


def kernel_classes(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Milliseconds of big kernels against small ones inside three spans of the 1024 edit."""
    big = [kind_sum(d, "edit_1024", name, MATMUL_KINDS) for name, _ in KERNEL_SPANS]
    small = [kind_sum(d, "edit_1024", name, SMALL_KINDS) for name, _ in KERNEL_SPANS]
    spans = [span_sum(d, "edit_1024", (name,)) for name, _ in KERNEL_SPANS]
    srcs = tuple(p.src for p in (*big, *small, *spans))
    fig, ax = canvas(th, sh, sh.pick(4.4, 5.6),
                     f"Small kernels take {small[0].value:.0f} of the QKV span's {spans[0].value:.0f} ms",
                     "Milliseconds of GPU kernel time inside three spans of one Turbo edit at 1024, by kernel kind.",
                     "Small kernels are elementwise ops, copies and layout conversions: norms, RoPE, activations, "
                     "casts. Lean port, one profiled request, RTX PRO 6000 Blackwell.",
                     left=sh.pick(2.0, 1.9), right=sh.pick(1.0, 0.9),
                     key=[(th.other, "matmul, attention or convolution kernels"), (th.hero, "small kernels")])
    n = len(KERNEL_SPANS)
    for i, ((_, label), b, s, span) in enumerate(zip(KERNEL_SPANS, big, small, spans)):
        y = n - 1 - i
        bar(ax, th, y, 0, b.value, th.other, 0.52)
        bar(ax, th, y, b.value, b.value + s.value, th.hero, 0.52)
        text = f"{s.value:.0f} ms small, {pct(s.value, span.value):.0f}%" if sh.wide else f"{s.value:.0f} ms"
        tag(ax, b.value + s.value, y, text, sh, th)
        row_label(ax, th, sh, y, label)
    clean(ax)
    ax.set_xlim(0, 1300)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 400, 800, 1200], ["0", "400 ms", "800 ms", "1200 ms"])
    return fig, srcs


def fit_32gb(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Peak GiB of a Turbo edit: official pipeline, lean port, and lean port without the text encoder, against 32 GB."""
    off = [get(d, TURBO, OFFICIAL, c, "peak_gib") for c, _ in SIZE_ROWS]
    full = [get(d, TURBO, LEAN, c, "peak_gib") for c, _ in SIZE_ROWS]
    lean = [Pt(max(d["baseline"]["magpie"][c]["stage_peak_gib"].values()), f"baseline.magpie[{c}].stage_peak_gib(max)")
            for c, _ in SIZE_ROWS]
    srcs = tuple(p.src for p in (*off, *full, *lean))
    fig, ax = canvas(th, sh, sh.pick(5.0, 6.4),
                     f"Without the text encoder an edit peaks at {lean[0].value:.0f} to {lean[-1].value:.0f} GiB, "
                     "under a 32 GB card",
                     "Peak GPU memory in GiB during one Turbo (8 steps) edit, batch 1, bf16. Lower is better.",
                     "Dashed line: the 32 GB RTX 5090, plotted as 32 GiB. Measured on the 96 GB RTX PRO 6000 "
                     "Blackwell, not on a 5090. DiT and VAE only means text states were computed beforehand.",
                     left=sh.pick(1.3, 1.2), right=sh.pick(1.0, 0.9),
                     key=[(th.other, "official pipeline"), (th.peer, "lean port"), (th.hero, "lean port, DiT and VAE only")])
    n = len(SIZE_ROWS)
    for i, ((_, label), o, f, l) in enumerate(zip(SIZE_ROWS, off, full, lean)):
        y = n - 1 - i
        for dy, p, color, ink in ((0.27, o, th.other, th.muted), (0, f, th.peer, th.muted), (-0.27, l, th.hero, None)):
            bar(ax, th, y + dy, 0, p.value, color, 0.25)
            tag(ax, p.value, y + dy, gib(p.value), sh, th, color=ink, size=11)
        row_label(ax, th, sh, y, label)
    top = n - 0.4
    ref_line(ax, th, RTX_5090_GB)
    ref_text(ax, th, sh, RTX_5090_GB + 1, top + 0.25, "RTX 5090, 32 GB", "left")
    clean(ax)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, top + 0.6)
    ax.set_xticks([0, 20, 40, 60, 80], ["0", "20", "40", "60", "80 GiB"])
    return fig, srcs


def floor(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Measured milliseconds per span of the 1024 edit against the same work at the card's measured rates and at its datasheet rates."""
    roof = d["roofline"]
    case = roof["cases"]["edit_1024"]
    pt = lambda name, field: Pt(case["spans"][name][field], f"roofline.cases[edit_1024].spans[{name}].{field}")
    took, reached, least = ([pt(name, f) for name, _ in FLOOR_SPANS] for f in ("measured_ms", "reached_ms", "floor_ms"))
    total = [Pt(case[f], f"roofline.cases[edit_1024].{f}") for f in ("request_ms", "floor_request_ms")]
    srcs = tuple(p.src for p in (*took, *reached, *least, *total))
    fig, ax = canvas(th, sh, sh.pick(7.4, 9.8),
                     f"On datasheet rates the same work takes {total[1].value / 1000:.2f} s; it takes {total[0].value / 1000:.2f} s",
                     "Milliseconds of GPU kernel time per span of one Turbo edit at 1024: measured, and the same work at two rates.",
                     f"Datasheet: {roof['datasheet']['bf16_tflops'] / 1000:.0f} PFLOPS of bf16 tensor-core peak and "
                     f"{roof['datasheet']['bandwidth_gbs']:,.0f} GB/s. Fastest measured: {roof['reached']['gemm_8192_bf16_tflops']:.0f} TFLOP/s "
                     f"and {roof['reached']['copy_gb_per_s']:,.0f} GB/s. Elementwise stages count as one read and one write. Lean port, "
                     "one profiled request, RTX PRO 6000 Blackwell Server Edition.",
                     left=sh.pick(2.3, 2.1), right=sh.pick(1.0, 0.9),
                     key=[(th.other, "measured"), (th.peer, "at the fastest rates measured on the card"), (th.hero, "at datasheet rates")])
    n = len(FLOOR_SPANS)
    for i, ((_, label), t, r, f) in enumerate(zip(FLOOR_SPANS, took, reached, least)):
        y = n - 1 - i
        for dy, p, color, ink in ((0.27, t, th.other, th.muted), (0, r, th.peer, th.muted), (-0.27, f, th.hero, None)):
            bar(ax, th, y + dy, 0, p.value, color, 0.25)
            tag(ax, p.value, y + dy, f"{p.value:,.0f} ms", sh, th, color=ink, size=11)
        row_label(ax, th, sh, y, label)
    clean(ax)
    ax.set_xlim(0, 1300)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 400, 800, 1200], ["0", "400 ms", "800 ms", "1200 ms"])
    return fig, srcs


def ladder(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """One bar per change: milliseconds of a 1024 Turbo edit after each step, labelled with the saving over the step before."""
    steps = [(s["step"], s["ms"]) for s in d["ladder"]["steps"]]
    first, last = steps[0][1], steps[-1][1]
    fig, ax = canvas(th, sh, sh.pick(7.0, 9.8),
                     f"Ten changes took a 1024² edit from {first / 1000:.2f} s to {last / 1000:.2f} s, "
                     f"{first / last:.1f} x faster",
                     "Milliseconds per Turbo (8 steps) edit at 1024², DiT + VAE, median of 5 warm requests, RTX PRO 6000 "
                     "Blackwell. Lower is better.",
                     "One change per row, in the order they were made. The last step misses the DINOv2 0.98 line on one "
                     "image; the current default measures 1,054 ms.",
                     left=sh.pick(2.5, 3.0), right=sh.pick(1.6, 0.5),
                     key=[(th.other, "port baseline"), (th.soft, "earlier steps"), (th.hero, "last step")])
    n = len(steps)
    for i, (name, v) in enumerate(steps):
        y = n - 1 - i
        saved = steps[i - 1][1] - v if i else None
        bar(ax, th, y, 0, v, th.other if i == 0 else th.hero if i == n - 1 else th.soft)
        tag(ax, v, y, f"{v:,} ms" if saved is None else f"{v:,} ms, -{saved:,} ms", sh, th,
            size=12 if i in (0, n - 1) else 11.5)
        row_label(ax, th, sh, y, fit(name, 12 * sh.k, sh.pick(2.3, 2.7)))
    clean(ax)
    ax.set_xlim(0, sh.pick(3300, 5000))
    ax.set_ylim(-0.6, n - 0.4)
    ticks = [0, 500, 1000, 1500, 2000, 2500] if sh.wide else [0, 2000, 4000]
    ax.set_xticks(ticks, [f"{t:,} ms" for t in ticks])
    return fig, ("ladder.steps",)


def stage_panel(ax: Axes, th: Theme, sh: Shape, rows: Sequence[StageRow], xlim: float, ticks: Sequence[float],
                title: str) -> None:
    """One request size of the stage breakdown: a now bar over a port bar for each stage, the total on the last row."""
    n = len(rows)
    for i, (name, port, now) in enumerate(rows):
        y = n - 1 - i
        bar(ax, th, y + 0.2, 0, now, th.hero, 0.36)
        bar(ax, th, y - 0.2, 0, port, th.other, 0.36)
        tag(ax, now, y + 0.2, f"{now:,} ms", sh, th, size=11)
        tag(ax, port, y - 0.2, f"{port:,} ms", sh, th, color=th.muted, size=11)
        row_label(ax, th, sh, y, name)
    clean(ax)
    ax.set_xlim(0, xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.set_title(title, loc="left", fontsize=12.5 * sh.k, color=th.ink, pad=4)
    ax.tick_params(labelsize=10.5 * sh.k, labelcolor=th.muted)


def stages_then_now(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Stage times of one traced edit, port against now, at 1024 and 2048 as two stacked panels on one canvas."""
    rows = {size: [(s, p, n) for s, p, n in stage] for size, stage in d["ladder"]["stages"].items()}
    (_, p1, n1), (_, p2, n2) = rows["1024"][-1], rows["2048"][-1]
    fig, ax = canvas(th, sh, sh.pick(10.6, 13.0),
                     f"The traced request drops {p1 / n1:.1f} x at 1024² and {p2 / n2:.1f} x at 2048²",
                     "Milliseconds per Turbo (8 steps) edit request, by stage, DiT + VAE, one traced request per size. "
                     "Lower is better.",
                     "RTX PRO 6000 Blackwell. Stages do not add to the total; the rest is overhead. Glue reads 0 ms "
                     "in the current build.",
                     left=sh.pick(2.4, 2.2), right=sh.pick(0.3, 0.3),
                     key=[(th.other, "port, the starting point"), (th.hero, "now")])
    x0, y0, w0, h0 = ax.get_position().bounds
    ax.remove()
    fh = fig.get_size_inches()[1]
    gap, cap = 0.9 / fh, 0.42 / fh
    half = (h0 - gap - cap) / 2
    top = fig.add_axes((x0, y0 + half + gap, w0, half))
    bottom = fig.add_axes((x0, y0, w0, half))
    stage_panel(top, th, sh, rows["1024"], sh.pick(2800, 3400), [0, 500, 1000, 1500, 2000, 2500] if sh.wide
                else [0, 1000, 2000], "1024² edit")
    stage_panel(bottom, th, sh, rows["2048"], sh.pick(19000, 21000), [0, 4000, 8000, 12000, 16000] if sh.wide
                else [0, 8000, 16000], "2048² edit")
    return fig, ("ladder.stages.1024", "ladder.stages.2048")


def verdict(base: float, now: float, lower: bool) -> str:
    """How the current build compares: the ratio when it is better, the percent it is worse by otherwise."""
    better = base / now if lower else now / base
    return f"{better:.1f} x better" if better >= 1 else f"{100 / better - 100:.0f} % worse"


def now_vs_baseline(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """Each metric of the current build as a share of the port baseline (100 %), labelled with its real values."""
    pairs = d["ladder"]["pairs"]
    speed = [p["base"] / p["now"] for p in pairs[:2]]
    fig, ax = canvas(th, sh, sh.pick(7.6, 10.6),
                     f"Edits run {min(speed):.1f} to {max(speed):.1f} x faster than the port baseline; peak memory at "
                     "2048² is the one regression",
                     "Current build as a share of the port baseline, which is 100 %. Turbo (8 steps). Lower is better, "
                     "except compute rate.",
                     "RTX PRO 6000 Blackwell, edit, DiT + VAE unless the row says otherwise. The compute rate was "
                     "measured on the 1,218 ms build, not the current default. The text encoder reads a 1024² copy of "
                     "the input.",
                     left=sh.pick(3.0, 2.2), right=sh.pick(2.4, 1.0),
                     key=[(th.other, "port baseline"), (th.hero, "current build")])
    n = len(pairs)
    for i, p in enumerate(pairs):
        y = n - 1 - i
        share = 100 * p["now"] / p["base"]
        bar(ax, th, y - 0.2, 0, 100, th.other, 0.36)
        bar(ax, th, y + 0.2, 0, share, th.hero, 0.36)
        tag(ax, 100, y - 0.2, f"{p['base']:,} {p['unit']}", sh, th, color=th.muted, size=11)
        text = f"{p['now']:,} {p['unit']}" + (f", {verdict(p['base'], p['now'], p['lower'])}" if sh.wide else "")
        tag(ax, share, y + 0.2, text, sh, th, size=11.5)
        row_label(ax, th, sh, y, fit(p["metric"], 12 * sh.k, sh.pick(2.7, 2.0)))
    ref_line(ax, th, 100)
    ref_text(ax, th, sh, 100, n + 0.1, "port baseline, 100 %", "center")
    clean(ax)
    ax.set_xlim(0, 240)
    ax.set_ylim(-0.6, n + 0.5)
    ticks = [0, 50, 100, 150, 200] if sh.wide else [0, 100, 200]
    ax.set_xticks(ticks, [f"{t} %" for t in ticks])
    return fig, ("ladder.pairs",)


def quality_gate(th: Theme, sh: Shape, d: Summary) -> tuple[Figure, tuple[str, ...]]:
    """DINOv2 similarity of each frozen-set edit to its baseline image, at 1024 and 2048, zoomed on the pass line."""
    gate = d["ladder"]["gate"]
    line, g1, g2 = gate["floor"], gate["1024"], gate["2048"]
    n = len(g1)
    fig, ax = canvas(th, sh, sh.pick(8.4, 11.2),
                     f"{sum(v >= line for v in g1)} of {n} 1024² edits clear the {line} DINOv2 gate, and "
                     f"{sum(v >= line for v in g2)} of {n} at 2048²",
                     "DINOv2 similarity of each edit to its frozen baseline image, per item. Higher is better. Axis "
                     "zoomed to 0.97 to 1.00.",
                     "Frozen set of 16 edits, RTX PRO 6000 Blackwell, Turbo (8 steps), edit, DiT + VAE. A VLM judge "
                     "called item 0 identical. Item 15 is the lettering poster.",
                     left=sh.pick(2.0, 1.8), right=sh.pick(0.6, 0.9),
                     key=[(th.hero, "1024²"), (th.peer, "2048²"), (th.soft, "under the line")])
    for i, (a, b) in enumerate(zip(g1, g2)):
        y = n - 1 - i
        ax.scatter([a], [y + 0.2], s=130 * sh.k, color=th.soft if a < line else th.hero, edgecolor=th.bg,
                   linewidth=2, zorder=3, clip_on=False)
        ax.scatter([b], [y - 0.2], s=130 * sh.k, color=th.peer, edgecolor=th.bg, linewidth=2, zorder=3, clip_on=False)
        if a < line:
            tag(ax, a, y + 0.2, f"{a:.4f}", sh, th, ha="right", dx=10, size=11.5)
        row_label(ax, th, sh, y, f"item {i}" + (", poster" if i == n - 1 else ""))
    ref_line(ax, th, line)
    ref_text(ax, th, sh, line, n + 0.1, f"gate {line}", "center")
    clean(ax)
    ax.set_xlim(0.968, 1.002)
    ax.set_ylim(-0.6, n + 0.5)
    ticks = [0.97, 0.98, 0.99, 1.0]
    ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
    return fig, ("ladder.gate.1024", "ladder.gate.2048")


PLOTS: dict[str, Plot] = {
    "stage-split": stage_split, "request-latency": request_latency, "prefix-cache": prefix_cache,
    "utilisation": utilisation, "peak-memory": peak_memory, "op-groups": op_groups,
    "span-split": span_split, "kernel-classes": kernel_classes, "fit-32gb": fit_32gb,
    "floor": floor, "ladder": ladder, "stages-then-now": stages_then_now, "now-vs-baseline": now_vs_baseline,
    "quality-gate": quality_gate,
}


def render(name: str, plot: Plot, d: Summary, th: Theme, sh: Shape) -> tuple[str, ...]:
    """Draw one chart in one theme and shape, save it, and return the summary paths it plotted."""
    style(th)
    fig, srcs = plot(th, sh, d)
    OUT.mkdir(parents=True, exist_ok=True)
    file = f"{PREFIX}_{name}{sh.suffix}{th.name}.png"
    fig.savefig(OUT / file, dpi=200)
    plt.close(fig)
    print(f"wrote {file}")
    return srcs


def write_sources(charts: dict[str, tuple[str, ...]]) -> None:
    """Record the summary hash, the repo head (null while the repo has no commits) and each chart's source fields."""
    path = OUT / "sources.json"
    prev = json.loads(path.read_text())["charts"] if path.exists() else {}
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    doc = {"generated": dt.date.today().isoformat(), "repo": {"root": str(ROOT), "head": head or None},
           **{name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in (("summary", SUMMARY), ("kernels", KERNELS), ("baseline", BASELINE), ("roofline", FLOORS),
                                 ("ladder", LADDER))},
           "charts": {**prev, **{k: list(dict.fromkeys(v)) for k, v in charts.items()}}}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote sources.json")


def main(names: list[str]) -> None:
    """Render the requested charts (all by default) in every theme and shape, then write their provenance."""
    d = {**json.loads(SUMMARY.read_text()), "kernels": json.loads(KERNELS.read_text()),
         "baseline": json.loads(BASELINE.read_text()), "roofline": json.loads(FLOORS.read_text()),
         "ladder": json.loads(LADDER.read_text())}
    charts: dict[str, tuple[str, ...]] = {}
    for name in names or list(PLOTS):
        for th in (LIGHT, DARK):
            for sh in (WIDE, TALL):
                charts[name] = render(name, PLOTS[name], d, th, sh)
    write_sources(charts)


if __name__ == "__main__":
    main(sys.argv[1:])
