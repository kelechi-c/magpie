"""Results charts for magpie: Qwen-Image 2.1 on one RTX PRO 6000, official pipeline against the lean port.

    cd /home/tensor/Shared/code/ml/magpie && uv run --no-project --with matplotlib python tools/make_charts.py [chart ...]

Every plotted number is read from out/bench/summary.json. Writes qwen21_<chart>[_mobile][_dark].png and sources.json
into assets/charts at 200 dpi.
"""

import datetime as dt
import hashlib
import json
import pathlib as pl
import subprocess
import sys
from collections.abc import Callable
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, "/home/tensor/.claude/skills/chart-craft")
from chartkit import DARK, LIGHT, TALL, WIDE, Shape, Theme, bar, canvas, clean, ms, row_label, style, tag  # noqa: E402

ROOT = pl.Path("/home/tensor/Shared/code/ml/magpie")
SUMMARY = ROOT / "out/bench/summary.json"
OUT = ROOT / "assets/charts"
PREFIX = "qwen21"
TURBO, BASE, OFFICIAL, LEAN = "turbo", "base", "diffusers", "magpie"
RTX_5090_GB = 32.0
ROOFLINE: tuple[str, ...] = ("gemm_8192_bf16_tflops", "mlp_4096x4096x12288_bf16_tflops", "sdpa_16384_bf16_tflops")
STAGE_FIELDS: tuple[str, ...] = ("total_ms", "text_ms", "dit_ms", "vae_encode_ms", "vae_decode_ms")
CaseRow = tuple[str, str, str]  # (variant, case, label)
CASE_ROWS: tuple[CaseRow, ...] = (
    (TURBO, "t2i_1024", "Turbo (8 steps)\ntext-to-image"), (TURBO, "edit_1024", "Turbo (8 steps)\nedit"),
    (BASE, "t2i_1024", "base (40 steps)\ntext-to-image"), (BASE, "edit_1024", "base (40 steps)\nedit"),
)
SIZE_ROWS: tuple[tuple[str, str], ...] = (
    ("t2i_1024", "text-to-image\n1024"), ("edit_1024", "edit\n1024"),
    ("t2i_2048", "text-to-image\n2048"), ("edit_2048", "edit\n2048"),
)
SCALE_PAIRS: tuple[tuple[str, str], ...] = (("t2i_2048", "t2i_1024"), ("edit_2048", "edit_1024"))
OP_LABELS: dict[str, str] = {
    "matmul (linear layers)": "matmul\n(linear layers)", "attention": "attention",
    "convolution (vae)": "convolution\n(VAE)", "elementwise, casts and copies": "elementwise, casts\nand copies",
}

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
                     f"The DiT takes {min(dit_share):.0f} to {max(dit_share):.0f} percent of every official request",
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
    """End-to-end seconds, Turbo against base, for text-to-image and edit at 1024 and 2048."""
    tt = {c: get(d, TURBO, OFFICIAL, c, "total_ms") for c, _ in SIZE_ROWS}
    bb = {c: get(d, BASE, OFFICIAL, c, "total_ms") for c, _ in SIZE_ROWS}
    srcs = tuple(p.src for p in (*tt.values(), *bb.values()))
    speed = [bb[c].value / tt[c].value for c, _ in SIZE_ROWS]
    cost = [row[big].value / row[small].value for row in (tt, bb) for big, small in SCALE_PAIRS]
    fig, ax = canvas(th, sh, sh.pick(5.6, 7.4),
                     f"Turbo is {min(speed):.1f} to {max(speed):.1f} x faster than base, end to end",
                     "Seconds per request, warm p50, official pipeline, batch 1. Lower is better.",
                     f"At 2048 each request costs {min(cost):.1f} to {max(cost):.1f} times its 1024 time. Edit rows "
                     "include the VAE encode of the input image. One RTX PRO 6000 Blackwell, bf16.",
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
                     f"At 1024, DiT steps run at {pct(lo, square):.0f} to {pct(hi, square):.0f} "
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
    fig, ax = canvas(th, sh, sh.pick(5.6, 7.4),
                     f"The lean port needs {min(saved):.1f} to {max(saved):.1f} GiB less at peak",
                     "Peak GPU memory in GiB during one Turbo (8 steps) request, batch 1. Lower is better.",
                     f"Dotted line: resident weights, {gib(weights.value)}. Dashed: the 32 GB RTX 5090, plotted as 32 "
                     f"GiB, and this card at {gib(card.value)}. The lowest lean-port peak, "
                     f"{gib(min(l.value for l in lean))}, does not fit a 32 GB card even unquantised.",
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
    """GPU time per operation group for one text-to-image request at 1024, from the profile in summary.json."""
    ops = {k: Pt(v, f"op_groups_ms.t2i_1024.{k}") for k, v in d["op_groups_ms"]["t2i_1024"].items()}
    total = sum(p.value for p in ops.values())
    srcs = tuple(p.src for p in ops.values())
    order = sorted(ops, key=lambda k: -ops[k].value)
    matmul = ops["matmul (linear layers)"].value
    elem = ops["elementwise, casts and copies"].value
    attn = ops["attention"].value
    fig, ax = canvas(th, sh, sh.pick(4.6, 5.6),
                     f"Matmul is {pct(matmul, total):.0f} percent of op time; elementwise, casts and copies cost "
                     f"{elem / attn:.1f} times attention",
                     "Milliseconds of GPU time in one Turbo text-to-image request at 1024, by operation group.",
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


PLOTS: dict[str, Plot] = {
    "stage-split": stage_split, "request-latency": request_latency, "prefix-cache": prefix_cache,
    "utilisation": utilisation, "peak-memory": peak_memory, "op-groups": op_groups,
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
           "summary": {"path": str(SUMMARY), "sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest()},
           "charts": {**prev, **{k: list(dict.fromkeys(v)) for k, v in charts.items()}}}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote sources.json")


def main(names: list[str]) -> None:
    """Render the requested charts (all by default) in every theme and shape, then write their provenance."""
    d = json.loads(SUMMARY.read_text())
    charts: dict[str, tuple[str, ...]] = {}
    for name in names or list(PLOTS):
        for th in (LIGHT, DARK):
            for sh in (WIDE, TALL):
                charts[name] = render(name, PLOTS[name], d, th, sh)
    write_sources(charts)


if __name__ == "__main__":
    main(sys.argv[1:])
