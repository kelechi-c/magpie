# Break the labelled GPU spans of the turbo traces into kernel classes and write out/trace/kernels.json.
# The profiler's own key_averages() double-counts nested labels and reports ~0 for outer spans, so span
# time is rebuilt here from the trace events: a kernel belongs to a span when it starts inside it.
#   uv run --no-project python tools/trace_kernels.py
import gzip
import json
import pathlib as pl
from bisect import bisect_left
from collections import Counter
from typing import Any

type Event = dict[str, Any]

ROOT = pl.Path(__file__).resolve().parents[1]
TRACES = ROOT / "out/trace/turbo"
OUT = ROOT / "out/trace/kernels.json"
CLASSES: tuple[tuple[str, str], ...] = (  # first matching substring wins
    ("flash_fwd", "attention"), ("attend_kernel", "attention"), ("nchwToNhwc", "layout"), ("nhwcToNchw", "layout"), ("xmma", "conv"),
    ("cudnn", "conv"), ("gemm", "gemm"), ("copy_kernel", "copy"),
)


# Name of the kernel class a CUDA kernel belongs to; anything unmatched is elementwise or reduction work.
def kind(name: str) -> str:
    return next((label for needle, label in CLASSES if needle in name), "elementwise")


# Per labelled span: calls, gpu ms, and ms per kernel class; plus how much of the request the gpu was busy.
def breakdown(events: list[Event]) -> dict[str, Any]:
    kernels = sorted((e["ts"], e["dur"], e["name"]) for e in events if e.get("cat") == "kernel")
    starts = [k[0] for k in kernels]
    spans: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("cat") != "gpu_user_annotation":
            continue
        row = spans.setdefault(e["name"], {"calls": 0, "gpu_ms": 0.0, "kernels": Counter()})
        row["calls"] += 1
        row["gpu_ms"] += e["dur"] / 1e3
        for _, dur, name in kernels[bisect_left(starts, e["ts"]):bisect_left(starts, e["ts"] + e["dur"])]:
            row["kernels"][kind(name)] += dur / 1e3
    busy = sum(k[1] for k in kernels)
    wall = kernels[-1][0] + kernels[-1][1] - kernels[0][0]
    return {"kernel_ms": round(busy / 1e3, 1), "wall_ms": round(wall / 1e3, 1), "gpu_busy": round(busy / wall, 4),
            "launches": len(kernels),
            "spans": {name: {"calls": row["calls"], "gpu_ms": round(row["gpu_ms"], 1),
                             "kernels_ms": {k: round(v, 1) for k, v in row["kernels"].most_common()}}
                      for name, row in sorted(spans.items(), key=lambda item: -item[1]["gpu_ms"])}}


def main() -> None:
    report = {path.name.removesuffix(".json.gz"): breakdown(json.load(gzip.open(path))["traceEvents"])
              for path in sorted(TRACES.glob("*.json.gz"))}
    OUT.write_text(json.dumps(report, indent=1))
    for case, row in report.items():
        print(f"{case}: {row['kernel_ms']} ms in {row['launches']} kernels, gpu busy {row['gpu_busy']:.1%}")


if __name__ == "__main__":
    main()
