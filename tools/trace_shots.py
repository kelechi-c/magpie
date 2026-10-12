# Crop magpie profiler traces into zoomed Perfetto views, screenshot them, and write assets/traces/manifest.json.
#   uv run --no-project --with pillow --with numpy python tools/trace_shots.py
import datetime as dt
import gzip
import json
import pathlib as pl
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from math import inf
from typing import Any, Literal

import numpy as np
from PIL import Image

type Event = dict[str, Any]
Source = Literal["t2i_1024", "edit_1024", "t2i_2048"]

ROOT = pl.Path(__file__).resolve().parents[1]
TRACES: dict[Source, pl.Path] = {
    "t2i_1024": ROOT / "out/trace/turbo/t2i_1024.json.gz",
    "edit_1024": ROOT / "out/trace/turbo/edit_1024.json.gz",
    "t2i_2048": ROOT / "out/trace/turbo/t2i_2048.json.gz",
}
CROPS = ROOT / "out/trace/crops"
OUT = ROOT / "assets/traces"
SHOTS_JS = ROOT / "tools/shots.mjs"
CHROME = "/usr/bin/chromium"
HTTP_PORT = 8080
CDP_PORT = 9333
CHROME_FLAGS = (
    "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
    f"--remote-debugging-port={CDP_PORT}", "--window-size=1920,1080", "--disable-web-security",
    "--allow-running-insecure-content",
    "--disable-features=BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessSendPreflights",
)
KEPT_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"})
DROPPED_PH = frozenset({"s", "f"})  # flow events (ac2g arrows) are dropped: they dangle once events are clipped
DECODE_STEP = 4  # 0-based over all dit steps; step 0 is the prefill, so this is the 4th decode step
BLOCK = 10
PAD_US = 100.0  # margin around zoom targets
MIN_STDDEV = 5.0  # blank or loading-screen guard for the pixel check
MIN_COLORS = 1000
NAMED = frozenset(
    {"layer norm", "qkv + qk-norm + rope", "attention (sdpa)", "attention out", "mlp (swiglu)",
     "time embed", "text embed", "modulation", "head", "vae encode", "vae decode",
     "vae encoder mid", "vae decoder mid"}
    | {f"block {i:02d}" for i in range(32)}
    | {f"vae down {i}" for i in range(5)}
    | {f"vae up {i}" for i in range(5)}
)


@dataclass(frozen=True)
class Span:
    """A GPU-side named interval in original trace microseconds."""
    name: str
    ts: float
    dur: float

    @property
    def end(self) -> float:
        """Start plus duration."""
        return self.ts + self.dur


@dataclass(frozen=True)
class Shot:
    """One requested screenshot: the source trace, the window to keep, and a note for the manifest."""
    name: str
    source: Source
    window: Span
    note: str = ""


class CorsHandler(SimpleHTTPRequestHandler):
    """Static handler that adds the CORS and private-network headers Perfetto needs to fetch the crops."""

    def end_headers(self) -> None:
        """Attach CORS and private-network headers to every response."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Answer CORS preflight with 204."""
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging."""
        return


def load_events(path: pl.Path) -> list[Event]:
    """Read the traceEvents list from a gzipped Chrome trace."""
    with gzip.open(path, "rt") as f:
        return json.load(f)["traceEvents"]


def gpu_spans(events: list[Event]) -> tuple[Span, ...]:
    """GPU-side named annotations (gpu_user_annotation), sorted by start time."""
    spans = (Span(e["name"], e["ts"], e.get("dur", 0.0)) for e in events if e.get("cat") == "gpu_user_annotation")
    return tuple(sorted(spans, key=lambda s: s.ts))


def named(spans: tuple[Span, ...], name: str) -> tuple[Span, ...]:
    """All spans with exactly this name, in start-time order."""
    return tuple(s for s in spans if s.name == name)


def single(spans: tuple[Span, ...], name: str) -> Span:
    """The only span with this name; errors if there is not exactly one."""
    hits = named(spans, name)
    if len(hits) != 1:
        raise ValueError(f"{name}: expected 1 span, found {len(hits)}")
    return hits[0]


def child(spans: tuple[Span, ...], parent: Span, name: str) -> Span:
    """The only span with this name that starts inside the parent window; errors otherwise."""
    hits = [s for s in named(spans, name) if parent.ts <= s.ts <= parent.end]
    if len(hits) != 1:
        raise ValueError(f"{name}: expected 1 span in window, found {len(hits)}")
    return hits[0]


def step_windows(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    """One window per dit step: from its block 00 start to the end of its block 31 (GPU side)."""
    starts = [s.ts for s in named(spans, "block 00")]
    block31 = named(spans, "block 31")
    stops = [*starts[1:], inf]
    return tuple(
        Span("dit step", start, max(b.end for b in block31 if start <= b.ts < stop) - start)
        for start, stop in zip(starts, stops)
    )


def cover(spans: Iterable[Span], name: str = "") -> Span:
    """Smallest window that contains all the given spans."""
    items = list(spans)
    start = min(s.ts for s in items)
    return Span(name, start, max(s.end for s in items) - start)


def pad(span: Span, us: float) -> Span:
    """Widen a span by `us` microseconds on each side."""
    return Span(span.name, span.ts - us, span.dur + 2 * us)


def plan_shots(spans: dict[Source, tuple[Span, ...]]) -> tuple[Shot, ...]:
    """Choose the window of each of the twelve requested shots from GPU-side spans."""
    t2i, edit, big = spans["t2i_1024"], spans["edit_1024"], spans["t2i_2048"]
    t2i_steps, edit_steps, big_steps = step_windows(t2i), step_windows(edit), step_windows(big)
    step = t2i_steps[DECODE_STEP]
    block = child(t2i, step, f"block {BLOCK:02d}")
    b31 = child(t2i, step, "block 31")
    b00 = child(t2i, t2i_steps[DECODE_STEP + 1], "block 00")
    up = max(("vae up 3", "vae up 4"), key=lambda n: sum(s.dur for s in named(t2i, n)))
    up_ms = sum(s.dur for s in named(t2i, up)) / 1000
    edit_block = child(edit, edit_steps[0], f"block {BLOCK:02d}")
    big_block = child(big, big_steps[DECODE_STEP], f"block {BLOCK:02d}")
    return (
        Shot("mg_01-request", "t2i_1024", cover(t2i, "request"), "all gpu annotations, 8 dit steps + vae decode"),
        Shot("mg_02-step", "t2i_1024", step, f"dit step index {DECODE_STEP} (4th decode step)"),
        Shot("mg_03-block", "t2i_1024", block, f"block {BLOCK:02d} of dit step {DECODE_STEP}"),
        Shot("mg_04-qkv", "t2i_1024",
             pad(child(t2i, block, "qkv + qk-norm + rope"), PAD_US), f"zoom: block {BLOCK:02d} qkv span"),
        Shot("mg_05-mlp", "t2i_1024",
             pad(child(t2i, block, "mlp (swiglu)"), PAD_US), f"zoom: block {BLOCK:02d} mlp span"),
        Shot("mg_06-step-boundary", "t2i_1024", Span("boundary", b31.ts, b00.end - b31.ts),
             f"block 31 of dit step {DECODE_STEP} to block 00 of dit step {DECODE_STEP + 1}, one block context each side"),
        Shot("mg_07-vae-decode", "t2i_1024", single(t2i, "vae decode"), "whole vae decode"),
        Shot("mg_08-vae-up", "t2i_1024", pad(cover(named(t2i, up), up), PAD_US),
             f"zoom: {up} chosen as the more busy of vae up 3/4 ({up_ms:.3f} ms gpu total)"),
        Shot("mg_09-edit-request", "edit_1024", cover(edit, "request"), "all gpu annotations of the edit request"),
        Shot("mg_10-edit-prefill-block", "edit_1024", edit_block,
             f"block {BLOCK:02d} of dit step 0 (prefill)"),
        Shot("mg_11-vae-encode", "edit_1024", single(edit, "vae encode"), "whole vae encode"),
        Shot("mg_12-block-2048", "t2i_2048", big_block, f"block {BLOCK:02d} of dit step {DECODE_STEP}, 16384 tokens"),
    )


def clip(e: Event, window: Span) -> Event | None:
    """Intersect an event with the window and shift it so the window starts at 0; None when outside."""
    lo = max(e["ts"], window.ts)
    hi = min(e["ts"] + e.get("dur", 0.0), window.end)
    if hi < lo:
        return None
    return {**e, "ts": lo - window.ts, **({"dur": hi - lo} if "dur" in e else {})}


def crop(events: list[Event], window: Span) -> list[Event]:
    """Metadata verbatim first, then kept categories overlapping the window, clipped and shifted."""
    meta = [e for e in events if e.get("ph") == "M"]
    kept = [e for e in events if "ts" in e and e.get("ph") not in DROPPED_PH and e.get("cat") in KEPT_CATS]
    return meta + [c for e in kept if (c := clip(e, window)) is not None]


def named_inside(spans: tuple[Span, ...], window: Span) -> dict[str, dict[str, float | int]]:
    """Count and total GPU ms of each named span lying fully inside the window (dit step spans excluded)."""
    inside = [s for s in spans if s.name in NAMED and s.ts >= window.ts and s.end <= window.end]
    return {
        name: {"count": len(group), "total_ms": round(sum(s.dur for s in group) / 1000, 3)}
        for name in sorted({s.name for s in inside})
        if (group := [s for s in inside if s.name == name])
    }


def category_counts(events: list[Event]) -> dict[str, int]:
    """Event count per category; metadata events are counted under 'M'."""
    return dict(sorted(Counter(e.get("cat", e.get("ph")) for e in events).items()))


def pixel_stats(path: pl.Path) -> dict[str, float | int | bool]:
    """Stddev and distinct-colour count of a PNG; flags blank or loading-screen captures."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    r, g, b = (rgb[..., i].astype(np.uint32) for i in range(3))
    colors = int(np.unique((r << 16) | (g << 8) | b).size)
    stddev = float(rgb.std())
    return {
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "stddev": round(stddev, 2),
        "distinct_colors": colors,
        "ok": stddev >= MIN_STDDEV and colors >= MIN_COLORS,
    }


def port_open(port: int) -> bool:
    """True when something already listens on 127.0.0.1:port."""
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_cdp(timeout_s: float = 30.0) -> None:
    """Block until chromium answers on its DevTools port."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=1)
            return
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"chromium did not open cdp port {CDP_PORT}")
            time.sleep(0.5)


def serve_and_shoot(plan_path: pl.Path) -> None:
    """Serve the crops on :8080, run headless chromium, screenshot via shots.mjs, then stop only what it started."""
    if port_open(HTTP_PORT) or port_open(CDP_PORT):
        raise RuntimeError(f"port {HTTP_PORT} or {CDP_PORT} already in use; not touching other processes")
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), partial(CorsHandler, directory=str(CROPS)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix="magpie-chrome-")
    chrome = subprocess.Popen([CHROME, *CHROME_FLAGS, f"--user-data-dir={profile}", "about:blank"])
    try:
        wait_for_cdp()
        subprocess.run(["node", str(SHOTS_JS), str(plan_path)], check=True)
    finally:
        chrome.terminate()
        chrome.wait()
        server.shutdown()
        server.server_close()


def build_manifest(
    shots: tuple[Shot, ...],
    spans: dict[Source, tuple[Span, ...]],
    counts: dict[Source, dict[str, int]],
    cropped: dict[str, list[Event]],
    pixels: dict[str, dict[str, float | int | bool]],
) -> dict[str, Any]:
    """Assemble the manifest dict: one entry per shot with window, categories, named GPU spans and pixel check."""
    entries = []
    for shot in shots:
        w = shot.window
        kept = category_counts(cropped[shot.name])
        entries.append({
            "file": f"assets/traces/{shot.name}.png",
            "crop": f"out/trace/crops/{shot.name}.json",
            "source": f"out/trace/turbo/{shot.source}.json.gz",
            "note": shot.note,
            "window_start_us": round(w.ts, 3),
            "window_end_us": round(w.end, 3),
            "shift_us": round(w.ts, 3),
            "window_ms": round(w.dur / 1000, 3),
            "categories_kept": kept,
            "categories_dropped": sorted(c for c in counts[shot.source] if c not in KEPT_CATS | {"M"}),
            "gpu_named_spans": named_inside(spans[shot.source], w),
            "pixels": pixels[shot.name],
        })
    return {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rules": {
            "clip": "kept categories clipped to window; events outside window and other categories (incl. instants) dropped",
            "dropped_phases": sorted(DROPPED_PH),
            "kept_categories": sorted(KEPT_CATS | {"M"}),
            "dit_step_spans": "outer 'dit step' spans excluded from gpu_named_spans (gpu durations not meaningful)",
            "step_windows": "block 00 start to block 31 end, gpu side",
            "zoom_pad_us": PAD_US,
        },
        "shots": entries,
    }


def main() -> None:
    """Load the traces, write one crop per shot, screenshot them, check pixels, and write the manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    CROPS.mkdir(parents=True, exist_ok=True)
    loaded = {src: load_events(path) for src, path in TRACES.items()}
    spans = {src: gpu_spans(ev) for src, ev in loaded.items()}
    counts = {src: category_counts(ev) for src, ev in loaded.items()}
    shots = plan_shots(spans)
    cropped = {shot.name: crop(loaded[shot.source], shot.window) for shot in shots}
    for shot in shots:
        (CROPS / f"{shot.name}.json").write_text(json.dumps({"traceEvents": cropped[shot.name]}))
        print(f"crop {shot.name}: {len(cropped[shot.name])} events, {shot.window.dur / 1000:.3f} ms")
    plan_path = CROPS / "plan.json"
    plan_path.write_text(json.dumps({
        "out": str(OUT),
        "shots": [{"name": s.name, "file": f"{s.name}.json"} for s in shots],
    }))
    serve_and_shoot(plan_path)
    pixels = {shot.name: pixel_stats(OUT / f"{shot.name}.png") for shot in shots}
    for name, p in pixels.items():
        print(f"pixels {name}: stddev {p['stddev']} colors {p['distinct_colors']} ok {p['ok']}")
    manifest = build_manifest(shots, spans, counts, cropped, pixels)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failed = [n for n, p in pixels.items() if not p["ok"]]
    if failed:
        raise SystemExit(f"failed pixel check: {', '.join(failed)}")
    print(f"done: {len(shots)} shots in {OUT}")


if __name__ == "__main__":
    main()
