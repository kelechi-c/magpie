# Cut the perfetto screenshots in assets/traces down to article figures in assets/traces/fig:
# the time ruler plus the track rows, without the sidebar. Detailed views are also folded into two
# rows (left half over right half) so kernel names stay readable at article width.
#   uv run --no-project --with pillow python tools/trace_figures.py
import pathlib as pl

from PIL import Image

ROOT = pl.Path(__file__).resolve().parents[1]
SHOTS = ROOT / "assets/traces"
OUT = SHOTS / "fig"
LEFT, RIGHT = 262, 1906  # track area, right of the perfetto sidebar
RULER, TRACKS = (112, 142), (190, 256)  # pixel rows of the time ruler and of the span + kernel tracks
OVERVIEWS = frozenset({"01-request", "02-step", "12-request-2048"})  # too dense to gain from folding
GAP = 6


# Ruler and tracks of one screenshot stacked into a single strip.
def strip(shot: Image.Image) -> Image.Image:
    rows = [shot.crop((LEFT, top, RIGHT, bottom)) for top, bottom in (RULER, TRACKS)]
    out = Image.new("RGB", (RIGHT - LEFT, sum(r.height for r in rows)), "white")
    out.paste(rows[0], (0, 0))
    out.paste(rows[1], (0, rows[0].height))
    return out


# Left half of a strip above its right half.
def folded(wide: Image.Image) -> Image.Image:
    half = wide.width // 2
    out = Image.new("RGB", (half, wide.height * 2 + GAP), "white")
    out.paste(wide.crop((0, 0, half, wide.height)), (0, 0))
    out.paste(wide.crop((half, 0, wide.width, wide.height)), (0, wide.height + GAP))
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for path in sorted(SHOTS.glob("mg_*.png")):
        name = path.stem.removeprefix("mg_")
        wide = strip(Image.open(path).convert("RGB"))
        wide.save(OUT / f"trace_{name}_full.png", optimize=True)
        (wide if name in OVERVIEWS else folded(wide)).save(OUT / f"trace_{name}.png", optimize=True)
        print(f"wrote trace_{name}.png")


if __name__ == "__main__":
    main()
