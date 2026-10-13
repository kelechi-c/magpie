"""A silent looping animation of how many 1024² edits each runtime finishes in the same seconds.

    cd /home/tensor/Shared/code/ml/magpie && uv run --no-project --with pillow python tools/make_animation.py

Request times come from out/bench/summary.json (official diffusers pipeline) and baseline/compare.json (sglang, magpie).
Writes assets/charts/qwen21_speed-race[_dark].webp in the chart palette.
"""

import json
import pathlib as pl
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

ROOT = pl.Path("/home/tensor/Shared/code/ml/magpie")
FONT = "/home/tensor/Shared/code/miscellaneous/sciel/sciel/sfpromedium.otf"
OUT = ROOT / "assets/charts"
WIDTH, HEIGHT, PAD = 1600, 900, 64
SECONDS, FPS, HOLD = 16, 20, 2.5  # clock length, frames per second, seconds the last frame stays up
LABEL, TOP, LANE, TILE, GAP = 360, 250, 138, 58, 8  # label column, first lane's top, lane pitch, tile size and spacing


class Theme(NamedTuple):
    name: str
    bg: str
    ink: str
    muted: str
    grid: str
    hero: str
    other: str


class Lane(NamedTuple):
    name: str
    note: str
    ms: float
    hero: bool


LIGHT = Theme("", "#FAF9F5", "#141413", "#6B6A60", "#E4E2D8", "#9C4A2C", "#B0AEA1")
DARK = Theme("_dark", "#141413", "#ECEAE2", "#A6A49A", "#2B2A27", "#C46A4A", "#6C6A61")


# the four runtimes, slowest first: (name, how it was timed, ms per 1024² turbo edit, is it ours).
def lanes() -> list[Lane]:
    rows = json.loads((ROOT / "out/bench/summary.json").read_text())["rows"]
    official = next(r for r in rows if (r["variant"], r["engine"], r["case"]) == ("turbo", "diffusers", "edit_1024"))["total_ms"]
    cases = json.loads((ROOT / "baseline/compare.json").read_text())["cases"]["edit 1024"]
    return [Lane("diffusers pipeline", "official, in-process", official, False),
            Lane("sglang default", "over HTTP", cases["sglang default"]["http_ms"], False),
            Lane("sglang fp8 + sage", "in-process", cases["sglang fp8 + sage"]["stages_ms"], False),
            Lane("magpie", "in-process", cases["magpie"]["total_ms"], True)]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


# one frame at clock time `t` seconds: per lane, a solid tile per finished image and a tile filling up for the one in flight.
def frame(t: float, rows: list[Lane], th: Theme) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), th.bg)
    draw = ImageDraw.Draw(image)
    draw.text((PAD, PAD - 8), f"Images finished in {SECONDS} seconds", font=font(46), fill=th.ink)
    draw.text((PAD, PAD + 58), "One 1024² Turbo edit after another, text encoder included, one RTX PRO 6000 Blackwell. Each square is one image.",
              font=font(22), fill=th.muted)
    draw.text((WIDTH - PAD, PAD + 4), f"{t:4.1f} s", font=font(40), fill=th.ink, anchor="ra")
    left = PAD + LABEL
    for index, row in enumerate(rows):
        y = TOP + index * LANE
        colour = th.hero if row.hero else th.other
        done, part = divmod(t * 1000 / row.ms, 1)
        draw.text((PAD, y + 4), row.name, font=font(28), fill=th.ink)
        draw.text((PAD, y + 42), f"{row.ms:,.0f} ms, {row.note}", font=font(20), fill=th.muted)
        for slot in range(int(SECONDS * 1000 / row.ms) + 1):
            x = left + slot * (TILE + GAP)
            draw.rounded_rectangle((x, y, x + TILE, y + TILE), 8, fill=colour if slot < done else th.grid)
            if slot == int(done) and part > 0:
                draw.rounded_rectangle((x, y, x + max(TILE * part, 16), y + TILE), 8, fill=colour)
        draw.text((WIDTH - PAD, y + 14), f"{int(done)}", font=font(40), fill=th.hero if row.hero else th.ink, anchor="ra")
    draw.text((PAD, HEIGHT - PAD - 12), "Request times are medians of 5 warm requests. sglang 0.5.21 and magpie on the same card and day; diffusers on the same card model, earlier. Real speed.",
              font=font(20), fill=th.muted)
    return image


def main() -> None:
    rows = lanes()
    for th in (LIGHT, DARK):
        frames = [frame(index / FPS, rows, th) for index in range(SECONDS * FPS + 1)]
        durations = [1000 // FPS] * (len(frames) - 1) + [int(HOLD * 1000)]
        path = OUT / f"qwen21_speed-race{th.name}.webp"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, quality=82, method=4)
        print(f"wrote {path.name} {path.stat().st_size // 1024} kb")


if __name__ == "__main__":
    main()
