"""Render magpie art with the sciel pencil engine. Previews go to the cache, exports go next to this file.

    uv run --no-project --with pillow --with numpy --with scipy python3 assets/build.py --preview thumb light
    uv run --no-project --with pillow --with numpy --with scipy python3 assets/build.py --export
"""
from __future__ import annotations

import argparse
import pathlib as pl
import sys

HERE = pl.Path(__file__).resolve().parent
ENGINE = pl.Path("/home/tensor/Shared/code/miscellaneous/sciel/scripts/art")
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(HERE))

from PIL import Image  # noqa: E402
from pencil import Sheet  # noqa: E402

import scene_magpie as scene  # noqa: E402

NAME = "magpie"
PREVIEW_DIR = pl.Path.home() / ".cache" / "pencil-art-preview"
SIZES: dict[str, tuple[int, int]] = {"banner": (2100, 700), "thumb": (1200, 675), "square": (1040, 1004)}
EXPORTS: tuple[tuple[str, str, str], ...] = (
    ("banner", "light", "banner"), ("banner", "dark", "banner-dark"), ("thumb", "light", "thumb"),
    ("thumb", "dark", "thumb-dark"), ("square", "dark", "square-dark"),
)


def render(kind: str, theme: str, scale: float = 1.0, ss: int = 3) -> Image.Image:
    """Draw the scene at the size of kind, scaled by scale, and return the RGB image."""
    W, H = (int(v * scale) for v in SIZES[kind])
    seed = sum(ord(c) for c in NAME + kind) * 7919
    S = Sheet(W, H, scene.PAL, theme, ss=ss, seed=seed, grid=getattr(scene, "GRID", None))
    for attr in ("smudge", "tooth", "weight"):
        if hasattr(scene, attr.upper()):
            setattr(S, attr, getattr(scene, attr.upper()))
    scene.draw(S)
    return S.render()


def main() -> None:
    """--preview KIND THEME writes one cached PNG, --export writes all five WebPs here."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs=2, metavar=("KIND", "THEME"))
    ap.add_argument("--scale", type=float, default=.5)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.preview:
        kind, theme = a.preview
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out = PREVIEW_DIR / f"{NAME}-{kind}-{theme}.png"
        render(kind, theme, a.scale, ss=2 if a.scale < .75 else 3).save(out)
        print(out)
    if a.export:
        for kind, theme, suffix in EXPORTS:
            out = HERE / f"{NAME}-{suffix}.webp"
            render(kind, theme).save(out, "WEBP", quality=74, method=6)
            print(f"{out.stat().st_size // 1024:5d} KB  {out}")


if __name__ == "__main__":
    main()
