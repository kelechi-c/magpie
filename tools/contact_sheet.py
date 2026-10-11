import pathlib as pl
import sys

from PIL import Image

Row = list[pl.Path]


# pastes rows of images into one sheet, each cell `cell` px square, for eyeballing a run.
def sheet(rows: list[Row], cell: int) -> Image.Image:
    canvas = Image.new("RGB", (cell * max(len(row) for row in rows), cell * len(rows)), "white")
    for y, row in enumerate(rows):
        for x, path in enumerate(row):
            canvas.paste(Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS), (x * cell, y * cell))
    return canvas


if __name__ == "__main__":
    out, cell, *groups = sys.argv[1:]
    rows = [sorted(pl.Path(group).glob("*.[jp][pn]g")) for group in groups]
    sheet(rows, int(cell)).save(out, quality=90)
    print(f"wrote {out}")
