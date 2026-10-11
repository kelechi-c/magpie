"""magpie (Qwen-Image 2.1): an edit drawn as one thread, read left to right. A framed drawing (the input)
is cut into tiles and strung on a cord behind a few round beads (the prompt). The cord sags, loops once and
runs through a rack of plates (the transformer blocks, fed by one shared rail). It leaves carrying scribbled
tiles that clear up, and lands in a second frame: the same view at night, the moon in the accent. Below:
8 tall and 40 short tally marks (Turbo and base steps) and a stopwatch."""
from __future__ import annotations

import math

from pencil import Sheet, catmull, ellipse_pts, point_at, rgb
from common import ring_ticks

Pt = tuple[float, float]

PAL = {
    "light": dict(paper=rgb("#E6E2D8"), grid=rgb("#000000"), aux=rgb("#7B7A7F"), ink=rgb("#23242B"),
                  acc=rgb("#9B2F6B"), hi=rgb("#FFFFFF"), glow=rgb("#F5F1E6"), opacity={"aux": .85}),
    "dark": dict(paper=rgb("#19191C"), grid=rgb("#000000"), aux=rgb("#8E8D93"), ink=rgb("#E9E5DB"),
                 acc=rgb("#E07AAE"), hi=rgb("#FFFFFF"), glow=rgb("#2C2B30"), opacity={"aux": .85}),
}
SMUDGE = .15

PLATES = 8            # blocks drawn (the model has 32)
TURBO, BASE = 8, 40   # sampling steps, drawn as tally marks
RIDGE: list[Pt] = [(-.5, .12), (-.3, -.06), (-.16, -.26), (-.04, -.04), (.1, -.2), (.2, -.34), (.34, -.08), (.5, .1)]


def box(x0: float, y0: float, x1: float, y1: float) -> list[Pt]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def tile(c: Pt, ang: float, half: float) -> list[Pt]:
    """A square of half-side `half` centred on c, turned to lie along the cord."""
    cos, sin = math.cos(ang), math.sin(ang)
    return [(c[0] + x * half * cos - y * half * sin, c[1] + x * half * sin + y * half * cos)
            for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def view(S: Sheet, cx: float, cy: float, w: float, h: float, night: bool) -> None:
    """A framed landscape: hatched mat, two peaks with shaded flanks, pines, water. By day a rayed sun in
    graphite; at night an accent crescent, stars and snow on the peaks."""
    k = S.k
    mat = w * .07
    outer, inner = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), box(cx - w / 2 + mat, cy - h / 2 + mat, cx + w / 2 - mat, cy + h / 2 - mat)
    S.erase(outer)
    S.hatch(outer, ang=58, gap=3.2, w=.9, lv=2, brk=.1, angj=.6)
    S.erase(inner)
    S.poly(outer, w=2.6, lv=3, wob=.5)
    S.poly(inner, w=1.9, lv=3, wob=.45)
    iw, ih = w - 2 * mat, h - 2 * mat
    ridge = [(cx + x * iw * .96, cy + y * ih + ih * .1) for x, y in RIDGE]
    S.stroke(catmull(ridge, 6), w=2.2, lv=3, wob=.6)
    for a, b in ((2, 3), (5, 6)):  # shaded flanks
        S.hatch([ridge[a], ridge[b], (ridge[b][0], ridge[a + 1][1] + ih * .2), (ridge[a][0] + iw * .03, ridge[a][1] + ih * .28)],
                ang=70, gap=3.4, w=.9, lv=2, brk=.15, inset=.5)
    ground = cy + ih * .26
    S.stroke(catmull([(cx - iw * .48, ground), (cx - iw * .1, ground + 3 * k), (cx + iw * .48, ground - 2 * k)], 8), w=1.7, lv=3, wob=.5)
    for j in range(3):
        y = ground + ih * (.07 + .055 * j)
        S.line(cx - iw * (.36 - .1 * j), y, cx + iw * (.3 - .08 * j), y, w=1.0, lv=2, wob=.4, passes=1, dropout=.2)
    for px, s in ((-.36, 1.0), (-.27, .75)):  # pines: a trunk and a zigzag
        x, top, base = cx + px * iw, ground - ih * .3 * s, ground
        S.line(x, base, x, top, w=1.3, lv=3, wob=.2, passes=1)
        S.stroke([(x + sgn * iw * .045 * s * (i + 1) / 4, top + (base - top) * (i + .6) / 4.6) for i in range(4) for sgn in (-1, 1)],
                 w=1.4, lv=3, wob=.3, passes=1)
    sx, sy, sr = cx + iw * .3, cy - ih * .3, ih * .1
    if not night:
        S.circle(sx, sy, sr, w=1.8, lv=3, wob=.3)
        for i in range(10):
            a = i * math.tau / 10
            S.line(sx + math.cos(a) * sr * 1.4, sy + math.sin(a) * sr * 1.4, sx + math.cos(a) * sr * 1.9, sy + math.sin(a) * sr * 1.9,
                   w=1.2, lv=2, wob=.2, passes=1)
        return
    S.wash(sx, sy, sr * 3.2, sr * 3.2, .6 if S.theme == "light" else .5)
    moon = ellipse_pts(sx, sy, sr * 1.15, sr * 1.15, n=40)
    S.hatch(moon, ang=40, gap=2.8, w=1.0, ink="acc", lv=3, brk=.05, inset=.5)
    S.erase_ellipse(sx + sr * .55, sy - sr * .3, sr * .95, sr * .95)
    S.stroke(ellipse_pts(sx, sy, sr * 1.15, sr * 1.15, math.radians(35), math.radians(270)), ink="acc", w=2.3, lv=3, wob=.25)
    S.stroke(ellipse_pts(sx + sr * .55, sy - sr * .3, sr * .95, sr * .95, math.radians(80), math.radians(215)), ink="acc", w=2.0, lv=3, wob=.25)
    for fx, fy in ((-.3, -.36), (-.05, -.4), (.08, -.28)):
        S.sparkle(cx + fx * iw, cy + fy * ih, 5 * k, ink="ink", lv=3, w=1.0)
    for a in (2, 5):  # snow caps
        px, py = ridge[a]
        S.stroke([(px - iw * .05, py + ih * .1), (px - iw * .02, py + ih * .06), (px, py + ih * .11), (px + iw * .025, py + ih * .06),
                  (px + iw * .055, py + ih * .1)], w=1.3, lv=3, wob=.3, passes=1)


def draw(S: Sheet) -> None:
    W, H, k = S.W, S.H, S.k
    rng = S.rng
    wide = W / H > 2
    light = S.theme == "light"
    by = H * (.47 if wide else .44)                     # the cord's height at the frames and the rack
    fw = min(H * .6, W * (.215 if wide else .23))        # frame width
    fh = fw / 1.3
    lx, rx = W * (.135 if wide else .145), W * (.865 if wide else .855)
    mx = W * .5
    ph = H * .5 if wide else fw * 1.25                       # plate height
    pw, pitch = min(15 * k, W * .009), min(33 * k, W * .017)
    r0 = mx - pitch * (PLATES - 1) / 2
    ty = H * (.9 if wide else .885)

    # ---- sky, and the glow behind the rack ----
    for fy in (.06, .12, .19):
        a = rng.uniform(.25, .55) * W
        S.line(a, H * fy, a + rng.uniform(.15, .3) * W, H * fy, w=.9, ink="aux", lv=2, wob=.4, passes=1)
    S.wash(mx, by, pitch * PLATES * 1.5, ph * .9, .62 if light else .52)

    # ---- the cord: out of the first frame, a sag and one lazy loop, through the rack, up into the second frame ----
    a0, a1 = lx + fw / 2, r0 - pw
    b0, b1 = r0 + pitch * (PLATES - 1) + pw * 2, rx - fw / 2
    d, e = a1 - a0, b1 - b0
    sag = min(H * .13, d * .45)
    lead = catmull([(a0, by), (a0 + d * .22, by + sag * .9), (a0 + d * .5, by + sag * 1.1), (a0 + d * .68, by + sag * .2),
                    (a0 + d * .55, by - sag * .75), (a0 + d * .42, by + sag * .1), (a0 + d * .74, by + sag * .75), (a1, by)]
                   if wide else [(a0, by), (a0 + d * .35, by + sag), (a0 + d * .72, by + sag * .7), (a1, by)], 16)
    tail = catmull([(b0, by), (b0 + e * .3, by + sag * .8), (b0 + e * .62, by + sag * .5), (b0 + e * .85, by - sag * .3), (b1, by)], 16)
    S.stroke(lead, w=2.3, lv=3, wob=.7)
    S.stroke(tail, w=2.3, lv=3, wob=.7)

    half = min(22 * k, d * (.05 if wide else .085))
    for f in ((.06, .1, .14) if wide else (.1, .22)):  # the prompt: round beads
        p, _ = point_at(lead, f)
        S.erase_ellipse(p[0], p[1], half * .62, half * .62)
        S.circle(p[0], p[1], half * .62, w=1.8, lv=3, wob=.3)
    for f in ((.21, .29, .37, .8, .9) if wide else (.42, .65, .88)):  # the input drawing: hatched tiles
        p, ang = point_at(lead, f)
        sq = tile(p, ang, half)
        S.erase(sq)
        S.hatch(sq, ang=62, gap=3.2, w=.9, lv=2, brk=.1, inset=.5, angj=.5)
        S.poly(sq, w=2.0, lv=3, wob=.35)
    marks = (.1, .27, .44, .61, .78, .93) if wide else (.14, .4, .66, .9)
    for i, f in enumerate(marks):  # the output: scribble that clears along the cord
        p, ang = point_at(tail, f)
        sq = tile(p, ang, half)
        S.erase(sq)
        for _ in range(int(30 * (1 - i / (len(marks) - 1)) ** 1.5)):
            x, y, t = p[0] + rng.uniform(-.8, .8) * half, p[1] + rng.uniform(-.8, .8) * half, rng.uniform(0, math.tau)
            S.line(x, y, x + math.cos(t) * half * .32, y + math.sin(t) * half * .32, w=1.3, lv=3, wob=.2, passes=1)
        if i >= len(marks) / 2:
            S.stroke([(sq[3][0] * .8 + sq[0][0] * .2, sq[3][1] * .8 + sq[0][1] * .2), ((sq[0][0] + sq[1][0]) / 2, (sq[0][1] + sq[1][1]) / 2 + half * .4),
                      (sq[2][0] * .8 + sq[1][0] * .2, sq[2][1] * .8 + sq[1][1] * .2)], w=1.2, lv=3, wob=.3, passes=1)
        S.poly(sq, w=2.0, lv=3, wob=.35)

    # ---- the rack: plates in section, slotted for the cord, hung from one shared rail ----
    rail = by - ph / 2 - 30 * k
    S.line(r0 - pitch, rail, r0 + pitch * PLATES, rail, w=2.0, lv=3, wob=.3)
    S.circle(r0 - pitch - 9 * k, rail, 8 * k, w=1.6, lv=3, wob=.3)
    for i in range(PLATES):
        x = r0 + i * pitch
        S.line(x + pw / 2, rail, x + pw / 2, by - ph / 2, w=1.2, lv=2, wob=.2, passes=1)
        for y0, y1 in ((by - ph / 2, by - half * 1.5), (by + half * 1.5, by + ph / 2)):
            plate = box(x, y0, x + pw, y1)
            S.erase(plate)
            S.hatch(plate, ang=118, gap=2.8, w=.9, lv=2, brk=.08, angj=.5)
            S.poly(plate, w=2.1, lv=3, wob=.3)
    S.dashed([(r0 - pw, by), (b0, by)], dash=9, gap=7, w=1.4, lv=3, wob=.2, passes=1)

    # ---- before and after ----
    view(S, lx, by, fw, fh, night=False)
    view(S, rx, by, fw, fh, night=True)

    # ---- bottom: 8 tall and 40 short tally marks, and a stopwatch ----
    dcx, dr = W * (.9 if wide else .86), min(H * (.075 if wide else .085), W * .055)
    S.line(W * .05, ty, dcx - dr * 1.7, ty, ink="aux", lv=2, w=1.0, wob=.2, passes=1)
    step = min(13 * k, W * .0105)
    x = mx - step * (TURBO + BASE + 2.2) / 2
    for count, h in ((TURBO, 40 * k), (BASE, 20 * k)):
        for _ in range(count):
            S.line(x, ty - 3 * k, x + rng.uniform(-1, 1) * k, ty - 3 * k - h, w=2.0, lv=3, wob=.15, passes=1)
            x += step
        x += step * 2.2
    S.circle(dcx, ty, dr, w=1.5, lv=3)
    S.circle(dcx, ty, dr * .86, w=.9, lv=2)
    ring_ticks(S, dcx, ty, dr * .86, dr * .86, n=40, length=6, a0=math.radians(135), a1=math.radians(405), long_every=5)
    S.stroke([(dcx, ty - dr), (dcx, ty - dr * 1.2)], w=1.6, lv=3, wob=0, passes=1)
    S.stroke([(dcx - dr * .2, ty - dr * 1.2), (dcx + dr * .2, ty - dr * 1.2)], w=1.6, lv=3, wob=0, passes=1)
    ha = math.radians(135 + 270 * .2)
    S.line(dcx, ty, dcx + math.cos(ha) * dr * .72, ty + math.sin(ha) * dr * .72, ink="acc", lv=3, w=1.9, wob=.1, passes=1)
    S.dot(dcx, ty, 2.4, "ink", 3)

    # ---- sparkles ----
    for sx, sy, r in ((.045, .1, 11), (.955, .1, 14), (.05, .8, 9)):
        S.sparkle(W * sx, H * sy, r * k, ink="ink", lv=3, w=1.3)
