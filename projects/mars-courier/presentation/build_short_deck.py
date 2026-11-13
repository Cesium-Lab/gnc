"""Build the short (~11 slides, ~10 min) Mars Courier talk.

Usage (from the repo root):
    venv/bin/python projects/mars-courier/presentation/build_short_deck.py            # .pptx + .pdf
    venv/bin/python projects/mars-courier/presentation/build_short_deck.py --no-pdf   # .pptx only

Reuses the theme, helpers and font-embedding PDF export of build_deck.py (which is imported, not
modified). Figures are downscaled into a temporary build directory before insertion so the PDF stays
around 1-2 MB; the original figures are never touched. Every number comes from
PRESENTATION_BRIEF.md; deck decisions (Round 8 and earlier) from DECK_DECISIONS.md. No animations.
"""

import math
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.dont_write_bytecode = True  # keep presentation/ free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as bd
from build_deck import (
    BLUE,
    BLUE_D,
    BLUE_L,
    BLUE_T,
    CARD,
    INK,
    INK2,
    LINE,
    MARS,
    MARS_L,
    MARS_T,
    MUTED,
    ORANGE,
    ORANGE_T,
    SECTIONS,
    WHITE,
    Deck,
    arrow,
    box,
    bullets,
    card,
    circle,
    eyebrow,
    headline,
    slide_number,
    tag,
    tx,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "mars_courier_short.pptx"

MAX_W = 2000         # px; figures are downscaled to this width
JPEG_OVER = 350_000  # bytes; an optimized PNG larger than this is re-encoded as JPEG
JPEG_Q = 90

BUILD = {"dir": None, "cache": {}}
CLOCK = {"t": 0.0}


# ---------------------------------------------------------------------------------------------
# Figures: downscale into a temp dir, then insert at native aspect ratio via build_deck.fig
# ---------------------------------------------------------------------------------------------
def small(rel):
    """Return the path of a downscaled copy of figures/<rel> in the build dir."""
    if rel in BUILD["cache"]:
        return BUILD["cache"][rel]
    src = ROOT / rel
    im = Image.open(src)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    stem = rel.replace("/", "__").rsplit(".", 1)[0]
    out = BUILD["dir"] / f"{stem}.png"
    im.save(out, optimize=True)
    if out.stat().st_size > JPEG_OVER:
        jpg = out.with_suffix(".jpg")
        bg = Image.new("RGB", im.size, (252, 252, 251))
        bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        bg.save(jpg, quality=JPEG_Q, optimize=True, progressive=True, subsampling=0)
        out.unlink()
        out = jpg
    BUILD["cache"][rel] = out
    return out


def fig(sh, rel, x, y, w, h, alt, **kw):
    return bd.fig(sh, small(rel), x, y, w, h, alt=alt, **kw)


# ---------------------------------------------------------------------------------------------
# Furniture
# ---------------------------------------------------------------------------------------------
def tracker(s, current=(), y=7.03):
    """The full deck's six-chip phase tracker; `current` may highlight several sections."""
    g = s.shapes.add_group_shape()
    x = 0.6
    for i, name in enumerate(SECTIONS):
        active = i in current
        fill, col = (MARS, WHITE) if active else ("ECEEF1", "5B6270")
        box(g.shapes, x, y, 1.5, 0.27, fill=fill, shape="round", radius=0.13, margin=0.02,
            text=[[(name, {"bold": active})]], size=10, color=col, font=bd.HEAD, align="c")
        x += 1.56
    g.name = "Phase tracker"


def footer(s, current=None):
    if current is not None:
        tracker(s, current)
    slide_number(s.shapes, 11.9, 7.03, 0.83, 0.27, MUTED)


def mmss(minutes):
    sec = round(minutes * 60)
    return f"{sec // 60}:{sec % 60:02d}"


def notes(s, minutes, points):
    t0 = CLOCK["t"]
    CLOCK["t"] += minutes
    head = f"Timing: ~{minutes:g} min  (clock {mmss(t0)} → {mmss(CLOCK['t'])})"
    s.notes_slide.notes_text_frame.text = "\n".join([head] + ["• " + p for p in points])


# ---------------------------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------------------------
def build():
    d = Deck()

    # 1 ---------------- Title ----------------
    d.section("Opening")
    s = d.new("Mars Courier", size=54, title_box=(0.6, 1.75, 6.6, 0.95))
    sh = s.shapes
    fig(sh, "figures/summary/heliocentric_transfer.png", 6.75, 0.55, 6.4, 6.6, ha="r", alpha=0.42,
        alt="Heliocentric Earth-to-Mars transfer, faded background")
    tx(sh, 0.6, 2.8, 5.9, 1.2, "End-to-end mission design from launch window to docking", size=28,
       color=INK, font=bd.HEAD, line=1.05)
    tx(sh, 0.6, 4.1, 6.0, 0.9,
       "Earth departure, interplanetary transfer, B-plane capture, and J2-aware rendezvous",
       size=18, color=INK2)
    tx(sh, 0.6, 5.35, 6.0, 0.45, [[("Colin Skinner", {"bold": True})]], size=20, color=INK,
       font=bd.HEAD)
    tx(sh, 0.6, 5.85, 6.0, 0.4, "[Course]  ·  [Date]", size=16, color=MUTED, name="Course and date")
    tx(sh, 0.6, 0.45, 4.0, 0.3, [[("MISSION DESIGN REVIEW", {"bold": True})]], size=11, color=MARS,
       font=bd.HEAD)
    slide_number(sh, 11.9, 7.03, 0.83, 0.27, MUTED)
    notes(s, 0.3, [
        ("A cargo courier that launches in June 2035, arrives at Mars on Christmas Day, and docks "
         "with a spacecraft already in Mars orbit."),
        "Ten minutes, in mission order; almost half of it goes to rendezvous, where the guidance work is.",
        "Fill in the course and date placeholders before presenting.",
    ])

    # 2 ---------------- Mission concept + timeline ----------------
    d.section("Overview")
    s = d.new("Mission concept")
    sh = s.shapes
    eyebrow(s, "Overview")
    headline(s, "Deliver a Perseverance-class payload to a spacecraft waiting in a 300 km Mars orbit")
    steps = [
        ("Launch", "2035-06-14", "Cape Canaveral, 300 km parking orbit, escape burn"),
        ("Cruise", "193 days", "Lambert transfer with one midcourse correction"),
        ("Capture", "2035-12-25", "B-plane-targeted direct injection into the target's orbit"),
        ("Dock", "2035-12-26", "~42 h of autonomous rendezvous, final miss 7–13 m"),
    ]
    for i, (name, big, body) in enumerate(steps):
        bx = 0.6 + i * 3.12
        rpo = i == 3
        box(sh, bx, 1.68, 2.82, 1.45, fill=BLUE_T if rpo else CARD, shape="round", radius=0.1,
            line=BLUE if rpo else None, lw=1.25)
        tx(sh, bx + 0.2, 1.76, 2.5, 0.3, [[(name.upper(), {"bold": True})]], size=11,
           color=BLUE_D if rpo else MARS, font=bd.HEAD)
        tx(sh, bx + 0.2, 2.04, 2.5, 0.42, [[(big, {"bold": True})]], size=20, color=INK,
           font=bd.HEAD)
        tx(sh, bx + 0.2, 2.5, 2.5, 0.6, body, size=13, color=INK2)
        if i < 3:
            arrow(sh, bx + 2.86, 2.4, bx + 3.08, 2.4, w=2.0)
    fig(sh, "figures/summary/mission_timeline.png", 0.6, 3.3, 12.1, 3.2,
        alt="Two-panel mission timeline: days since escape, and RPO hours since capture")
    tx(sh, 0.6, 6.55, 12.1, 0.35,
       "Escape 2035-06-14 22:04  ·  capture 2035-12-25 00:00  ·  docking 2035-12-26 17:36 (nominal) "
       "/ 18:57 (dispersed), TDB", size=13, color=MUTED, align="c")
    footer(s, ())
    notes(s, 0.8, [
        "Frame it as logistics: the target spacecraft and its orbit are fixed; we have to get to it.",
        ("Launch 2035-06-14 22:04 TDB, capture 2035-12-25 00:00 TDB, dock 2035-12-26 17:36 TDB "
         "(nominal) or 18:57 TDB (dispersed)."),
        "Timeline: left panel is days since the escape burn; right panel zooms into the RPO in hours.",
        "193 days of cruise, then about 43 hours to dock; phasing is most of those hours.",
        "Light remark: launch on my birthday, arrival on Christmas Day.",
    ])

    # 3 ---------------- Launch window ----------------
    d.section("Launch window")
    s = d.new("Finding the launch window")
    sh = s.shapes
    eyebrow(s, "01 · Launch window")
    headline(s, "Porkchop search over 2034–2036: a low-C3 type-I basin near June 2035")
    fig(sh, "figures/interplanetary/porkchop_c3.png", 0.6, 1.68, 8.1, 4.05, ha="l",
        alt="Porkchop plot of departure C3 over departure date and time of flight, 2034 to 2036")
    tx(sh, 0.6, 5.95, 12.1, 0.9, bullets([
        "Lambert solved on a **579 × 1029-day grid**; color = departure C3 (km²/s²)",
        "Constraints on C3 and departure declination leave **441 feasible transfers**",
    ], size=15, after=6))
    box(sh, 9.0, 1.68, 3.7, 2.75, fill=BLUE_T, shape="round", radius=0.1)
    tx(sh, 9.2, 1.8, 3.3, 0.35, [[("Selected transfer", {"bold": True, "color": BLUE_D})]],
       size=16, font=bd.HEAD)
    tx(sh, 9.2, 2.25, 3.35, 2.9, [
        "Depart **2035-06-14**",
        "Arrive **2035-12-25**",
        "Time of flight **193.5 days**",
        "Departure C3 **10.62 km²/s²**",
        "Declination **4.06°**: easy from the Cape",
        "Arrival v∞ **2.82 km/s**",
    ], size=15, color=INK, after=6)
    box(sh, 9.0, 4.63, 3.7, 1.1, fill=MARS_T, shape="round", radius=0.1, margin=0.18,
        text=[[("Aside: ", {"bold": True, "color": MARS}),
               ("launch on my birthday, arrive on Christmas Day. It still sits in the low-C3 band.",
                {})]], size=14, color=INK, anchor="m")
    footer(s, (0,))
    notes(s, 0.8, [
        "Axes: departure date on x, time of flight on y, color = departure C3 = v∞², the launcher's energy.",
        ("Two basins: type I (short way, ~150–250 days) and type II (long way, ~350–500 days); we use "
         "type I at C3 ≈ 10–11 km²/s²."),
        "579 × 1029-day grid at 1-day steps; 441 feasible transfers after C3 and declination limits.",
        ("Selected: 2035-06-14 → 2035-12-25, 193.5 days; C3 10.62 km²/s², DLA 4.06°; arrival v∞ "
         "2.820 km/s. A 0.01-day fine search confirmed it."),
    ])

    # 4 ---------------- Departure + transfer ----------------
    d.section("Departure and cruise")
    s = d.new("Leaving Earth and crossing to Mars")
    sh = s.shapes
    eyebrow(s, "02 · Departure  ·  03 · Cruise")
    headline(s, "One 3.68 km/s escape burn, timed onto the Lambert arc, then 193.5 days of cruise")
    fig(sh, "figures/summary/heliocentric_transfer.png", 0.6, 1.62, 6.0, 5.25, ha="l",
        alt="Top-down ecliptic view of the Earth-to-Mars transfer with dated markers")
    cards = [
        ("3.68 km/s", "escape burn",
         "Tangential at periapsis of the 300 km, 49.1° parking orbit: 7.73\u00a0→\u00a011.40\u00a0km/s", CARD, INK),
        ("564 → 132 m/s", "midcourse correction",
         "Launching 0.42 day after the Lambert reference lines the escape hyperbola up with the arc",
         BLUE_T, BLUE_D),
        ("8.0 min", "one-way light time at arrival",
         "0.97 AU from Earth: docking has to run onboard, not from the ground", ORANGE_T, ORANGE),
    ]
    for i, (big, lab, body, fill, col) in enumerate(cards):
        y0 = 1.68 + i * 1.75
        box(sh, 6.9, y0, 5.8, 1.55, fill=fill, shape="round", radius=0.1)
        tx(sh, 7.1, y0 + 0.12, 5.4, 0.5, [[(big, {"bold": True, "color": col}),
                                          ("   " + lab, {"color": MUTED, "size": 14})]],
           size=24, font=bd.HEAD)
        tx(sh, 7.1, y0 + 0.68, 5.45, 0.8, body, size=15, color=INK2)
    footer(s, (1, 2))
    notes(s, 0.9, [
        ("Escape: 3,675.6 m/s ideal tangential burn from the 300 km parking orbit, 7.726 → 11.401 km/s; "
         "hyperbola e = 1.178, v∞ 3.258 km/s."),
        ("Lambert is heliocentric and ignores Earth's gravity well, so the hyperbola must be timed onto "
         "the arc: launching 0.42 day late cut the midcourse from 564.2 to 131.8 m/s."),
        "Transfer: type I, 193.5 days, 1.0 → 1.48 AU; Mars leads Earth by 31° at launch.",
        ("At arrival the spacecraft is 0.97 AU from Earth, 8.0 min one way: this is why RPO must be "
         "autonomous."),
    ])

    # 5 ---------------- Arrival: the target-plane lever ----------------
    d.section("Arrival")
    s = d.new("The biggest lever: choosing the target plane")
    sh = s.shapes
    eyebrow(s, "04 · Arrival")
    headline(s, "Aligning the target orbit with the arrival geometry cut capture Δv from 5.52 to 2.27 km/s")
    fig(sh, "figures/summary/capture_trade.png", 0.6, 1.6, 12.1, 3.45, va="t",
        alt="Useful payload versus capture delta-v with coplanar, current and earlier design points")
    for i, (lbl, vals, col, tint, rov) in enumerate([
        ("Earlier design: target RAAN 15°", ["49.7°", "5.52", "3.35", "1.3 t"], ORANGE, ORANGE_T,
         "1.3"),
        ("Current design: target RAAN 125°", ["6.8°", "2.27", "0.09", "6.7 t"], BLUE, BLUE_T, "6.6"),
    ]):
        bx = 0.6 + i * 6.2
        box(sh, bx, 5.15, 5.9, 1.72, fill=tint, shape="round", radius=0.1,
            line=BLUE if i == 1 else None, lw=1.25)
        tx(sh, bx + 0.2, 5.22, 5.5, 0.35, [[(lbl, {"bold": True, "color": col})]], size=15,
           font=bd.HEAD)
        labels = ["plane-change angle β", "capture Δv, km/s", "km/s above coplanar",
                  f"payload: {rov} rovers"]
        for k, (v, lab) in enumerate(zip(vals, labels)):
            sx = bx + 0.2 + k * 1.42
            tx(sh, sx, 5.62, 1.4, 0.45, [[(v, {"bold": True})]], size=22, color=INK, font=bd.HEAD)
            tx(sh, sx, 6.12, 1.35, 0.7, lab, size=13, color=MUTED)
    footer(s, (3,))
    notes(s, 1.2, [
        "Takeaway 1: matching the target plane to the arrival geometry is the single biggest lever.",
        ("Direct injection does the plane change at periapsis, the most expensive place, so the angle β "
         "matters enormously. Same 45° inclination, different node."),
        "Earlier (RAAN 15°): β 49.7°, capture 5,524 m/s, a 3.35 km/s plane-change penalty.",
        "Current (RAAN 125°): β 6.8°, capture 2,268.5 m/s, only 93 m/s above the 2,175 m/s coplanar ideal.",
        ("Payload per Falcon Heavy: 1.30 t (1.3 Perseverance rovers) → 6.73 t (6.6 rovers). Napkin "
         "rocket equation: assumed 320 s Isp, 10% tankage."),
    ])

    # 6 ---------------- RPO architecture ----------------
    d.section("Rendezvous")
    s = d.new("Rendezvous guidance architecture")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous")
    headline(s, "Five-stage impulsive guidance, flown on a two-body + Mars J2 truth model")
    stage_txt = [
        ("Plane change", "At the node, optionally pre-biased against J2 nodal drift", "0 / 120.5"),
        ("Hohmann", "Transfer to the target radius", "16.8 / 24.9"),
        ("Phase entry", "Merged with the Hohmann circularization", "(with Hohmann)"),
        ("Closed-loop phasing", "20 revolutions, re-planned every rev from the actual state",
         "9.7 / 9.8"),
        ("Exit + HCW", "Phase exit, then two-impulse HCW with one retarget",
         "18.3 + 25.1 / 15.8 + 14.9"),
    ]
    for i, (name, desc, dv) in enumerate(stage_txt):
        bx = 0.6 + i * 2.48
        box(sh, bx, 1.68, 2.24, 2.7, fill=CARD, shape="round", radius=0.1)
        circle(sh, bx + 0.38, 2.06, 0.2, fill=BLUE)
        tx(sh, bx + 0.18, 1.86, 0.4, 0.4, [[(str(i + 1), {"bold": True})]], size=14,
           color=WHITE, font=bd.HEAD, align="c", anchor="m")
        tx(sh, bx + 0.18, 2.38, 1.95, 0.6, [[(name, {"bold": True})]], size=15, font=bd.HEAD)
        tx(sh, bx + 0.18, 2.95, 1.95, 0.85, desc, size=13, color=INK2)
        box(sh, bx + 0.12, 3.8, 2.0, 0.46, fill=WHITE, shape="round", radius=0.08, line=LINE,
            margin=0.04, text=[[(dv, {"bold": True})]], size=13, color=BLUE_D, align="c")
        if i < 4:
            arrow(sh, bx + 2.27, 3.03, bx + 2.45, 3.03, w=2.0)
    tx(sh, 0.6, 4.45, 12.1, 0.3, "Δv in m/s, nominal / dispersed", size=12, color=MUTED, align="r")
    card(sh, 0.6, 4.9, 3.0, 1.95, "Nominal", "Exact capture: chaser coplanar with the target, "
                                             "same radius", fill=BLUE_T, size=14)
    card(sh, 3.85, 4.9, 5.2, 1.95, "Dispersed: one seeded draw",
         ("Capture errors within ±100 km in a, e ≤ 0.01, ±3° in angles → a +73 km, e 0.0086, "
          "i +1.87°, RAAN −1.43°, aop −2.5°. Not a Monte Carlo."), fill=ORANGE_T, size=14)
    card(sh, 9.3, 4.9, 3.4, 1.95, "One propagated truth",
         "Every coast continues from the last state; every burn uses current elements", size=14)
    footer(s, (4,))
    notes(s, 1.4, [
        ("Inputs straight from the arrival stage: capture epoch, post-capture state, target orbit "
         "(300 km / 45° / 125°, target 13° true anomaly at capture, phase gap 107.9°)."),
        ("Five stages: plane change at the node, Hohmann, phase entry merged with circularization, 20 "
         "revs of closed-loop phasing, phase exit + two-impulse HCW with a retarget."),
        "Settings: HCW handoff threshold 20 km, HCW transfer 40 min, docking threshold 50 m.",
        ("Dispersed case: one example drawn with a fixed, reproducible random seed, not a Monte Carlo. "
         "Offsets a +73 km, e 0.0086, i +1.87°, RAAN −1.43°, aop −2.5°."),
        "No navigation errors or finite burns are modeled in either case.",
    ])

    # 7 ---------------- RPO results ----------------
    s = d.new("Rendezvous results")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous")
    headline(s, "Both cases dock within 7–13 m, about 42 hours after capture")
    _, (fx, fy, fw, fh) = fig(sh, "figures/rpo/rsw_terminal.png", 0.6, 1.62, 7.6, 2.62, ha="l",
                              alt="Terminal approach in the target's RSW frame, dispersed case")
    tag(sh, fx + fw - 1.75, fy + 0.1, "Dispersed case", fill=ORANGE_T, color=ORANGE, w=1.6)
    fig(sh, "figures/summary/rpo_log_range.png", 0.6, fy + fh + 0.12, 7.6, 2.3, ha="l",
        alt="Chaser-target range on a log scale for both cases with 15 km and 200 m reference lines")
    for i, (name, vals, tint, col) in enumerate([
        ("Nominal", ["69.8 m/s", "25", "41.6 h", "13.5 m"], BLUE_T, BLUE_D),
        ("Dispersed", ["186.0 m/s", "27", "43.0 h", "7.3 m"], ORANGE_T, ORANGE),
    ]):
        y0 = 1.68 + i * 1.72
        box(sh, 8.5, y0, 4.2, 1.55, fill=tint, shape="round", radius=0.1)
        tx(sh, 8.7, y0 + 0.1, 3.8, 0.35, [[(name, {"bold": True, "color": col})]], size=15,
           font=bd.HEAD)
        for k, (v, lab) in enumerate(zip(vals, ["total Δv", "burns", "to dock", "final miss"])):
            sx = 8.7 + (k % 2) * 1.95
            sy = y0 + 0.47 + (k // 2) * 0.5
            tx(sh, sx, sy, 1.95, 0.5, [[(v, {"bold": True}), ("  " + lab, {"color": MUTED,
                                                                         "size": 12})]],
               size=17, color=INK, font=bd.HEAD)
    tx(sh, 8.5, 5.18, 4.2, 0.3, [[("Open items", {"bold": True})]], size=15, font=bd.HEAD)
    tx(sh, 8.5, 5.55, 4.2, 1.35, bullets([
        "Dispersed HCW handoff at **18.2 km**, outside the 15 km linear region",
        "Final approach crosses the **200 m keep-out**",
    ], size=14, after=6, bcolor=ORANGE))
    footer(s, (4,))
    notes(s, 1.5, [
        ("Top: terminal approach in the target's RSW frame (dispersed case); along-track horizontal, "
         "radial up; dotted circle = 15 km HCW linear region."),
        ("Bottom: range on a log scale for both cases; phasing closes thousands of km slowly, HCW the "
         "last ~18 km in 1.3 h. Reference lines at 15 km and 200 m."),
        "Nominal: 69.8 m/s, 25 burns, 41.6 h, 13.5 m miss. Dispersed: 186.0 m/s, 27 burns, 43.0 h, 7.3 m.",
        ("Be upfront: dispersed HCW starts at 18.2 km, outside the 15 km region (retargets at 290 m); "
         "both paths cross the 200 m keep-out because HCW aims straight at the target."),
    ])

    # 8 ---------------- What broke and the fix ----------------
    s = d.new("What broke, and how I fixed it")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Design logic")
    headline(s, "J2 broke open-loop phasing; closed-loop replanning plus a J2 pre-bias fixed it")
    box(sh, 0.6, 1.62, 12.1, 0.42, fill="ECEEF1", shape="round", radius=0.08, margin=0.15,
        text=[[("Development history · ", {"bold": True}),
               (("earlier test geometry (target RAAN 15°, synthetic chaser offset). Not comparable with "
                 "current results."), {})]], size=14, color=INK2, anchor="m")
    dev = [
        ("Open loop", "Incremental phasing burns, planned once", "51.5 km miss", "202 m/s", 51.5),
        ("Closed loop", "Vis-viva burn from the actual state every rev", "12.2 km", "165 m/s", 12.2),
        ("Diagnose", "Residual mostly cross-track (~9.7 km): J2 differential nodal regression", None,
         None, None),
        ("Pre-bias", "Offset the plane change by the predicted drift", "5.5 km", "152 m/s", 5.5),
    ]
    for i, (t, b, miss, dv, mv) in enumerate(dev):
        bx = 0.6 + i * 3.12
        diag = miss is None
        box(sh, bx, 2.2, 2.82, 2.6, fill=MARS_T if diag else CARD, shape="round", radius=0.1)
        tx(sh, bx + 0.2, 2.3, 2.5, 0.4, [[(t, {"bold": True})]], size=16, font=bd.HEAD,
           color=MARS if diag else INK)
        tx(sh, bx + 0.2, 2.72, 2.5, 1.05, b, size=14, color=INK2)
        if diag:
            tx(sh, bx + 0.2, 3.75, 2.5, 1.0, "Phasing orbit ~16 km higher → node regresses more "
                                             "slowly (dΩ/dt ∝ a^(−7/2))", size=13, color=INK2)
        else:
            tx(sh, bx + 0.2, 3.68, 2.5, 0.25, "handoff miss", size=12, color=MUTED)
            box(sh, bx + 0.2, 3.96, 2.35 * mv / 51.5, 0.18, fill=ORANGE, shape="rect")
            tx(sh, bx + 0.2, 4.2, 2.5, 0.5, [[(miss, {"bold": True}), ("  ·  " + dv, {})]],
               size=15, color=INK, font=bd.HEAD)
        if i < 3:
            arrow(sh, bx + 2.86, 3.5, bx + 3.08, 3.5, w=2.0)
    box(sh, 0.6, 5.0, 4.6, 1.85, fill=CARD, shape="round", radius=0.1)
    tx(sh, 0.8, 5.1, 4.2, 0.35, [[("Tried and dropped", {"bold": True})]], size=15, font=bd.HEAD)
    tx(sh, 0.8, 5.5, 4.2, 1.3, "Hohmann shape corrections during phasing: ~2% saving for 3–18 "
                               "extra burns and 6–30 h, and sometimes worse", size=14, color=INK2)
    box(sh, 5.45, 5.0, 7.25, 1.85, fill=BLUE_T, shape="round", radius=0.1, line=BLUE, lw=1.25)
    tx(sh, 5.65, 5.1, 6.9, 0.35, [[("Current design (this mission)", {"bold": True,
                                                                      "color": BLUE_D})]],
       size=15, font=bd.HEAD)
    tx(sh, 5.65, 5.5, 6.9, 1.3, bullets([
        "Pre-bias **saves 50.4 m/s** in the dispersed case (236.4 → 186.0 m/s)",
        "In the nominal case it would **cost 31.6 m/s**, so the planner picks the cheaper plan",
    ], size=14, after=4, bcolor=BLUE))
    footer(s, (4,))
    notes(s, 1.5, [
        ("Takeaway 2. Development history on an earlier test geometry (target RAAN 15°, synthetic chaser "
         "offset); do not compare these numbers with the 69.8 / 186.0 m/s results."),
        "Open-loop incremental phasing under J2: 51.5 km handoff miss, 202.1 m/s.",
        "Closed loop, re-planned each rev from the actual state: 12.2 km, 164.9 m/s.",
        ("Residual ≈9.7 km cross-track from J2 differential nodal regression: the phasing orbit sits "
         "~16 km higher, so its node regresses more slowly. Pre-bias: 5.5 km, 152.2 m/s."),
        ("Current design: pre-bias saves 50.4 m/s dispersed, would cost 31.6 m/s nominal, so it is "
         "chosen per case, not always on."),
    ])

    # 9 ---------------- Δv budget ----------------
    d.section("Mission summary")
    s = d.new("Mission Δv budget")
    sh = s.shapes
    eyebrow(s, "06 · Mission summary")
    headline(s, "The spacecraft needs 2.47–2.59 km/s; the launcher supplies 11.47 km/s (ideal)")
    fig(sh, "figures/summary/dv_budget.png", 0.6, 1.6, 12.1, 3.0,
        alt="Full mission delta-v by maneuver and vehicle")
    tx(sh, 0.6, 4.72, 6.0, 0.35, [[("Spacecraft budget, step by step", {"bold": True})]],
       size=15, font=bd.HEAD)
    tx(sh, 6.2, 4.72, 6.5, 0.35, [[("92%", {"bold": True, "color": BLUE}),
                                   (" of the spacecraft budget is Mars capture", {})]],
       size=15, color=INK2, align="r")
    x0, wmax = 2.4, 10.3
    scale = wmax / 2586.3
    rows = [("Midcourse", 0, 131.8, "132 m/s", BLUE_L),
            ("Mars capture", 131.8, 2268.5, "2,269 m/s", BLUE),
            ("RPO", 131.8 + 2268.5, 186.0, "70 (nominal) to 186 (dispersed) m/s", BLUE_D)]
    for k, (lab, start_, val, vs, col) in enumerate(rows):
        y0 = 5.15 + k * 0.4
        tx(sh, 0.6, y0, 1.7, 0.32, lab, size=14, color=INK2, anchor="m")
        if lab == "RPO":
            box(sh, x0 + start_ * scale, y0 + 0.04, 69.8 * scale, 0.26, fill=col)
            box(sh, x0 + (start_ + 69.8) * scale, y0 + 0.04, (186.0 - 69.8) * scale, 0.26,
                fill="B9C9DE")
            tx(sh, x0 + start_ * scale - 4.1, y0, 4.0, 0.32, [[(vs, {"bold": True})]], size=13,
               color=INK, anchor="m", align="r")
        elif lab == "Mars capture":
            box(sh, x0 + start_ * scale, y0 + 0.04, val * scale, 0.26, fill=col)
            tx(sh, x0 + start_ * scale + 0.1, y0, 3.0, 0.32, [[(vs, {"bold": True})]], size=13,
               color=WHITE, anchor="m")
        else:
            box(sh, x0 + start_ * scale, y0 + 0.04, val * scale, 0.26, fill=col)
            tx(sh, x0 + (start_ + val) * scale + 0.12, y0, 3.0, 0.32, [[(vs, {"bold": True})]],
               size=13, color=INK, anchor="m")
    yt = 5.15 + 3 * 0.4
    tx(sh, 0.6, yt, 1.7, 0.32, [[("Total", {"bold": True})]], size=14, anchor="m")
    box(sh, x0, yt + 0.04, 2470.2 * scale, 0.26, fill=INK)
    box(sh, x0 + 2470.2 * scale, yt + 0.04, (2586.3 - 2470.2) * scale, 0.26, fill="8A919C")
    tx(sh, x0 + 0.1, yt, 4.0, 0.32, [[("2.47 (nominal) to 2.59 (dispersed) km/s",
                                       {"bold": True})]], size=13, color=WHITE, anchor="m")
    footer(s, (5,))
    notes(s, 0.9, [
        "Takeaway 3: the spacecraft needs about 2.5 km/s, and Mars capture is 92% of it.",
        "Spacecraft: midcourse 131.8 + capture 2,268.5 + RPO 69.8 / 186.0 = 2,470.2 / 2,586.3 m/s.",
        ("Launch vehicle: ascent 7,790.7 + escape 3,675.6 = 11,466.3 m/s. These are ideal lower bounds, "
         "not real launch costs (no drag or gravity losses)."),
        "RPO is small (70–186 m/s) but it is where the precision engineering lives.",
        "Green bar: Mars surface ↔ orbit, 3.53 km/s ideal, a reference for a future lander, not in totals.",
    ])

    # 10 ---------------- Limitations + next steps ----------------
    s = d.new("Limitations and next steps")
    sh = s.shapes
    eyebrow(s, "06 · Mission summary")
    headline(s, "What a reviewer should ask next, and what I would build next")
    card(sh, 0.6, 1.68, 5.9, 4.3, "Limitations", bullets([
        "Patched conics, impulsive burns, no navigation errors",
        "Mars J2 only in the rendezvous",
        "One seeded dispersion case, not a Monte Carlo",
        "Dispersed HCW handoff (18.2 km) outside the 15 km linear region",
        "HCW path crosses the 200 m keep-out",
        "Ascent Δv is an ideal lower bound; payload uses assumed Isp and tankage",
        "131.8° launch azimuth not checked against Cape range safety",
    ], size=15, after=8, bcolor=ORANGE), fill=ORANGE_T, size=15, tsize=17)
    card(sh, 6.8, 1.68, 5.9, 4.3, "Next steps", bullets([
        ("**Docking:** hand off to HCW inside the linear region, hold at the 200 m keep-out "
        "boundary, then a slow V-bar or R-bar approach"),
        ("**GNC:** relative navigation (sensors + filter) and closed-loop control such as MPC, "
        "with contact, capture and abort logic"),
        "**Robustness:** Monte Carlo dispersions on every segment",
        "**Fixed target planes:** high-ellipse capture with the plane change near apoapsis",
        "**Models:** atmospheric ascent and descent; ground link budget",
    ], size=15, after=8, bcolor=BLUE), fill=BLUE_T, size=15, tsize=17)
    box(sh, 0.6, 6.2, 12.1, 0.65, fill=CARD, shape="round", radius=0.1, margin=0.18,
        text=[[("First up: ", {"bold": True, "color": BLUE_D}),
               (("the two rendezvous open items, via the proposed docking design (not implemented "
                "yet)."), {})]], size=15, color=INK, anchor="m")
    footer(s, (5,))
    notes(s, 0.6, [
        "Be upfront: these are the next questions a reviewer would ask.",
        ("The two RPO open items (handoff outside the linear region, keep-out crossing) drive the first "
         "next step: a proposed docking design, not implemented."),
        "Monte Carlo replaces the single seeded draw; high-ellipse capture covers fixed target planes.",
        "Range-safety check for the 131.8° azimuth is still open.",
    ])

    # 11 ---------------- Questions ----------------
    s = d.new("Questions", dark=True, size=48, title_box=(0.9, 0.95, 6.0, 1.0))
    sh = s.shapes
    circle(sh, 10.95, 3.6, 3.85, line="3A4150", lw=1.25)
    circle(sh, 11.3, 3.75, 2.9, fill=MARS)
    ang = math.radians(150)  # spacecraft dot on the ring, upper left
    circle(sh, 10.95 + 3.85 * math.cos(ang), 3.6 - 3.85 * math.sin(ang), 0.1, fill=WHITE)
    tx(sh, 0.9, 2.2, 5.5, 0.35, [[("THREE TAKEAWAYS", {"bold": True})]], size=13, color=MARS_L,
       font=bd.HEAD)
    take = [
        ("Choose the target plane", ("RAAN 15° → 125° cut capture from 5.52 to 2.27\u00a0km/s: "
                                    "1.3\u00a0→\u00a06.6\u00a0rovers per Falcon Heavy")),
        ("Closed loop, J2-aware", ("Re-plan every rev; the J2 pre-bias saves 50.4\u00a0m/s in the "
                                  "dispersed case")),
        ("About 2.5 km/s of spacecraft Δv", ("Capture dominates; RPO is 70–186\u00a0m/s with a "
                                           "7–13\u00a0m final miss")),
    ]
    for i, (t, b) in enumerate(take):
        y0 = 2.75 + i * 1.33
        circle(sh, 1.12, y0 + 0.25, 0.22, fill=MARS)
        tx(sh, 0.9, y0 + 0.03, 0.44, 0.44, [[(str(i + 1), {"bold": True})]], size=14, color=WHITE,
           font=bd.HEAD, align="c", anchor="m")
        tx(sh, 1.6, y0, 5.45, 1.25, [{"t": [(t, {"bold": True})], "size": 17, "font": bd.HEAD,
                                     "color": WHITE, "after": 3},
                                    {"t": b, "size": 15, "color": "C9CED6"}])
    slide_number(sh, 11.9, 7.03, 0.83, 0.27, "A9B0BB")
    notes(s, 0.3, [
        "Recap the three takeaways in one breath each, then open for questions.",
        ("Likely questions: keep-out crossing, linear-region handoff, Monte Carlo, range-safety azimuth. "
         "The full deck's backup slides cover burn tables, validation and costs."),
    ])
    return d


def main():
    d_tmp = Path(tempfile.mkdtemp(prefix="mars_short_"))
    BUILD["dir"] = d_tmp
    try:
        d = build()
        prs = d.prs
        bd.set_theme_fonts(prs)
        bd.add_sections(prs, d.sections)
        cp = prs.core_properties
        cp.title = "Mars Courier: End-to-End Mission Design (short talk)"
        cp.author = "Colin Skinner"
        prs.save(OUT)
        print(f"wrote {OUT} ({len(prs.slides)} slides, ~{CLOCK['t']:.1f} min of notes timing)")
    finally:
        shutil.rmtree(d_tmp, ignore_errors=True)
    if "--no-pdf" not in sys.argv:
        bd.export_pdf(OUT)


if __name__ == "__main__":
    main()
