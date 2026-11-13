"""Build the ~10-minute Mars Courier talk for the SpaceX GNC onsite interview.

Usage (from the repo root):
    venv/bin/python projects/mars-courier/presentation/build_interview_deck.py            # .pptx + .pdf
    venv/bin/python projects/mars-courier/presentation/build_interview_deck.py --no-pdf   # .pptx only

Look: matches `Starlink Presentation 2.pdf` (navy title/closing slides, cream content slides, navy
serif titles top-left, sans bullets with ● / ○, header label top-right, page number bottom-right).
That deck uses Merriweather + DM Sans, which are not installed here; the closest installed faces are
used instead: Georgia (titles) and Inter (body). Both are embedded in the PDF through an intermediate
ODP, like build_deck.py does, because the sandboxed LibreOffice cannot see the system font folders.

Content follows Round 9 of DECK_DECISIONS.md; every number comes from PRESENTATION_BRIEF.md. Slide 3
(About Me) is copied verbatim from the Starlink deck, and its photo is extracted from that PDF at
build time. Figures are downscaled (and only their baked-in title strips cropped) into a temporary
directory so the PDF stays around 1-2 MB; the original figures are never touched.
build_deck.py is imported for a few generic helpers only and is not modified.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Pt

sys.dont_write_bytecode = True  # keep presentation/ free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_deck import E, _strip_style, add_sections, md_runs, poly, rgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # projects/mars-courier
OUT = HERE / "mars_courier_interview.pptx"
STARLINK = HERE / "Starlink Presentation 2.pdf"

# Canvas: same 10 x 5.625 in (720 x 405 pt) page as the Starlink deck, so sizes match one to one.
SW, SH = 10.0, 5.625

# Colors sampled from the Starlink renders (pdftoppm -r 100).
NAVY = "005088"    # title-slide background, titles, body text
CREAM = "F3F0DF"   # content background; text on navy
GREEN = "D9EAD3"   # diagram box fill
PINK = "EAD1DC"    # diagram highlight fill
EDGE = "3E373A"    # diagram outline / diagram text
SOFT = "4F7FA6"    # muted navy for captions and labels
WHITE = "FFFFFF"
LV = "9DB9CF"      # launch-vehicle bars (light navy)

HEAD = "Georgia"   # stands in for Merriweather
BODY = "Inter"     # stands in for DM Sans
LABEL = "SpaceX GNC Onsite Interview"
TITLE_W = 7.75  # in; content titles must end left of the header label

FONT_FILES = {
    HEAD: ["/System/Library/Fonts/Supplemental/Georgia.ttf",
           "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
           "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
           "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf"],
    BODY: [str(Path.home() / "Library/Fonts" / f) for f in
           ("Inter-Regular.otf", "Inter-Bold.otf", "Inter-Italic.otf", "Inter-BoldItalic.otf")],
}

MAX_W = 2000         # px; figures are downscaled to this width
JPEG_OVER = 350_000  # bytes; an optimized PNG larger than this is re-encoded as JPEG
JPEG_Q = 90

BUILD = {"dir": None, "cache": {}}
CLOCK = {"t": 0.0}
ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}

# Bullet geometry, measured on the Starlink About Me slide: ● at 0.50 in, text at 0.81 in;
# ○ at 0.97 in, text at 1.31 in (body boxes start at x = 0.30 in with zero inset).
LEVELS = {0: ("●", 0.51, 0.31), 1: ("○", 1.01, 0.34), 2: ("■", 1.51, 0.34)}


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def small(rel, crop_top=0.0):
    """Downscaled copy of figures/<rel> in the build dir; crop_top drops the baked-in title strip."""
    key = (rel, crop_top)
    if key in BUILD["cache"]:
        return BUILD["cache"][key]
    im = Image.open(ROOT / rel)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if crop_top:
        im = im.crop((0, round(im.height * crop_top), im.width, im.height))
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    stem = rel.replace("/", "__").rsplit(".", 1)[0] + (f"_c{round(crop_top * 100)}" if crop_top else "")
    out = BUILD["dir"] / f"{stem}.png"
    im.save(out, optimize=True)
    if out.stat().st_size > JPEG_OVER:
        jpg = out.with_suffix(".jpg")
        bg = Image.new("RGB", im.size, (252, 252, 251))
        bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        bg.save(jpg, quality=JPEG_Q, optimize=True, progressive=True, subsampling=0)
        out.unlink()
        out = jpg
    BUILD["cache"][key] = out
    return out


def about_photo():
    """Extract Colin's photo from slide 3 of the Starlink PDF (the only image on that page)."""
    subprocess.run(["pdfimages", "-j", "-f", "3", "-l", "3", str(STARLINK),
                    str(BUILD["dir"] / "about")], check=True)
    src = min(BUILD["dir"].glob("about-*"))
    im = Image.open(src).convert("RGB")
    im = im.resize((700, round(im.height * 700 / im.width)), Image.LANCZOS)
    out = BUILD["dir"] / "about_photo.jpg"
    im.save(out, quality=88, optimize=True, progressive=True)
    return out


def picture(sh, path, x, y, w, h, ha="c", va="t", alt=""):
    """Insert an image at native aspect ratio, fitted inside the (x, y, w, h) box."""
    iw, ih = Image.open(path).size
    r = iw / ih
    fw, fh = (h * r, h) if w / h > r else (w, w / r)
    fx = x + {"l": 0, "c": (w - fw) / 2, "r": w - fw}[ha]
    fy = y + {"t": 0, "m": (h - fh) / 2, "b": h - fh}[va]
    pic = sh.add_picture(str(path), E(fx), E(fy), E(fw), E(fh))
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic, (fx, fy, fw, fh)


def fig(sh, rel, x, y, w, h, alt, crop_top=0.0, **kw):
    return picture(sh, small(rel, crop_top), x, y, w, h, alt=alt, **kw)


# ---------------------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------------------
def _bullet(p, lvl, color):
    char, mar, ind = LEVELS[lvl]
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(E(mar))))
    pPr.set("indent", str(-int(E(ind))))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buSzPct")).set("val", "70000")
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", BODY)
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def _number(p, color, font, mar=0.51, ind=0.41):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(E(mar))))
    pPr.set("indent", str(-int(E(ind))))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", font)
    etree.SubElement(pPr, qn("a:buAutoNum")).set("type", "arabicPeriod")


def fill(tf, paras, size=14, color=NAVY, font=BODY, align="l", after=0, line=1.0):
    """paras: str | list of (str | runs | dict). Dict keys: t, lvl, num, size, color, font, bold,
    u (underline), after, before, align, line. '**x**' marks bold inside a string."""
    if isinstance(paras, str):
        paras = [paras]
    for i, item in enumerate(paras):
        spec = item if isinstance(item, dict) else {"t": item}
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ALIGN[spec.get("align", align)]
        p.space_after = Pt(spec.get("after", after))
        if spec.get("before"):
            p.space_before = Pt(spec["before"])
        p.line_spacing = spec.get("line", line)
        content = spec.get("t", "")
        runs = md_runs(content) if isinstance(content, str) else content
        col = spec.get("color", color)
        for text, opts in runs:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = opts.get("font", spec.get("font", font))
            f.size = Pt(opts.get("size", spec.get("size", size)))
            f.bold = opts.get("bold", False) or spec.get("bold", False)
            f.italic = opts.get("italic", spec.get("italic", False))
            f.underline = opts.get("u", spec.get("u", False))
            f.color.rgb = rgb(opts.get("color", col))
            if opts.get("sup"):
                r._r.get_or_add_rPr().set("baseline", "30000")
        if spec.get("lvl") is not None:
            _bullet(p, spec["lvl"], col)
        elif spec.get("num"):
            _number(p, col, spec.get("font", font))
    return tf


def tx(sh, x, y, w, h, paras, anchor="t", name=None, **kw):
    tb = sh.add_textbox(E(x), E(y), E(w), E(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = ANCHOR[anchor]
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    fill(tf, paras, **kw)
    if name:
        tb.name = name
    return tb


def B(text, lvl=0, **kw):
    """One bullet paragraph."""
    return {"t": text, "lvl": lvl, **kw}


def box(sh, x, y, w, h, fill_color=GREEN, line=EDGE, lw=0.75, text=None, shape="rect",
        dash=None, name=None, **kw):
    kind = {"rect": MSO_SHAPE.RECTANGLE, "oval": MSO_SHAPE.OVAL}[shape]
    s = sh.add_shape(kind, E(x), E(y), E(w), E(h))
    if fill_color is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill_color)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = dash
    _strip_style(s)
    tf = s.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = ANCHOR[kw.pop("anchor", "m")]
    m = kw.pop("margin", 0.05)
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", E(m))
    if text is not None:
        kw.setdefault("color", EDGE)
        kw.setdefault("align", "c")
        fill(tf, text, **kw)
    if name:
        s.name = name
    return s


def arrow(sh, x1, y1, x2, y2, color=EDGE, w=1.0, dash=None):
    return poly(sh, [(x1, y1), (x2, y2)], color=color, w=w, dash=dash, head=True)


# ---------------------------------------------------------------------------------------------
# Slide furniture
# ---------------------------------------------------------------------------------------------
def page_number(sh, color=NAVY):
    tb = sh.add_textbox(E(9.15), E(5.22), E(0.6), E(0.2))
    tf = tb.text_frame
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, qn("a:fld"))
    fld.set("id", "{" + str(uuid.uuid4()).upper() + "}")
    fld.set("type", "slidenum")
    rpr = etree.SubElement(fld, qn("a:rPr"))
    rpr.set("lang", "en-US")
    rpr.set("sz", "800")
    sf = etree.SubElement(rpr, qn("a:solidFill"))
    etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
    etree.SubElement(rpr, qn("a:latin")).set("typeface", BODY)
    etree.SubElement(fld, qn("a:t")).text = "‹#›"
    tb.name = "Slide number"


def header_label(sh, text=LABEL, color=NAVY):
    tx(sh, 6.75, 0.24, 3.0, 0.2, text, size=8, color=color, align="r", name="Header label")


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = E(SW)
        self.prs.slide_height = E(SH)
        self.layout = self.prs.slide_layouts[5]  # "Title Only"
        self.sections = []

    def section(self, name):
        self.sections.append((name, []))

    def _slide(self, bg):
        s = self.prs.slides.add_slide(self.layout)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(bg)
        self.sections[-1][1].append(s.slide_id)
        return s

    def _title(self, s, text, box_, size, color, font, bold, align="l", anchor="t"):
        t = s.shapes.title
        x, y, w, h = box_
        t.left, t.top, t.width, t.height = E(x), E(y), E(w), E(h)
        tf = t.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = ANCHOR[anchor]
        for side in ("left", "right", "top", "bottom"):
            setattr(tf, f"margin_{side}", 0)
        p = tf.paragraphs[0]
        p.alignment = ALIGN[align]
        p.line_spacing = 1.0
        r = p.add_run()
        r.text = text
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = rgb(color)

    def content(self, title, size=28):
        """Cream content slide: navy serif title top-left, header label, page number."""
        width = ImageFont.truetype(FONT_FILES[HEAD][1], 1000).getlength(title) * size / 1000 / 72
        if width > TITLE_W:
            raise ValueError(f"title {title!r} is {width:.2f} in wide; it would hit the header label")
        s = self._slide(CREAM)
        self._title(s, title, (0.3, 0.22, TITLE_W, 0.55), size, NAVY, HEAD, True)
        header_label(s.shapes)
        page_number(s.shapes)
        return s

    def dark(self, title, box_, size, bold=True, align="l", anchor="t", number=False):
        s = self._slide(NAVY)
        self._title(s, title, box_, size, CREAM, HEAD, bold, align, anchor)
        if number:
            page_number(s.shapes, CREAM)
        return s


def mmss(minutes):
    sec = round(minutes * 60)
    return f"{sec // 60}:{sec % 60:02d}"


def notes(s, minutes, points):
    t0 = CLOCK["t"]
    CLOCK["t"] += minutes
    head = f"Timing: ~{minutes:g} min  (clock {mmss(t0)} → {mmss(CLOCK['t'])})"
    s.notes_slide.notes_text_frame.text = "\n".join([head] + ["• " + p for p in points])


def bar_chart(sh, x, y, w, h, cats, vals, fmt, color=NAVY, max_v=None):
    """Native horizontal bar chart, styled to sit on the cream background."""
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("handoff miss", vals)
    gf = sh.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, E(x), E(y), E(w), E(h), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = BODY
    ch.font.size = Pt(12)
    ch.font.color.rgb = rgb(NAVY)
    plot = ch.plots[0]
    plot.gap_width = 55
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(13)
    dl.font.bold = True
    dl.font.color.rgb = rgb(NAVY)
    ser = plot.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(color)
    ca = ch.category_axis
    ca.reverse_order = True  # first category on top
    ca.format.line.color.rgb = rgb(EDGE)
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.color.rgb = rgb(NAVY)
    va = ch.value_axis
    va.visible = False
    va.has_major_gridlines = False
    va.minimum_scale = 0
    if max_v:
        va.maximum_scale = max_v
    gf.name = "Handoff miss chart"
    return gf


def dv_chart(sh, x, y, w, h):
    """Native mission Δv budget (PRESENTATION_BRIEF.md §2.7), one bar per maneuver, in km/s."""
    rows = [("Earth ascent to LEO", 7790.7, "7.79 km/s", LV),
            ("Earth escape", 3675.6, "3.68 km/s", LV),
            ("Midcourse correction", 131.8, "132 m/s", NAVY),
            ("Mars capture", 2268.5, "2.27 km/s", NAVY),
            ("RPO, nominal", 69.8, "70 m/s", NAVY),
            ("RPO, dispersed", 186.0, "186 m/s", NAVY)]
    cd = CategoryChartData()
    cd.categories = [r[0] for r in rows]
    cd.add_series("Δv", [r[1] / 1000 for r in rows])
    gf = sh.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, E(x), E(y), E(w), E(h), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = BODY
    ch.font.size = Pt(12)
    ch.font.color.rgb = rgb(NAVY)
    plot = ch.plots[0]
    plot.gap_width = 45
    ser = plot.series[0]
    for i, (_, _, label, col) in enumerate(rows):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(col)
        dl = pt.data_label
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        tf = dl.text_frame
        tf.text = label
        f = tf.paragraphs[0].runs[0].font
        f.size = Pt(12)
        f.bold = True
        f.name = BODY
        f.color.rgb = rgb(NAVY)
    ca = ch.category_axis
    ca.reverse_order = True
    ca.format.line.color.rgb = rgb(EDGE)
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.color.rgb = rgb(NAVY)
    va = ch.value_axis
    va.visible = False
    va.has_major_gridlines = False
    va.minimum_scale = 0
    va.maximum_scale = 9.0
    gf.name = "Mission delta-v budget chart"
    return gf


# ---------------------------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------------------------
def build():
    d = Deck()

    # 1 ---------------- Title ----------------
    d.section("Opening")
    s = d.dark("Mars Courier Mission Design", (0.3, 0.24, 8.6, 0.75), 40)
    sh = s.shapes
    tx(sh, 8.4, 0.26, 1.35, 0.2, "[Date]", size=8, color=CREAM, align="r", name="Date")
    tx(sh, 0.32, 2.45, 4.2, 0.9, "A Technical Presentation for my SpaceX GNC Onsite Interview",
       size=18, color=CREAM, line=1.08)
    tx(sh, 0.46, 4.39, 3.5, 0.3, "Colin Skinner", size=15, color=CREAM)
    fig(sh, "figures/summary/heliocentric_transfer.png", 4.74, 1.6, 4.0, 3.55, crop_top=0.075,
        ha="l", alt="Top-down ecliptic view of the 193.5-day Earth-to-Mars transfer")
    notes(s, 0.3, [
        ("Mars Courier: a cargo courier that launches in June 2035, arrives at Mars on Christmas Day, "
         "and docks with a spacecraft already in Mars orbit. Design, code and results are my own."),
        "About ten minutes; roughly 40% of it goes to the rendezvous guidance, where the GNC work is.",
        "Fill in the [Date] placeholder (top right) before presenting.",
    ])

    # 2 ---------------- Overview ----------------
    d.section("Overview")
    s = d.content("Overview")
    items = ["About Me", "Project Scope and What I Built", "Mission Design: the Biggest Lever",
             "Rendezvous Guidance Architecture", "Rendezvous Results", "What Broke and How I Fixed It",
             "Validation and Δv Budget", "Limitations and Next Steps"]
    tx(s.shapes, 0.3, 0.92, 7.0, 4.2, [{"t": t, "num": True, "font": HEAD, "bold": True,
                                         "after": 12.5} for t in items], size=16)
    notes(s, 0.3, [
        "Problem → what I tried → result, in mission order.",
        "Slides 5–7 are the rendezvous (RPO) work: the guidance design, results, and what broke.",
    ])

    # 3 ---------------- About Me (verbatim from the Starlink deck, slide 3) ----------------
    d.section("About Me")
    s = d.content("About Me")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 6.55, 4.3, [
        B("AstroAstro MS @ Stanford"),
        B("Main GNC work centers around Navigation"),
        B("Spacecraft experience across software and hardware", u=True),
        B("Lunar GNC — Blue Origin", 1),
        B("Flight Software —Vast", 1),
        B("Integration and Test — Varda", 1),
        B("Falcon Avionics Systems — SpaceX (Hawthorne)", 1),
        B("Other roles", u=True),
        B("Avionics / Flight Software Lead (UCLA Rocket Project and Cubesat research)", 1),
        B("Relevant coursework", u=True),
        B("Optimal Control, State Estimation, Advanced Feedback Control Design, GPS, Lunar PNT, RPi "
          "Bare-metal OS", 1),
        B("Interests", u=True),
        B("Spacecraft GNC and simulations", 1),
        B("Intersection of math and applying it to spacecraft behavior", 1),
        B("Flight software and embedded systems", 1),
    ], size=14)
    picture(sh, about_photo(), 6.92, 0.61, 2.37, 4.33, alt="Photo of Colin Skinner")
    notes(s, 0.8, [
        "Stanford AeroAstro MS; my GNC work centers on navigation.",
        ("Industry: lunar GNC at Blue Origin, flight software at Vast, I&T at Varda, Falcon avionics "
        "systems at SpaceX Hawthorne."),
        "This project is the guidance and mission-design side of GNC, complementing my navigation work.",
    ])

    # 4 ---------------- Project scope ----------------
    d.section("Project")
    s = d.content("I Built the Pipeline End to End")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 9.4, 2.4, [
        B("Goal: deliver a Perseverance-class payload to a spacecraft waiting in a 300 km Mars orbit"),
        B("Launch June 2035 from Cape Canaveral, capture on Christmas Day, dock ~42 h later", 1),
        B("Track every burn's Δv and size the mission against a real launch vehicle", 1),
        B("Built from scratch: **cesium**, my Python astrodynamics library"),
        B("Lambert and porkchop search, B-plane targeting, departure and arrival planners, J2 "
          "propagation, RPO guidance (HCW), plotting", 1),
        B("**272 unit tests**; one command regenerates every archive and figure", 1),
        B("Six notebooks run in order; each stage writes a mission archive the next stage reads"),
    ], size=14)
    stages = ["Interplanetary", "Departure", "Ground segment", "Arrival", "RPO", "Summary"]
    subs = ["porkchop, Lambert", "escape, midcourse", "station coverage", "B-plane capture",
            "J2-aware guidance", "Δv budget"]
    bw, gap, y0 = 1.34, 0.272, 3.55
    for i, (name, sub) in enumerate(zip(stages, subs)):
        bx = 0.5 + i * (bw + gap)
        rpo = name == "RPO"
        box(sh, bx, y0, bw, 0.7, fill_color=PINK if rpo else GREEN,
            text=[{"t": name, "bold": rpo, "size": 13}], name=f"Stage {name}")
        tx(sh, bx - 0.1, y0 + 0.8, bw + 0.2, 0.25, sub, size=11, color=SOFT, align="c")
        if i < len(stages) - 1:
            arrow(sh, bx + bw + 0.03, y0 + 0.35, bx + bw + gap - 0.03, y0 + 0.35)
    tx(sh, 0.5, y0 + 1.25, 9.0, 0.25, "Each arrow is a mission archive file: one stage's output is the "
       "next stage's input", size=11, color=SOFT, align="c")
    notes(s, 1.0, [
        "Everything here is my own: the cesium library, the six-stage notebook pipeline, and the tests.",
        ("cesium covers Lambert, porkchop grids, B-plane targeting, departure/arrival planners, the RPO "
         "timeline with J2 propagation, HCW guidance, and plotting. 272 unit tests in the library."),
        ("Pipeline: interplanetary → departure → ground segment → arrival → RPO → summary. Each stage writes "
         "a trajectories/<stage>-<mission>.mission archive; one command regenerates every archive and "
         "figure in ~2 minutes."),
        ("Dates (TDB): escape 2035-06-14 22:04, capture 2035-12-25 00:00, docking 2035-12-26 17:36 "
         "(nominal) / 18:57 (dispersed). Cruise 193.1 days escape to capture; Lambert TOF 193.5 days."),
        "Light remark: launch on my birthday, arrival on Christmas Day.",
        "Keep this one short; it establishes scope and ownership.",
    ])

    # 5 ---------------- Mission design: the biggest lever ----------------
    d.section("Mission design")
    s = d.content("Matching the Target Plane Paid Most")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 9.4, 1.6, [
        B("Problem: direct injection rotates the plane at periapsis, the costliest place to do it"),
        B("First design (target RAAN 15°): plane change 49.7°, capture **5.52 km/s**", 1),
        B("Decision: move the target node to match the arrival (RAAN 125°, same 45° inclination)"),
        B("Plane change 6.8°, capture **2.27 km/s**: **1.3 → 6.6 rovers** per Falcon Heavy", 1),
        B("Same idea at departure: timing launch onto the Lambert arc cut midcourse **564 → 132 m/s**"),
    ], size=14)
    fig(sh, "figures/summary/capture_trade.png", 0.4, 2.42, 9.2, 2.78, crop_top=0.1,
        alt="Useful payload versus capture delta-v with coplanar, current and earlier design points")
    tx(sh, 0.4, 5.24, 8.0, 0.2, "Payload: napkin rocket-equation estimate (assumed 320 s Isp, 10% tankage)",
       size=9, color=SOFT)
    notes(s, 1.3, [
        "Takeaway 1: choosing the target plane to match the arrival geometry is the single biggest lever.",
        ("Direct injection does the plane change at periapsis, so β (asymptote vs target plane) matters "
         "enormously. Earlier: RAAN 15°, β 49.7°, capture 5,524 m/s, a 3.35 km/s penalty."),
        ("Current: RAAN 125°, β 6.8°, capture 2,268.5 m/s, only 93 m/s above the 2,175 m/s coplanar "
         "ideal. Aim point |B| = 7,312 km for the 300 km periapsis."),
        ("Payload per Falcon Heavy (16.8 t to Mars injection): 1.30 t (1.3 Perseverance rovers) → 6.73 t "
         "(6.6 rovers). Napkin numbers: assumed 320 s Isp, 10% tankage, 1,025 kg rover."),
        ("Departure: Lambert is heliocentric and ignores Earth's well; launching 0.42 day after the "
         "Lambert reference cut the midcourse from 564.2 to 131.8 m/s."),
        ("If the target plane were fixed, the fallback is a high-ellipse capture with the plane change "
         "near apoapsis (future work)."),
        ("Cut from slides, if asked: porkchop 579 × 1029-day grid, 441 feasible transfers; C3 10.62 km²/s², "
         "DLA 4.06°, RLA 4.79°; arrival v∞ 2.820 km/s, asymptote DEC 8.90°."),
    ])

    # 6 ---------------- RPO guidance architecture ----------------
    d.section("Rendezvous")
    s = d.content("Closed-Loop Guidance in Five Stages")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 9.4, 1.0, [
        B("Problem: dock autonomously (8-minute one-way light time) under Mars J2 perturbations"),
        B("Impulsive guidance flown on a two-body + J2 truth model; every coast continues from the last "
          "state, and every burn is planned from the current osculating elements", 1),
    ], size=14)
    flow = [("1  Plane change", "at the node; J2 pre-bias if it is cheaper"),
            ("2  Hohmann", "to the target radius"),
            ("3  Phase entry", "merged with the circularization"),
            ("4  Closed-loop phasing", "20 revs, re-planned every rev"),
            ("5  Exit + HCW", "two-impulse terminal, with a retarget")]
    bw, gap, y0 = 1.64, 0.25, 2.06
    for i, (name, sub) in enumerate(flow):
        bx = 0.5 + i * (bw + gap)
        key = i in (0, 3)
        box(sh, bx, y0, bw, 0.6, fill_color=PINK if key else GREEN,
            text=[{"t": name, "bold": True, "size": 12}], name=f"RPO stage {i + 1}")
        tx(sh, bx, y0 + 0.68, bw, 0.5, sub, size=11, color=SOFT, align="c")
        if i < len(flow) - 1:
            arrow(sh, bx + bw + 0.03, y0 + 0.3, bx + bw + gap - 0.03, y0 + 0.3)
    tx(sh, 0.3, 3.4, 9.4, 1.7, [
        B("Two cases from the arrival stage's post-capture state"),
        B("Nominal: exact capture, so the chaser starts coplanar with the target", 1),
        B("Dispersed: one seeded draw of capture errors (±100 km in a, e ≤ 0.01, ±3° in angles), "
          "not a Monte Carlo", 1),
        B("Design choice: the planner prices both plane-change options and flies the cheaper one"),
    ], size=14)
    notes(s, 1.4, [
        ("Inputs come straight from the arrival archive: capture epoch, post-capture state, target orbit "
         "300 km / 45° / RAAN 125°; target at 13° true anomaly at capture; phase gap 107.9°."),
        ("Five stages: plane change at the node (optionally pre-biased against J2 nodal drift), Hohmann to "
         "the target radius, phase entry merged with the circularization, 20 revolutions of closed-loop "
         "phasing, phase exit + two-impulse HCW terminal rendezvous with retargeting."),
        ("Why closed loop: every coast is propagated with J2 from the last true state; every burn is "
         "planned from current osculating elements, so model error does not accumulate open-loop."),
        "Settings: HCW handoff threshold 20 km, HCW transfer 40 min, docking threshold 50 m.",
        ("Dispersed draw (fixed, reproducible seed, one example): a +73 km, e 0.0086, i +1.87°, "
         "RAAN −1.43°, aop −2.5°. No navigation errors or finite burns are modeled."),
        ("Per-stage Δv (nominal / dispersed, m/s): plane change 0 / 120.5; Hohmann + entry 16.8 / 24.9; "
         "phasing 9.7 / 9.8; phase exit 18.3 / 15.8; HCW 25.1 / 14.9. Burns: 25 / 27."),
    ])

    # 7 ---------------- RPO results ----------------
    s = d.content("Both Cases Dock Within 7–13 m")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 4.75, 1.5, [
        B("Outcome"),
        B("Nominal: **13.5 m** miss after 41.6 h", 1),
        B("Dispersed: **7.3 m** miss after 43.0 h", 1),
        B("HCW closes the last ~18 km in 1.3 h", 1),
    ], size=14)
    tx(sh, 5.1, 0.84, 4.6, 1.5, [
        B("Honest limits"),
        B("Dispersed handoff at **18.2 km**, outside the 15 km linear region", 1),
        B("Both cross the **200 m keep-out**", 1),
    ], size=14)
    fig(sh, "figures/summary/rpo_log_range.png", 0.4, 2.3, 9.2, 2.9, crop_top=0.1,
        alt="Chaser-target range on a log scale for both cases with 15 km and 200 m reference lines")
    notes(s, 1.3, [
        "Range on a log scale, both cases; dotted lines at the 15 km HCW linear region and the 200 m keep-out.",
        ("Nominal: 69.8 m/s total, 25 burns, 41.6 h, 13.5 m miss. Dispersed: 186.0 m/s, 27 burns, 43.0 h, "
         "7.3 m miss."),
        "Phasing is ~35 of the ~43 RPO hours, closing thousands of km slowly; HCW closes the last ~18 km in about 1.3 h.",
        ("Be upfront: dispersed HCW starts at 18.2 km (outside the 15 km linear region), retargets at "
         "290 m, stops at 7.3 m. Both paths cross the 200 m keep-out because HCW aims straight at the "
         "target; a hold point is the fix (slide 10)."),
        ("If asked for the frame view: rsw_terminal.png shows the dispersed terminal approach in RSW, "
         "along-track horizontal (+V-bar left), radial up, dotted 15 km circle."),
    ])

    # 8 ---------------- What broke ----------------
    s = d.content("Closed Loop + J2 Pre-Bias Fixed It")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 5.4, 3.3, [
        B("Problem: open-loop phasing under J2 missed the HCW handoff by **51.5 km**"),
        B("Tried: re-plan the phasing burn every rev from the actual state → **12.2 km**"),
        B("Residual was mostly cross-track (~9.7 km)", 1),
        {"t": [(("Cause: J2 differential nodal regression; the phasing orbit sits ~16 km higher, so its "
                "node drifts more slowly (dΩ/dt ~ a"), {}), ("−7/2", {"sup": True}), (")", {})],
         "lvl": 1},
        B("Fix: pre-bias the plane change by the predicted drift → **5.5 km**"),
        B("Dropped: Hohmann shape corrections (~2% saving, many extra burns)"),
    ], size=14)
    tx(sh, 6.0, 0.86, 3.7, 0.25, [[("Phasing handoff miss", {"bold": True})]], size=13)
    tx(sh, 6.0, 1.1, 3.7, 0.2, "development history, earlier test geometry", size=10, color=SOFT)
    bar_chart(sh, 5.9, 1.32, 3.85, 2.3, ["Open loop", "Closed loop", "+ J2 pre-bias"],
              [51.5, 12.2, 5.5], '0.0" km"', max_v=80)
    box(sh, 0.5, 4.25, 9.0, 0.85, fill_color=PINK, margin=0.12, anchor="m", text=[
        {"t": [("Current mission: the pre-bias is conditional. ", {"bold": True}),
               (("It saves 50.4 m/s in the dispersed case but would cost 31.6 m/s in the nominal case, "
                "so the planner prices both and flies the cheaper plan."), {})],
         "size": 14, "align": "l"}], name="Conditional pre-bias")
    notes(s, 1.4, [
        "Takeaway 2: the guidance had to be closed-loop and J2-aware to work.",
        ("Development history on an earlier test geometry (target RAAN 15°, synthetic chaser offset); do "
         "not compare with the current 69.8 / 186.0 m/s results."),
        ("Open-loop incremental phasing: 51.5 km handoff miss, 202.1 m/s. Closed loop (vis-viva burn from "
         "the actual state each rev): 12.2 km, 164.9 m/s. + J2 RAAN pre-bias: 5.5 km, 152.2 m/s."),
        ("Root cause: Ω̇ ∝ a^(−7/2). The phasing orbit is ~16 km above the target, so its node regresses "
         "more slowly; the residual was ≈9.7 km cross-track."),
        ("Hohmann shape corrections during phasing: ~2% saving for 3–18 extra burns and 6–30 h, and "
         "sometimes worse, so dropped."),
        ("Current mission: pre-bias saves 50.4 m/s dispersed (236.4 → 186.0 m/s) but would cost 31.6 m/s "
         "nominal, so it is chosen per case, not always on. RAAN drift during phasing, measured vs "
         "analytic: 0.514° / 0.438° nominal, 0.527° / 0.399° dispersed (first-order formula covers only "
         "the phasing leg)."),
    ])

    # 9 ---------------- Validation + Δv budget ----------------
    d.section("Validation")
    s = d.content("Validated: Spacecraft Needs ~2.5 km/s")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 4.75, 1.5, [
        B("Validation against theory"),
        B("B-plane periapsis radius error **0 m**", 1),
        B("Escape and Lambert arcs re-propagated", 1),
        B("272 unit tests in cesium", 1),
    ], size=14)
    tx(sh, 5.1, 0.84, 4.6, 1.5, [
        B("Spacecraft Δv budget"),
        B("**2.47–2.59 km/s** (nominal–dispersed)", 1),
        B("Mars capture is 92% of it", 1),
        B("Launcher: 11.47 km/s (ideal)", 1),
    ], size=14)
    dv_chart(sh, 0.4, 2.3, 9.2, 2.75)
    tx(sh, 0.4, 5.1, 8.0, 0.2, [[("■ ", {"color": NAVY}), ("spacecraft     ", {}),
                                 ("■ ", {"color": LV}), ("launch vehicle (ideal, no losses)", {})]],
       size=10, color=SOFT)
    notes(s, 1.0, [
        ("Validation: arrival asymptote error 0.0000°, periapsis radius error 0 m, post-burn e ≈ 3×10⁻¹⁶, "
         "i and RAAN exact. Escape: propagated impact parameter matches theory to ~1 cm, C3 to "
         "3.6×10⁻⁸ km²/s². Lambert re-propagation within 594 m over the transfer."),
        ("Spacecraft: midcourse 131.8 + capture 2,268.5 + RPO 69.8 / 186.0 = 2,470.2 / 2,586.3 m/s. "
         "Capture is 92% of it."),
        ("Launch vehicle: ideal ascent 7,790.7 + escape 3,675.6 = 11,466.3 m/s. Ideal lower bounds, not "
         "real launch costs (no drag or gravity losses; real vehicles add ~1.5–2 km/s)."),
        ("Not shown: Mars surface ↔ orbit, 3,531.5 m/s ideal, a reference for a future lander and not in "
         "the totals."),
        "Takeaway 3: RPO is small in Δv but it is where the precision engineering lives (7–13 m miss).",
    ])

    # 10 ---------------- Limitations + next steps ----------------
    d.section("Next steps")
    s = d.content("Next: a Hold Point and Relative Nav")
    sh = s.shapes
    tx(sh, 0.3, 0.84, 4.6, 4.3, [
        B("Limitations"),
        B("Patched conics, impulsive burns, no navigation errors", 1),
        B("Mars J2 only in the rendezvous", 1),
        B("One seeded dispersion, not a Monte Carlo", 1),
        B("HCW handoff outside the linear region; path crosses the keep-out", 1),
        B("Ideal ascent Δv; launch azimuth not checked against Cape range safety", 1),
    ], size=14)
    tx(sh, 5.1, 0.84, 4.6, 2.2, [
        B("What I would build next"),
        B("Hand off inside 15 km, hold at the keep-out boundary, then a slow V-bar or R-bar approach", 1),
        B("Relative navigation (sensors + filter) and closed-loop control such as MPC, with abort logic", 1),
        B("Monte Carlo on every segment", 1),
    ], size=14)
    # Proposed docking concept: V-bar approach with a hold point on the keep-out boundary
    tx(sh, 5.2, 3.32, 3.0, 0.22, [[("Proposed docking concept", {"bold": True})],
                                  ], size=11)
    tx(sh, 5.2, 3.53, 3.0, 0.2, "not implemented", size=10, color=SOFT)
    cx, cy, r = 8.85, 4.42, 0.5
    box(sh, cx - r, cy - r, 2 * r, 2 * r, fill_color=None, line=EDGE, lw=0.75, shape="oval",
        dash=MSO_LINE_DASH_STYLE.DASH, name="Keep-out sphere")
    box(sh, cx - 0.1, cy - 0.1, 0.2, 0.2, fill_color=PINK, name="Target")
    box(sh, cx - r - 0.07, cy - 0.07, 0.14, 0.14, fill_color=GREEN, shape="oval", name="Hold point")
    arrow(sh, 5.3, cy, cx - r - 0.1, cy, dash=MSO_LINE_DASH_STYLE.DASH)
    arrow(sh, cx - r + 0.08, cy, cx - 0.13, cy, color=NAVY, w=1.5)
    tx(sh, 5.3, cy - 0.27, 2.8, 0.2, "HCW handoff inside 15 km", size=10, color=SOFT)
    tx(sh, 5.3, cy + 0.08, 2.95, 0.2, "hold point", size=10, color=SOFT, align="r")
    tx(sh, cx - 0.8, cy - r - 0.24, 1.6, 0.2, "200 m keep-out", size=10, color=SOFT, align="c")
    tx(sh, cx - 0.4, cy + 0.13, 0.8, 0.2, "target", size=10, color=SOFT, align="c")
    notes(s, 0.9, [
        "Be upfront: these are the questions a reviewer would ask next.",
        ("Limitations: patched conics and two-body arcs, impulsive burns, analytic ephemeris, Mars J2 only "
         "in RPO, no navigation errors, no finite-burn or attitude effects; one seeded dispersion case."),
        ("RPO open items: dispersed HCW handoff at 18.2 km, outside the 15 km linear region; HCW crosses "
         "the 200 m keep-out."),
        ("Proposed docking (not done): hand off to HCW inside the linear region; hold at the 200 m "
         "keep-out boundary; slow V-bar or R-bar final approach; relative navigation (sensors + filter); "
         "closed-loop control such as MPC; contact/capture and abort logic."),
        ("Also next: Monte Carlo on every segment, high-ellipse capture for fixed target planes, "
         "atmospheric ascent/descent models, ground link budget, range-safety check of the 131.8° launch "
         "azimuth."),
    ])

    # 11 ---------------- Questions ----------------
    d.section("Questions")
    s = d.dark("Questions?", (0.5, 2.1, 9.0, 0.9), 32, bold=False, align="c", anchor="m",
               number=True)
    notes(s, 0.3, [
        ("Recap in one breath: pick the target plane (5.52 → 2.27 km/s capture); closed-loop, J2-aware "
         "guidance (7–13 m miss); ~2.5 km/s of spacecraft Δv."),
        "Likely questions: keep-out crossing, linear-region handoff, Monte Carlo, nav filter, MPC.",
    ])
    return d


# ---------------------------------------------------------------------------------------------
# Package finishing and PDF export
# ---------------------------------------------------------------------------------------------
def set_theme_fonts(prs):
    theme_part = prs.slide_master.part.part_related_by(RT.THEME)
    root = etree.fromstring(theme_part.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    root.find(".//a:majorFont/a:latin", ns).set("typeface", HEAD)
    root.find(".//a:minorFont/a:latin", ns).set("typeface", BODY)
    theme_part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _arc(path):
    return "Fonts/" + Path(path).name.replace(" ", "_")


def _embed_fonts_in_odp(src, dst):
    """Embed the deck fonts in the ODP so a sandboxed LibreOffice renders (and embeds) them."""
    def face(name, files):
        uris = "".join(
            f'<svg:font-face-uri xlink:href="{_arc(f)}" xlink:type="simple">'
            f'<svg:font-face-format svg:string="truetype"/></svg:font-face-uri>' for f in files)
        fam = f"&apos;{name}&apos;" if " " in name else name
        return (f'<style:font-face style:name="{name}" svg:font-family="{fam}">'
                f"<svg:font-face-src>{uris}</svg:font-face-src></style:font-face>")

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in ("content.xml", "styles.xml"):
                xml = data.decode()
                for name, files in FONT_FILES.items():
                    pat = r'<style:font-face style:name="' + re.escape(name) + r'"[^>]*/>'
                    xml = re.sub(pat, face(name, files), xml)
                data = xml.encode()
            elif item.filename == "META-INF/manifest.xml":
                entries = "".join(
                    f' <manifest:file-entry manifest:full-path="{_arc(f)}" '
                    f'manifest:media-type="application/x-font-ttf"/>\n'
                    for files in FONT_FILES.values() for f in files)
                data = data.replace(b"</manifest:manifest>",
                                    entries.encode() + b"</manifest:manifest>")
            comp = zipfile.ZIP_STORED if item.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(item, data, compress_type=comp)
        for files in FONT_FILES.values():
            for f in files:
                zout.write(f, _arc(f), compress_type=zipfile.ZIP_DEFLATED)


def export_pdf(pptx):
    """pptx -> odp (soffice) -> embed fonts -> pdf (soffice --headless --convert-to pdf)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        conv = ["soffice", "--headless", "--convert-to"]
        subprocess.run(conv + ["odp", "--outdir", str(tmp), str(pptx)], check=True,
                       capture_output=True)
        odp = tmp / (pptx.stem + ".odp")
        embedded = tmp / "embedded" / odp.name
        embedded.parent.mkdir()
        _embed_fonts_in_odp(odp, embedded)
        subprocess.run(conv + ["pdf", "--outdir", str(tmp), str(embedded)], check=True,
                       capture_output=True)
        out = pptx.with_suffix(".pdf")
        shutil.move(tmp / (pptx.stem + ".pdf"), out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def main():
    BUILD["dir"] = Path(tempfile.mkdtemp(prefix="mars_interview_"))
    try:
        d = build()
        prs = d.prs
        set_theme_fonts(prs)
        add_sections(prs, d.sections)
        cp = prs.core_properties
        cp.title = "Mars Courier Mission Design: SpaceX GNC Onsite Interview"
        cp.author = "Colin Skinner"
        prs.save(OUT)
        print(f"wrote {OUT} ({len(prs.slides)} slides, ~{CLOCK['t']:.1f} min of notes timing)")
    finally:
        shutil.rmtree(BUILD["dir"], ignore_errors=True)
    if "--no-pdf" not in sys.argv:
        export_pdf(OUT)


if __name__ == "__main__":
    main()
