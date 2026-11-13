"""Build the Mars Courier presentation (.pptx) from the content in this script.

Usage (from the repo root):
    venv/bin/python projects/mars-courier/presentation/build_deck.py            # .pptx + .pdf
    venv/bin/python projects/mars-courier/presentation/build_deck.py --no-pdf   # .pptx only

The PDF is exported with `soffice --headless --convert-to`, going through an intermediate ODP with
Inter and Source Sans 3 embedded, so the PDF gets the real fonts even when LibreOffice cannot see
the system font folder (fonts: `brew install --cask font-inter font-source-sans-3`).

Every number on the slides comes from PRESENTATION_BRIEF.md; deck decisions come from
DECK_DECISIONS.md. Figures are inserted at their native aspect ratio (never stretched).
Click-build animations are injected as <p:timing> XML; a PDF export shows every slide fully built.
"""

import math
import uuid
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # projects/mars-courier
OUT = HERE / "mars_courier.pptx"

# ---------------------------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------------------------
HEAD = "Inter"
BODY = "Source Sans 3"

BG = "FCFCFB"
INK = "1D2228"
INK2 = "3B4350"
MUTED = "5A6270"
LINE = "D5D9DF"
CARD = "F0F1F3"
WHITE = "FFFFFF"
BLUE = "2A78D6"
BLUE_D = "1B4F91"
BLUE_L = "8DB8EC"
ORANGE = "EB6834"
AQUA = "1BAF7A"
MARS = "C1440E"
MARS_L = "E06A3B"  # lighter Mars red for text on dark backgrounds
DARK = "111418"
DARK2 = "2A2F38"
BLUE_T = "E7F0FB"
ORANGE_T = "FDEEE6"
AQUA_T = "E2F5EE"
MARS_T = "F7E6DE"
SUN = "E8A33D"

SW, SH = 13.333, 7.5
SECTIONS = ["Launch window", "Departure", "Cruise & comms", "Arrival", "Rendezvous", "Mission summary"]
STAGES = ["Plane change", "Hohmann", "Phase entry", "Phasing", "Exit + HCW"]

ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}


def E(inches):
    return Emu(round(inches * 914400))


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


# ---------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------
def md_runs(s):
    """'plain **bold** plain' -> [(text, {"bold": bool})]."""
    parts = s.split("**")
    return [(t, {"bold": i % 2 == 1}) for i, t in enumerate(parts) if t]


def _set_bullet(p, color, indent=0.24, level=0, char="•"):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(E(indent * (level + 1)))))
    pPr.set("indent", str(-int(E(indent))))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", BODY)
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def fill_tf(tf, paras, size=16, color=INK, font=BODY, bold=False, align="l", after=4, italic=False,
            bullet_color=None, line=None):
    """Fill a text frame. paras: str | list of (str | list-of-runs | dict)."""
    if isinstance(paras, str):
        paras = [paras]
    first = True
    for item in paras:
        spec = item if isinstance(item, dict) else {"t": item}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = ALIGN[spec.get("align", align)]
        p.space_after = Pt(spec.get("after", after))
        if spec.get("before"):
            p.space_before = Pt(spec["before"])
        if line or spec.get("line"):
            p.line_spacing = spec.get("line", line)
        content = spec.get("t", "")
        runs = md_runs(content) if isinstance(content, str) else content
        for text, opts in runs:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = opts.get("font", spec.get("font", font))
            f.size = Pt(opts.get("size", spec.get("size", size)))
            f.bold = opts.get("bold", False) or spec.get("bold", bold)
            f.italic = opts.get("italic", spec.get("italic", italic))
            f.color.rgb = rgb(opts.get("color", spec.get("color", color)))
            if opts.get("baseline"):
                r._r.get_or_add_rPr().set("baseline", str(opts["baseline"]))
        if spec.get("bullet"):
            _set_bullet(p, spec.get("bcolor", bullet_color or MARS), level=spec.get("level", 0))
    return tf


def tx(sh, x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, align="l", anchor="t",
       after=4, italic=False, margin=0.0, name=None, bullet_color=None, line=None):
    box = sh.add_textbox(E(x), E(y), E(w), E(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = ANCHOR[anchor]
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", E(margin))
    fill_tf(tf, paras, size=size, color=color, font=font, bold=bold, align=align, after=after,
            italic=italic, bullet_color=bullet_color, line=line)
    if name:
        box.name = name
    return box


def bullets(items, size=16, color=INK, bcolor=MARS, after=8):
    return [{"t": s, "bullet": True, "size": size, "color": color, "bcolor": bcolor, "after": after}
            for s in items]


# ---------------------------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------------------------
def box(sh, x, y, w, h, fill=CARD, line=None, lw=1.0, shape="rect", radius=0.08, dash=None,
        text=None, name=None, **tkw):
    kind = {"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE,
            "oval": MSO_SHAPE.OVAL, "chevron": MSO_SHAPE.CHEVRON, "pent": MSO_SHAPE.PENTAGON}[shape]
    s = sh.add_shape(kind, E(x), E(y), E(w), E(h))
    if shape == "round":
        s.adjustments[0] = min(0.5, radius / min(w, h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
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
    m = tkw.pop("margin", 0.1)
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", E(m))
    tf.vertical_anchor = ANCHOR[tkw.pop("anchor", "m")]
    if text is not None:
        fill_tf(tf, text, **tkw)
    if name:
        s.name = name
    return s


def _strip_style(shape):
    """Drop the theme style reference (no inherited shadows); fill and line are explicit."""
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def _ln_end(ln, head=True, tail=False):
    if tail:
        e = etree.SubElement(ln, qn("a:headEnd"))
        e.set("type", "triangle")
        e.set("w", "med")
        e.set("len", "med")
    if head:
        e = etree.SubElement(ln, qn("a:tailEnd"))
        e.set("type", "triangle")
        e.set("w", "med")
        e.set("len", "med")


def arrow(sh, x1, y1, x2, y2, color=MUTED, w=1.75, head=True, tail=False, dash=None):
    return poly(sh, [(x1, y1), (x2, y2)], color=color, w=w, dash=dash, head=head, tail=tail)


def poly(sh, pts, color=BLUE, w=2.0, dash=None, closed=False, fill=None, head=False, tail=False):
    """Polyline/polygon through points given in inches."""
    b = sh.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
    b.add_line_segments([(E(x), E(y)) for x, y in pts[1:]], close=closed)
    s = b.convert_to_shape()
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    if color is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = rgb(color)
        s.line.width = Pt(w)
        if dash:
            s.line.dash_style = dash
        if head or tail:
            _ln_end(s.line._get_or_add_ln(), head=head, tail=tail)
    _strip_style(s)
    return s


def circle(sh, cx, cy, r, fill=None, line=None, lw=1.25, dash=None):
    return box(sh, cx - r, cy - r, 2 * r, 2 * r, fill=fill, line=line, lw=lw, shape="oval", dash=dash)


def fig(sh, rel, x, y, w, h, ha="c", va="t", alt=None, alpha=None):
    """Insert a figure at native aspect ratio, fitted inside the (x, y, w, h) box."""
    path = ROOT / rel
    iw, ih = Image.open(path).size
    r = iw / ih
    fw, fh = (h * r, h) if w / h > r else (w, w / r)
    fx = x + {"l": 0, "c": (w - fw) / 2, "r": w - fw}[ha]
    fy = y + {"t": 0, "m": (h - fh) / 2, "b": h - fh}[va]
    pic = sh.add_picture(str(path), E(fx), E(fy), E(fw), E(fh))
    pic._element.nvPicPr.cNvPr.set("descr", alt or rel)
    if alpha is not None:
        blip = pic._element.blipFill.find(qn("a:blip"))
        etree.SubElement(blip, qn("a:alphaModFix")).set("amt", str(int(alpha * 100000)))
    return pic, (fx, fy, fw, fh)


def group(slide):
    return slide.shapes.add_group_shape()


# ---------------------------------------------------------------------------------------------
# Slide furniture
# ---------------------------------------------------------------------------------------------
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = E(SW)
        self.prs.slide_height = E(SH)
        self.layout_title = self.prs.slide_layouts[5]  # "Title Only"
        self.sections = []  # (name, [slide ids])
        self.main_count = 0

    def section(self, name):
        self.sections.append((name, []))

    def new(self, title, dark=False, size=30, title_box=(0.6, 0.48, 12.1, 0.62), color=None,
            font=HEAD, bold=True, align="l", anchor="t"):
        s = self.prs.slides.add_slide(self.layout_title)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(DARK if dark else BG)
        t = s.shapes.title
        x, y, w, h = title_box
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
        r = p.add_run()
        r.text = title
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = rgb(color or (WHITE if dark else INK))
        self.sections[-1][1].append(s.slide_id)
        return s


def eyebrow(s, text, color=MARS):
    tx(s.shapes, 0.6, 0.2, 8.0, 0.26, [[(text.upper(), {"bold": True})]], size=11, color=color,
       font=HEAD, name="Eyebrow")


def headline(s, text, y=1.08, size=18):
    tx(s.shapes, 0.6, y, 12.1, 0.45, text, size=size, color=INK2, name="Headline")


def slide_number(sh, x, y, w, h, color, size=11):
    tb = sh.add_textbox(E(x), E(y), E(w), E(h))
    tf = tb.text_frame
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, qn("a:fld"))
    fld.set("id", "{" + str(uuid.uuid4()).upper() + "}")
    fld.set("type", "slidenum")
    rpr = etree.SubElement(fld, qn("a:rPr"))
    rpr.set("lang", "en-US")
    rpr.set("sz", str(size * 100))
    sf = etree.SubElement(rpr, qn("a:solidFill"))
    etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
    etree.SubElement(rpr, qn("a:latin")).set("typeface", BODY)
    etree.SubElement(fld, qn("a:t")).text = "‹#›"
    tb.name = "Slide number"
    return tb


def tracker(s, current, dark=False, y=7.03):
    """Mission-phase tracker: six chips, the current section highlighted."""
    g = s.shapes.add_group_shape()
    x = 0.6
    for i, name in enumerate(SECTIONS):
        active = i == current
        if dark:
            fill, col = (MARS, WHITE) if active else (DARK2, "A9B0BB")
        else:
            fill, col = (MARS, WHITE) if active else ("ECEEF1", "5B6270")
        box(g.shapes, x, y, 1.5, 0.27, fill=fill, shape="round", radius=0.13, margin=0.02,
            text=[[(name, {"bold": active})]], size=10, color=col, font=HEAD, align="c")
        x += 1.56
    g.name = "Phase tracker"
    return g


def footer(s, section=None, dark=False, backup=False):
    if backup:
        box(s.shapes, 0.6, 7.03, 1.1, 0.27, fill="ECEEF1", shape="round", radius=0.13, margin=0.02,
            text=[[("Backup", {"bold": True})]], size=10, color="5B6270", font=HEAD, align="c")
    elif section is not None:
        tracker(s, section, dark=dark)
    slide_number(s.shapes, 11.9, 7.03, 0.83, 0.27, "A9B0BB" if dark else MUTED)


def notes(s, minutes, points):
    head = f"Timing: ~{minutes} min" if minutes else "Backup slide"
    s.notes_slide.notes_text_frame.text = "\n".join([head] + ["• " + p for p in points])


def stage_stepper(s, current):
    """Small five-stage stepper at top right for the RPO stage slides."""
    x = 12.73 - 5 * 1.28
    for i, name in enumerate(STAGES):
        active = i in current
        box(s.shapes, x, 0.16, 1.24, 0.26, fill=BLUE if active else "ECEEF1", shape="round",
            radius=0.13, margin=0.01, text=[[(f"{i + 1} {name}", {"bold": active})]], size=10,
            color=WHITE if active else "5B6270", font=HEAD, align="c")
        x += 1.28


def tag(sh, x, y, text, fill=MARS_T, color=MARS, w=None):
    w = w or (0.2 + 0.085 * len(text))
    return box(sh, x, y, w, 0.3, fill=fill, shape="round", radius=0.15, margin=0.04,
               text=[[(text.upper(), {"bold": True})]], size=10, color=color, font=HEAD, align="c")


def card(sh, x, y, w, h, title, body, fill=CARD, tcolor=INK, size=15, tsize=16, line=None,
         badge=None, badge_color=MARS):
    g_box = box(sh, x, y, w, h, fill=fill, line=line, shape="round", radius=0.1)
    tx0 = x + 0.2
    if badge is not None:
        circle(sh, x + 0.38, y + 0.38, 0.2, fill=badge_color)
        tx(sh, x + 0.18, y + 0.18, 0.4, 0.4, [[(str(badge), {"bold": True})]], size=14, color=WHITE,
           font=HEAD, align="c", anchor="m")
        tx0 = x + 0.7
    tx(sh, tx0, y + 0.2, x + w - tx0 - 0.15, 0.4, [[(title, {"bold": True})]], size=tsize,
       color=tcolor, font=HEAD)
    tx(sh, x + 0.2, y + 0.66, w - 0.4, h - 0.8, body, size=size, color=INK2, after=6)
    return g_box


def stat(sh, x, y, w, value, label, vcolor=INK, vsize=28, lsize=12, h=1.1, align="l"):
    tx(sh, x, y, w, 0.55, [[(value, {"bold": True})]], size=vsize, color=vcolor, font=HEAD, align=align)
    tx(sh, x, y + 0.06 + vsize / 60, w, h - 0.5, label, size=lsize, color=MUTED, align=align)


def simple_table(sh, x, y, col_w, rows, size=14, row_h=0.38, head_fill=INK, head_color=WHITE,
                 zebra=("FFFFFF", "F3F4F6"), bold_first_col=False, aligns=None, highlight=None):
    nrows, ncols = len(rows), len(rows[0])
    gf = sh.add_table(nrows, ncols, E(x), E(y), E(sum(col_w)), E(row_h * nrows))
    tbl = gf.table
    tbl_pr = tbl._tbl.tblPr
    tbl_pr.set("firstRow", "0")
    tbl_pr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = E(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = E(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            if i == 0 and head_fill:
                cell.fill.fore_color.rgb = rgb(head_fill)
            elif highlight and i in highlight:
                cell.fill.fore_color.rgb = rgb(highlight[i])
            else:
                cell.fill.fore_color.rgb = rgb(zebra[(i + 1) % 2])
            cell.margin_left = cell.margin_right = E(0.08)
            cell.margin_top = cell.margin_bottom = E(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            is_head = i == 0 and head_fill
            al = (aligns[j] if aligns else "l")
            fill_tf(tf, str(val), size=size, color=head_color if is_head else INK,
                    font=HEAD if is_head else BODY, bold=bool(is_head) or (bold_first_col and j == 0),
                    align=al, after=0)
    return gf


# ---------------------------------------------------------------------------------------------
# Animations: click-builds via <p:timing>
# ---------------------------------------------------------------------------------------------
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def add_builds(slide, steps):
    """steps: list of click steps; each step is a list of shapes that fade in together."""
    counter = [2]

    def nid():
        counter[0] += 1
        return counter[0]

    click_pars = []
    for step in steps:
        o1, o2 = nid(), nid()
        effects = []
        for k, shp in enumerate(step):
            a, b, c = nid(), nid(), nid()
            node = "clickEffect" if k == 0 else "withEffect"
            sid = shp.shape_id
            effects.append(
                f'<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" '
                f'fill="hold" grpId="0" nodeType="{node}"><p:stCondLst><p:cond delay="0"/>'
                f'</p:stCondLst><p:childTnLst><p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold">'
                f'<p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl>'
                f'<p:spTgt spid="{sid}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility'
                f'</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to>'
                f'</p:set><p:animEffect transition="in" filter="fade"><p:cBhvr>'
                f'<p:cTn id="{c}" dur="400"/><p:tgtEl><p:spTgt spid="{sid}"/></p:tgtEl></p:cBhvr>'
                f'</p:animEffect></p:childTnLst></p:cTn></p:par>'
            )
        click_pars.append(
            f'<p:par><p:cTn id="{o1}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
            f'</p:stCondLst><p:childTnLst><p:par><p:cTn id="{o2}" fill="hold"><p:stCondLst>'
            f'<p:cond delay="0"/></p:stCondLst><p:childTnLst>{"".join(effects)}</p:childTnLst>'
            f'</p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
        )
    bld = "".join(
        f'<p:bldP spid="{shp.shape_id}" grpId="0" animBg="1"/>'
        for step in steps for shp in step if shp._element.tag == qn("p:sp")
    )
    xml = (
        f'<p:timing xmlns:p="{P_NS}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" '
        f'restart="never" nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek">'
        f'<p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>{"".join(click_pars)}'
        f'</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl>'
        f'<p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" '
        f'delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
        f'</p:childTnLst></p:cTn></p:par></p:tnLst>'
        + (f"<p:bldLst>{bld}</p:bldLst>" if bld else "")
        + "</p:timing>"
    )
    timing = etree.fromstring(xml)
    sld = slide._element
    clr = sld.find(qn("p:clrMapOvr"))
    clr.addnext(timing)


# ---------------------------------------------------------------------------------------------
# Native schematics
# ---------------------------------------------------------------------------------------------
def conic_pts(cx, cy, scale, p, e, th0, th1, rot=0.0, n=160, flip_x=False):
    """Points of r = p / (1 + e cos th), focus at (cx, cy); angles in degrees, math orientation."""
    pts = []
    for k in range(n + 1):
        th = math.radians(th0 + (th1 - th0) * k / n)
        r = p / (1 + e * math.cos(th))
        x, y = r * math.cos(th + math.radians(rot)), r * math.sin(th + math.radians(rot))
        if flip_x:
            x = -x
        pts.append((cx + scale * x, cy - scale * y))
    return pts


def lambert_schematic(sh, cx=3.35, cy=4.3, au=1.42):
    circle(sh, cx, cy, au, line=AQUA, lw=1.25, dash=MSO_LINE_DASH_STYLE.DASH)
    circle(sh, cx, cy, 1.48 * au, line=ORANGE, lw=1.25, dash=MSO_LINE_DASH_STYLE.DASH)
    e = 0.2116  # schematic conic through 1.0 AU and 1.48 AU over a 149 deg sweep
    pts = conic_pts(cx, cy, au, 1 + e, e, 0, 149, rot=-110)
    poly(sh, pts, color=BLUE, w=3.0, head=True)
    circle(sh, cx, cy, 0.17, fill=SUN)
    ex, ey = pts[0]
    mx, my = pts[-1]
    circle(sh, ex, ey, 0.11, fill=AQUA)
    circle(sh, mx, my, 0.12, fill=MARS)
    tx(sh, ex - 1.25, ey + 0.12, 2.5, 0.55, ["Earth at departure", "position r₁ at time t₁"],
       size=13, color=INK2, align="c", after=0)
    tx(sh, mx - 0.2, my - 0.75, 2.6, 0.55, ["Mars at arrival", "position r₂ at time t₂"],
       size=13, color=INK2, after=0)
    tx(sh, cx - 0.6, cy + 0.22, 1.2, 0.3, "Sun", size=12, color=MUTED, align="c")
    tx(sh, cx - 1.2, cy - 1.0, 2.1, 0.6,
       [[("Lambert arc (blue)", {"bold": True, "color": BLUE})], "flight time Δt = t₂ − t₁"],
       size=13, color=INK2, after=0)
    tx(sh, 0.6, 6.55, 5.6, 0.3, "Schematic, not to scale", size=11, color=MUTED, italic=True)


def escape_schematic(sh, cx=2.0, cy=5.35, s=0.72):
    e = 1.1779  # brief: escape hyperbola eccentricity
    circle(sh, cx, cy, s, line=MUTED, lw=1.25, dash=MSO_LINE_DASH_STYLE.DASH)
    circle(sh, cx, cy, 0.5, fill=AQUA)
    pts = conic_pts(cx, cy, s, 1 + e, e, 0, 118, flip_x=True, rot=0)
    poly(sh, pts, color=BLUE, w=3.0)
    x1, y1 = pts[-1]
    th = math.radians(148.1)  # asymptote true anomaly
    dx, dy = -math.cos(th), -math.sin(th)
    arrow(sh, x1, y1, x1 + 1.25 * dx, y1 + 1.25 * dy, color=MARS, w=3.0)
    tx(sh, x1 + 0.6, y1 - 0.05, 2.6, 0.6, [[("v∞", {"bold": True, "color": MARS})],
                                           "along the departure asymptote"], size=13, color=INK2,
       after=0)
    px, py = pts[0]
    circle(sh, px, py, 0.1, fill=ORANGE)
    tx(sh, px - 2.0, py - 0.15, 1.85, 0.6, [[("Δv burn", {"bold": True, "color": ORANGE})],
                                           "at periapsis"], size=13, color=INK2, align="r", after=0)
    tx(sh, cx - 0.9, cy + 0.8, 1.8, 0.5, "parking orbit", size=13, color=MUTED, align="c")
    tx(sh, cx + 1.6, cy - 1.6, 2.2, 0.6, [[("escape hyperbola", {"bold": True, "color": BLUE})]],
       size=13, color=INK2)
    tx(sh, 0.6, 6.55, 5.6, 0.3, "Schematic, not to scale", size=11, color=MUTED, italic=True)


def bplane_schematic(sh):
    # Side view: incoming asymptote offset by B from the planet center.
    e = 1.6864  # brief: arrival hyperbola eccentricity
    fx, fy, s = 2.75, 4.35, 0.62
    th_inf = math.degrees(math.acos(-1 / e))
    a = 1 / (e - 1)
    # rotate so the incoming travel direction is +x (left -> right)
    vin = (-math.cos(math.radians(th_inf)), math.sin(math.radians(th_inf)))
    rot = -math.degrees(math.atan2(vin[1], vin[0]))
    pts = conic_pts(fx, fy, s, 1 + e, e, -th_inf + 38, th_inf - 22, rot=rot, n=200)
    c_th = math.radians(rot)
    ccx, ccy = fx + s * a * e * math.cos(c_th), fy - s * a * e * math.sin(c_th)
    # incoming asymptote line through the hyperbola centre, travel direction +x
    poly(sh, [(0.7, ccy), (ccx, ccy)], color=MUTED, w=1.25, dash=MSO_LINE_DASH_STYLE.DASH)
    poly(sh, pts, color=BLUE, w=3.0, head=True)
    circle(sh, fx, fy, 0.34, fill=MARS)
    arrow(sh, fx - 1.3, fy, fx - 1.3, ccy, color=INK, w=1.5, head=True, tail=True)
    tx(sh, fx - 2.15, (fy + ccy) / 2 - 0.2, 0.8, 0.4, [[("B", {"bold": True})]], size=16, color=INK,
       font=HEAD, align="r")
    poly(sh, [(fx - 1.45, fy), (fx + 0.0, fy)], color=LINE, w=1.0)
    tx(sh, 0.7, ccy + 0.08, 3.2, 0.35, "incoming asymptote", size=13, color=MUTED)
    tx(sh, 0.6, 6.15, 4.0, 0.6, [[("Side view", {"bold": True})], "B = miss distance of the asymptote"],
       size=13, color=INK2, after=0)
    # Face-on B-plane
    ox, oy, rm = 6.85, 4.1, 0.52
    r_imp, r_ring = rm * 6936 / 3396, rm * 7312 / 3396  # brief: impact radius, |B|, Mars radius
    circle(sh, ox, oy, r_ring, line=BLUE, lw=2.5)
    circle(sh, ox, oy, r_imp, line=ORANGE, lw=1.5, dash=MSO_LINE_DASH_STYLE.DASH)
    circle(sh, ox, oy, rm, fill=MARS)
    arrow(sh, ox, oy, ox + r_ring + 0.55, oy, color=INK, w=1.25)
    arrow(sh, ox, oy, ox, oy + r_ring + 0.5, color=INK, w=1.25)
    tx(sh, ox + r_ring + 0.5, oy - 0.38, 0.4, 0.35, [[("T", {"bold": True})]], size=14, font=HEAD)
    tx(sh, ox + 0.1, oy + r_ring + 0.3, 0.4, 0.35, [[("R", {"bold": True})]], size=14, font=HEAD)
    ang = math.atan2(-4735, 5572)  # brief aim point direction (B·T, B·R)
    axp, ayp = ox + r_ring * math.cos(ang), oy - r_ring * math.sin(ang)
    arrow(sh, ox, oy, axp, ayp, color=BLUE_D, w=2.0)
    circle(sh, axp, ayp, 0.09, fill=INK)
    tx(sh, axp + 0.12, ayp - 0.05, 1.2, 0.35, [[("aim point", {"bold": True})]], size=12, color=INK)
    tx(sh, 4.95, 6.15, 3.9, 0.6, [[("Face-on", {"bold": True})], "looking down the asymptote"],
       size=13, color=INK2, after=0)


def j2_schematic(sh, cx=3.2, cy=4.25):
    def ellipse_pts(rx, ry, rot, t0=0, t1=360, n=120):
        out = []
        for k in range(n + 1):
            t = math.radians(t0 + (t1 - t0) * k / n)
            x, y = rx * math.cos(t), ry * math.sin(t)
            c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            out.append((cx + x * c - y * s_, cy + x * s_ + y * c))
        return out

    eq = (2.75, 0.6, 0)
    orb = (2.95, 0.95, -20)
    orb2 = (2.95, 0.95, -32)
    # back halves (upper half in screen = behind), then the planet, then front halves
    poly(sh, ellipse_pts(*eq, 180, 360), color="9AA1AC", w=1.25, dash=MSO_LINE_DASH_STYLE.DASH)
    poly(sh, ellipse_pts(*orb2, 180, 360), color=BLUE_L, w=1.5, dash=MSO_LINE_DASH_STYLE.DASH)
    poly(sh, ellipse_pts(*orb, 180, 360), color=BLUE, w=2.0)
    box(sh, cx - 1.55, cy - 1.32, 3.1, 2.64, fill=MARS, shape="oval")
    poly(sh, ellipse_pts(*eq, 0, 180), color="F3D2C3", w=1.25, dash=MSO_LINE_DASH_STYLE.DASH)
    poly(sh, ellipse_pts(*orb2, 0, 180), color=BLUE_L, w=1.5, dash=MSO_LINE_DASH_STYLE.DASH)
    poly(sh, ellipse_pts(*orb, 0, 180), color=BLUE, w=3.0)

    # node: where the (front) orbit crosses the equator ring
    best = None
    for p in ellipse_pts(*orb, 0, 180, n=720):
        tq = math.atan2((p[1] - cy) / eq[1], (p[0] - cx) / eq[0])
        q = (cx + eq[0] * math.cos(tq), cy + eq[1] * math.sin(tq))
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if best is None or d < best[0]:
            best = (d, p, math.degrees(tq))
    _, node, tnode = best
    t_arc = ellipse_pts(eq[0], eq[1], 0, tnode, tnode + 28, n=30)
    poly(sh, t_arc, color="FFD27A", w=3.5, head=True)
    circle(sh, node[0], node[1], 0.11, fill="FFD27A", line=INK, lw=1.0)
    tx(sh, cx - 1.5, cy + 1.5, 3.0, 0.6, [[("ascending node drifts", {"bold": True,
                                                                      "color": ORANGE})],
                                         "(nodal regression, yellow arrow)"], size=13, color=INK2,
       after=0, align="c")
    tx(sh, cx - 1.2, cy - 2.15, 2.4, 0.35, "equatorial bulge (exaggerated)", size=12, color=MUTED,
       align="c")
    tx(sh, 0.6, 6.55, 5.6, 0.3, "Schematic, not to scale", size=11, color=MUTED, italic=True)


def rsw_schematic(sh):
    mx, my, rm, ro = 3.4, 8.7, 3.0, 4.45
    pts_mars = [(mx + rm * math.cos(math.radians(a)), my - rm * math.sin(math.radians(a)))
                for a in range(55, 126, 2)]
    poly(sh, pts_mars, color=None, fill=MARS, closed=True)
    orbit = [(mx + ro * math.cos(math.radians(a)), my - ro * math.sin(math.radians(a)))
             for a in range(62, 119, 2)]
    poly(sh, orbit, color=MUTED, w=1.5, dash=MSO_LINE_DASH_STYLE.DASH)
    tx(sh, mx - 0.6, my - rm + 0.15, 1.2, 0.35, "Mars", size=13, color=WHITE, align="c")
    tx_, ty_ = mx, my - ro
    arrow(sh, tx_, ty_, tx_, ty_ - 1.75, color=AQUA, w=3.0)
    arrow(sh, tx_, ty_, tx_ + 1.6, ty_, color=BLUE, w=3.0)
    circle(sh, tx_, ty_, 0.17, fill=WHITE, line=INK, lw=1.5)
    circle(sh, tx_, ty_, 0.05, fill=INK)
    tx(sh, tx_ + 0.15, ty_ - 1.85, 2.6, 0.35, [[("R", {"bold": True, "color": AQUA}),
                                               ("  radial (R-bar)", {})]], size=14, color=INK2)
    tx(sh, tx_ + 0.35, ty_ - 0.42, 2.8, 0.35, [[("S", {"bold": True, "color": BLUE}),
                                               ("  along-track (V-bar)", {})]], size=14, color=INK2)
    tx(sh, tx_ - 1.6, ty_ + 0.55, 3.2, 0.35, [[("W", {"bold": True}), ("  cross-track (out of page)", {})]],
       size=14, color=INK2, align="c")
    tx(sh, tx_ - 0.6, ty_ + 0.22, 1.2, 0.35, [[("target", {"bold": True})]], size=13, align="c")
    cxp, cyp = tx_ - 1.7, ty_ - 1.0
    circle(sh, cxp, cyp, 0.1, fill=ORANGE)
    arrow(sh, tx_ - 0.17, ty_ - 0.1, cxp + 0.1, cyp + 0.06, color=ORANGE, w=1.5,
          dash=MSO_LINE_DASH_STYLE.DASH)
    tx(sh, cxp - 0.95, cyp - 0.5, 1.9, 0.35, [[("chaser", {"bold": True, "color": ORANGE})]],
       size=13, align="c")
    tx(sh, cxp - 1.1, cyp + 0.3, 1.3, 0.55, ["relative", "position (x, y, z)"], size=12, color=MUTED,
       after=0)
    tx(sh, 0.6, 6.55, 6.0, 0.3, "Frame rides with the target; schematic, not to scale", size=11,
       color=MUTED, italic=True)


def flow(sh, x, y, w, h, items, gap=0.32, fill=CARD, size=14, tsize=15, badge_color=MARS,
         arrow_color=MUTED, fills=None):
    """Left-to-right boxes with arrows. items: list of (title, body). Returns list of groups' parts."""
    n = len(items)
    bw = (w - gap * (n - 1)) / n
    parts = []
    for i, (title, body) in enumerate(items):
        bx = x + i * (bw + gap)
        parts.append((bx, bw))
        card(sh, bx, y, bw, h, title, body, fill=(fills[i] if fills else fill), size=size, tsize=tsize,
             badge=i + 1, badge_color=badge_color)
        if i < n - 1:
            arrow(sh, bx + bw + 0.04, y + h / 2, bx + bw + gap - 0.04, y + h / 2, color=arrow_color,
                  w=2.0)
    return parts


# ---------------------------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------------------------
def divider(d, idx, title, question, minutes, talk, lines=1):
    y_title = 3.0 if lines == 1 else 2.6
    s = d.new(title, dark=True, size=46 if lines == 1 else 40,
              title_box=(0.9, y_title, 6.9, 0.8 * lines + 0.2), anchor="t")
    sh = s.shapes
    ring = circle(sh, 10.95, 3.6, 3.85, line="3A4150", lw=1.25)
    ring.name = "Orbit ring"
    circle(sh, 11.3, 3.75, 2.9, fill=MARS)
    ang = math.radians(230 - 20 * idx)  # the spacecraft dot advances around the ring per section
    circle(sh, 10.95 + 3.85 * math.cos(ang), 3.6 - 3.85 * math.sin(ang), 0.1, fill=WHITE)
    tx(sh, 0.9, y_title - 1.55, 4.0, 1.4, [[(f"{idx + 1:02d}", {"bold": True})]], size=88,
       color=MARS_L, font=HEAD)
    tx(sh, 0.9, y_title + 0.8 * lines + 0.45, 6.9, 1.0, question, size=22, color="C9CED6")
    footer(s, idx, dark=True)
    notes(s, minutes, talk)
    return s


def build():
    d = Deck()

    # ---------------- Opening ----------------
    d.section("Opening")
    s = d.new("Mars Courier", size=54, title_box=(0.6, 1.75, 6.6, 0.95))
    sh = s.shapes
    fig(sh, "figures/summary/heliocentric_transfer.png", 6.75, 0.55, 6.4, 6.6, ha="r",
        alt="Heliocentric Earth-to-Mars transfer, faded background", alpha=0.42)
    tx(sh, 0.6, 2.8, 5.9, 1.2, "End-to-end mission design from launch window to docking", size=28,
       color=INK, font=HEAD, line=1.05)
    tx(sh, 0.6, 4.1, 6.0, 0.9,
       "Earth departure, interplanetary transfer, B-plane capture, and J2-aware rendezvous",
       size=18, color=INK2)
    tx(sh, 0.6, 5.35, 6.0, 0.45, [[("Colin Skinner", {"bold": True})]], size=20, color=INK, font=HEAD)
    tx(sh, 0.6, 5.85, 6.0, 0.4, "[Course]  ·  [Date]", size=16, color=MUTED, name="Course and date")
    tx(sh, 0.6, 0.45, 4.0, 0.3, [[("MISSION DESIGN REVIEW", {"bold": True})]], size=11, color=MARS,
       font=HEAD)
    slide_number(sh, 11.9, 7.03, 0.83, 0.27, MUTED)
    notes(s, 0.5, [
        ("This project designs a cargo-courier mission that launches in June 2035, arrives at Mars on "
        "Christmas Day, and docks with a spacecraft already in Mars orbit."),
        ("I'll walk through each phase in mission order and finish with what it costs, what I learned, "
        "and what I'd improve."),
        "Most of the time goes to rendezvous and docking: that's where the guidance work lives.",
        "Fill in course and date placeholders before presenting.",
    ])

    # Mission concept
    s = d.new("Mission concept and goals")
    sh = s.shapes
    eyebrow(s, "Overview")
    headline(s, "Deliver a Perseverance-class payload to a spacecraft waiting in a 300 km Mars orbit")
    tx(sh, 0.6, 1.8, 5.9, 3.9, bullets([
        "Launch from Cape Canaveral into a 300 km parking orbit, then escape onto a Lambert transfer",
        "193-day cruise with one midcourse correction",
        "B-plane-targeted, direct-injection capture into the target's orbit",
        "Rendezvous and docking: plane change, Hohmann, phasing, HCW terminal guidance",
        "Track every burn's Δv and size it against a real launch vehicle",
    ], size=17, after=10))
    box(sh, 0.6, 5.75, 5.9, 1.05, fill=MARS_T, shape="round", radius=0.1, margin=0.18,
        text=[[("A logistics problem: ", {"bold": True, "color": MARS}),
               ("the target spacecraft is fixed. We have to get to it, on its terms.", {})]],
        size=16, color=INK, anchor="m")
    tx(sh, 7.0, 1.72, 5.7, 0.35, [[("Mission at a glance", {"bold": True})]], size=15, font=HEAD)
    simple_table(sh, 7.0, 2.12, [2.25, 3.45], [
        ["Item", "Value"],
        ["Launch (escape burn)", "2035-06-14 22:04 TDB"],
        ["Mars capture", "2035-12-25 00:00 TDB"],
        ["Docking", "2035-12-26 17:36 (nominal) / 18:57 (dispersed) TDB"],
        ["Cruise", "193.1 days (escape to capture)"],
        ["Earth parking orbit", "300 km circular, 49.1° inclination"],
        ["Mars target orbit", "300 km circular, i = 45°, RAAN = 125°"],
        ["Spacecraft Δv", "2.47 (nominal) – 2.59 (dispersed) km/s"],
        ["Launch vehicle Δv", "11.47 km/s ideal (ascent 7.79 + escape 3.68)"],
        ["Launch vehicle", "Falcon Heavy, expendable: 16.8 t to Mars injection"],
    ], size=13, row_h=0.44, bold_first_col=True)
    footer(s)
    notes(s, 1.2, [
        ("Frame the courier as a logistics problem: the target spacecraft and its orbit are given; we "
        "have to get to it."),
        ("Launch 2035-06-14 22:04 TDB, capture 2035-12-25 00:00 TDB, dock 2035-12-26 17:36 TDB "
        "(nominal) or 18:57 TDB (dispersed)."),
        "Cruise is 193.1 days escape-to-capture; the Lambert time of flight is 193.5 days.",
        "Target orbit: 300 km circular, i = 45°, RAAN = 125°.",
        "Spacecraft Δv 2,470.2 m/s nominal, 2,586.3 m/s dispersed; launch vehicle 11,466.3 m/s ideal.",
        ("Light remark: launch falls on my birthday and arrival on Christmas Day (mission files are "
        "named birthday_christmas_earlier)."),
    ])

    # Agenda
    s = d.new("Agenda")
    sh = s.shapes
    eyebrow(s, "Overview")
    headline(s, "Six phases in mission order: 193 days of cruise, then 43 hours to dock")
    blurbs = ["Porkchops and picking the date", "Parking orbit, escape, launch timing",
              "Ground contacts and light delay", "B-plane aim and the target plane",
              "Guidance, results, what broke", "Δv budget, payload, next steps"]
    x = 0.6
    for i, (name, blurb) in enumerate(zip(SECTIONS, blurbs)):
        rpo = i == 4
        box(sh, x, 1.7, 1.95, 1.5, fill=BLUE_T if rpo else CARD, shape="round", radius=0.1,
            line=BLUE if rpo else None, lw=1.5)
        tx(sh, x + 0.15, 1.8, 1.0, 0.4, [[(f"{i + 1:02d}", {"bold": True})]], size=18, color=MARS,
           font=HEAD)
        tx(sh, x + 0.15, 2.2, 1.7, 0.35, [[(name, {"bold": True})]], size=14, font=HEAD)
        tx(sh, x + 0.15, 2.55, 1.7, 0.6, blurb, size=13, color=INK2)
        if rpo:
            tag(sh, x + 0.9, 1.82, "Focus", fill=BLUE, color=WHITE, w=0.9)
        x += 2.07
    fig(sh, "figures/summary/mission_timeline.png", 0.6, 3.4, 12.1, 3.0,
        alt="Two-panel mission timeline: days since escape, and RPO hours since capture")
    tx(sh, 0.6, 6.5, 12.1, 0.35,
       "Escape 2035-06-14 22:04  ·  midcourse 06-17 21:36  ·  capture 12-25 00:00  ·  docking 12-26 "
       "17:36 (nominal) / 18:57 (dispersed), TDB", size=13, color=MUTED, align="c")
    footer(s)
    notes(s, 0.8, [
        "Six sections in mission order; the tracker at the bottom of each slide shows where we are.",
        "Rendezvous is the focus: about half of the talk.",
        "Timeline: left panel is days since the escape burn; right panel zooms into the RPO in hours.",
        "Phasing takes ~35 of the ~43 RPO hours. Every burn is modeled as impulsive.",
        "Midcourse correction at 2035-06-17 21:36 (brief section 2.2 time).",
    ])

    # ---------------- 01 Launch window ----------------
    d.section("Launch window")
    divider(d, 0, "Launch window", "When do we leave Earth, and how long is the trip?", 0.2, [
        "Section 1 (~3.5 min): finding a launch window with porkchop plots.",
        "Question to answer: when do we leave, and how long is the trip?",
        "Ends with the selected transfer: 2035-06-14 → 2035-12-25, 193.5 days.",
    ])

    s = d.new("Primer: Lambert's problem, porkchop plots, and C3")
    sh = s.shapes
    eyebrow(s, "01 · Launch window  ·  Primer")
    headline(s, "Pick a departure date and a flight time; Lambert's problem gives the orbit that connects them")
    lambert_schematic(sh)
    steps = []
    for i, (title, body) in enumerate([
        ("Lambert's problem", ("Given where we start (Earth on the departure date), where we end "
                              "(Mars on the arrival date) and the flight time, solve for the transfer "
                              "orbit that connects them.")),
        ("C3 = v∞²", ("The transfer leaves Earth with a speed relative to Earth, v∞. Its square, C3, "
                     "is the energy the launcher must supply: lower is cheaper.")),
        ("Porkchop plot", ("Repeat for every departure date × flight time and color the grid by C3. "
                          "Low-C3 basins are the launch windows: a short-way (type I) and a long-way "
                          "(type II) basin.")),
    ]):
        g = group(s)
        card(g.shapes, 7.0, 1.75 + i * 1.7, 5.7, 1.55, title, body, size=14, badge=i + 1,
             badge_color=BLUE)
        steps.append([g])
    add_builds(s, steps)
    footer(s, 0)
    notes(s, 0.8, [
        "Primer for non-specialists; self-contained, delete if the audience knows Lambert/C3.",
        "Lambert: two positions plus a flight time give the transfer conic (one per transfer type).",
        ("v∞ is the hyperbolic excess speed relative to the planet; C3 = v∞² is how launch-vehicle "
        "capability is quoted."),
        ("A porkchop plot is just Lambert solved over a grid of departure date × time of flight, colored "
        "by C3."),
        "Type I = short way around the Sun; type II = long way.",
    ])

    s = d.new("Finding the window: the porkchop plot")
    sh = s.shapes
    eyebrow(s, "01 · Launch window")
    headline(s, "The 2035 opportunity has a clear low-C3 basin near June departures with ~190–200-day flights")
    fig(sh, "figures/interplanetary/porkchop_c3.png", 0.6, 1.7, 8.4, 5.1, va="t",
        alt="Porkchop plot of departure C3 over departure date and time of flight, 2034 to 2036")
    tx(sh, 9.3, 1.85, 3.45, 4.9, [
        {"t": "**How to read it:** x = departure date, y = time of flight, color = departure C3",
         "size": 15, "after": 14, "color": INK2},
    ] + bullets([
        "Lambert solved over a **579 × 1029-day grid** of departure date × time of flight",
        "Two basins: **type I** (short way, ~150–250 days) and **type II** (long way, ~350–500 days)",
        "Chosen region: type I, C3 ≈ 10–11 km²/s²",
    ], size=15, after=12))
    footer(s, 0)
    notes(s, 0.8, [
        "Axes: departure date on x, time of flight on y, color = departure C3 (km²/s²).",
        "Grid: 579 × 1029 days at 1-day resolution.",
        "Two basins: type I (short way, ~150–250 days) and type II (long way, ~350–500 days).",
        "The minimum sits around 8–12 km²/s² in mid-2035; we work in the type-I basin at C3 ≈ 10–11.",
    ])

    s = d.new("How I decided: the launch window")
    sh = s.shapes
    eyebrow(s, "01 · Launch window  ·  Design logic")
    headline(s, "Constraints on C3 and departure declination narrow it to a few-day window; I picked June 14 → Dec 25")
    steps = []
    for i, (title, body) in enumerate([
        ("Search", "579 × 1029-day grid, 1-day steps"),
        ("Constrain", "Limits on C3 and departure declination: 441 feasible transfers"),
        ("Pick", "Inside the low-C3 band, 10.5–11 km²/s² contours"),
        ("Refine", "Fine search at 0.01-day resolution around the window"),
    ]):
        g = group(s)
        bx = 0.6 + i * 3.12
        card(g.shapes, bx, 1.7, 2.82, 1.42, title, body, size=14, badge=i + 1)
        if i < 3:
            arrow(g.shapes, bx + 2.86, 2.41, bx + 3.08, 2.41, w=2.0)
        steps.append([g])
    fig(sh, "figures/interplanetary/launch_window.png", 0.6, 3.3, 7.1, 3.55, ha="l",
        alt="Zoomed C3 contours with the selected transfer star")
    box(sh, 8.2, 3.3, 4.5, 2.3, fill=BLUE_T, shape="round", radius=0.1)
    tx(sh, 8.4, 3.4, 4.1, 0.35, [[("Selected transfer", {"bold": True, "color": BLUE_D})]], size=15,
       font=HEAD)
    tx(sh, 8.4, 3.8, 4.2, 1.8, [
        "Depart **2035-06-14**, arrive **2035-12-25**",
        "Time of flight **193.5 days**",
        "Departure C3 **10.62 km²/s²**",
        "DLA **4.06°**: easy from the Cape (28.5° N)",
        "Arrival v∞ **2.82 km/s**",
    ], size=14, color=INK, after=3)
    aside = group(s)
    box(aside.shapes, 8.2, 5.75, 4.5, 1.1, fill=MARS_T, shape="round", radius=0.1, margin=0.15,
        text=[[("Aside: ", {"bold": True, "color": MARS}),
               (("the date is personal. Launch on my birthday, arrive on Christmas Day. It still sits "
                "inside the low-C3 band."), {})]], size=14, color=INK, anchor="m")
    steps[2].append(aside)
    add_builds(s, steps)
    footer(s, 0)
    notes(s, 1.2, [
        "Decision chain: search the full grid, apply C3 and departure-declination constraints, pick, refine.",
        "441 feasible transfers survive the constraints.",
        "Selection: Lambert references 2035-06-14 12:00 TDB → 2035-12-25 00:00 TDB, 193.5 days.",
        "Departure v∞ 3.258 km/s, C3 10.62 km²/s², RLA 4.79°, DLA 4.06°; arrival v∞ 2.820 km/s, C3 7.95 km²/s².",
        ("The pick was personal (birthday to Christmas) but sits in the 10.5–11 km²/s² contour band of "
        "the zoomed plot; a 0.01-day fine grid confirmed it."),
    ])

    s = d.new("The transfer")
    sh = s.shapes
    eyebrow(s, "01 · Launch window")
    headline(s, "A 193.5-day type-I transfer from 1.0 AU to 1.48 AU")
    fig(sh, "figures/summary/heliocentric_transfer.png", 0.6, 1.65, 5.6, 5.2, ha="l",
        alt="Top-down ecliptic view of the transfer with dated markers")
    fig(sh, "figures/interplanetary/transfer_3d.png", 5.9, 1.6, 6.8, 3.4, ha="r",
        alt="3D heliocentric transfer with Earth, Mars and Sun")
    tx(sh, 6.4, 5.1, 6.3, 1.8, bullets([
        "Mars leads Earth by **31°** at launch",
        "The spacecraft sweeps **149°** around the Sun and arrives at **1.48 AU**",
        "Re-propagating the Lambert solution with two-body dynamics reproduces it within **594 m**",
    ], size=15, after=8))
    footer(s, 0)
    notes(s, 0.7, [
        ("Left: top-down ecliptic view with launch, Mars-at-launch and arrival markers. Right: same "
        "transfer in 3D."),
        "Mars leads Earth by 31° at launch; the spacecraft sweeps 149° around the Sun.",
        "Arrives at 1.48 AU after 193.5 days.",
        "Check: re-propagating the Lambert solution with two-body dynamics reproduced it within 594 m.",
    ])

    # ---------------- 02 Departure ----------------
    d.section("Departure")
    divider(d, 1, "Earth departure", "How do we leave Earth on exactly the right asymptote?", 0.2, [
        "Section 2 (~3.5 min): from the parking orbit onto the interplanetary trajectory.",
        "Key idea: the escape hyperbola has to deliver the v∞ that Lambert asks for.",
        "Design decision in this section: the 0.42-day launch-time offset.",
    ])

    s = d.new("Primer: escaping Earth on a hyperbola")
    sh = s.shapes
    eyebrow(s, "02 · Departure  ·  Primer")
    headline(s, "Burn once, deep in the gravity well; what is left far from Earth is the excess speed v∞")
    escape_schematic(sh)
    steps = []
    for i, (title, body) in enumerate([
        ("Parking orbit", "Circle Earth at 300 km first. Circular speed there is 7.726 km/s."),
        ("One burn at periapsis", ("Speed up to vₚ = √(v∞² + 2μ/rₚ). Burning low in the gravity well "
                                  "makes each m/s count most (the Oberth effect).")),
        ("Aim the asymptote", ("Far from Earth the path straightens along the asymptote. Its direction "
                              "(RLA/DLA) must match the Lambert departure v∞.")),
    ]):
        g = group(s)
        card(g.shapes, 7.0, 1.75 + i * 1.7, 5.7, 1.55, title, body, size=14, badge=i + 1,
             badge_color=BLUE)
        steps.append([g])
    add_builds(s, steps)
    footer(s, 1)
    notes(s, 0.8, [
        "Primer; self-contained and safe to delete.",
        ("Spacecraft starts in a circular parking orbit, then one tangential burn at periapsis puts it "
        "on an escape hyperbola."),
        ("vₚ = √(v∞² + 2μ/rₚ): for v∞ = 3.258 km/s at r = 6678.1 km this is 11.401 km/s, so the burn is "
        "11.401 − 7.726 = 3.676 km/s."),
        "The asymptote direction (RLA/DLA) is what Lambert hands us; the hyperbola has to deliver it.",
    ])

    s = d.new("Escape from the parking orbit")
    sh = s.shapes
    eyebrow(s, "02 · Departure")
    headline(s, "A 3.68 km/s burn from a 300 km parking orbit puts us on the departure asymptote")
    fig(sh, "figures/departure/escape_hyperbola_eci.png", 0.6, 1.65, 7.0, 3.5, ha="l",
        alt="Escape hyperbola in Earth-centered inertial frame with Sun vector and Cape Canaveral")
    tx(sh, 7.85, 1.8, 4.85, 3.4, bullets([
        "v∞ **3.258 km/s** → hyperbola e = 1.178, aiming radius 23,369 km",
        "Tangential burn at periapsis: **7.726 → 11.401 km/s**",
        "v∞ and the launch site fix the parking-orbit plane: **49.1°** inclination",
        "Validated: impact parameter to ~1 cm, C3 to 3.6×10⁻⁸ km²/s²",
    ], size=15, after=10))
    g = group(s)
    box(g.shapes, 0.6, 5.35, 12.1, 1.5, fill=CARD, shape="round", radius=0.1)
    tx(g.shapes, 0.85, 5.45, 4.0, 0.35, [[("Getting to the parking orbit", {"bold": True})]],
       size=15, font=HEAD)
    tx(g.shapes, 0.85, 5.82, 3.2, 0.9, "Ideal lower bound: no drag, no gravity losses. Real vehicles "
                                       "add ~1.5–2 km/s.", size=13, color=MUTED)
    stat(g.shapes, 4.3, 5.48, 2.6, "7.79 km/s", "ideal ascent, Cape → 300 km in the 49.1° plane",
         vcolor=ORANGE, vsize=24)
    stat(g.shapes, 7.0, 5.48, 2.5, "300 m/s", "help from Earth's rotation for this plane",
         vcolor=AQUA, vsize=24)
    stat(g.shapes, 9.6, 5.48, 3.0, "131.8°", "launch azimuth (southeast). Open item: not yet checked "
                                            "against Cape range safety", vcolor=MARS, vsize=24)
    add_builds(s, [[g]])
    footer(s, 1)
    notes(s, 1.2, [
        ("Departure v∞ 3.258 km/s → escape hyperbola e = 1.1779, aiming radius 23,369 km, asymptote "
        "true anomaly 148.1°."),
        "Burn 3.676 km/s from 7.726 to 11.401 km/s at r = 6678.1 km.",
        ("The escape-hyperbola plane is defined by v∞ and the launch site; that fixes the 49.1° parking "
        "orbit."),
        "Validation: propagated h/v∞ matches theory to ~1 cm; C3 matches Lambert to 3.6×10⁻⁸ km²/s².",
        ("Ascent (click): ideal 7.791 km/s; due-east reference 7.682, non-rotating 8.091; rotation helps "
        "300 m/s (site moves east at 409 m/s). This is a lower bound, not a launch cost."),
        "Azimuth 131.8° from north is likely outside the Cape's usual range-safety corridor: open item.",
    ])

    s = d.new("How I decided: launch timing")
    sh = s.shapes
    eyebrow(s, "02 · Departure  ·  Design logic")
    headline(s, "Shifting launch by 0.42 day cut the midcourse correction from 564 to 132 m/s")
    box(sh, 0.6, 1.75, 12.1, 0.95, fill=CARD, shape="round", radius=0.1, margin=0.2,
        text=[[("The problem: ", {"bold": True}),
               (("Lambert's arc is heliocentric and ignores Earth's gravity well, so the escape hyperbola "
                "has to be timed to hand off onto it."), {})]], size=16, color=INK, anchor="m")
    opts = []
    for i, (lbl, sub, val, num, col, tint) in enumerate([
        ("Option A", "Launch at the Lambert reference time (2035-06-14 12:00 TDB)", "564 m/s", 564.2,
         ORANGE, ORANGE_T),
        ("Option B: chosen", "Launch 0.42 day later (2035-06-14 22:04 TDB); correct on 2035-06-17 21:36",
         "131.8 m/s", 131.8, BLUE, BLUE_T),
    ]):
        g = group(s)
        y0 = 2.95 + i * 1.6
        box(g.shapes, 0.6, y0, 12.1, 1.4, fill=tint, shape="round", radius=0.1)
        tx(g.shapes, 0.85, y0 + 0.15, 4.2, 0.4, [[(lbl, {"bold": True, "color": col})]], size=16,
           font=HEAD)
        tx(g.shapes, 0.85, y0 + 0.55, 4.3, 0.8, sub, size=14, color=INK2)
        bw = 5.2 * num / 564.2
        box(g.shapes, 5.4, y0 + 0.42, bw, 0.56, fill=col, shape="rect")
        tx(g.shapes, 5.4 + bw + 0.15, y0 + 0.35, 1.9, 0.7, [[(val, {"bold": True})]], size=24,
           color=col, font=HEAD, anchor="m")
        tx(g.shapes, 5.4, y0 + 1.0, 4.0, 0.3, "midcourse correction", size=12, color=MUTED)
        opts.append([g])
    note = group(s)
    tx(note.shapes, 0.6, 6.25, 12.1, 0.6,
       "The correction is velocity-only, onto the nearest Lambert sample (26,286 km away). A full "
       "patched-conic handoff is future work.", size=14, color=MUTED)
    opts[1].append(note)
    add_builds(s, opts)
    footer(s, 1)
    notes(s, 1.2, [
        ("Lambert is heliocentric; it doesn't know about Earth's gravity well. The escape hyperbola must "
        "be timed so it lands on the Lambert arc."),
        "Launch at the Lambert reference (2035-06-14 12:00 TDB) → 564.2 m/s midcourse.",
        "Launch 0.42 day later (22:04 TDB) → 131.8 m/s midcourse at 2035-06-17 21:36.",
        "Velocity-only correction onto the nearest Lambert sample, 26,286 km away.",
        "Proper fix is a full patched-conic handoff (future work).",
    ])

    # ---------------- 03 Cruise & comms ----------------
    d.section("Cruise & comms")
    divider(d, 2, "Cruise and communications", "Who can hear the spacecraft, and how long does a "
                                               "message take?", 0.2, [
        "Section 3 (~1.5 min): ground coverage during departure and the light-time delay at Mars.",
        "Keep it brisk: one slide.",
        "Its punchline (8-minute one-way delay) motivates autonomous RPO later.",
    ], lines=2)

    s = d.new("Ground coverage and light time")
    sh = s.shapes
    eyebrow(s, "03 · Cruise & comms")
    headline(s, "Up to five stations in view during departure; by arrival, an 8-minute one-way delay")
    fig(sh, "figures/ground_segment/network_visibility.png", 0.6, 1.65, 6.1, 3.08, ha="l",
        alt="Per-station visibility bars and available-station count, 2035-06-15 to 06-17")
    tx(sh, 7.0, 1.75, 5.7, 3.0, bullets([
        "Seven stations: Cape Canaveral, Svalbard, Punta Arenas, Goldstone, Madrid, Canberra, Vandenberg",
        "Elevation masks 10° (Madrid 5°); 71.5 h of departure propagated",
        "Cape Canaveral: 3 passes of ~10.3–10.6 h, peak elevation 60.7°–64.4°",
        "Zero stations in view at the first sample, then at least one for most of the period",
    ], size=14, after=7))
    fig(sh, "figures/summary/cruise_range.png", 0.6, 4.85, 8.0, 2.05, ha="l",
        alt="Spacecraft range from Earth and Mars versus day of cruise")
    g = group(s)
    box(g.shapes, 8.85, 4.9, 3.85, 1.95, fill=MARS_T, shape="round", radius=0.1)
    tx(g.shapes, 9.05, 4.98, 3.5, 0.75, [[("8.0 min", {"bold": True})]], size=36, color=MARS,
       font=HEAD)
    tx(g.shapes, 9.05, 5.75, 3.5, 1.05, "one-way light time at arrival (0.97 AU). 16 min round trip, "
                                        "so docking must run onboard.", size=14, color=INK)
    add_builds(s, [[g]])
    footer(s, 2)
    notes(s, 1.2, [
        "Seven-station network; masks 10° except Madrid at 5°; 71.5 h of the departure propagated.",
        "Max five stations simultaneously; zero at the very first sample (spacecraft starts out of view).",
        "Cape Canaveral: 3 passes of ~10.3–10.6 h, peak elevation 60.7°–64.4°.",
        ("The ground track spirals off as the spacecraft leaves Earth, then settles near the asymptote "
        "declination (backup slide)."),
        ("At arrival the spacecraft is 0.97 AU from Earth: 8.0 min one way, 16 min round trip. That's "
        "why RPO must be autonomous; ground-in-the-loop docking at Mars isn't practical."),
        "Link budget is future work.",
    ])

    # ---------------- 04 Arrival ----------------
    d.section("Arrival")
    divider(d, 3, "Mars arrival", "Where do we aim, and which orbit plane do we capture into?", 0.2, [
        "Section 4 (~4.5 min): B-plane targeting and the single biggest design lever in the mission.",
        "Primer on the B-plane, then the aim point, then how the target plane was chosen.",
        "Payoff: capture 5.52 → 2.27 km/s and 1.3 → 6.6 rovers per Falcon Heavy.",
    ])

    s = d.new("Primer: the B-plane")
    sh = s.shapes
    eyebrow(s, "04 · Arrival  ·  Primer")
    headline(s, "Aim at a point on a plane through Mars; that point sets periapsis altitude and the orbit plane")
    bplane_schematic(sh)
    steps = []
    for i, (title, body) in enumerate([
        ("The aim point", ("The B-plane passes through Mars, perpendicular to the incoming asymptote. "
                          "B is where the asymptote pierces it.")),
        ("Gravity focusing", ("Mars pulls the path in, so its effective target (impact radius, dashed) "
                             "is bigger than its disk.")),
        ("The ring", ("Every point on the ring gives the same periapsis altitude. Where on the ring you "
                     "aim sets the arrival orbit plane.")),
    ]):
        g = group(s)
        card(g.shapes, 9.05, 1.75 + i * 1.7, 3.65, 1.55, title, body, size=13, tsize=15)
        steps.append([g])
    add_builds(s, steps)
    footer(s, 3)
    notes(s, 0.8, [
        "Primer; self-contained and safe to delete.",
        ("The B-plane is perpendicular to the incoming asymptote through the planet center; B·T and B·R "
        "locate the aim point."),
        "|B| sets periapsis radius; gravity focusing makes the impact radius larger than the planet.",
        ("The ring of aim points with the same periapsis is a circle; the angle around it rotates the "
        "arrival plane. That's the lever on the next slides."),
    ])

    s = d.new("The B-plane and the aim point")
    sh = s.shapes
    eyebrow(s, "04 · Arrival")
    headline(s, "Aim for |B| = 7,312 km to get a 300 km periapsis, only ~376 km outside the impact radius")
    fig(sh, "figures/summary/arrival_bplane.png", 0.6, 1.65, 7.0, 5.25, ha="l",
        alt="Arrival B-plane with Mars disk, impact radius, 300 km periapsis ring and aim point")
    tx(sh, 7.9, 1.85, 4.8, 4.9, bullets([
        "Arrival v∞ **2.82 km/s**",
        "Gravity focusing: impact radius **6,936 km** (Mars radius 3,396 km)",
        "Aim point B·T **5,572 km**, B·R **−4,735 km**",
        "Validated: asymptote error 0.0000°, periapsis radius error 0 m",
        "Where we sit on the ring sets the arrival plane. That choice is next.",
    ], size=16, after=12))
    footer(s, 3)
    notes(s, 1.0, [
        "Arrival v∞ 2.820 km/s; asymptote RA 305.76°, DEC 8.90° (Mars-equatorial).",
        "Hyperbola e = 1.6864, aiming radius 7,312 km, pre-burn periapsis speed 5.579 km/s.",
        "Impact radius 6,936 km; the 300 km-periapsis ring is only ~376 km outside it.",
        "Aim point B·T 5,572 km, B·R −4,735 km, |B| 7,312 km.",
        ("Validation: asymptote error 0.0000°, periapsis radius error 0 m, post-burn e ≈ 3×10⁻¹⁶, i and "
        "RAAN exact."),
    ])

    s = d.new("How I decided: the target orbit plane")
    sh = s.shapes
    eyebrow(s, "04 · Arrival  ·  Design logic")
    headline(s, "The transfer fixes the arrival direction; the target's orbit plane is the free choice")
    steps = []
    items = [
        ("Transfer", "Fixes v∞: 2.82 km/s from RA 305.76°, DEC 8.90°"),
        ("Altitude", "300 km periapsis fixes the |B| = 7,312 km ring"),
        ("Physics", "Direct injection rotates the plane at periapsis, where it costs most"),
        ("Free choice", "Target RAAN: 15° → 125°, keeping i = 45°"),
        ("Result", "β 49.7° → 6.8°; capture 5.52 → 2.27 km/s"),
    ]
    for i, (title, body) in enumerate(items):
        g = group(s)
        bx = 0.6 + i * 2.48
        last = i == len(items) - 1
        card(g.shapes, bx, 1.75, 2.24, 2.1, title, body, size=14, tsize=15, badge=i + 1,
             fill=BLUE_T if i == 3 else (MARS_T if last else CARD),
             badge_color=BLUE if i == 3 else MARS)
        if not last:
            arrow(g.shapes, bx + 2.27, 2.8, bx + 2.45, 2.8, w=2.0)
        steps.append([g])
    key = group(s)
    box(key.shapes, 0.6, 4.1, 7.4, 2.75, fill=CARD, shape="round", radius=0.1)
    tx(key.shapes, 0.85, 4.25, 7.0, 0.4, [[("Why it works", {"bold": True})]], size=16, font=HEAD)
    tx(key.shapes, 0.85, 4.7, 7.0, 2.1, bullets([
        "β is the angle between the arrival asymptote and the target plane",
        "Same inclination (45°), different node: the target plane now nearly contains the asymptote",
        "Capture is only 93 m/s above the coplanar ideal (2.175 km/s)",
    ], size=16, after=8))
    alt = group(s)
    box(alt.shapes, 8.3, 4.1, 4.4, 2.75, fill=ORANGE_T, shape="round", radius=0.1)
    tx(alt.shapes, 8.5, 4.25, 4.0, 0.4, [[("If the target plane were fixed", {"bold": True})]],
       size=16, font=HEAD, color=INK)
    tx(alt.shapes, 8.5, 4.7, 4.0, 2.0, "Capture into a high ellipse and do the plane change near "
                                        "apoapsis, where it is cheap. Future work.", size=15, color=INK2)
    steps[-1] += [key, alt]
    add_builds(s, steps)
    footer(s, 3)
    notes(s, 1.0, [
        "The transfer fixes the incoming asymptote (RA 305.76°, DEC 8.90°, v∞ 2.820 km/s).",
        ("The 300 km periapsis fixes |B| = 7,312 km; only the angle around the ring is free, and that "
        "sets the arrival plane."),
        ("Direct injection does the plane change at periapsis, the most expensive place, so β matters "
        "enormously."),
        "Moving target RAAN 15° → 125° (same 45° inclination) cut β from 49.7° to 6.8°.",
        "Capture 2,268.5 m/s vs coplanar 2,175.2 m/s: 93.3 m/s for the residual plane change.",
        ("Fallback if the target plane were fixed: high-ellipse capture, plane change near apoapsis "
        "(future work)."),
    ])

    s = d.new("The biggest lever: choosing the target plane")
    sh = s.shapes
    eyebrow(s, "04 · Arrival")
    headline(s, "Aligning the target orbit with the arrival geometry cut capture Δv from 5.52 to 2.27 km/s")
    fig(sh, "figures/summary/capture_trade.png", 0.6, 1.6, 12.1, 3.45, va="t",
        alt="Useful payload versus capture delta-v with coplanar, current and earlier design points")
    steps = []
    for i, (lbl, vals, col, tint) in enumerate([
        ("Earlier design: target RAAN 15°", ["49.7°", "5.52", "3.35", "1.3 t"], ORANGE, ORANGE_T),
        ("Current design: target RAAN 125°", ["6.8°", "2.27", "0.09", "6.7 t"], BLUE, BLUE_T),
    ]):
        g = group(s)
        bx = 0.6 + i * 6.2
        box(g.shapes, bx, 5.15, 5.9, 1.72, fill=tint, shape="round", radius=0.1)
        tx(g.shapes, bx + 0.2, 5.22, 5.5, 0.35, [[(lbl, {"bold": True, "color": col})]], size=15,
           font=HEAD)
        labels = ["plane-change angle β", "capture Δv, km/s", "km/s above coplanar (2.18)",
                  f"payload ({'1.3' if i == 0 else '6.6'} rovers)"]
        for k, (v, lab) in enumerate(zip(vals, labels)):
            sx = bx + 0.2 + k * 1.42
            tx(g.shapes, sx, 5.62, 1.4, 0.45, [[(v, {"bold": True})]], size=20, color=INK, font=HEAD)
            tx(g.shapes, sx, 6.08, 1.35, 0.75, lab, size=12, color=MUTED)
        steps.append([g])
    add_builds(s, steps)
    footer(s, 3)
    notes(s, 2, [
        "Takeaway 1: matching the target plane to the arrival geometry is the single biggest lever.",
        ("Earlier (RAAN 15°): β 49.7°, capture 5,524 m/s, 3.35 km/s plane-change penalty; aim B·T 886 km, "
        "B·R 7,258 km."),
        "Current (RAAN 125°): β 6.8°, capture 2,268.5 m/s, 93 m/s above the 2,175.2 m/s coplanar ideal.",
        "Payload per Falcon Heavy: 1.30 t (1.3 rovers) → 6.73 t (6.6 rovers); coplanar ideal 6.98 t (6.8).",
        "Payload comes from a napkin rocket-equation estimate (assumed 320 s Isp, 10% tankage).",
    ])

    # ---------------- 05 RPO ----------------
    d.section("Rendezvous")
    divider(d, 4, "Rendezvous and proximity operations",
            "How does the chaser find the target and dock, on its own, 8 light-minutes from Earth?", 0.2, [
                "Section 5, the centerpiece: about 20 of the ~40 talk minutes.",
                ("Plan: two primers (J2, RSW/HCW), the architecture, the dispersed case, each stage, "
                "results, what broke, and how I'd really dock."),
                "Motivation from section 3: an 8-minute light delay means the chaser must do this onboard.",
            ], lines=2)

    s = d.new("Primer: J2 and nodal regression")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Primer")
    headline(s, "Mars's equatorial bulge makes orbit planes slowly swivel, and the rate depends on altitude")
    j2_schematic(sh)
    steps = []
    for i, (title, body) in enumerate([
        ("What J2 is", ("The oblateness term in Mars's gravity field: the planet is wider at the equator "
                       "than pole to pole.")),
        ("What it does", ("It torques the orbit, so the node (RAAN, Ω) slides around the equator: "
                         "dΩ/dt ∝ −J2 · cos i · a^(−7/2).")),
        ("Why RPO cares", ("Two orbits at different altitudes regress at different rates. Their planes "
                          "drift apart, which shows up as cross-track error at rendezvous.")),
    ]):
        g = group(s)
        card(g.shapes, 7.0, 1.75 + i * 1.7, 5.7, 1.55, title, body, size=14, badge=i + 1,
             badge_color=BLUE)
        steps.append([g])
    add_builds(s, steps)
    footer(s, 4)
    notes(s, 1.2, [
        "Primer; self-contained and safe to delete.",
        "J2 is the planetary oblateness term; it makes orbit planes precess (nodal regression).",
        "dΩ/dt ∝ a^(−7/2): a higher orbit regresses more slowly than a lower one.",
        "For rendezvous that means chaser and target planes drift apart during a long phasing coast.",
        "In our phasing the measured RAAN drift is ~0.5° (0.514° nominal, 0.527° dispersed).",
    ])

    s = d.new("Primer: the RSW frame and HCW guidance")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Primer")
    headline(s, "Describe the chaser relative to the target; close in, that motion is linear and solvable")
    rsw_schematic(sh)
    steps = []
    g = group(s)
    box(g.shapes, 7.0, 1.75, 5.7, 2.35, fill=CARD, shape="round", radius=0.1)
    tx(g.shapes, 7.2, 1.85, 5.3, 0.4, [[("HCW equations (circular target orbit)", {"bold": True})]],
       size=15, font=HEAD)
    tx(g.shapes, 7.2, 2.3, 5.3, 1.1, ["ẍ − 2nẏ − 3n²x = 0", "ÿ + 2nẋ = 0", "z̈ + n²z = 0"], size=17, font=HEAD,
       color=INK, after=2)
    tx(g.shapes, 7.2, 3.42, 5.3, 0.6, "x radial, y along-track, z cross-track; n = target mean motion",
       size=13, color=MUTED)
    steps.append([g])
    g = group(s)
    card(g.shapes, 7.0, 4.25, 5.7, 1.2, "Valid only close in",
         "Linearized about the target: here we treat 15 km as the linear region.", size=14)
    steps.append([g])
    g = group(s)
    card(g.shapes, 7.0, 5.6, 5.7, 1.25, "Two-impulse transfer",
         "Pick a transfer time, solve for the burn that arrives at the target, then burn to stop.",
         size=14)
    steps.append([g])
    add_builds(s, steps)
    footer(s, 4)
    notes(s, 1.2, [
        "Primer; self-contained and safe to delete.",
        "RSW: R radial, S along-track (V-bar), W cross-track. Our plots put +V-bar to the left.",
        "HCW (Hill–Clohessy–Wiltshire): linearized relative motion about a circular target orbit.",
        "Only valid near the target: we treat the 15 km circle as the linear region.",
        "Two-impulse HCW: choose a transfer time (40 min here), solve for Δv₁ to arrive, Δv₂ to stop.",
    ])

    s = d.new("RPO guidance architecture")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous")
    headline(s, "Five-stage impulsive guidance, flown on a J2 truth model")
    stage_txt = [
        ("Plane change", "At the node, optionally pre-biased against J2 nodal drift", "0 / 120.5"),
        ("Hohmann", "Transfer to the target radius", "16.8 / 24.9"),
        ("Phase entry", "Merged with the Hohmann circularization", "(with Hohmann)"),
        ("Closed-loop phasing", "20 revolutions, re-planned every rev from the actual state",
         "9.7 / 9.8"),
        ("Exit + HCW", "Phase exit, then two-impulse HCW with one retarget",
         "18.3 + 25.1 / 15.8 + 14.9"),
    ]
    steps = []
    for i, (name, desc, dv) in enumerate(stage_txt):
        g = group(s)
        bx = 0.6 + i * 2.48
        box(g.shapes, bx, 1.7, 2.24, 2.75, fill=CARD, shape="round", radius=0.1)
        circle(g.shapes, bx + 0.38, 2.08, 0.2, fill=BLUE)
        tx(g.shapes, bx + 0.18, 1.88, 0.4, 0.4, [[(str(i + 1), {"bold": True})]], size=14,
           color=WHITE, font=HEAD, align="c", anchor="m")
        tx(g.shapes, bx + 0.18, 2.4, 1.95, 0.65, [[(name, {"bold": True})]], size=15, font=HEAD)
        tx(g.shapes, bx + 0.18, 3.0, 1.95, 0.9, desc, size=13, color=INK2)
        box(g.shapes, bx + 0.12, 3.85, 2.0, 0.48, fill=WHITE, shape="round", radius=0.08,
            line=LINE, margin=0.04, text=[[(dv, {"bold": True})]], size=12, color=BLUE_D,
            font=BODY, align="c")
        if i < 4:
            arrow(g.shapes, bx + 2.27, 3.07, bx + 2.45, 3.07, w=2.0)
        steps.append([g])
    tx(sh, 0.6, 4.55, 12.1, 0.3, "Δv in m/s, nominal / dispersed", size=12, color=MUTED, align="r")
    box(sh, 0.6, 4.95, 12.1, 0.7, fill=BLUE_T, shape="round", radius=0.1, margin=0.2,
        text=[[("One propagated truth: ", {"bold": True, "color": BLUE_D}),
               (("every coast continues from the last state; every burn is planned from current "
                "osculating elements."), {})]], size=16, color=INK, anchor="m")
    tx(sh, 0.6, 5.85, 5.9, 1.0, [
        ("**Setup:** target in 300 km / 45° / 125°, 13° true anomaly at capture; chaser starts at the "
        "post-capture state; phase gap 107.9°"),
    ], size=14, color=INK2)
    tx(sh, 6.8, 5.85, 5.9, 1.0, [
        ("**Settings:** 20 phasing revs · HCW handoff threshold 20 km · HCW transfer 40 min · docking "
        "threshold 50 m · truth: two-body + Mars J2"),
    ], size=14, color=INK2)
    add_builds(s, steps)
    footer(s, 4)
    notes(s, 1.6, [
        "Inputs come straight from the arrival file: capture epoch, post-capture state, target orbit.",
        ("Stages build on click: plane change, Hohmann, phase entry (merged with circularization), "
        "closed-loop phasing, phase exit + HCW."),
        "Truth model: two-body + Mars J2. Every coast continues from the last propagated state.",
        ("Settings: 20 phasing revolutions, HCW handoff threshold 20 km, HCW transfer 40 min, docking "
        "threshold 50 m."),
        ("Two cases: nominal (perfect capture) and dispersed (capture errors). Next slide: what "
        "'dispersed' means."),
    ])

    s = d.new("The dispersed case: one seeded draw, not a Monte Carlo")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous")
    headline(s, "Capture errors are drawn once, within fixed bounds, from a fixed and reproducible random seed")
    rows = [
        ("Semi-major axis a", -100, 100, 73, "+73 km", "±100 km"),
        ("Eccentricity e", 0, 0.01, 0.0086, "0.0086", "0 to 0.01"),
        ("Inclination i", -3, 3, 1.87, "+1.87°", "±3°"),
        ("RAAN", -3, 3, -1.43, "−1.43°", "±3°"),
        ("Arg. of periapsis", -3, 3, -2.5, "−2.5°", "±3°"),
    ]
    tx(sh, 2.75, 1.68, 5.0, 0.3, "drawn value within its bound", size=12, color=MUTED)
    for k, (name, lo, hi, v, vs, bnd) in enumerate(rows):
        y0 = 2.05 + k * 0.92
        tx(sh, 0.6, y0, 2.1, 0.45, [[(name, {"bold": True})]], size=15, color=INK, anchor="m")
        x0, wt = 2.75, 4.6
        box(sh, x0, y0 + 0.12, wt, 0.22, fill="E3E6EA", shape="round", radius=0.11)
        if lo < 0:
            zx = x0 + wt * (0 - lo) / (hi - lo)
            poly(sh, [(zx, y0 + 0.02), (zx, y0 + 0.44)], color=MUTED, w=1.25)
        vx = x0 + wt * (v - lo) / (hi - lo)
        circle(sh, vx, y0 + 0.23, 0.14, fill=ORANGE, line=WHITE, lw=1.5)
        tx(sh, vx - 0.6, y0 + 0.42, 1.2, 0.3, [[(vs, {"bold": True})]], size=13, color=ORANGE,
           align="c")
        tx(sh, x0 + wt + 0.15, y0, 1.0, 0.45, bnd, size=13, color=MUTED, anchor="m")
    g = group(s)
    card(g.shapes, 8.85, 1.75, 3.85, 1.55, "Nominal", "Exact capture: chaser coplanar with the "
                                                      "target, same radius", fill=BLUE_T, size=14)
    card(g.shapes, 8.85, 3.45, 3.85, 1.75, "Dispersed", "Needs a 120.5 m/s plane change; RPO total "
                                                        "186.0 vs 69.8 m/s", fill=ORANGE_T, size=14)
    card(g.shapes, 8.85, 5.35, 3.85, 1.5, "Read with care", "One sample, not a distribution. "
                                                           "Monte Carlo is future work.", size=14)
    add_builds(s, [[g]])
    footer(s, 4)
    notes(s, 1.2, [
        "Dispersion bounds: ±100 km in a, e between 0 and 0.01, ±3° in i, RAAN and argument of periapsis.",
        "One example drawn with a fixed, reproducible random seed; it is not a Monte Carlo.",
        "Resulting offsets: a +73 km, e 0.0086, i +1.87°, RAAN −1.43°, aop −2.5°.",
        "Nominal: coplanar, same radius. Dispersed needs a 120.5 m/s plane change; RPO 186.0 vs 69.8 m/s.",
        "No navigation errors or finite burns are modeled in either case.",
    ])

    # Stage 1
    s = d.new("Stage 1: plane change at the node")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Guidance stage")
    stage_stepper(s, [0])
    headline(s, "Fix the plane first, and pre-bias it against J2 drift only when that is cheaper")
    steps = []
    g = group(s)
    box(g.shapes, 0.6, 1.75, 5.6, 0.75, fill=CARD, shape="round", radius=0.1, margin=0.15,
        text="Burn at the node line between the chaser and target planes", size=15, anchor="m")
    arrow(g.shapes, 1.9, 2.52, 1.9, 2.82, w=2.0)
    arrow(g.shapes, 4.9, 2.52, 4.9, 2.82, w=2.0)
    box(g.shapes, 0.6, 2.85, 2.65, 1.25, fill=WHITE, line=LINE, shape="round", radius=0.1,
        margin=0.12, text=[[("Plan A", {"bold": True}), (": rotate onto the target plane", {})]],
        size=14, anchor="m")
    box(g.shapes, 3.55, 2.85, 2.65, 1.25, fill=WHITE, line=LINE, shape="round", radius=0.1,
        margin=0.12, text=[[("Plan B", {"bold": True}),
                            (": rotate onto the target plane offset by the measured J2 RAAN drift",
                             {})]], size=14, anchor="m")
    arrow(g.shapes, 1.9, 4.12, 1.9, 4.42, w=2.0)
    arrow(g.shapes, 4.9, 4.12, 4.9, 4.42, w=2.0)
    box(g.shapes, 0.6, 4.45, 5.6, 0.75, fill=CARD, shape="round", radius=0.1, margin=0.15,
        text="Fly both through the full RPO on the J2 truth model", size=15, anchor="m")
    arrow(g.shapes, 3.4, 5.22, 3.4, 5.52, w=2.0)
    box(g.shapes, 0.6, 5.55, 5.6, 0.75, fill=BLUE_T, shape="round", radius=0.1, margin=0.15,
        text=[[("Keep the cheaper plan", {"bold": True, "color": BLUE_D})]], size=15, anchor="m")
    steps.append([g])
    simple_table(sh, 6.6, 1.75, [2.2, 1.95, 1.95], [
        ["", "Nominal", "Dispersed"],
        ["Plane-change Δv", "0 (not needed)", "120.5 m/s"],
        ["Pre-bias effect", "costs 31.6 m/s", "saves 50.4 m/s (236.4 → 186.0)"],
        ["Decision", "not used", "used"],
        ["RAAN drift in phasing, measured / analytic", "0.514° / 0.438°", "0.527° / 0.399°"],
    ], size=14, row_h=0.62, bold_first_col=True, highlight={3: "E7F0FB"})
    g2 = group(s)
    tx(g2.shapes, 6.6, 5.1, 6.1, 1.75, bullets([
        "Dispersed Burn 1: **120.5 m/s** at 0.24 h after capture",
        "Nominal needs none: direct injection already put the chaser in the target plane",
        "The pre-bias is not always a win, so the planner checks both",
    ], size=15, after=6))
    steps.append([g2])
    add_builds(s, steps)
    footer(s, 4)
    notes(s, 2.0, [
        ("Plane change at the node between chaser and target planes; done first because it is cheapest "
        "to fix the plane early, before phasing."),
        ("Planner runs the RPO once to measure J2 RAAN drift during phasing, re-runs with Burn 1 pre-biased "
        "by that drift, and keeps the cheaper."),
        ("Dispersed: pre-bias saves 50.4 m/s (236.4 → 186.0 m/s) → used. Nominal: it would cost 31.6 m/s "
        "→ not used."),
        ("RAAN drift during phasing measured / analytic: nominal 0.514° / 0.438°, dispersed 0.527° / 0.399°. "
        "The first-order formula only covers the phasing leg."),
        "Burn 1 (dispersed) 120.5 m/s at 0.24 h, 2035-12-25 00:12:57 UTC.",
    ])

    # Stage 2-3
    s = d.new("Stages 2–3: Hohmann transfer and phase entry")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Guidance stage")
    stage_stepper(s, [1, 2])
    headline(s, "Two burns fix the orbit size and start phasing: 16.8 m/s nominal, 24.9 m/s dispersed")
    cx0, cy0 = 2.35, 4.25
    circle(sh, cx0, cy0, 0.55, fill=MARS)
    circle(sh, cx0, cy0, 1.25, line=MUTED, lw=1.5, dash=MSO_LINE_DASH_STYLE.DASH)
    circle(sh, cx0, cy0, 1.55, line=ORANGE, lw=1.5)
    ell = []
    a_e, b_e = 1.4, math.sqrt(1.55 * 1.25)
    for k in range(91):
        t = math.radians(180 * k / 90)
        ell.append((cx0 + 0.15 + a_e * math.cos(t), cy0 - b_e * math.sin(t)))
    poly(sh, ell, color=BLUE, w=3.0, head=True)
    for bx_, num in ((cx0 + 1.55, "2"), (cx0 - 1.25, "3")):
        circle(sh, bx_, cy0, 0.19, fill=INK)
        tx(sh, bx_ - 0.19, cy0 - 0.19, 0.38, 0.38, [[(num, {"bold": True})]], size=12, color=WHITE,
           font=HEAD, align="c", anchor="m")
    tx(sh, 0.6, 6.0, 3.6, 0.6, [[("chaser orbit (outer)", {"color": ORANGE, "bold": True})],
                               [("target radius (dashed)", {"color": MUTED, "bold": True})]], size=12,
       after=0)
    tx(sh, 0.6, 6.55, 4.0, 0.3, "Schematic, not to scale", size=11, color=MUTED, italic=True)
    g = group(s)
    cols3 = [("Burn 2: Hohmann", "14.4 m/s at 0.35 h (dispersed)"),
             ("Burn 3: circularize + phase entry", "10.5 m/s at 1.32 h (dispersed); two jobs, one burn"),
             ("Stage total", "16.8 m/s nominal · 24.9 m/s dispersed")]
    for k, (t, b) in enumerate(cols3):
        bx = 4.5 + k * 2.8
        box(g.shapes, bx, 1.72, 2.6, 1.5, fill=BLUE_T if k == 2 else CARD, shape="round", radius=0.1)
        tx(g.shapes, bx + 0.15, 1.8, 2.3, 1.4, [{"t": [(t, {"bold": True})], "size": 15, "after": 3},
                                                {"t": b, "size": 14, "color": INK2}])
    add_builds(s, [[g]])
    fig(sh, "figures/rpo/rsw_3d.png", 4.5, 3.35, 8.2, 3.2, ha="c", va="t",
        alt="3D view of the chaser relative to the target in RSW with key burns")
    tx(sh, 4.5, 6.55, 8.2, 0.3, "Dispersed case in the target's RSW frame, key burns marked",
       size=12, color=MUTED, align="c")
    footer(s, 4)
    notes(s, 1.2, [
        "Dispersed chaser starts a +73 km, e 0.0086 off the target orbit, so first fix the size.",
        "Burn 2 (Hohmann) 14.4 m/s at 0.35 h, 2035-12-25 00:19:45 UTC.",
        "Burn 3 (circularize + phase entry) 10.5 m/s at 1.32 h, 01:17:43 UTC: merging the two saves a burn.",
        "Stage totals: 16.8 m/s nominal, 24.9 m/s dispersed.",
        "Figure: the dispersed RPO in 3D RSW with key burns.",
    ])

    # Stage 4
    s = d.new("Stage 4: closed-loop phasing")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Guidance stage")
    stage_stepper(s, [3])
    headline(s, "Re-plan every revolution from the actual state: 20 small burns close the phase gap for ~10 m/s")
    fig(sh, "figures/rpo/separation.png", 0.6, 1.62, 12.1, 3.2,
        alt="Chaser-to-target separation versus time with shaded phases and burn callouts")
    loop = [("Propagate", "one rev on the J2 truth"), ("Measure", "phase error from the actual state"),
            ("Re-solve", "the phasing orbit (vis-viva)"), ("Burn", "small correction")]
    g = group(s)
    for i, (t1, t2) in enumerate(loop):
        bx = 0.6 + i * 1.95
        box(g.shapes, bx, 5.0, 1.7, 1.0, fill=AQUA_T, shape="round", radius=0.1, margin=0.08,
            text=[{"t": [(t1, {"bold": True})], "size": 14, "after": 0},
                  {"t": t2, "size": 12, "color": INK2}], align="c", anchor="m")
        if i < 3:
            arrow(g.shapes, bx + 1.72, 5.5, bx + 1.93, 5.5, w=2.0)
    poly(g.shapes, [(6.15, 6.02), (6.15, 6.35), (1.45, 6.35), (1.45, 6.04)], color=AQUA, w=2.0,
         head=True)
    tx(g.shapes, 2.6, 6.38, 2.6, 0.3, "next revolution (×20)", size=12, color=MUTED, align="c")
    g2 = group(s)
    tx(g2.shapes, 8.6, 5.0, 4.1, 1.9, bullets([
        "Phase gap at capture: **107.9°**",
        "Burns 4.0–4.19 from 3.2 to 39.7 h",
        "**9.7 m/s** nominal · **9.8 m/s** dispersed",
        "~35 of the ~43 RPO hours",
    ], size=15, after=5))
    add_builds(s, [[g], [g2]])
    footer(s, 4)
    notes(s, 2.0, [
        "Figure: dispersed case, separation vs time with phases shaded and burn callouts.",
        ("Closed loop: each revolution, propagate on the J2 truth, measure the phase error from the actual "
        "osculating state, re-solve the phasing orbit with vis-viva, burn."),
        "Open-loop phasing failed under J2 in development (later slide); re-planning fixed it.",
        "20 revolutions; burns 4.0–4.19 run 3.2–39.7 h after capture.",
        "Cheap: 9.7 m/s nominal, 9.8 m/s dispersed. Time-dominant: ~35 of ~43 h.",
    ])

    # Stage 5
    s = d.new("Stage 5: phase exit and HCW terminal rendezvous")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Guidance stage")
    stage_stepper(s, [4])
    headline(s, "Inside the 20 km handoff, two 40-minute HCW legs take the chaser from 18 km to a 7 m stop")
    fig(sh, "figures/rpo/rsw_terminal.png", 0.6, 1.62, 7.7, 2.65, ha="l",
        alt="Terminal approach in RSW: phase exit and HCW rendezvous versus 15 km linear region")
    tx(sh, 0.6, 4.3, 7.7, 0.3, "Dispersed case; dotted circle = 15 km linear region", size=12,
       color=MUTED)
    simple_table(sh, 8.6, 1.7, [1.7, 1.2, 1.2], [
        ["Δv", "Nominal", "Dispersed"],
        ["Phase exit", "18.3 m/s", "15.8 m/s"],
        ["HCW terminal", "25.1 m/s", "14.9 m/s"],
        ["Final miss", "13.5 m", "7.3 m"],
    ], size=14, row_h=0.42, bold_first_col=True)
    tx(sh, 8.6, 3.5, 4.1, 0.9, "Settings: handoff threshold 20 km · each HCW leg 40 min · docking "
                               "threshold 50 m", size=13, color=MUTED)
    # event timeline (dispersed)
    g = group(s)
    ev = [("41.62 h", "Phase exit + HCW start", "15.8 + 6.5 m/s", "18.2 km"),
          ("42.29 h", "HCW retarget", "8.2 m/s", "290 m"),
          ("42.95 h", "HCW stop", "0.2 m/s", "7.3 m")]
    poly(g.shapes, [(1.2, 5.05), (7.7, 5.05)], color=LINE, w=3.0)
    for i, (t, name, dv, rng_) in enumerate(ev):
        ex = 1.2 + i * 3.25
        circle(g.shapes, ex, 5.05, 0.13, fill=BLUE)
        tx(g.shapes, ex - 0.6, 4.62, 2.6, 0.3, [[(t, {"bold": True})]], size=13, color=BLUE_D)
        tx(g.shapes, ex - 0.6, 5.25, 2.9, 1.0, [{"t": [(name, {"bold": True})], "size": 14, "after": 0},
                                                 {"t": f"{dv} · range {rng_}", "size": 13,
                                                  "color": INK2}])
        if i < 2:
            tx(g.shapes, ex + 0.9, 4.72, 1.6, 0.3, "40 min", size=12, color=MUTED, align="c")
    warn = group(s)
    box(warn.shapes, 0.6, 6.2, 12.1, 0.65, fill=ORANGE_T, shape="round", radius=0.1, margin=0.18,
        text=[[("Open items: ", {"bold": True, "color": ORANGE}),
               (("the 18.2 km handoff is outside the 15 km linear region, and the path crosses the "
                "200 m keep-out."), {})]], size=15, color=INK, anchor="m")
    add_builds(s, [[g], [warn]])
    footer(s, 4)
    notes(s, 2.0, [
        ("Phase exit when inside the 20 km HCW handoff threshold; then two-impulse HCW with a 40-minute "
        "transfer, retargeting once."),
        "Dispersed: HCW starts at 18.2 km, retargets at 290 m, stops at 7.3 m.",
        ("Burns: phase exit 15.8 m/s and HCW start 6.5 m/s at 41.62 h (17:36:00 UTC); retarget 8.2 m/s at "
        "42.29 h; stop 0.2 m/s at 42.95 h (18:56:00 UTC)."),
        "Nominal: phase exit 18.3, HCW 25.1 m/s, miss 13.5 m. Dispersed: 15.8, 14.9 m/s, miss 7.3 m.",
        ("Be upfront: the dispersed handoff at 18.2 km is outside the 15 km linear region; the path also "
        "crosses the 200 m keep-out."),
    ])

    # Results
    s = d.new("RPO results in the target's frame")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous")
    headline(s, "Both cases dock within 7–13 m, about 42 hours after capture")
    fig(sh, "figures/rpo/rsw_full.png", 0.6, 1.62, 7.6, 2.6, ha="l",
        alt="Full RPO in the target's RSW frame with Mars surface for context")
    fig(sh, "figures/rpo/rsw_close.png", 0.6, 4.3, 7.6, 2.6, ha="l",
        alt="Final approach inside 1 km against the 200 m keep-out")
    for i, (name, vals, tint, col) in enumerate([
        ("Nominal", ["69.8 m/s", "25", "41.6 h", "13.5 m"], BLUE_T, BLUE_D),
        ("Dispersed", ["186.0 m/s", "27", "43.0 h", "7.3 m"], ORANGE_T, ORANGE),
    ]):
        y0 = 1.7 + i * 1.75
        box(sh, 8.5, y0, 4.2, 1.55, fill=tint, shape="round", radius=0.1)
        tx(sh, 8.7, y0 + 0.1, 3.8, 0.35, [[(name, {"bold": True, "color": col})]], size=15, font=HEAD)
        for k, (v, lab) in enumerate(zip(vals, ["total Δv", "burns", "to dock", "final miss"])):
            sx = 8.7 + (k % 2) * 1.95
            sy = y0 + 0.45 + (k // 2) * 0.52
            tx(sh, sx, sy, 1.95, 0.5, [[(v, {"bold": True}), ("  " + lab, {"color": MUTED, "size": 12})]],
               size=17, color=INK, font=HEAD)
    tx(sh, 8.5, 5.3, 4.2, 1.6, bullets([
        "Top: the dip “toward Mars” is curved-frame geometry, not a descent",
        "Bottom: the final approach passes inside the 200 m keep-out",
    ], size=14, after=6))
    tx(sh, 0.6, 6.92, 7.6, 0.1, "", size=4)
    footer(s, 4)
    notes(s, 1.5, [
        "RSW frame: along-track horizontal (+V-bar to the left), radial up. Figures show the dispersed case.",
        ("Top: the whole RPO with the Mars surface for context. The curve dipping toward Mars is real "
        "Cartesian-RSW geometry of a rotating frame, not a descent."),
        "Bottom: final 1 km against the 200 m keep-out. The path does cross it (no hold point yet).",
        "Nominal: 69.8 m/s, 25 burns, 41.6 h, 13.5 m miss. Dispersed: 186.0 m/s, 27 burns, 43.0 h, 7.3 m.",
    ])

    # Log range
    s = d.new("Closing the distance: range on a log scale")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous")
    headline(s, "Phasing closes thousands of kilometers slowly; HCW closes the last ~18 km in 1.3 hours")
    fig(sh, "figures/summary/rpo_log_range.png", 0.6, 1.62, 12.1, 3.63,
        alt="Chaser-target range on a log scale for both cases with 15 km and 200 m reference lines")
    steps = []
    for i, (t, b, tint) in enumerate([
        ("15 km line", "HCW linear region. Above it, phasing does the work.", CARD),
        ("200 m line", "Keep-out. Both trajectories cross it: HCW aims straight at the target.", ORANGE_T),
        ("Fix", "A hold point at the keep-out boundary, then a slow final approach. Future work.", BLUE_T),
    ]):
        g = group(s)
        card(g.shapes, 0.6 + i * 4.1, 5.4, 3.9, 1.45, t, b, fill=tint, size=14)
        steps.append([g])
    add_builds(s, steps)
    footer(s, 4)
    notes(s, 1, [
        "Log range makes all scales visible: thousands of km during phasing down to metres.",
        "Point out the 15 km (HCW linear region) and 200 m (keep-out) reference lines.",
        "HCW closes the last ~18 km in 1.3 hours.",
        ("Both trajectories cross the 200 m keep-out because HCW aims straight at the target; a hold "
        "point is future work."),
        "Final miss: 13.5 m nominal, 7.3 m dispersed.",
    ])

    # Δv by phase
    s = d.new("Where the RPO Δv goes")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous")
    headline(s, "Capture dispersions cost a 120 m/s plane change; everything else is small")
    fig(sh, "figures/summary/rpo_by_phase.png", 0.6, 1.62, 12.1, 3.75,
        alt="RPO delta-v by phase, nominal versus dispersed")
    simple_table(sh, 0.6, 5.55, [1.9, 1.7, 2.2, 1.5, 1.6, 1.6, 1.6], [
        ["m/s", "Plane change", "Hohmann + entry", "Phasing", "Phase exit", "HCW", "Total"],
        ["Nominal", "0", "16.8", "9.7", "18.3", "25.1", "69.8"],
        ["Dispersed", "120.5", "24.9", "9.8", "15.8", "14.9", "186.0"],
    ], size=14, row_h=0.42, bold_first_col=True, aligns=["l", "r", "r", "r", "r", "r", "r"])
    footer(s, 4)
    notes(s, 1, [
        ("Dispersed breakdown: plane change 120.5 (64.8%), Hohmann + entry 24.9, phasing 9.8, phase exit "
        "15.8, HCW 14.9 m/s."),
        "Phasing itself is cheap: ~10 m/s over 20 revs.",
        "Nominal needs no plane change because direct injection already put the chaser in the target plane.",
        "Takeaway 3: RPO is small (70–186 m/s) next to capture, but it is where the precision lives.",
    ])

    # What broke
    s = d.new("How I decided: what broke, and how I fixed it")
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
    steps = []
    for i, (t, b, miss, dv, mv) in enumerate(dev):
        g = group(s)
        bx = 0.6 + i * 3.12
        diag = miss is None
        box(g.shapes, bx, 2.2, 2.82, 2.6, fill=MARS_T if diag else CARD, shape="round", radius=0.1)
        tx(g.shapes, bx + 0.2, 2.3, 2.5, 0.4, [[(t, {"bold": True})]], size=16, font=HEAD,
           color=MARS if diag else INK)
        tx(g.shapes, bx + 0.2, 2.72, 2.5, 1.05, b, size=14, color=INK2)
        if diag:
            tx(g.shapes, bx + 0.2, 3.75, 2.5, 1.0, "Phasing orbit ~16 km higher → node regresses more "
                                                  "slowly (dΩ/dt ∝ a^(−7/2))", size=13, color=INK2)
        else:
            box(g.shapes, bx + 0.2, 3.95, 2.35 * mv / 51.5, 0.18, fill=ORANGE, shape="rect")
            tx(g.shapes, bx + 0.2, 4.18, 2.5, 0.55, [[(miss, {"bold": True}), ("  ·  " + dv, {})]],
               size=15, color=INK, font=HEAD)
            tx(g.shapes, bx + 0.2, 3.7, 2.5, 0.25, "handoff miss", size=11, color=MUTED)
        if i < 3:
            arrow(g.shapes, bx + 2.86, 3.5, bx + 3.08, 3.5, w=2.0)
        steps.append([g])
    g = group(s)
    box(g.shapes, 0.6, 5.0, 4.6, 1.85, fill=CARD, shape="round", radius=0.1)
    tx(g.shapes, 0.8, 5.1, 4.2, 0.35, [[("Tried and dropped", {"bold": True})]], size=15, font=HEAD)
    tx(g.shapes, 0.8, 5.5, 4.2, 1.3, "Hohmann shape corrections during phasing: ~2% saving for 3–18 "
                                     "extra burns and 6–30 h, and sometimes worse", size=14, color=INK2)
    steps.append([g])
    g = group(s)
    box(g.shapes, 5.45, 5.0, 7.25, 1.85, fill=BLUE_T, shape="round", radius=0.1, line=BLUE, lw=1.25)
    tx(g.shapes, 5.65, 5.1, 6.9, 0.35, [[("Current design (this mission)", {"bold": True,
                                                                            "color": BLUE_D})]],
       size=15, font=HEAD)
    tx(g.shapes, 5.65, 5.5, 6.9, 1.3, bullets([
        "Pre-bias **saves 50.4 m/s** in the dispersed case (236.4 → 186.0 m/s)",
        "In the nominal case it would **cost 31.6 m/s**, so the planner picks the cheaper plan",
    ], size=14, after=4, bcolor=BLUE))
    steps.append([g])
    add_builds(s, steps)
    footer(s, 4)
    notes(s, 2, [
        ("Development history on an earlier test geometry (target RAAN 15°, synthetic chaser offset). Do "
        "not compare these Δv numbers with the 69.8 / 186.0 m/s results."),
        "Open-loop incremental phasing under J2: 51.5 km handoff miss, 202.1 m/s total.",
        "Closed loop (re-plan each rev from the actual state): 12.2 km, 164.9 m/s.",
        ("Root cause of the residual: ≈9.7 km cross-track from J2 differential nodal regression. The "
        "phasing orbit sits ~16 km higher, so its node regresses more slowly (dΩ/dt ∝ a^(−7/2))."),
        "Pre-bias the plane change by the predicted drift: 5.5 km, 152.2 m/s.",
        "Hohmann shape corrections: ~2% saving for 3–18 extra burns and 6–30 h; sometimes worse; dropped.",
        ("Current design: measured drift 0.53° vs analytic 0.40° in the dispersed case; the first-order "
        "formula only covers the phasing leg."),
    ])

    # How I'd actually dock
    s = d.new("How I would actually dock")
    sh = s.shapes
    eyebrow(s, "05 · Rendezvous  ·  Proposed")
    headline(s, "Next design: hand off inside the linear region, hold at the keep-out, then close slowly on sensors")
    tag(sh, 0.6, 1.62, "Proposed · not implemented", fill=ORANGE_T, color=ORANGE, w=2.9)
    # RSW schematic: target at centre, keep-out circle, hold point on V-bar, approach corridor
    ox, oy = 3.9, 4.45
    circle(sh, ox, oy, 2.15, line="9AA1AC", lw=1.25, dash=MSO_LINE_DASH_STYLE.ROUND_DOT)
    circle(sh, ox, oy, 0.85, line=MARS, lw=1.75, dash=MSO_LINE_DASH_STYLE.DASH)
    poly(sh, [(ox, oy), (ox - 0.85, oy - 0.3), (ox - 0.85, oy + 0.3)], color=None, fill="C9DDF5",
         closed=True)
    poly(sh, [(0.75, oy - 1.25), (ox - 2.15 * 0.8, oy - 1.0), (ox - 1.2, oy - 0.25), (ox - 0.85, oy)],
         color=MUTED, w=1.75, dash=MSO_LINE_DASH_STYLE.DASH, head=True)
    circle(sh, ox - 2.15 * 0.8, oy - 1.0, 0.09, fill=INK)
    circle(sh, ox - 0.85, oy, 0.12, fill=ORANGE)
    arrow(sh, ox - 0.72, oy, ox - 0.12, oy, color=BLUE, w=2.5)
    circle(sh, ox, oy, 0.1, fill=INK)
    tx(sh, ox - 2.9, oy - 1.65, 2.2, 0.55, [[("1 ", {"bold": True}), ("handoff inside 15 km", {})]],
       size=13, color=INK2)
    tx(sh, ox - 1.95, oy + 0.15, 1.6, 0.55, [[("2 ", {"bold": True}), ("hold point", {})]], size=13,
       color=ORANGE)
    tx(sh, ox - 0.55, oy + 0.95, 2.6, 0.35, [[("200 m keep-out", {})]], size=13, color=MARS)
    tx(sh, ox - 1.1, oy + 2.2, 2.2, 0.35, "15 km linear region", size=13, color=MUTED, align="c")
    tx(sh, ox - 0.95, oy - 1.25, 2.2, 0.35, [[("3 ", {"bold": True}), ("V-bar corridor", {})]],
       size=13, color=BLUE_D)
    tx(sh, 0.6, 6.55, 6.0, 0.3, "Schematic, not to scale", size=11, color=MUTED, italic=True)
    items = [
        ("Hand off to HCW inside the linear region", "so the HCW model holds"),
        ("Hold point at the 200 m keep-out boundary", "a safe place to stop and check"),
        ("Slow final approach along V-bar or R-bar", "inside an approach corridor"),
        ("Relative navigation", "sensors plus a filter, instead of perfect knowledge"),
        ("Closed-loop control", "for example MPC, instead of open impulsive legs"),
        ("Contact, capture, and abort logic", "back out to the hold point on a fault"),
    ]
    steps = []
    for i, (t, b) in enumerate(items):
        g = group(s)
        y0 = 1.65 + i * 0.86
        circle(g.shapes, 7.45, y0 + 0.33, 0.2, fill=BLUE if i < 3 else AQUA)
        tx(g.shapes, 7.25, y0 + 0.13, 0.4, 0.4, [[(str(i + 1), {"bold": True})]], size=13,
           color=WHITE, font=HEAD, align="c", anchor="m")
        tx(g.shapes, 7.85, y0 + 0.02, 4.85, 0.8, [{"t": [(t, {"bold": True})], "size": 15, "after": 0},
                                                  {"t": b, "size": 13, "color": MUTED}])
        steps.append([g])
    add_builds(s, [steps[0] + steps[1] + steps[2], steps[3] + steps[4] + steps[5]])
    footer(s, 4)
    notes(s, 1.5, [
        "Clearly a proposal: none of this is implemented yet.",
        ("Fix the two open items: hand off to HCW inside the 15 km linear region; stop at a hold point on "
        "the 200 m keep-out boundary."),
        "Then a slow final approach along V-bar or R-bar inside a corridor.",
        ("Second click: the GNC that real docking needs: relative navigation (sensors + filter), "
        "closed-loop control such as MPC, and contact/capture plus abort logic."),
        "These map to the RPO half of the roadmap.",
    ])

    # ---------------- 06 Mission summary ----------------
    d.section("Mission summary")
    divider(d, 5, "Mission summary", "What does it all cost, what did I learn, and what comes next?",
            0.2, ["Section 6 (~5 min): Δv budget, payload, limitations, lessons and roadmap.",
                  ("Bring the three takeaways back: target plane, closed-loop J2-aware guidance, "
                  "~2.5 km/s spacecraft budget."),
                  "Costs live in backup only."])

    s = d.new("Mission Δv budget")
    sh = s.shapes
    eyebrow(s, "06 · Mission summary")
    headline(s, "The spacecraft needs 2.47–2.59 km/s; the launcher supplies 11.47 km/s (ideal)")
    fig(sh, "figures/summary/dv_budget.png", 0.6, 1.6, 12.1, 3.0,
        alt="Full mission delta-v by maneuver and vehicle")
    g = group(s)
    tx(g.shapes, 0.6, 4.72, 6.0, 0.35, [[("Spacecraft budget, step by step", {"bold": True})]],
       size=15, font=HEAD)
    tx(g.shapes, 6.2, 4.72, 6.5, 0.35, [[("92%", {"bold": True, "color": BLUE}),
                                         (" of the spacecraft budget is Mars capture", {})]],
       size=15, color=INK2, align="r")
    x0, wmax = 2.4, 10.3
    scale = wmax / 2586.3
    rows = [("Midcourse", 0, 131.8, "132 m/s", BLUE_L), ("Mars capture", 131.8, 2268.5, "2,269 m/s", BLUE),
            ("RPO", 131.8 + 2268.5, 186.0, "70 (nominal) to 186 (dispersed) m/s", BLUE_D)]
    for k, (lab, start_, val, vs, col) in enumerate(rows):
        y0 = 5.15 + k * 0.4
        tx(g.shapes, 0.6, y0, 1.7, 0.32, lab, size=14, color=INK2, anchor="m")
        if lab == "RPO":
            box(g.shapes, x0 + start_ * scale, y0 + 0.04, 69.8 * scale, 0.26, fill=col)
            box(g.shapes, x0 + (start_ + 69.8) * scale, y0 + 0.04, (186.0 - 69.8) * scale, 0.26,
                fill="B9C9DE")
            tx(g.shapes, x0 + start_ * scale - 4.1, y0, 4.0, 0.32, [[(vs, {"bold": True})]], size=13,
               color=INK, anchor="m", align="r")
        elif lab == "Mars capture":
            box(g.shapes, x0 + start_ * scale, y0 + 0.04, val * scale, 0.26, fill=col)
            tx(g.shapes, x0 + start_ * scale + 0.1, y0, 3.0, 0.32, [[(vs, {"bold": True})]], size=13,
               color=WHITE, anchor="m")
        else:
            box(g.shapes, x0 + start_ * scale, y0 + 0.04, val * scale, 0.26, fill=col)
            tx(g.shapes, x0 + (start_ + val) * scale + 0.12, y0, 3.0, 0.32, [[(vs, {"bold": True})]],
               size=13, color=INK, anchor="m")
    yt = 5.15 + 3 * 0.4
    tx(g.shapes, 0.6, yt, 1.7, 0.32, [[("Total", {"bold": True})]], size=14, anchor="m")
    box(g.shapes, x0, yt + 0.04, 2470.2 * scale, 0.26, fill=INK)
    box(g.shapes, x0 + 2470.2 * scale, yt + 0.04, (2586.3 - 2470.2) * scale, 0.26, fill="8A919C")
    tx(g.shapes, x0 + 0.1, yt, 4.0, 0.32, [[("2.47 (nominal) to 2.59 (dispersed) km/s", {"bold": True})]],
       size=13, color=WHITE, anchor="m")
    add_builds(s, [[g]])
    footer(s, 5)
    notes(s, 1.2, [
        ("Launch vehicle (ideal): ascent 7,790.7 + escape 3,675.6 = 11,466.3 m/s. Ideal lower bounds, "
        "not real launch costs."),
        "Spacecraft: midcourse 131.8 + capture 2,268.5 + RPO 69.8 / 186.0 = 2,470.2 / 2,586.3 m/s.",
        "Mars capture is 92% of the spacecraft budget.",
        "All-up: 13,936.5 m/s nominal, 14,052.6 m/s dispersed.",
        ("Green bar in the figure: Mars surface ↔ orbit, 3,531.5 m/s ideal, a reference for a future "
        "lander and not in the totals."),
    ])

    s = d.new("What one Falcon Heavy can deliver")
    sh = s.shapes
    eyebrow(s, "06 · Mission summary")
    headline(s, "About 6.7 t to the target orbit: roughly 6.6 Perseverance-class rovers")
    fig(sh, "figures/summary/mass_history.png", 0.6, 1.6, 12.1, 3.5,
        alt="Spacecraft mass after each burn, current versus coplanar capture")
    chain = [("16.8 t", "injected toward Mars by a Falcon Heavy (expendable)"),
             ("9.2 t", "spacecraft propellant burned (Isp 320 s, assumed)"),
             ("6.73 t", "useful payload after tanks and engines (10% of propellant, assumed)"),
             ("6.6", "Perseverance-class rovers (1,025 kg each)")]
    g = group(s)
    for i, (v, lab) in enumerate(chain):
        bx = 0.6 + i * 3.12
        box(g.shapes, bx, 5.25, 2.82, 1.2, fill=BLUE_T if i >= 2 else CARD, shape="round", radius=0.1)
        tx(g.shapes, bx + 0.18, 5.3, 2.5, 0.5, [[(v, {"bold": True})]], size=22, font=HEAD,
           color=BLUE_D if i >= 2 else INK)
        tx(g.shapes, bx + 0.18, 5.8, 2.55, 0.65, lab, size=12, color=INK2)
        if i < 3:
            arrow(g.shapes, bx + 2.86, 5.85, bx + 3.08, 5.85, w=2.0)
    tx(g.shapes, 0.6, 6.5, 12.1, 0.4,
       "For scale: Mars 2020 launched at 4,050 kg on an Atlas V and entered the atmosphere directly (no "
       "propulsive capture); 4.1× that mass fits on a Falcon Heavy to Mars.", size=12, color=MUTED)
    add_builds(s, [[g]])
    footer(s, 5)
    notes(s, 0.8, [
        ("Napkin estimate: rocket equation with assumed 320 s Isp (storable bipropellant) and tanks/engines "
        "= 10% of propellant mass. Order of magnitude."),
        "16.8 t injected → 9.2 t propellant (mass ratio 2.20) → 6.73 t useful → 6.6 rovers at 1,025 kg.",
        "Dispersed: 6.43 t, 6.3 rovers. Coplanar ideal: 6.98 t, 6.8 rovers.",
        ("The figure compares current capture with the coplanar ideal; the earlier RAAN 15° design (1.30 t) "
        "is on the capture-trade slide."),
        "Costs are in backup; they rest on assumed prices.",
    ])

    s = d.new("Limitations and assumptions")
    sh = s.shapes
    eyebrow(s, "06 · Mission summary")
    headline(s, "These are the next questions a reviewer would ask")
    card(sh, 0.6, 1.7, 5.9, 4.3, "Modeling", bullets([
        "Patched conics and two-body arcs",
        "Impulsive burns; no finite-burn or attitude effects",
        "Analytic planetary ephemeris",
        "TDB ≈ TT; UTC uses 37 leap seconds",
        "Mars J2 only in the RPO",
        "No navigation errors",
    ], size=17, after=12), size=17)
    card(sh, 6.8, 1.7, 5.9, 4.3, "Results to read carefully", bullets([
        "Ascent and descent Δv are ideal lower bounds",
        "Payload estimate uses assumed Isp and tankage",
        "Dispersed HCW handoff (18.2 km) sits outside the 15 km linear region",
        "The HCW path crosses the 200 m keep-out",
        "One seeded dispersion case, not a Monte Carlo",
        "131.8° launch azimuth not checked against Cape range safety",
    ], size=17, after=12, bcolor=ORANGE), fill=ORANGE_T, size=17)
    box(sh, 0.6, 6.2, 12.1, 0.65, fill=BLUE_T, shape="round", radius=0.1, margin=0.18,
        text=[[("Next: ", {"bold": True, "color": BLUE_D}),
               ("each of these maps to an item on the two roadmap slides.", {})]], size=15,
        color=INK, anchor="m")
    footer(s, 5)
    notes(s, 0.8, [
        "Be upfront: these are the next questions a reviewer would ask.",
        ("Modeling: patched conics, impulsive burns, analytic ephemeris, TDB ≈ TT, 37 leap seconds "
        "(TT − UTC = 69.184 s), J2 only in RPO, no navigation errors."),
        ("Results: ascent/descent Δv are ideal; payload uses assumed Isp and tankage; HCW handoff at 18.2 km "
        "is outside 15 km; path crosses the 200 m keep-out; one seeded dispersion; azimuth unchecked."),
    ])

    s = d.new("Lessons learned")
    sh = s.shapes
    eyebrow(s, "06 · Mission summary")
    headline(s, "Geometry beat propulsion, closed loops beat plans, and small pipelines beat big notebooks")
    cols = [
        ("Technical", BLUE_T, BLUE, [
            "Choosing the target plane saved more than any guidance tweak (5.52 → 2.27 km/s capture)",
            "Under J2, plans made once drift; re-plan from the actual state",
            ("Small altitude differences matter: dΩ/dt ∝ a^(−7/2) turned a ~16 km offset into ~9.7 km "
            "cross-track (development geometry)"),
            "Validate each stage against theory (594 m Lambert check, ~1 cm impact parameter)",
        ]),
        ("Process and tooling", AQUA_T, AQUA, [
            "One archive per stage that the next notebook reads",
            "One command regenerates every archive and figure in ~2 minutes",
            "272 unit tests in the library underneath the notebooks",
            "One plotting palette across every figure",
        ]),
        ("What I'd do differently", ORANGE_T, ORANGE, [
            "Choose the target plane from the arrival geometry at the start",
            "Build dispersions in from day one, not one seeded case at the end",
            "Design the RPO around the keep-out and linear-region limits from the beginning",
            "Hand off between stages with a patched-conic model, not a launch-time offset",
        ]),
    ]
    steps = []
    for i, (t, tint, col, items) in enumerate(cols):
        g = group(s)
        card(g.shapes, 0.6 + i * 4.1, 1.7, 3.9, 5.15, t, bullets(items, size=14, after=8, bcolor=col),
             fill=tint, size=14)
        steps.append([g])
    add_builds(s, steps)
    footer(s, 5)
    notes(s, 1, [
        "Draft wording for Colin to edit: technical, process/tooling, and what I'd do differently.",
        ("Technical: the target plane (RAAN 15° → 125°) was the biggest lever; closed-loop phasing; J2 "
        "differential regression; validate each stage."),
        ("Process: stage archives (trajectories/<stage>-<mission>.mission), one-command regeneration in "
        "~2 min, 272 unit tests."),
        ("Differently: pick the plane early, dispersions from day one, RPO around constraints, "
        "patched-conic handoff."),
    ])

    def roadmap(title, head, rows_, note_pts):
        s_ = d.new(title)
        sh_ = s_.shapes
        eyebrow(s_, "06 · Mission summary  ·  Next steps")
        headline(s_, head)
        tx(sh_, 0.6, 1.68, 1.6, 0.3, [[("PHASE", {"bold": True})]], size=11, color=MUTED, font=HEAD)
        tx(sh_, 2.35, 1.68, 7.0, 0.3, [[("ITEM", {"bold": True})]], size=11, color=MUTED, font=HEAD)
        tx(sh_, 9.75, 1.68, 1.4, 0.3, [[("PRIORITY", {"bold": True})]], size=11, color=MUTED,
           font=HEAD, align="c")
        tx(sh_, 11.3, 1.68, 1.4, 0.3, [[("EFFORT", {"bold": True})]], size=11, color=MUTED,
           font=HEAD, align="c")
        pr_col = {"High": (MARS, WHITE), "Medium": (ORANGE_T, ORANGE), "Low": ("ECEEF1", "5B6270")}
        ef_col = {"High": (INK, WHITE), "Medium": ("8A919C", WHITE), "Low": ("ECEEF1", "5B6270")}
        rh = min(0.66, 4.85 / len(rows_))
        for k, (ph, item, pr, ef) in enumerate(rows_):
            y0 = 2.02 + k * rh
            box(sh_, 0.6, y0, 12.1, rh - 0.08, fill=CARD if k % 2 == 0 else WHITE, shape="round",
                radius=0.06)
            tx(sh_, 0.75, y0, 1.6, rh - 0.08, [[(ph, {"bold": True})]], size=14, color=INK2, anchor="m")
            tx(sh_, 2.35, y0, 7.2, rh - 0.08, item, size=15, color=INK, anchor="m")
            for xx, lvl, cmap in ((9.85, pr, pr_col), (11.4, ef, ef_col)):
                f, c = cmap[lvl]
                box(sh_, xx, y0 + (rh - 0.08 - 0.32) / 2, 1.2, 0.32, fill=f, shape="round", radius=0.16,
                    margin=0.02, text=[[(lvl, {"bold": True})]], size=12, color=c, font=HEAD, align="c")
        tx(sh_, 0.6, 6.62, 12.1, 0.3, "Priority and effort are draft ratings for review.", size=12,
           color=MUTED, italic=True)
        footer(s_, 5)
        notes(s_, 0.6, note_pts)

    roadmap("Roadmap (1/2): mission design", "Close the modeling gaps from launch through capture", [
        ("Launch", "Cape range-safety check for the 131.8° launch azimuth", "High", "Low"),
        ("Launch", "Atmospheric ascent model: drag and gravity losses", "Medium", "Medium"),
        ("Departure", "Full patched-conic handoff onto the Lambert arc", "Medium", "Medium"),
        ("Cruise", "Dispersions on every segment, with midcourse burns during cruise", "High", "High"),
        ("Comms", "Link budget for the ground segment", "Low", "Medium"),
        ("Arrival", "High-ellipse capture with the plane change near apoapsis (fixed target planes)",
         "Medium", "Medium"),
        ("Arrival", "Atmospheric descent model for Mars surface ↔ orbit", "Low", "High"),
    ], [
        "Grouped by mission phase; priority/effort ratings are a draft for Colin to adjust.",
        "Quick win: check the 131.8° azimuth against Cape range-safety limits.",
        "Biggest item: dispersions on every segment (Monte Carlo) with cruise midcourse burns.",
        "Ascent/descent models would turn the ideal lower bounds (7.79 / 3.53 km/s) into realistic numbers.",
        "High-ellipse capture is the fallback if a target's plane can't be chosen.",
    ])
    roadmap("Roadmap (2/2): rendezvous and GNC", "Turn the RPO from a guidance demo into a docking design", [
        ("RPO", "Hand off to HCW inside the 15 km linear region", "High", "Low"),
        ("RPO", "Hold point at the 200 m keep-out, then a V-bar or R-bar final approach", "High",
         "Medium"),
        ("GNC", "Relative navigation: sensors and a filter", "High", "High"),
        ("GNC", "Closed-loop control (MPC)", "Medium", "High"),
        ("RPO", "Contact, capture, and abort logic", "Medium", "Medium"),
        ("All", "Monte Carlo over the RPO instead of one seeded case", "High", "Medium"),
        ("All", "Finite burns and attitude effects", "Medium", "Medium"),
    ], [
        "This is the 'how I would actually dock' slide turned into work items.",
        "First two items fix today's open items (18.2 km handoff, keep-out crossing).",
        "Navigation and control are the big lifts: right now the chaser knows its state perfectly.",
        "Priority/effort ratings are a draft.",
    ])

    s = d.new("Questions", size=48, title_box=(0.6, 2.6, 4.6, 1.0))
    sh = s.shapes
    tx(sh, 0.6, 3.65, 4.4, 1.5, "Thank you. Backup slides follow: porkchops, ground segment, burn "
                                "tables, validation, mass and cost, and the software pipeline.",
       size=16, color=INK2)
    _, (_qx, qy, _qw, qh) = fig(sh, "figures/rpo/mci_3d.png", 5.2, 1.0, 7.5, 5.4, ha="r", va="m",
                              alt="Arrival hyperbola flowing into the RPO, Mars-centered, dispersed case")
    tx(sh, 5.2, qy + qh + 0.1, 7.5, 0.35, "Arrival hyperbola flowing into the RPO, Mars-centered (dispersed case)",
       size=12, color=MUTED, align="r")
    footer(s, 5)
    notes(s, 5, [
        "Q&A (~5 min).",
        "Figure: the arrival hyperbola flowing into the RPO, Mars-centered, dispersed case.",
        ("Likely questions: keep-out crossing, linear-region handoff, Monte Carlo, range-safety azimuth, "
        "cost assumptions (backup)."),
    ])
    d.main_count = len(d.prs.slides)

    # ---------------- Backup ----------------
    d.section("Backup")
    s = d.new("Backup", dark=True, size=46, title_box=(0.9, 3.0, 7.4, 1.0))
    sh = s.shapes
    circle(sh, 10.95, 3.6, 3.85, line="3A4150", lw=1.25)
    circle(sh, 11.3, 3.75, 2.9, fill=MARS)
    tx(sh, 0.9, 4.15, 7.2, 1.0, "Detail for questions", size=22, color="C9CED6")
    slide_number(sh, 11.9, 7.03, 0.83, 0.27, "A9B0BB")
    notes(s, None, ["Backup section."])

    def backup_fig(title, rel, note_pts, caption=None, head=None, alt=None):
        s_ = d.new(title)
        eyebrow(s_, "Backup")
        top = 1.62
        if head:
            headline(s_, head)
        else:
            top = 1.25
        fig(s_.shapes, rel, 0.6, top, 12.1, 6.7 - top - (0.4 if caption else 0), alt=alt)
        if caption:
            tx(s_.shapes, 0.6, 6.45, 12.1, 0.4, caption, size=13, color=MUTED, align="c")
        footer(s_, backup=True)
        notes(s_, None, note_pts)
        return s_

    backup_fig("Porkchop: departure declination", "figures/interplanetary/porkchop_declination.png",
               [("DLA over the same grid; the selected transfer's DLA is 4.06°, well within reach from the "
                "Cape's 28.5° latitude.")],
               head="The selected transfer's DLA is 4.06°, well within reach from the Cape at 28.5° N")
    backup_fig("Porkchop: arrival v∞", "figures/interplanetary/porkchop_vinf_arrival.png",
               ["Arrival excess speed drives capture cost; the selection has v∞ = 2.82 km/s."],
               head="Arrival excess speed drives capture cost; the selected transfer arrives at 2.82 km/s")
    backup_fig("Constrained C3 porkchop", "figures/interplanetary/porkchop_c3_constrained.png",
               ["C3 masked by constraints and overlaid with declination / right-ascension contours.",
                "441 feasible transfers remain after constraints."],
               head="C3 masked by the constraints, with declination and right-ascension contours overlaid")

    s = d.new("Departure asymptote geometry")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Earth-centered view of the departure direction, and the numbers behind it")
    fig(sh, "figures/interplanetary/departure_direction_eci.png", 0.6, 1.65, 7.0, 5.2, ha="l", va="t",
        alt="Earth-centered view of the departure direction")
    simple_table(sh, 7.85, 1.7, [2.55, 2.3], [
        ["Quantity", "Value"],
        ["Departure v∞ / C3", "3.258 km/s / 10.62 km²/s²"],
        ["Asymptote RLA / DLA", "4.79° / 4.06°"],
        ["Parking orbit", "300 km (r = 6678.1 km)"],
        ["Circular speed", "7.726 km/s"],
        ["Escape burn (ideal)", "3.676 km/s"],
        ["Post-burn periapsis speed", "11.401 km/s"],
        ["Hyperbola e", "1.1779"],
        ["Aiming radius", "23,369 km"],
        ["Asymptote true anomaly", "148.1°"],
    ], size=13, row_h=0.42, bold_first_col=True)
    footer(s, backup=True)
    notes(s, None, ["Departure numbers from brief §2.1–2.2.",
                    "Launch moved 0.42 day after the Lambert reference to line up with the trajectory."])

    s = d.new("Ideal ascent from the Cape")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Lossless lower bounds only: real vehicles add ~1.5–2 km/s of gravity and drag losses")
    tx(sh, 0.6, 1.75, 6.0, 0.35, [[("Ideal ascent, Cape → 300 km (km/s)", {"bold": True})]], size=15,
       font=HEAD)
    for k, (lab, v, col) in enumerate([("Due-east reference", 7.682, AQUA),
                                       ("Into the 49.1° plane (used)", 7.791, ORANGE),
                                       ("Non-rotating Earth reference", 8.091, "8A919C")]):
        y0 = 2.3 + k * 0.85
        tx(sh, 0.6, y0, 2.6, 0.5, lab, size=14, color=INK2, anchor="m")
        bw = 3.0 * v / 8.091
        box(sh, 3.3, y0 + 0.08, bw, 0.36, fill=col)
        tx(sh, 3.3 + bw + 0.12, y0, 1.2, 0.5, [[(f"{v:.3f}", {"bold": True})]], size=15, anchor="m")
    tx(sh, 7.6, 1.8, 5.1, 4.9, bullets([
        "Launch site: Cape Canaveral, 28.49° N, 80.58° W",
        "Earth's rotation helps **300 m/s** for this plane (the site moves east at 409 m/s)",
        "Required launch azimuth **131.8°** from north (southeast)",
        "Open item: 131.8° has not been checked against Cape range-safety limits",
        "No drag, no gravity losses: a lower bound, not a launch cost",
    ], size=15, after=10))
    footer(s, backup=True)
    notes(s, None, ["Ideal ascent 7,790.7 m/s is the launch-vehicle ascent line in the Δv budget.",
                    "Due-east 7.682 km/s; non-rotating 8.091 km/s; Earth-rotation benefit 300 m/s."])

    s = d.new("Departure in Earth-fixed axes")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "The ground track spirals off from periapsis and settles near the asymptote declination")
    fig(sh, "figures/ground_segment/departure_ecef_3d.png", 0.6, 1.65, 6.3, 4.9, ha="l",
        alt="Departure trajectory in Earth-fixed axes")
    fig(sh, "figures/ground_segment/ground_track.png", 7.1, 1.65, 5.6, 4.9, ha="r",
        alt="Departure ground track")
    footer(s, backup=True)
    notes(s, None, [("Left: departure in ECEF axes. Right: the ground track spirals from periapsis toward "
                    "a near-equatorial line, near the asymptote declination.")])

    s = d.new("Sky tracks: Cape Canaveral and Svalbard")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Azimuth and elevation over the first ~3 days of departure")
    fig(sh, "figures/ground_segment/sky_track_cape.png", 0.6, 1.65, 5.9, 5.2,
        alt="Sky track from Cape Canaveral")
    fig(sh, "figures/ground_segment/sky_track_station.png", 6.8, 1.65, 5.9, 5.2,
        alt="Sky track from Svalbard")
    footer(s, backup=True)
    notes(s, None, [("Sky tracks show azimuth/elevation from Cape Canaveral and Svalbard over the first "
                    "~3 days."), "Cape: 3 passes of ~10.3–10.6 h, peak elevation 60.7°–64.4°."])

    s = d.new("Arrival hyperbola and capture numbers")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Mars-centered view of the arrival, target orbit, Sun vector and capture Δv direction")
    fig(sh, "figures/arrival/arrival_hyperbola_mci.png", 0.6, 1.65, 6.6, 5.2, ha="l",
        alt="Arrival hyperbola in Mars-centered inertial frame")
    simple_table(sh, 7.45, 1.7, [2.6, 2.65], [
        ["Quantity", "Value"],
        ["Capture periapsis", "300 km (r = 3696 km)"],
        ["Circular speed at 300 km", "3.404 km/s"],
        ["Hyperbola e / aiming radius", "1.6864 / 7,312 km"],
        ["Pre-burn periapsis speed", "5.579 km/s"],
        ["Coplanar (ideal) capture", "2.175 km/s"],
        ["Current, RAAN 125°", "β 6.8°, 2.269 km/s"],
        ["Earlier, RAAN 15°", "β 49.7°, 5.524 km/s"],
        ["Earlier aim point", "B·T 886 km, B·R 7,258 km"],
        ["Post-burn e", "≈ 3×10⁻¹⁶"],
    ], size=13, row_h=0.43, bold_first_col=True)
    footer(s, backup=True)
    notes(s, None, ["Validation: asymptote error 0.0000°, periapsis radius error 0 m, i and RAAN exact.",
                    ("Ideal Mars surface ↔ 300 km orbit (Perseverance site, 18.4° N): 3.531 km/s, reference "
                    "only.")])

    s = d.new("RPO: nominal versus dispersed")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Current design, both cases (target 300 km / 45° / 125°; truth: two-body + Mars J2)")
    simple_table(sh, 0.6, 1.65, [3.6, 3.4, 5.1], [
        ["Quantity", "Nominal (exact capture)", "Dispersed (capture errors)"],
        ["Initial chaser vs target", "coplanar, same radius",
         "a +73 km, e 0.0086, i +1.87°, RAAN −1.43°, aop −2.5°"],
        ["Total RPO Δv", "69.8 m/s", "186.0 m/s"],
        ["Plane change", "0 (not needed)", "120.5 m/s"],
        ["Hohmann + phase entry", "16.8 m/s", "24.9 m/s"],
        ["Phasing (20 burns)", "9.7 m/s", "9.8 m/s"],
        ["Phase exit", "18.3 m/s", "15.8 m/s"],
        ["HCW terminal", "25.1 m/s", "14.9 m/s"],
        ["Number of burns", "25", "27"],
        ["Time to docking", "41.6 h", "43.0 h"],
        ["Final miss", "13.5 m", "7.3 m"],
        ["J2 RAAN pre-bias", "costs 31.6 m/s → not used", "saves 50.4 m/s (236.4 → 186.0) → used"],
        ["RAAN drift in phasing, measured / analytic", "0.514° / 0.438°", "0.527° / 0.399°"],
    ], size=13, row_h=0.39, bold_first_col=True)
    footer(s, backup=True)
    notes(s, None, ["Brief §2.5 table. Phase gap at capture 107.9°; target 13° true anomaly at capture.",
                    ("Dispersed HCW: starts 18.2 km (outside the 15 km linear region), retargets at 290 m, "
                    "stops at 7.3 m.")])

    s = d.new("Dispersed-case burn timeline")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Times in UTC (TT − UTC = 69.184 s assumed); 27 burns in all")
    simple_table(sh, 0.6, 1.65, [0.9, 2.1, 3.2, 1.6, 4.3], [
        ["#", "After capture", "UTC", "Δv", "Burn"],
        ["0", "0.24 h", "2035-12-25 00:12:57", "120.5 m/s", "Burn 1 (plane change)"],
        ["1", "0.35 h", "2035-12-25 00:19:45", "14.4 m/s", "Burn 2 (Hohmann)"],
        ["2", "1.32 h", "2035-12-25 01:17:43", "10.5 m/s", "Burn 3 (circularize + phase entry)"],
        ["3–22", "3.2–39.7 h", "", "9.8 m/s total", "Burns 4.0–4.19 (phasing)"],
        ["23", "41.62 h", "2035-12-26 17:36:00", "15.8 m/s", "Burn 5 (phase exit)"],
        ["24", "41.62 h", "2035-12-26 17:36:00", "6.5 m/s", "HCW start"],
        ["25", "42.29 h", "2035-12-26 18:16:00", "8.2 m/s", "HCW retarget"],
        ["26", "42.95 h", "2035-12-26 18:56:00", "0.2 m/s", "HCW stop"],
    ], size=14, row_h=0.46, aligns=["l", "r", "l", "r", "l"])
    tx(sh, 0.6, 6.4, 12.1, 0.4, "Every burn with TDB Julian date: figures/rpo/burn_table.txt",
       size=12, color=MUTED)
    footer(s, backup=True)
    notes(s, None, ["Condensed from brief §2.5; the full 27-row list is in figures/rpo/burn_table.txt."])

    s = d.new("RPO development history")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Earlier test geometry (target RAAN 15°, synthetic chaser offset): not comparable with current results")
    simple_table(sh, 0.6, 1.7, [5.6, 2.6, 3.9], [
        ["Step", "Phasing handoff", "Total RPO Δv"],
        ["Open-loop phasing (incremental burns)", "51.5 km miss", "202.1 m/s"],
        ["Closed-loop phasing (re-plan each rev from actual state)", "12.2 km", "164.9 m/s"],
        ["+ J2 RAAN pre-bias at the plane change", "5.5 km", "152.2 m/s"],
        ["Hohmann shape corrections during phasing (tried, dropped)", "—",
         "~2% saving for 3–18 extra burns and 6–30 h; sometimes worse"],
    ], size=14, row_h=0.6, bold_first_col=False)
    box(sh, 0.6, 5.0, 12.1, 1.3, fill=MARS_T, shape="round", radius=0.1, margin=0.2,
        text=[[("Root cause: ", {"bold": True, "color": MARS}),
               (("the residual error was mostly cross-track (≈9.7 km) from J2 differential nodal "
                "regression. The phasing orbit sits ~16 km higher than the target, so its node "
                "regresses more slowly (dΩ/dt ∝ a^(−7/2))."), {})]], size=15, color=INK, anchor="m")
    footer(s, backup=True)
    notes(s, None, ["Brief §2.6. Development results only; keep separate from 69.8 / 186.0 m/s."])

    s = d.new("Validation checks")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Each stage was checked against theory before the next one used its output")
    simple_table(sh, 0.6, 1.7, [2.2, 5.8, 4.1], [
        ["Stage", "Check", "Result"],
        ["Interplanetary", "Re-propagate the Lambert solution (two-body)", "594 m position difference"],
        ["Departure", "Propagated impact parameter h/v∞ vs theory", "~1 cm"],
        ["Departure", "Propagated vs Lambert C3", "3.6×10⁻⁸ km²/s²"],
        ["Arrival", "Asymptote direction error", "0.0000°"],
        ["Arrival", "Periapsis radius error", "0 m"],
        ["Arrival", "Post-burn eccentricity; i and RAAN", "e ≈ 3×10⁻¹⁶; i, RAAN exact"],
        ["RPO", "J2 RAAN drift, measured vs analytic (dispersed)", "0.527° vs 0.399°"],
    ], size=14, row_h=0.5, bold_first_col=True)
    tx(sh, 0.6, 5.9, 12.1, 0.8, "The RPO drift check differs because the first-order formula only covers "
                                "the phasing leg.", size=14, color=MUTED)
    footer(s, backup=True)
    notes(s, None, ["Collected validation results from brief §2.1, §2.2, §2.4, §2.5."])

    s = d.new("Mission Δv budget table")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Launch-vehicle lines are ideal (no drag, gravity losses, aerobraking or parachutes)")
    simple_table(sh, 0.6, 1.7, [4.8, 2.5, 2.4, 2.4], [
        ["Maneuver", "Paid by", "Nominal", "Dispersed"],
        ["Earth ascent to LEO (ideal)", "launch vehicle", "7,790.7 m/s", "7,790.7 m/s"],
        ["Earth escape", "launch vehicle", "3,675.6 m/s", "3,675.6 m/s"],
        ["Midcourse correction", "spacecraft", "131.8 m/s", "131.8 m/s"],
        ["Mars capture (direct injection)", "spacecraft", "2,268.5 m/s", "2,268.5 m/s"],
        ["RPO", "spacecraft", "69.8 m/s", "186.0 m/s"],
        ["Total, spacecraft", "", "2,470.2 m/s", "2,586.3 m/s"],
        ["Total, launch vehicle", "", "11,466.3 m/s", "11,466.3 m/s"],
        ["Total, all", "", "13,936.5 m/s", "14,052.6 m/s"],
        ["Reference: Mars surface ↔ orbit (ideal, not in totals)", "", "3,531.5 m/s", "3,531.5 m/s"],
    ], size=14, row_h=0.46, aligns=["l", "l", "r", "r"],
        highlight={6: "E7F0FB", 7: "E7F0FB"})
    footer(s, backup=True)
    notes(s, None, [("Brief §2.7. Coplanar capture minimum 2,175.2 m/s; direct injection pays 93.3 m/s "
                    "for the plane change.")])

    s = d.new("Napkin mass and cost estimate")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Order of magnitude only: prices are assumptions, not quotes")
    simple_table(sh, 0.6, 1.65, [3.4, 1.55, 1.35, 1.5, 1.6, 1.1, 1.6], [
        ["Case", "Spacecraft Δv", "Mass ratio", "Propellant", "Useful payload", "Rovers", "$ / useful kg"],
        ["Current design, nominal", "2.47 km/s", "2.20", "9.2 t", "6.73 t", "6.6", "$22.6k"],
        ["Current design, dispersed", "2.59 km/s", "2.28", "9.4 t", "6.43 t", "6.3", "$23.6k"],
        ["Coplanar-capture ideal", "2.38 km/s", "2.13", "8.9 t", "6.98 t", "6.8", "$21.7k"],
        ["Earlier design (RAAN 15°)", "~5.7 km/s", "~6.2", "~14 t", "1.30 t", "1.3", "~$120k"],
    ], size=14, row_h=0.45, aligns=["l", "r", "r", "r", "r", "r", "r"])
    card(sh, 0.6, 4.1, 6.0, 2.75, "Assumptions", bullets([
        "Falcon Heavy expendable, 16.8 t to Mars injection",
        "$150M launch (assumed)",
        "Isp 320 s, storable bipropellant (assumed)",
        "Tanks/engines = 10% of propellant mass (assumed)",
        "Spacecraft propellant $200/kg (assumed); Perseverance 1,025 kg",
    ], size=13, after=3), size=13)
    card(sh, 6.8, 4.1, 5.9, 2.75, "What dominates", bullets([
        "Falcon Heavy propellant: ~1,331 t, about $1.22M, 0.8% of the launch price",
        "Spacecraft propellant: about $1.8–1.9M",
        "The launch price dominates the cost",
        "Mars 2020: 4,050 kg on an Atlas V, direct atmospheric entry",
    ], size=13, after=3), size=13)
    footer(s, backup=True)
    notes(s, None, [("The only slide with dollar figures. All prices are assumptions; treat as order of "
                    "magnitude, not quotes."),
                    "4.1× the Mars 2020 launch mass fits on a Falcon Heavy to Mars."])

    s = d.new("Software and data pipeline")
    sh = s.shapes
    eyebrow(s, "Backup")
    headline(s, "Six notebooks, one archive per stage, one command to regenerate everything")
    nbs = ["interplanetary", "departure", "ground segment", "arrival", "RPO", "summary"]
    for i, nb in enumerate(nbs):
        bx = 0.6 + i * 2.07
        box(sh, bx, 1.8, 1.8, 0.9, fill=BLUE_T if i == 4 else CARD, shape="round", radius=0.1,
            text=[[(nb, {"bold": True})]], size=14, font=HEAD, align="c")
        if i < 5:
            arrow(sh, bx + 1.82, 2.25, bx + 2.05, 2.25, w=2.0)
    tx(sh, 0.6, 2.8, 12.1, 0.35, "each stage writes trajectories/<stage>-<mission>.mission; the next stage "
                                 "reads it", size=13, color=MUTED, align="c")
    tx(sh, 0.6, 3.45, 12.1, 3.3, bullets([
        ("Python library (cesium): Lambert, porkchop grids, B-plane targeting, departure/arrival "
        "planners, RPO timeline with J2 propagation, HCW guidance, plotting"),
        "Six notebooks run in order: interplanetary → departure → ground segment → arrival → RPO → summary",
        "One command regenerates every archive and figure in ~2 minutes",
        "272 unit tests in the library",
    ], size=16, after=10))
    footer(s, backup=True)
    notes(s, None, ["Brief §5 B9."])

    s = d.new("Glossary")
    sh = s.shapes
    eyebrow(s, "Backup")
    gl = [
        ("C3", "characteristic energy, v∞²; launch-vehicle capability is quoted against it"),
        ("v∞", "hyperbolic excess speed relative to a planet"),
        ("RLA / DLA", "right ascension / declination of the departure asymptote"),
        ("B-plane", ("plane through the planet center perpendicular to the incoming asymptote; the aim "
                    "point (B·T, B·R) sets periapsis radius and arrival plane")),
        ("β (beta)", ("angle between the arrival asymptote and the target orbit plane; drives the "
                     "plane-change cost of direct injection")),
        ("RAAN (Ω)", "right ascension of the ascending node; orientation of the orbit plane"),
        ("J2", "planetary oblateness term; makes orbit planes precess (nodal regression)"),
        ("RSW", "target-centered rotating frame: R radial, S along-track, W cross-track"),
        ("HCW", "Hill–Clohessy–Wiltshire linearized relative-motion equations, valid near the target"),
        ("Direct injection", "single capture burn at periapsis that also rotates the plane"),
        ("TDB / TT / UTC", "time scales; TT − UTC = 69.184 s here (32.184 s + 37 leap seconds)"),
    ]
    for k, (t, dfn) in enumerate(gl):
        col, row = divmod(k, 6)
        tx(sh, 0.6 + col * 6.2, 1.3 + row * 0.92, 5.9, 0.9,
           [{"t": [(t, {"bold": True})], "size": 15, "font": HEAD, "after": 1},
            {"t": dfn, "size": 13, "color": INK2}])
    footer(s, backup=True)
    notes(s, None, ["Brief §7."])

    return d


# ---------------------------------------------------------------------------------------------
# Package-level finishing: theme fonts, sections, properties
# ---------------------------------------------------------------------------------------------
def set_theme_fonts(prs):
    theme_part = prs.slide_master.part.part_related_by(RT.THEME)
    root = etree.fromstring(theme_part.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    root.find(".//a:majorFont/a:latin", ns).set("typeface", HEAD)
    root.find(".//a:minorFont/a:latin", ns).set("typeface", BODY)
    theme_part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def add_sections(prs, sections):
    p14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
    pres = prs.part._element
    ext_lst = pres.find(qn("p:extLst"))
    if ext_lst is None:
        ext_lst = etree.SubElement(pres, qn("p:extLst"))
    ext = etree.SubElement(ext_lst, qn("p:ext"))
    ext.set("uri", "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
    sec_lst = etree.SubElement(ext, f"{{{p14}}}sectionLst", nsmap={"p14": p14})
    for name, ids in sections:
        sec = etree.SubElement(sec_lst, f"{{{p14}}}section")
        sec.set("name", name)
        sec.set("id", "{" + str(uuid.uuid4()).upper() + "}")
        lst = etree.SubElement(sec, f"{{{p14}}}sldIdLst")
        for sid in ids:
            etree.SubElement(lst, f"{{{p14}}}sldId").set("id", str(sid))


# ---------------------------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------------------------
FONT_DIR = Path.home() / "Library/Fonts"
EMBED_FACES = {
    HEAD: ["Inter-Regular.otf", "Inter-Bold.otf", "Inter-Italic.otf", "Inter-BoldItalic.otf"],
    BODY: ["SourceSans3[wght].ttf", "SourceSans3-Italic[wght].ttf"],
}


def _embed_fonts_in_odp(src, dst):
    """Embed the deck fonts into an ODP so LibreOffice registers them in-process.

    A sandboxed (or font-less) LibreOffice cannot see ~/Library/Fonts; fonts embedded in an
    ODF document are always loaded. The PDF then embeds the real Inter / Source Sans 3.
    """
    import re
    import zipfile

    def arc(f):
        return "Fonts/" + f.replace("[", "_").replace("]", "_")

    def face(name, files):
        uris = "".join(
            f'<svg:font-face-uri xlink:href="{arc(f)}" xlink:type="simple">'
            f'<svg:font-face-format svg:string="truetype"/></svg:font-face-uri>' for f in files)
        fam = f"&apos;{name}&apos;" if " " in name else name
        return (f'<style:font-face style:name="{name}" svg:font-family="{fam}">'
                f"<svg:font-face-src>{uris}</svg:font-face-src></style:font-face>")

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in ("content.xml", "styles.xml"):
                xml = data.decode()
                for name, files in EMBED_FACES.items():
                    pat = r'<style:font-face style:name="' + re.escape(name) + r'"[^>]*/>'
                    xml = re.sub(pat, face(name, files), xml)
                data = xml.encode()
            elif item.filename == "META-INF/manifest.xml":
                entries = "".join(
                    f' <manifest:file-entry manifest:full-path="{arc(f)}" '
                    f'manifest:media-type="application/x-font-ttf"/>\n'
                    for files in EMBED_FACES.values() for f in files)
                data = data.replace(b"</manifest:manifest>", entries.encode() + b"</manifest:manifest>")
            comp = zipfile.ZIP_STORED if item.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(item, data, compress_type=comp)
        for files in EMBED_FACES.values():
            for f in files:
                zout.write(FONT_DIR / f, arc(f), compress_type=zipfile.ZIP_DEFLATED)


def export_pdf(pptx):
    """pptx -> odp (soffice) -> embed fonts -> pdf (soffice --headless --convert-to pdf)."""
    import shutil
    import subprocess
    import tempfile

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
    print(f"wrote {out}")


def main():
    import sys

    d = build()
    prs = d.prs
    set_theme_fonts(prs)
    add_sections(prs, d.sections)
    cp = prs.core_properties
    cp.title = "Mars Courier: End-to-End Mission Design from Launch Window to Docking"
    cp.author = "Colin Skinner"
    prs.save(OUT)
    n = len(prs.slides)
    print(f"wrote {OUT} ({d.main_count} main + {n - d.main_count} backup = {n} slides)")
    if "--no-pdf" not in sys.argv:
        export_pdf(OUT)


if __name__ == "__main__":
    main()
