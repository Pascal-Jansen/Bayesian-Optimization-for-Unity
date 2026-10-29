"""Generate images/ibo-overview-{light,dark}.svg and the front-end logos images/front-end-*.svg for the README
(icons in icons/: Simple Icons, CC0).

Usage: python tools/readme_figures/make_overview.py
"""
import re, sys, html
from pathlib import Path

ICONS = Path(__file__).resolve().parent / "icons"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "images"

def icon_path(name):
    return re.search(r'<path d="([^"]+)"', (ICONS / f"{name}.svg").read_text()).group(1)

THEMES = {
    "light": dict(card="#F6F8FA", stroke="#D0D7DE", text="#1F2328", muted="#59636E", band="#DDF4FF",
                  bandstroke="#54AEFF", accent="#0969DA", chip="#FFFFFF", unity="#000000", unreal="#0E1128", rust="#000000"),
    "dark": dict(card="#161B22", stroke="#3D444D", text="#E6EDF3", muted="#9198A1", band="#0C2D4B",
                 bandstroke="#1F6FEB", accent="#4493F8", chip="#0D1117", unity="#FFFFFF", unreal="#FFFFFF", rust="#FFFFFF"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

def text(x, y, s, size, fill, weight=400, anchor="start", family=FONT):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{html.escape(s)}</text>')

def icon(name, x, y, size, fill):
    s = size / 24
    return f'<g transform="translate({x},{y}) scale({s:.4f})"><path d="{icon_path(name)}" fill="{fill}"/></g>'

def width(s, size):  # rough text width for layout
    return sum(0.62 if c.isupper() else 0.53 if c.isalnum() else 0.33 for c in s) * size

def render(t):
    W, H = 1000, 450
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="iBO overview: Unity, Unreal Engine, Python and other clients talk to the iBO backend over the iBO protocol">']
    o.append(f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{t["accent"]}"/></marker></defs>')
    top, ch, gap = 58, 82, 14
    # column headers
    o.append(text(20, 38, "Any front end", 14, t["muted"], 600))
    o.append(text(500, 38, "iBO protocol v1", 14, t["muted"], 600, "middle"))
    o.append(text(650, 38, "Backend", 14, t["muted"], 600))
    cards = [
        ("unity", t["unity"], "Unity", "C# package, Inspector setup", "questionnaire add-on, examples"),
        ("unrealengine", t["unreal"], "Unreal Engine", "C++ plugin, Blueprint component", "analytic example actor"),
        ("python", "#3776AB", "Python", "interactive_bo client", "scripts, PsychoPy, lab hardware"),
        (None, t["text"], "Your own tool", "Godot, C++, Node.js, MATLAB, ...", "any language with TCP and JSON"),
    ]
    for i, (ic, col, name, l1, l2) in enumerate(cards):
        y = top + i * (ch + gap)
        # the open slot for other tools is outlined like a placeholder
        outline = f'stroke="{t["stroke"]}"' if ic else f'stroke="{t["accent"]}" stroke-width="1.5" stroke-dasharray="6 4"'
        o.append(f'<rect x="20" y="{y}" width="330" height="{ch}" rx="10" fill="{t["card"]}" {outline}/>')
        if ic:
            o.append(icon(ic, 38, y + 23, 36, col))
        else:
            o.append(f'<rect x="38" y="{y+23}" width="36" height="36" rx="8" fill="none" stroke="{t["text"]}" stroke-width="2"/>')
            o.append(text(56, y + 47, "{ }", 15, t["text"], 700, "middle", "ui-monospace, Menlo, Consolas, monospace"))
        o.append(text(90, y + 30, name, 17, t["text"], 600))
        o.append(text(90, y + 51, l1, 13, t["muted"]))
        o.append(text(90, y + 68, l2, 13, t["muted"]))
        cy = y + ch / 2
        o.append(f'<line x1="352" y1="{cy}" x2="388" y2="{cy}" stroke="{t["accent"]}" stroke-width="2" '
                 f'marker-start="url(#arr)" marker-end="url(#arr)"/>')
    # protocol band: the messages of a study in order, each arrow pointing from sender to receiver
    bh = 4 * ch + 3 * gap
    o.append(f'<rect x="390" y="{top}" width="220" height="{bh}" rx="12" fill="{t["band"]}" stroke="{t["bandstroke"]}"/>')
    o.append(text(402, top + 24, "front end", 12, t["muted"], 600))
    o.append(text(598, top + 24, "backend", 12, t["muted"], 600, "end"))
    lx, ly, lh = 398, top + 110, 152  # the repeated part: one design shown, one response back
    o.append(f'<rect x="{lx}" y="{ly}" width="204" height="{lh}" rx="8" fill="none" stroke="{t["bandstroke"]}" '
             f'stroke-width="1.2" stroke-dasharray="5 4"/>')
    o.append(text(500, ly + 18, "repeated for every design", 12, t["muted"], 600, "middle"))
    msgs = [(top + 62, True, "init", "parameters, objectives, settings"),
            (ly + 50, False, "parameters", "the next design to show"),
            (ly + 112, True, "objectives", "ratings, behavior, telemetry"),
            (top + 320, False, "progress, finished", "hypervolume or best value")]
    for y, to_backend, kind, detail in msgs:
        x1, x2 = (404, 596) if to_backend else (596, 404)
        o.append(text(500, y - 8, kind, 15, t["accent"], 700, "middle"))
        o.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{t["accent"]}" stroke-width="2" marker-end="url(#arr)"/>')
        o.append(text(500, y + 19, detail, 12.5, t["text"], 400, "middle"))
    # connector band -> backend
    o.append(f'<line x1="612" y1="{top + bh/2}" x2="648" y2="{top + bh/2}" stroke="{t["accent"]}" stroke-width="2" '
             f'marker-start="url(#arr)" marker-end="url(#arr)"/>')
    # backend card
    x0 = 650
    o.append(f'<rect x="{x0}" y="{top}" width="330" height="{bh}" rx="12" fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="1.5"/>')
    o.append(icon("python", x0 + 18, top + 18, 36, "#3776AB"))
    o.append(text(x0 + 66, top + 37, "iBO backend", 18, t["text"], 700))
    o.append(text(x0 + 66, top + 56, "Python 3.13, one process per study", 13, t["muted"]))
    o.append(text(x0 + 18, top + 96, "MODEL LIBRARY", 11.5, t["muted"], 600))
    o.append(icon("pytorch", x0 + 18, top + 106, 22, "#EE4C2C"))
    o.append(text(x0 + 46, top + 123, "BoTorch", 14, t["text"], 600))
    o.append(icon("rust", x0 + 150, top + 106, 22, t["rust"]))
    o.append(text(x0 + 178, top + 123, "botorch-rust", 14, t["text"], 600))
    o.append(text(x0 + 18, top + 164, "METHODS", 11.5, t["muted"], 600))
    chips = [["BO", "MOBO", "CABOP", "DBO"], ["Contextual (LCE-M)", "MetaTAF"]]
    for r, row in enumerate(chips):
        cx = x0 + 18
        for c in row:
            w = width(c, 13) + 20
            y = top + 174 + r * 34
            o.append(f'<rect x="{cx:.0f}" y="{y}" width="{w:.0f}" height="26" rx="13" fill="{t["chip"]}" stroke="{t["stroke"]}"/>')
            o.append(text(cx + w / 2, y + 18, c, 13, t["text"], 600, "middle"))
            cx += w + 8
    o.append(text(x0 + 18, top + 270, "RESULTS", 11.5, t["muted"], 600))
    for i, line in enumerate(["CSV logs per participant and condition", "Pareto front and hypervolume", "warm start from earlier runs"]):
        o.append(text(x0 + 18, top + 292 + i * 20, "• " + line, 13, t["text"]))
    o.append("</svg>")
    return "\n".join(o) + "\n"

def tile(name, color, label):
    """Front-end logo for the README table: a white mark on a brand-colored tile, readable in light and dark mode."""
    mark = (icon(name, 5, 5, 18, "#FFFFFF") if name else
            text(14, 19, "{ }", 12, "#FFFFFF", 700, "middle", "ui-monospace, Menlo, Consolas, monospace"))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28" role="img" '
            f'aria-label="{label}"><rect width="28" height="28" rx="6" fill="{color}"/>{mark}</svg>\n')


for name, theme in THEMES.items():
    (OUT / f"ibo-overview-{name}.svg").write_text(render(theme), encoding="utf-8")
    print("wrote", OUT / f"ibo-overview-{name}.svg")
# tile colors as in the README badges
for file, ic, color, label in [("unity", "unity", "#000000", "Unity"), ("unreal", "unrealengine", "#0E1128", "Unreal Engine"),
                               ("python", "python", "#3776AB", "Python"), ("protocol", None, "#6E7781", "iBO protocol")]:
    (OUT / f"front-end-{file}.svg").write_text(tile(ic, color, label), encoding="utf-8")
print("wrote", OUT / "front-end-{unity,unreal,python,protocol}.svg")
