"""Generate the iBO logo: images/ibo-logo-{light,dark}.svg (icon and wordmark) and images/ibo-icon.svg.

The icon is the human-in-the-loop cycle (the loop arrow) around a peak with the next design on top (the orange
point); the dot on the "i" repeats that point. The letters are drawn as strokes, so no font is needed except for
the subtitle. Usage: python tools/readme_figures/make_logo.py
"""
import math
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "images"
BLUE, VIOLET, ORANGE = "#2f81f7", "#a371f7", "#fb8f44"
THEMES = {"light": dict(text="#1f2328", muted="#59636e"), "dark": dict(text="#e6edf3", muted="#9198a1")}


def defs(prefix):
    return (f'<defs><linearGradient id="{prefix}g" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="{VIOLET}"/></linearGradient>'
            f'<linearGradient id="{prefix}r" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="{VIOLET}"/></linearGradient></defs>')


def icon(x, y, size, prefix):
    """Rounded square with the loop arrow, the uncertainty band, the estimate and the next design."""
    s = size / 140
    cx, cy, r = 70, 72, 46
    a0, sweep = math.radians(118), math.radians(292)  # gap at the bottom; y points down, so + is clockwise
    p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
    a1 = a0 + sweep
    p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
    tangent = (-math.sin(a1), math.cos(a1))
    normal = (math.cos(a1), math.sin(a1))
    tip = (p1[0] + 13 * tangent[0], p1[1] + 13 * tangent[1])
    wing = [(p1[0] + 9 * normal[0], p1[1] + 9 * normal[1]), (p1[0] - 9 * normal[0], p1[1] - 9 * normal[1])]
    peak = "M38 93 C54 93 58 52 70 52 C82 52 86 93 102 93"
    return (f'<g transform="translate({x} {y}) scale({s:.4f})">'
            f'<rect width="140" height="140" rx="32" fill="url(#{prefix}g)"/>'
            f'<path d="M{p0[0]:.1f} {p0[1]:.1f} A{r} {r} 0 1 1 {p1[0]:.1f} {p1[1]:.1f}" fill="none" stroke="#fff" '
            f'stroke-width="7" stroke-linecap="round"/>'
            f'<path d="M{tip[0]:.1f} {tip[1]:.1f} L{wing[0][0]:.1f} {wing[0][1]:.1f} L{wing[1][0]:.1f} {wing[1][1]:.1f} Z" '
            f'fill="#fff" stroke="#fff" stroke-width="3" stroke-linejoin="round"/>'
            f'<path d="{peak}" fill="none" stroke="#fff" stroke-opacity="0.28" stroke-width="20" stroke-linecap="round"/>'
            f'<path d="{peak}" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="70" cy="51" r="9.5" fill="{ORANGE}" stroke="#fff" stroke-width="3.5"/>'
            '</g>')


def wordmark(x, colors, prefix):
    stroke = f'fill="none" stroke="{colors["text"]}" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"'
    return (f'<g transform="translate({x} 0)">'
            f'<path d="M15 82 V141" {stroke}/>'
            f'<circle cx="15" cy="51" r="12" fill="{ORANGE}"/>'
            f'<path d="M50 41 V141 M50 41 H80 A24 24 0 0 1 80 89 H50 M50 89 H84 A26 26 0 0 1 84 141 H50" {stroke}/>'
            f'<circle cx="195" cy="91" r="50" fill="none" stroke="url(#{prefix}r)" stroke-width="18"/>'
            '</g>')


def logo(theme):
    c = THEMES[theme]
    prefix = f"ibo{theme}"
    subtitle = (f'<text x="182" y="198" font-family="-apple-system, \'Segoe UI\', \'Helvetica Neue\', Arial, sans-serif" '
                f'font-size="25" font-weight="600" letter-spacing="0.6" fill="{c["muted"]}">Interactive Bayesian Optimization</text>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 616 214" width="616" height="214" role="img" '
            'aria-label="iBO: Interactive Bayesian Optimization">'
            + defs(prefix) + icon(0, 30, 154, prefix) + wordmark(182, c, prefix) + subtitle + "</svg>\n")


if __name__ == "__main__":
    for theme in THEMES:
        (OUT / f"ibo-logo-{theme}.svg").write_text(logo(theme), encoding="utf-8")
    (OUT / "ibo-icon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140" width="140" height="140" '
                                      'role="img" aria-label="iBO">' + defs("iboicon") + icon(0, 0, 140, "iboicon") + "</svg>\n",
                                      encoding="utf-8")
    print("wrote", ", ".join(f"ibo-logo-{t}.svg" for t in THEMES), "and ibo-icon.svg in", OUT)
