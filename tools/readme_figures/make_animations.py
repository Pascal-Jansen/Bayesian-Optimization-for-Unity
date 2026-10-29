"""Draw the README animations from studies.json (recorded iBO runs), in a light and a dark version.

Usage: python tools/readme_figures/make_animations.py  (needs numpy, matplotlib, torch, botorch and ffmpeg)
The estimate in the first animation is the model of bo.py, refitted to the recorded ratings for each frame.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "images"
sys.path.insert(0, str(HERE))
from run_studies import DRIFT_EVALS, comfort, drifting, drifting_peak, preference, speed  # noqa: E402

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", grid="#d8dee4", accent="#0969da", next="#bc4c00",
                  best="#1a7f37", truth="#8c959f", other="#afb8c1", heat="#d4a72c"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#262c36", accent="#4493f8", next="#f0883e",
                 best="#3fb950", truth="#7d8590", other="#59636e", heat="#e3b341"),
}
FPS = 10
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11})
STUDIES = json.loads((HERE / "studies.json").read_text(encoding="utf-8"))


class Frames:
    def __init__(self, name, theme):
        self.name, self.theme, self.count = name, theme, 0
        self.dir = Path(tempfile.mkdtemp(prefix=name))

    def add(self, fig, hold=1):
        path = self.dir / f"{self.count:04d}.png"
        fig.savefig(path, facecolor=THEMES[self.theme]["bg"])
        plt.close(fig)
        self.count += 1
        for _ in range(hold - 1):
            shutil.copy(path, self.dir / f"{self.count:04d}.png")
            self.count += 1

    def gif(self):
        target = OUT / f"{self.name}-{self.theme}.gif"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(self.dir / "%04d.png"),
                        "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", str(target)], check=True)
        shutil.rmtree(self.dir)
        print(f"{target.name}: {target.stat().st_size / 1e6:.2f} MB, {self.count} frames")


def style(ax, c):
    ax.set_facecolor(c["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["muted"], length=0)
    ax.xaxis.label.set_color(c["muted"])
    ax.yaxis.label.set_color(c["muted"])
    ax.grid(color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def counter(fig, c, text):
    """The only text above a plot: the README gives each animation its title and explanation."""
    fig.text(0.97, 0.965, text, color=c["muted"], fontsize=11, va="top", ha="right")


def legend(fig, c, handles, ncol):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=9.5, labelcolor=c["muted"],
               handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))


# One parameter, one rating ------------------------------------------------------------------------------------------
def gp_curves(xs, ys, grid, low=0.0, high=10.0):
    """iBO's estimate after these ratings: the model of bo.py (a SingleTaskGP with its default transforms on
    parameters in [0, 1] and ratings mapped to [-1, 1]), converted back to rating units."""
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    torch.manual_seed(0)
    scale = (high - low) / 2
    train_y = torch.tensor([(y - low) / scale - 1 for y in ys], dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(torch.tensor(xs, dtype=torch.double).unsqueeze(-1), train_y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    with torch.no_grad():
        posterior = model.posterior(torch.tensor(grid, dtype=torch.double).unsqueeze(-1))
        mean = (posterior.mean.squeeze(-1).numpy() + 1) * scale + low
        return mean, posterior.variance.clamp_min(0).sqrt().squeeze(-1).numpy() * scale


def one_parameter(theme):
    c = THEMES[theme]
    runs = STUDIES["bo1d"]
    xs = [r["parameters"]["x"] for r in runs]
    ys = [r["objectives"]["rating"] for r in runs]
    n, sampling = len(xs), 3
    grid = np.linspace(0, 1, 400)
    truth = np.array([preference(x) for x in grid])
    models = [None, None] + [gp_curves(xs[:k], ys[:k], grid) for k in range(2, n + 1)]
    best = int(np.argmax(ys))
    frames = Frames("ibo-one-parameter", theme)

    def draw(k, model, next_index, final=False, just_rated=False):
        fig = plt.figure(figsize=(9, 3.9), dpi=100)
        ax = fig.add_axes([0.07, 0.26, 0.9, 0.64])
        style(ax, c)
        ax.plot(grid, truth, color=c["truth"], linestyle=(0, (4, 3)), linewidth=1.6)
        if model is not None:
            mean, sd = model
            ax.fill_between(grid, mean - 2 * sd, mean + 2 * sd, color=c["accent"], alpha=0.16, linewidth=0)
            ax.plot(grid, mean, color=c["accent"], linewidth=2.2)
        if k:
            ax.scatter(xs[:k], ys[:k], s=46, color=c["fg"], zorder=5, edgecolor=c["bg"], linewidth=1.2)
        if just_rated:
            x, y = xs[k - 1], ys[k - 1]
            ax.scatter([x], [y], s=190, facecolor="none", edgecolor=c["fg"], linewidth=1.6, zorder=6)
            ax.text(x + (0.025 if x < 0.85 else -0.025), y - 0.9, "just rated", color=c["fg"], fontsize=10,
                    ha="left" if x < 0.85 else "right")
        if next_index is not None:
            x = xs[next_index]
            ax.axvline(x, color=c["next"], linewidth=1.6, linestyle=(0, (2, 2)), zorder=4)
            ax.scatter([x], [0.15], marker="^", s=120, color=c["next"], zorder=6, clip_on=False)
            ax.text(x + (0.012 if x < 0.85 else -0.012), 10.4, "next design", color=c["next"], fontsize=10.5,
                    ha="left" if x < 0.85 else "right", va="top")
        if final:
            ax.scatter([xs[best]], [ys[best] + 1.1], marker="v", s=150, color=c["best"], zorder=7)
            ax.text(xs[best] + 0.03, ys[best] + 1.1, "best design found", color=c["best"], fontsize=11,
                    fontweight="bold", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["low", "high"])
        ax.set_yticks([5, 10])
        ax.set_xlabel("Design parameter (for example, text size)", labelpad=-4)
        ax.set_ylabel("Rating")
        counter(fig, c, f"{k} of {n} ratings")
        legend(fig, c, [
            Line2D([], [], color=c["truth"], linestyle=(0, (4, 3)), linewidth=1.6, label="participant's preference (hidden from iBO)"),
            Line2D([], [], color=c["accent"], linewidth=2.2, label="iBO's estimate"),
            Patch(color=c["accent"], alpha=0.25, label="uncertainty"),
            Line2D([], [], marker="o", linestyle="none", color=c["fg"], label="rated designs"),
            Line2D([], [], marker="^", linestyle="none", color=c["next"], label="next design")], ncol=5)
        return fig

    frames.add(draw(0, None, 0), hold=16)
    for k in range(1, n + 1):
        before, after = models[k - 1], models[k]
        steps = 3 if before is not None and after is not None else 1
        for s in range(1, steps + 1):
            a = s / steps
            model = after if before is None or after is None else tuple(b * (1 - a) + v * a for b, v in zip(before, after))
            frames.add(draw(k, model, None, just_rated=True), hold=2 if s == steps else 1)
        frames.add(draw(k, after, k if k < n else None), hold=10 if k < n else 1)
    frames.add(draw(n, models[n], None, final=True), hold=50)
    frames.gif()


# Two objectives: the best trade-offs --------------------------------------------------------------------------------
def pareto_mask(points):
    pts = np.asarray(points)
    return np.array([not np.any(np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)) for p in pts])


def trade_offs(theme):
    c = THEMES[theme]
    runs = STUDIES["mobo"]
    X = np.array([(r["parameters"]["x1"], r["parameters"]["x2"]) for r in runs])
    F = np.array([(r["objectives"]["comfort"], r["objectives"]["speed"]) for r in runs])
    n, sampling = len(X), 6
    t = np.linspace(0, 1, 300)[:, None]
    true_set = np.array([0.2, 0.3]) + t * np.array([0.6, 0.4])
    true_front = np.array([(comfort(*p), speed(*p)) for p in true_set])
    frames = Frames("ibo-trade-offs", theme)

    def draw(k, final=False):
        fig = plt.figure(figsize=(9.6, 4.1), dpi=100)
        left = fig.add_axes([0.07, 0.2, 0.37, 0.64])
        right = fig.add_axes([0.59, 0.2, 0.37, 0.64])
        for ax in (left, right):
            style(ax, c)
        left.plot(true_set[:, 0], true_set[:, 1], color=c["truth"], linestyle=(0, (4, 3)), linewidth=1.6)
        right.plot(true_front[:, 0], true_front[:, 1], color=c["truth"], linestyle=(0, (4, 3)), linewidth=1.6)
        best = 0
        if k:
            mask = pareto_mask(F[:k])
            best = int(mask.sum())
            for ax, data in ((left, X[:k]), (right, F[:k])):
                ax.scatter(data[~mask, 0], data[~mask, 1], s=38, color=c["other"], zorder=4, edgecolor=c["bg"], linewidth=1)
                ax.scatter(data[mask, 0], data[mask, 1], s=50, color=c["accent"], zorder=5, edgecolor=c["bg"], linewidth=1)
            if not final:
                for ax, point in ((left, X[k - 1]), (right, F[k - 1])):
                    ax.scatter([point[0]], [point[1]], s=190, facecolor="none", edgecolor=c["fg"], linewidth=1.6, zorder=6)
                fig.add_artist(ConnectionPatch(xyA=X[k - 1], coordsA=left.transData, xyB=F[k - 1], coordsB=right.transData,
                                               color=c["fg"], linewidth=1.1, linestyle=(0, (3, 3)), alpha=0.55))
        left.set_xlim(-0.04, 1.04)
        left.set_ylim(-0.04, 1.04)
        left.set_xticks([])
        left.set_yticks([])
        left.set_xlabel("Parameter 1, for example button size  →")
        left.set_ylabel("Parameter 2, for example contrast  →")
        left.set_title("The design iBO shows", color=c["fg"], fontsize=11.5, loc="left")
        right.set_xlim(3, 10.4)
        right.set_ylim(3, 10.4)
        right.set_xticks([4, 6, 8, 10])
        right.set_yticks([4, 6, 8, 10])
        right.set_xlabel("Comfort (rated, higher is better)")
        right.set_ylabel("Speed (measured, higher is better)")
        right.set_title("Its result", color=c["fg"], fontsize=11.5, loc="left")
        right.text(1.0, 1.02, f"best trade-offs found: {best}", transform=right.transAxes, ha="right", va="bottom",
                   color=c["accent"], fontsize=11, fontweight="bold")
        counter(fig, c, f"{k} of {n} ratings")
        legend(fig, c, [
            Line2D([], [], color=c["truth"], linestyle=(0, (4, 3)), linewidth=1.6, label="best possible (hidden from iBO)"),
            Line2D([], [], marker="o", linestyle="none", color=c["accent"], label="best trade-offs so far: no other design beats them on both goals"),
            Line2D([], [], marker="o", linestyle="none", color=c["other"], label="other designs")], ncol=3)
        return fig

    frames.add(draw(0), hold=12)
    for k in range(1, n + 1):
        frames.add(draw(k), hold=8)
    frames.add(draw(n, final=True), hold=50)
    frames.gif()


# A drifting optimum: DBO and standard BO ----------------------------------------------------------------------------
def drift(theme):
    c = THEMES[theme]
    picks = {name: np.array([r["parameters"]["x"] for r in STUDIES["drift_" + name]]) for name in ("dbo", "botorch")}
    n, sampling, band = DRIFT_EVALS, 4, 0.1
    grid = np.linspace(0, 1, 200)
    heat = np.array([[drifting(x, t) for t in range(n)] for x in grid])
    cmap = LinearSegmentedColormap.from_list("heat", [to_rgb(c["bg"]), to_rgb(c["heat"])])
    peaks = np.array([drifting_peak(t) for t in range(n)])
    frames = Frames("ibo-drift", theme)

    def draw(k, final=False):
        fig = plt.figure(figsize=(9.6, 4.1), dpi=100)
        axes = [fig.add_axes([0.07, 0.24, 0.4, 0.6]), fig.add_axes([0.56, 0.24, 0.4, 0.6])]
        for ax, name, title in zip(axes, ("dbo", "botorch"), ("Dynamic BO (DBO)", "Standard BO")):
            style(ax, c)
            ax.grid(False)
            ax.imshow(heat, origin="lower", aspect="auto", extent=(0.5, n + 0.5, 0, 1), cmap=cmap, vmin=1, vmax=10)
            ts, xs = np.arange(1, k + 1), picks[name][:k]
            ax.plot(ts, xs, color=c["accent"], linewidth=1, alpha=0.5)
            ax.scatter(ts, xs, s=36, color=c["accent"], zorder=5, edgecolor=c["bg"], linewidth=1)
            if k and not final:
                ax.scatter([k], [xs[-1]], s=170, facecolor="none", edgecolor=c["fg"], linewidth=1.5, zorder=6)
            if k > sampling:
                inside = int(np.sum(np.abs(xs[sampling:] - peaks[sampling:k]) < band))
                ax.text(1.0, 1.02, f"in the gold band: {inside} of {k - sampling}", transform=ax.transAxes,
                        ha="right", va="bottom", color=c["fg"], fontsize=10.5)
            ax.set_xlim(0.5, n + 0.5)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks([1, n // 2, n])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["low", "high"])
            ax.set_xlabel("Rating number (time in the session)")
            ax.set_title(title, color=c["fg"], fontsize=11.5, loc="left", fontweight="bold")
        axes[0].set_ylabel("Design parameter", labelpad=-10)
        counter(fig, c, f"{k} of {n} ratings")
        legend(fig, c, [Patch(color=c["heat"], label="rated highly at that moment"),
                        Line2D([], [], marker="o", color=c["accent"], linewidth=1, label="design shown at each rating")], ncol=2)
        return fig

    frames.add(draw(0), hold=14)
    for k in range(1, n + 1):
        frames.add(draw(k), hold=6)
    frames.add(draw(n, final=True), hold=70)
    frames.gif()


# Who does what: designer, participant, interactive system, iBO ------------------------------------------------------
ROLE_COLORS = {
    "light": dict(card="#f6f8fa", stroke="#d0d7de", screen="#ffffff", designer="#8250df", participant="#bc4c00",
                  control="#d0d7de", blank="#e6eaef"),
    "dark": dict(card="#161b22", stroke="#3d444d", screen="#0d1117", designer="#a371f7", participant="#f0883e",
                 control="#3d444d", blank="#262c36"),
}


def mix(a, b, t):
    a, b = np.array(to_rgb(a)), np.array(to_rgb(b))
    return tuple(a + (b - a) * min(1.0, max(0.0, t)))


def roles(theme):
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge
    from matplotlib.path import Path as MplPath
    c = dict(THEMES[theme], **ROLE_COLORS[theme])
    runs = STUDIES["roles"]
    n = len(runs)
    pareto = pareto_mask([(r["objectives"]["usability"], -r["objectives"]["time"]) for r in runs])
    best = int(np.argmax([r["objectives"]["usability"] for r in runs]))
    frames = Frames("ibo-roles", theme)
    arrows = {
        "setup": [(23.2, 33.0), (29.8, 33.0)],
        "to_ibo": [(61.2, 36.0), (68.8, 36.0)],
        "next": [(68.8, 26.0), (61.2, 26.0)],
        "show": [(38.0, 17.8), (38.0, 12.2)],
        "respond": [(44.0, 12.2), (44.0, 17.8)],
        "result": [(81.5, 44.2), (81.5, 45.9), (12.5, 45.9), (12.5, 42.2)],
    }

    def card(ax, x, y, w, h, active, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=c["card"],
                                    edgecolor=color if active else c["stroke"], linewidth=2.2 if active else 1))

    def person(ax, x, y, color, size=1.0):
        ax.add_patch(Circle((x, y + 1.25 * size), 0.85 * size, color=color))
        ax.add_patch(Wedge((x, y - 1.1 * size), 1.65 * size, 0, 180, color=color))

    def ibo_icon(ax, x, y, size):
        box = FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={size * 0.23}", facecolor="none",
                             edgecolor="none")
        ax.add_patch(box)
        u, v = np.meshgrid(np.linspace(0, 1, 64), np.linspace(1, 0, 64))
        g = np.clip((u + (1 - v)) / 2, 0, 1)[..., None]
        image = (1 - g) * np.array(to_rgb("#2f81f7")) + g * np.array(to_rgb("#a371f7"))
        ax.imshow(image, extent=(x, x + size, y, y + size), zorder=3, clip_path=box, clip_on=True)
        k = size / 140
        cx, cy, r = x + 70 * k, y + size - 72 * k, 46 * k
        t = np.radians(np.linspace(-118, -118 - 292, 80))
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color="white", linewidth=1.6, zorder=4, solid_capstyle="round")
        end, tan = np.radians(-410), np.array([np.sin(np.radians(-410)), -np.cos(np.radians(-410))])
        p1 = np.array([cx + r * np.cos(end), cy + r * np.sin(end)])
        nrm = np.array([np.cos(end), np.sin(end)])
        ax.add_patch(plt.Polygon([p1 + 13 * k * tan, p1 + 9 * k * nrm, p1 - 9 * k * nrm], color="white", zorder=4))
        xs = np.linspace(38, 102, 60)
        ax.plot(x + xs * k, y + size - (93 - 41 * np.exp(-((xs - 70) / 15) ** 2)) * k, color="white", linewidth=1.4, zorder=4)
        ax.add_patch(Circle((cx, y + size - 51 * k), 9.5 * k, facecolor="#fb8f44", edgecolor="white", linewidth=0.8, zorder=5))

    def arrow(ax, key, active, label, label_xy, ha="center", progress=None):
        pts = arrows[key]
        color = c["accent"] if active else c["muted"]
        alpha = 1 if active else 0.55
        if len(pts) == 2:
            ax.add_patch(FancyArrowPatch(pts[0], pts[1], arrowstyle="-|>", mutation_scale=14, color=color, alpha=alpha,
                                         linewidth=2 if active else 1.3, shrinkA=0, shrinkB=0))
        else:
            ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, alpha=alpha,
                    linewidth=2 if active else 1.3, solid_joinstyle="round")
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14, color=color, alpha=alpha,
                                         linewidth=2 if active else 1.3, shrinkA=0, shrinkB=0))
        if label:
            ax.text(*label_xy, label, color=color, fontsize=9.5, ha=ha, va="center", fontweight="bold" if active else "normal",
                    bbox=dict(facecolor=c["bg"], edgecolor="none", pad=1.5) if key == "result" else None)
        if active and progress is not None:
            seg = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
            d = progress * sum(seg)
            for (a, b), length in zip(zip(pts, pts[1:]), seg):
                if d <= length:
                    f = d / length
                    ax.add_patch(Circle((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), 0.75, color=c["accent"], zorder=6))
                    break
                d -= length

    def draw(state):
        fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
        fig.patch.set_facecolor(c["bg"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 96); ax.set_ylim(0, 54); ax.axis("off")
        active = state.get("active", set())
        ax.text(2.5, 51.4, "Who does what in an iBO study", color=c["fg"], fontsize=16, fontweight="bold", va="center")
        if state.get("final"):
            ax.text(93.5, 51.4, f"after {n} designs", color=c["muted"], fontsize=11.5, va="center", ha="right")
        elif state.get("design") is not None:
            ax.text(93.5, 51.4, f"design {state['design'] + 1} of {n}", color=c["muted"], fontsize=11.5, va="center", ha="right")

        # designer
        card(ax, 2, 22, 21, 20, "designer" in active, c["designer"])
        person(ax, 5.6, 37.6, c["designer"], 1.15)
        ax.text(8.6, 39.4, "Designer", color=c["fg"], fontsize=12.5, fontweight="bold", va="center")
        ax.text(8.6, 37.2, "researcher, developer", color=c["muted"], fontsize=9.5, va="center")
        lines = [("may change:", True), ("button size, contrast", False), ("counts as good:", True),
                 ("usability rating ↑, task time ↓", False)]
        for i, (line, bold) in enumerate(lines[:state.get("setup", 4)]):
            ax.text(3.6, 33.2 - i * 2.35 - (0.5 if i >= 2 else 0), line, color=c["muted"] if bold else c["fg"],
                    fontsize=9.5, fontweight="bold" if bold else "normal", va="center")
        if state.get("final"):
            ax.text(3.6, 23.9, "gets: best designs, logs", color=c["accent"], fontsize=9.5, fontweight="bold", va="center")

        # interactive system
        card(ax, 30, 18, 31, 26, "system" in active, c["accent"])
        for i in range(3):
            ax.add_patch(Circle((31.8 + i * 1.3, 42.4), 0.42, color=c["control"]))
        ax.text(35.8, 42.4, "Interactive system", color=c["fg"], fontsize=10.5, fontweight="bold", va="center")
        ax.add_patch(FancyBboxPatch((31.2, 22.4), 28.6, 18.4, boxstyle="round,pad=0,rounding_size=0.6", facecolor=c["screen"],
                                    edgecolor=c["stroke"], linewidth=0.8))
        ax.text(45.5, 39.3, "built with Unity, Unreal, Python or your own tool", color=c["muted"], fontsize=8.5, ha="center",
                va="center", style="italic")
        k = state.get("design")
        if k is None:
            ax.text(45.5, 31.6, "waiting for the first design", color=c["muted"], fontsize=9.5, ha="center", va="center", style="italic")
        else:
            size, contrast = runs[k]["parameters"]["size"], runs[k]["parameters"]["contrast"]
            w = 2.6 + 6.6 * size
            h = 1.2 + 2.0 * size
            face = mix(c["blank"], c["accent"], contrast)
            label = mix(c["muted"], "#ffffff", contrast)
            ax.add_patch(FancyBboxPatch((45.5 - w / 2, 31.8 - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={min(h / 2, 1.2)}",
                                        facecolor=face, edgecolor="none"))
            ax.text(45.5, 31.8, "Start", color=label, fontsize=6 + 5.5 * size, ha="center", va="center", fontweight="bold")
            ax.text(45.5, 24.2, f"size {size:.2f}  ·  contrast {contrast:.2f}", color=c["muted"], fontsize=9, ha="center", va="center")
            if state.get("final") and k == best:
                ax.text(45.5, 27.4, f"highest-rated design (design {k + 1})", color=c["best"], fontsize=9.5, fontweight="bold", ha="center", va="center")

        # participant and the response
        person(ax, 41, 7.2, c["participant"], 1.3)
        ax.text(35.8, 8.2, "Participant", color=c["fg"], fontsize=12.5, fontweight="bold", va="center", ha="right")
        ax.text(35.8, 5.9, "uses each design", color=c["muted"], fontsize=9.5, va="center", ha="right")
        if "participant" in active:
            ax.add_patch(Circle((41, 7.2), 3.4, facecolor="none", edgecolor=c["participant"], linewidth=1.6))
        if state.get("response") is not None:
            r = runs[state["response"]]["objectives"]
            card(ax, 47.5, 2.6, 16.5, 9.2, True, c["participant"])
            stars = int(round(r["usability"] / 20))
            ax.text(48.8, 9.4, "rating", color=c["muted"], fontsize=9, va="center")
            for i in range(5):
                ax.plot(53.6 + i * 1.65, 9.4, marker="*", markersize=10, color=c["participant"] if i < stars else c["control"],
                        markeredgewidth=0)
            ax.text(61.9, 9.4, f"{r['usability']:.0f}", color=c["fg"], fontsize=9.5, fontweight="bold", va="center", ha="center")
            ax.text(48.8, 6.6, "task time", color=c["muted"], fontsize=9, va="center")
            ax.text(55.4, 6.6, f"{r['time']:.1f} s", color=c["fg"], fontsize=10.5, fontweight="bold", va="center")
            ax.text(48.8, 4.1, "measured, not asked", color=c["muted"], fontsize=8, va="center", style="italic")

        # iBO
        card(ax, 69, 18, 25, 26, "ibo" in active, c["accent"])
        ibo_icon(ax, 70.6, 37.8, 4.8)
        ax.text(76.6, 41.4, "iBO", color=c["fg"], fontsize=13, fontweight="bold", va="center")
        ax.text(76.6, 39.2, "proposes the next design", color=c["muted"], fontsize=9.5, va="center")
        inset = fig.add_axes([0.76, 0.38, 0.19, 0.25])
        inset.set_facecolor(c["card"])
        for side in inset.spines.values():
            side.set_color(c["stroke"])
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_xlim(1.6, 9.6); inset.set_ylim(-8, 108)
        inset.set_xlabel("task time (lower is better)", color=c["muted"], fontsize=7.5, labelpad=2)
        inset.set_ylabel("rating (higher is better)", color=c["muted"], fontsize=7.5, labelpad=2)
        inset.set_title("each dot: one rated design", color=c["muted"], fontsize=8, pad=3, loc="left")
        m = state.get("learned", 0)
        if m:
            seen = pareto_mask([(r["objectives"]["usability"], -r["objectives"]["time"]) for r in runs[:m]])
            for i, r in enumerate(runs[:m]):
                o = r["objectives"]
                inset.scatter([o["time"]], [o["usability"]], s=26 if seen[i] else 18, color=c["accent"] if seen[i] else c["muted"],
                              zorder=3, edgecolor=c["card"], linewidth=0.6)
            if state.get("newest"):
                o = runs[m - 1]["objectives"]
                inset.scatter([o["time"]], [o["usability"]], s=110, facecolor="none", edgecolor=c["fg"], linewidth=1.1, zorder=4)
        ax.scatter([72.3], [16.4], s=26, color=c["accent"]); ax.text(73.1, 16.4, "best trade-offs", color=c["muted"], fontsize=8, va="center")
        ax.scatter([82.6], [16.4], s=18, color=c["muted"]); ax.text(83.4, 16.4, "others", color=c["muted"], fontsize=8, va="center")

        # arrows
        go = state.get("arrow")
        arrow(ax, "setup", go == "setup", "starts", (26.5, 34.6))
        arrow(ax, "to_ibo", go == "to_ibo", state.get("to_ibo_label", "responses"), (65.0, 37.6))
        arrow(ax, "next", go == "next", "next design", (65.0, 24.4))
        arrow(ax, "show", go == "show", "shows the design", (37.2, 15.0), ha="right")
        arrow(ax, "respond", go == "respond", "rating, behavior", (44.8, 15.0), ha="left")
        arrow(ax, "result", go == "result", "best designs and logs", (46.0, 45.9), progress=state.get("progress"))
        if go in ("setup", "to_ibo", "next", "show", "respond"):
            arrow(ax, go, True, None, (0, 0), progress=state.get("progress"))
        return fig

    def travel(state, steps):
        for s in range(1, steps + 1):
            frames.add(draw(dict(state, progress=s / steps)))

    # "phase" names each step of the story; the README explains it, the figure only shows it.
    base = dict(phase="Three roles: the designer sets up a study, the participant tries the designs, iBO picks them.", setup=0)
    frames.add(draw(base), hold=30)
    for i in range(1, 5):
        frames.add(draw(dict(base, setup=i, active={"designer"},
                             phase="The designer decides what may change and what counts as good.")), hold=10)
    setup = dict(setup=4, phase="The designer starts the study in the interactive system ...")
    travel(dict(setup, arrow="setup", active={"designer", "system"}), 10)
    travel(dict(setup, arrow="to_ibo", to_ibo_label="study setup", active={"system", "ibo"},
                phase="... which hands the study to iBO."), 10)
    learned = 0
    for k in range(n):
        detail = k == 0 or k == best
        st = dict(setup=4, learned=learned)
        travel(dict(st, arrow="next", active={"ibo", "system"}, design=k - 1 if k else None,
                    phase=f"iBO proposes design {k + 1}; the system shows it."), 8 if detail else 4)
        frames.add(draw(dict(st, design=k, active={"system"}, phase=f"iBO proposes design {k + 1}; the system shows it.")),
                   hold=16 if detail else 6)
        if detail:
            travel(dict(st, design=k, arrow="show", active={"system", "participant"},
                        phase="The participant just uses the design; iBO stays in the background."), 8)
        frames.add(draw(dict(st, design=k, response=k, active={"participant"},
                             phase="The participant rates the design; the system also measures the task time.")), hold=25 if detail else 7)
        if detail:
            travel(dict(st, design=k, response=k, arrow="respond", active={"participant", "system"},
                        phase="The participant rates the design; the system also measures the task time."), 7)
            travel(dict(st, design=k, response=k, arrow="to_ibo", active={"system", "ibo"},
                        phase="iBO learns from the response and picks the next design on its own."), 7)
        learned = k + 1
        frames.add(draw(dict(st, learned=learned, newest=True, design=k, response=k if detail else None, active={"ibo"},
                             phase="iBO learns from the response and picks the next design on its own." if detail else "... and the loop repeats.")),
                   hold=14 if detail else 4)
    final = dict(setup=4, learned=n, design=best, final=True)
    travel(dict(final, arrow="result", active={"ibo", "designer"},
                phase="At the end the designer gets the best designs and the logs of every response."), 14)
    frames.add(draw(dict(final, active={"designer"},
                         phase="At the end the designer gets the best designs and the logs of every response.")), hold=60)
    frames.gif()

if __name__ == "__main__":
    for theme in THEMES:
        roles(theme)
        one_parameter(theme)
        trade_offs(theme)
        drift(theme)
