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


def header(fig, c, title, phase, counter):
    fig.text(0.03, 0.94, title, color=c["fg"], fontsize=15, fontweight="bold", va="top")
    fig.text(0.03, 0.865, phase, color=c["muted"], fontsize=11.5, va="top")
    fig.text(0.97, 0.94, counter, color=c["muted"], fontsize=11.5, va="top", ha="right")


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
        fig = plt.figure(figsize=(9, 4.6), dpi=100)
        ax = fig.add_axes([0.07, 0.23, 0.9, 0.55])
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
        if final:
            phase = "Done: iBO settled on the best design; its last ratings cluster at the peak."
        elif next_index is not None and next_index < sampling:
            phase = "Phase 1, sampling: a few spread-out designs give a first overview."
        else:
            phase = "Phase 2, optimization: iBO tries designs where its estimate is high or still uncertain."
        header(fig, c, "How iBO finds the best design", phase, f"{k} of {n} ratings")
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
        fig = plt.figure(figsize=(9.6, 4.8), dpi=100)
        left = fig.add_axes([0.07, 0.17, 0.37, 0.58])
        right = fig.add_axes([0.59, 0.17, 0.37, 0.58])
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
        if final:
            phase = "Done: the blue designs are the best trade-offs; improving one goal further would cost the other."
        elif k < sampling:
            phase = "Phase 1, sampling: spread-out designs. The dotted line links each design to its result."
        else:
            phase = "Phase 2, optimization: iBO looks for designs that push the best trade-offs further out."
        header(fig, c, "Two goals that conflict: finding the best trade-offs", phase, f"{k} of {n} ratings")
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
        fig = plt.figure(figsize=(9.6, 4.8), dpi=100)
        axes = [fig.add_axes([0.07, 0.2, 0.4, 0.55]), fig.add_axes([0.56, 0.2, 0.4, 0.55])]
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
        if final:
            phase = "DBO trusts recent ratings more and follows the moving best design; standard BO weighs old and new alike."
        elif k <= sampling:
            phase = "The participant adapts, so the best design moves (gold band). Both start with the same spread-out designs."
        else:
            phase = "The participant adapts, so the best design moves (gold band). Which optimizer keeps up?"
        header(fig, c, "When the best design drifts during a session", phase, f"{k} of {n} ratings")
        legend(fig, c, [Patch(color=c["heat"], label="rated highly at that moment"),
                        Line2D([], [], marker="o", color=c["accent"], linewidth=1, label="design shown at each rating")], ncol=2)
        return fig

    frames.add(draw(0), hold=14)
    for k in range(1, n + 1):
        frames.add(draw(k), hold=6)
    frames.add(draw(n, final=True), hold=70)
    frames.gif()


if __name__ == "__main__":
    for theme in THEMES:
        one_parameter(theme)
        trade_offs(theme)
        drift(theme)
