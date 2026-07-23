"""Generate an animated schematic of the BELIEF-STATE pipeline for the README.

Produces ``figures/pipeline.gif`` -- a looping walkthrough of the four stages
(identify -> probe -> steer -> stress) drawn over a stylized residual stream.
This is an *explanatory schematic*, not a plot of results; the empirical numbers
live in results_*/ and the run-generated figures. Pure matplotlib + pillow so it
renders headless with no extra dependencies.

    python3 -m scripts.make_readme_gif
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Repo palette (matches beliefstate/figures.py).
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GREY = "#9aa3ad"
INK = "#1b2733"
BG = "#f7f8fa"

N_LAYERS = 12          # stylized residual-stream depth
BELIEF_LAYER = 8       # where the belief direction "lives" in the schematic
FPS = 20
HOLD = 22              # frames held on each fully-drawn stage


def _fade(c, t):
    """Blend hex color c toward the background by (1 - t)."""
    t = float(np.clip(t, 0.0, 1.0))
    c = np.array(matplotlib.colors.to_rgb(c))
    b = np.array(matplotlib.colors.to_rgb(BG))
    return tuple(b + (c - b) * t)


def _base_axes():
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def _draw_stream(ax, alpha=1.0):
    """The residual stream: a vertical stack of layer blocks."""
    x0, w = 0.7, 1.15
    ys = np.linspace(0.6, 5.2, N_LAYERS)
    for i, y in enumerate(ys):
        is_belief = i == BELIEF_LAYER
        fc = _fade(BLUE if is_belief else "#dfe4ea", alpha)
        ec = _fade(BLUE if is_belief else GREY, alpha)
        box = FancyBboxPatch((x0, y - 0.14), w, 0.28,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             fc=fc, ec=ec, lw=1.6 if is_belief else 1.0, zorder=3)
        ax.add_patch(box)
    ax.text(x0 + w / 2, 5.62, "residual stream", ha="center", va="bottom",
            fontsize=9.5, color=_fade(INK, alpha), style="italic")
    ax.text(x0 + w / 2, ys[BELIEF_LAYER] + 0.0, "L8", ha="center", va="center",
            fontsize=7.5, color=_fade("white", alpha), zorder=4, weight="bold")
    return x0, w, ys


def _title(ax, step_no, title, sub, alpha=1.0):
    ax.text(3.4, 5.55, f"{step_no}", fontsize=26, color=_fade(BLUE, alpha),
            weight="bold", va="center")
    ax.text(3.95, 5.72, title, fontsize=17, color=_fade(INK, alpha),
            weight="bold", va="center")
    ax.text(3.95, 5.28, sub, fontsize=10.5, color=_fade("#55606b", alpha),
            va="center")


def _header(ax):
    ax.text(0.7, 5.95, "BELIEF-STATE", fontsize=13, color=INK, weight="bold")


# ---------------------------------------------------------------------------
# Stage panels. Each returns nothing; draws onto ax at reveal fraction p in[0,1]
# ---------------------------------------------------------------------------

def stage_identify(ax, p):
    _header(ax)
    x0, w, ys = _draw_stream(ax, alpha=1.0)
    _title(ax, "1", "Identify", "difference-of-means over controlled contrasts", p)
    bx = 4.2
    # Two contrast conditions feeding a difference vector.
    for k, (lbl, col, dy) in enumerate([("history A", BLUE, 3.9),
                                         ("history B", ORANGE, 2.9)]):
        a = min(1.0, p * 2 - k * 0.2)
        if a <= 0:
            continue
        box = FancyBboxPatch((bx, dy - 0.28), 1.9, 0.56,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             fc=_fade("white", a), ec=_fade(col, a), lw=1.8, zorder=3)
        ax.add_patch(box)
        ax.text(bx + 0.95, dy, lbl, ha="center", va="center",
                fontsize=9.5, color=_fade(INK, a))
    if p > 0.55:
        a = (p - 0.55) / 0.45
        ax.add_patch(FancyArrowPatch((bx + 1.95, 3.4), (7.4, 3.4),
                     arrowstyle="-|>", mutation_scale=16,
                     color=_fade(GREEN, a), lw=2.2, zorder=4))
        ax.text(8.9, 3.4, "b", ha="center", va="center", fontsize=20,
                color=_fade(GREEN, a), weight="bold", style="italic")
        ax.text(8.9, 2.95, "belief\ndirection", ha="center", va="center",
                fontsize=8.5, color=_fade("#55606b", a))
        # Tie b back to the belief layer.
        ax.add_patch(FancyArrowPatch((8.9, 2.7), (x0 + w, ys[BELIEF_LAYER]),
                     arrowstyle="-|>", mutation_scale=10, connectionstyle="arc3,rad=0.25",
                     color=_fade(GREY, a), lw=1.2, ls=(0, (3, 3)), zorder=2))


def stage_probe(ax, p):
    _header(ax)
    x0, w, ys = _draw_stream(ax, alpha=1.0)
    _title(ax, "2", "Probe & localize", "belief > null separation, layer-corrected", p)
    # Per-layer separation bars growing to a peak at the belief layer.
    peak = BELIEF_LAYER
    bx = 4.3
    for i, y in enumerate(ys):
        d = np.exp(-((i - peak) ** 2) / 6.0)  # bell centered on belief layer
        a = min(1.0, max(0.0, p * 1.3 - i / N_LAYERS * 0.3))
        length = 3.4 * d * a
        col = GREEN if abs(i - peak) <= 2 else GREY
        ax.add_patch(FancyBboxPatch((bx, y - 0.09), max(0.001, length), 0.18,
                     boxstyle="round,pad=0.005,rounding_size=0.03",
                     fc=_fade(col, a), ec="none", zorder=3))
    if p > 0.6:
        a = (p - 0.6) / 0.4
        ax.text(bx + 3.5, ys[peak], "◄ significant\n   deep layers", fontsize=9,
                color=_fade(GREEN, a), va="center", weight="bold")
        ax.text(bx, 0.15, "separation from surface-only null (per layer)",
                fontsize=8.5, color=_fade("#55606b", a))


def stage_steer(ax, p):
    _header(ax)
    x0, w, ys = _draw_stream(ax, alpha=1.0)
    _title(ax, "3", "Steer & verify", "h' = h + αb  vs random / orthogonal / swap", p)
    yb = ys[BELIEF_LAYER]
    # The steering injection at the belief layer.
    if p > 0.15:
        a = min(1.0, (p - 0.15) / 0.3)
        ax.add_patch(FancyArrowPatch((x0 + w + 0.15, yb), (x0 + w - 0.02, yb),
                     arrowstyle="-|>", mutation_scale=16,
                     color=_fade(RED, a), lw=2.4, zorder=5))
        ax.text(x0 + w + 0.9, yb + 0.28, "+αb", fontsize=12, color=_fade(RED, a),
                weight="bold", ha="center")
    # Independent downstream behavioral readout responding.
    rx = 5.6
    if p > 0.4:
        a = min(1.0, (p - 0.4) / 0.4)
        ax.add_patch(FancyBboxPatch((rx, 2.6), 3.4, 1.4,
                     boxstyle="round,pad=0.04,rounding_size=0.1",
                     fc=_fade("white", a), ec=_fade(INK, a), lw=1.5, zorder=3))
        ax.text(rx + 1.7, 3.72, "independent readout", ha="center", fontsize=9.5,
                color=_fade(INK, a), weight="bold")
        ax.text(rx + 1.7, 3.44, "next-token  risk − calm", ha="center",
                fontsize=8.5, color=_fade("#55606b", a), style="italic")
        # A little needle that swings right as alpha is applied.
        swing = -0.7 + 1.4 * min(1.0, (p - 0.4) / 0.55)
        cx, cy = rx + 1.7, 2.95
        ax.plot([cx, cx + swing], [cy, cy + 0.28], color=_fade(RED, a), lw=2.6,
                solid_capstyle="round", zorder=4)
        ax.add_patch(plt.Circle((cx, cy), 0.05, color=_fade(INK, a), zorder=5))
        ax.text(cx - 1.15, cy - 0.02, "calm", fontsize=7.5, color=_fade(GREY, a), va="center")
        ax.text(cx + 1.15, cy - 0.02, "risk", fontsize=7.5, color=_fade(RED, a), va="center", ha="right")
    if p > 0.85:
        a = (p - 0.85) / 0.15
        ax.text(rx + 1.7, 2.35, "readout ≠ steering axis  (non-circular)",
                ha="center", fontsize=8, color=_fade(GREEN, a), weight="bold")


def stage_stress(ax, p):
    _header(ax)
    _draw_stream(ax, alpha=0.35)  # recede the stream; foreground the drift trace
    _title(ax, "4", "Stress", "belief-projection drift under regime shift", p)
    ax0x, ax0y, W, H = 4.1, 1.1, 5.0, 3.2
    # Axes frame.
    ax.plot([ax0x, ax0x, ax0x + W], [ax0y + H, ax0y, ax0y], color=_fade(INK, p), lw=1.3)
    ax.text(ax0x + W / 2, ax0y - 0.3, "reasoning step", ha="center", fontsize=9,
            color=_fade("#55606b", p))
    ax.text(ax0x - 0.25, ax0y + H / 2, "|projection|", rotation=90, va="center",
            ha="center", fontsize=9, color=_fade("#55606b", p))
    n = 40
    xs = np.linspace(ax0x + 0.1, ax0x + W - 0.1, n)
    tt = np.linspace(0, 1, n)
    stable = 0.35 + 0.02 * np.sin(tt * 9)          # flat baseline
    drift = 0.35 + 0.02 * np.sin(tt * 9) + 1.9 * tt ** 2.4   # accelerating drift
    show = int(np.clip(p * n, 1, n))
    ys_s = ax0y + stable * H
    ys_d = ax0y + drift * H
    ax.plot(xs[:show], ys_s[:show], color=_fade(GREY, 1), lw=2.0, label="stable")
    ax.plot(xs[:show], np.clip(ys_d[:show], None, ax0y + H), color=_fade(RED, 1),
            lw=2.4, label="regime shift")
    if show > 1:
        ax.scatter([xs[show - 1]], [min(ys_d[show - 1], ax0y + H)], s=28,
                   color=RED, zorder=5)
    if p > 0.75:
        a = (p - 0.75) / 0.25
        # Mark drift onset preceding failure.
        onset = int(0.55 * n)
        if show > onset:
            ax.axvline(xs[onset], ymin=0.18, ymax=0.9, color=_fade(ORANGE, a),
                       lw=1.4, ls=(0, (4, 3)))
            ax.text(xs[onset], ax0y + H + 0.08, "drift onset", ha="center",
                    fontsize=8.5, color=_fade(ORANGE, a), weight="bold")
    ax.text(ax0x + 0.15, ax0y + H - 0.12, "— regime shift", color=_fade(RED, p),
            fontsize=8.5, va="top")
    ax.text(ax0x + 0.15, ax0y + H - 0.42, "— stable", color=_fade(GREY, p),
            fontsize=8.5, va="top")


STAGES = [stage_identify, stage_probe, stage_steer, stage_stress]
REVEAL = 26  # frames to draw each stage in


def _render_frames():
    frames = []
    for stage in STAGES:
        for f in range(REVEAL + HOLD):
            p = min(1.0, (f + 1) / REVEAL)
            fig, ax = _base_axes()
            stage(ax, p)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(buf)
            plt.close(fig)
    return frames


def main():
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pipeline.gif")
    frames = _render_frames()
    try:
        import imageio.v2 as imageio  # noqa
        imageio.mimsave(out, frames, duration=1.0 / FPS, loop=0)
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {out}  ({len(frames)} frames, {len(frames)/FPS:.1f}s loop)")


if __name__ == "__main__":
    main()
