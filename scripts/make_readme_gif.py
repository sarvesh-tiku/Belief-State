"""Generate an animated dashboard schematic of the BELIEF-STATE pipeline.

Produces ``figures/pipeline.gif`` -- a looping, continuous dashboard that walks
through the four stages (identify -> localize -> steer -> stress) over a shared
frame: title, model badge, and a progress tracker that advances stage to stage.
Each panel is annotated with the *actual verified numbers* from the
Qwen2.5-1.5B-Instruct run (results_qwen/), so the animation carries information,
not just decoration. It remains a schematic -- the ground-truth figures live in
results_*/ and the run-generated plots.

Pure matplotlib + pillow; renders headless with no extra dependencies.

    python3 -m scripts.make_readme_gif
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ---- palette (matches beliefstate/figures.py) ----------------------------
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#7e57c2"
GREY = "#9aa3ad"
LGREY = "#c7ced6"
INK = "#1b2733"
MUT = "#66707a"
BG = "#f7f8fa"
PANEL = "#eef1f5"

# ---- model facts + verified results (results_qwen/) -----------------------
N_LAYERS = 28
PEAK = 27
SIGNIF = {18, 21, 22, 25, 27, 28}   # layers surviving Holm-Bonferroni
FPS = 20
REVEAL = 28
HOLD = 16


def ease(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def _fade(c, t):
    t = float(np.clip(t, 0.0, 1.0))
    c = np.array(matplotlib.colors.to_rgb(c))
    b = np.array(matplotlib.colors.to_rgb(BG))
    return tuple(b + (c - b) * t)


def _base_axes():
    fig, ax = plt.subplots(figsize=(12.0, 6.9), dpi=125)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    return fig, ax


def _chip(ax, x, y, w, text, fc=PANEL, ec=LGREY, tc=INK, a=1.0, fs=9.5,
          weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y - 0.19), w, 0.38,
                 boxstyle="round,pad=0.02,rounding_size=0.09",
                 fc=_fade(fc, a), ec=_fade(ec, a), lw=1.1, zorder=6))
    tx = x + w / 2 if ha == "center" else x + 0.14
    ax.text(tx, y, text, ha=ha, va="center", fontsize=fs,
            color=_fade(tc, a), zorder=7, weight=weight)


# ---------------------------------------------------------------------------
# Persistent chrome: title, model badge, 4-step progress tracker.
# ---------------------------------------------------------------------------

STEPS = [("1", "Identify"), ("2", "Localize"), ("3", "Steer"), ("4", "Stress")]
STEP_CX = [1.55, 4.15, 6.75, 9.35]


def draw_chrome(ax, active, within):
    ax.text(0.55, 6.86, "BELIEF-STATE", fontsize=17, weight="bold", color=INK)
    ax.text(0.55, 6.52, "temporal belief dynamics in agentic financial systems",
            fontsize=9.0, style="italic", color=MUT)

    # model badge (top-right)
    ax.add_patch(FancyBboxPatch((9.5, 6.44), 2.2, 0.5,
                 boxstyle="round,pad=0.02,rounding_size=0.1",
                 fc=PANEL, ec=LGREY, lw=1.0, zorder=5))
    ax.text(10.6, 6.78, "Qwen2.5-1.5B-Instruct", ha="center", fontsize=8.8,
            weight="bold", color=INK, zorder=6)
    ax.text(10.6, 6.55, "28 layers · d=1536 · real activations", ha="center",
            fontsize=6.8, color=MUT, zorder=6)

    y = 6.02
    # connectors
    for i in range(len(STEPS) - 1):
        done = i < active
        ax.plot([STEP_CX[i] + 0.22, STEP_CX[i + 1] + 0.22], [y, y],
                color=GREEN if done else LGREY, lw=2.2, zorder=2,
                solid_capstyle="round")
    for i, (num, name) in enumerate(STEPS):
        cx = STEP_CX[i]
        done, act = i < active, i == active
        fc = GREEN if done else (BLUE if act else "#ffffff")
        ec = GREEN if done else (BLUE if act else LGREY)
        r = 0.18 + (0.03 * np.sin(within * np.pi) if act else 0.0)
        ax.add_patch(Circle((cx, y), r, fc=fc, ec=ec, lw=2.0, zorder=4))
        lbl = "✓" if done else num
        ax.text(cx, y, lbl, ha="center", va="center", fontsize=10.5,
                color="white" if (done or act) else GREY, weight="bold", zorder=5)
        ax.text(cx + 0.34, y, name, ha="left", va="center", fontsize=10.5,
                color=INK if act else (MUT if not done else "#3d7a3d"),
                weight="bold" if act else "normal", zorder=5)

    ax.text(11.9, 0.16, "schematic · numbers from results_qwen/", ha="right",
            va="bottom", fontsize=7.0, color=MUT, style="italic")


# ---------------------------------------------------------------------------
# Residual stream column (persistent through stages 1-3).
# ---------------------------------------------------------------------------

def draw_stream(ax, alpha=1.0, mode="peak"):
    x0, w = 0.78, 0.9
    ys = np.linspace(0.55, 5.15, N_LAYERS)
    h = (ys[1] - ys[0]) * 0.72
    for i, yy in enumerate(ys):
        layer = i + 1
        sig = (mode == "signif") and layer in SIGNIF
        pk = layer == PEAK and mode in ("peak", "steer", "signif")
        if sig:
            fc, ec, lw = GREEN, GREEN, 1.4
        elif pk:
            fc, ec, lw = BLUE, BLUE, 1.6
        else:
            fc, ec, lw = "#e3e8ee", GREY, 0.9
        ax.add_patch(FancyBboxPatch((x0, yy - h / 2), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.03",
                     fc=_fade(fc, alpha), ec=_fade(ec, alpha), lw=lw, zorder=3))
    ax.text(x0 + w / 2, 5.45, "residual stream", ha="center", fontsize=8.5,
            style="italic", color=_fade(INK, alpha))
    ax.text(x0 + w / 2, 0.28, "layer 1  →  28", ha="center", fontsize=6.8,
            color=_fade(MUT, alpha))
    return x0, w, ys


def _content_header(ax, text, a):
    ax.text(2.35, 5.42, text, fontsize=10.5, color=_fade(INK, a), weight="bold")


# ---------------------------------------------------------------------------
# Stage 1 — Identify.
# ---------------------------------------------------------------------------

def stage_identify(ax, p):
    draw_chrome(ax, 0, p)
    x0, w, ys = draw_stream(ax, 1.0, mode="peak")
    a = ease(p)
    _content_header(ax, "Recover b by difference-of-means over controlled contrasts", a)

    rng = np.random.default_rng(7)
    # two point clouds: contrast condition A (blue) vs B (orange)
    cA = np.array([4.15, 3.55]); cB = np.array([4.95, 2.35])
    ptsA = cA + rng.normal(0, 0.34, (7, 2)) * [1.0, 0.8]
    ptsB = cB + rng.normal(0, 0.34, (7, 2)) * [1.0, 0.8]
    nshow = int(round(ease(min(1.0, p * 1.4)) * 7))
    for k in range(nshow):
        ax.scatter(*ptsA[k], s=42, color=BLUE, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=0.6)
        ax.scatter(*ptsB[k], s=42, color=ORANGE, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=0.6)
    if p > 0.15:
        ax.scatter(*cA, s=120, color=BLUE, marker="X", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.scatter(*cB, s=120, color=ORANGE, marker="X", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(cA[0] - 0.15, cA[1] + 0.32, "vary history\n(hold disclosure)",
                fontsize=7.5, color=BLUE, ha="center", zorder=6)
        ax.text(cB[0] + 0.15, cB[1] - 0.45, "vary disclosure\n(hold history)",
                fontsize=7.5, color=ORANGE, ha="center", zorder=6)
    if p > 0.5:
        aa = ease((p - 0.5) / 0.5)
        ax.add_patch(FancyArrowPatch(tuple(cB), tuple(cA), arrowstyle="-|>",
                     mutation_scale=18, color=_fade(GREEN, aa), lw=2.6, zorder=6))
        mid = (cA + cB) / 2
        ax.text(mid[0] + 0.95, mid[1], "b", fontsize=20, color=_fade(GREEN, aa),
                weight="bold", style="italic", zorder=7)
        ax.text(mid[0] + 0.98, mid[1] - 0.38, "belief\ndirection", fontsize=7.5,
                color=_fade(MUT, aa), ha="center", zorder=7)
        # tie b back to the peak layer
        ax.add_patch(FancyArrowPatch((mid[0] + 0.9, mid[1] + 0.28),
                     (x0 + w, ys[PEAK - 1]), arrowstyle="-|>", mutation_scale=9,
                     connectionstyle="arc3,rad=0.3", color=_fade(GREY, aa),
                     lw=1.1, ls=(0, (3, 3)), zorder=2))
    if p > 0.8:
        aa = ease((p - 0.8) / 0.2)
        _chip(ax, 6.9, 3.0, 4.6,
              "belief > null:   p = 6e-4    d = 0.70    N_eff = 16.6",
              fc="#eaf4ea", ec=GREEN, tc="#256b25", a=aa, weight="bold")
        _chip(ax, 6.9, 2.35, 4.6, "survives firm-level cluster permutation test",
              a=aa, fs=8.5, tc=MUT)


# ---------------------------------------------------------------------------
# Stage 2 — Localize.
# ---------------------------------------------------------------------------

def stage_localize(ax, p):
    draw_chrome(ax, 1, p)
    x0, w, ys = draw_stream(ax, 1.0, mode="signif")
    a = ease(p)
    _content_header(ax, "Test belief-vs-null separation at every layer, corrected", a)

    depth = np.clip((np.arange(N_LAYERS) - 13) / 14.0, 0, 1)
    sep = 0.12 + 0.30 * depth
    for l in SIGNIF:
        sep[l - 1] = 0.78 + 0.05 * ((l * 7) % 5)
    bx = 2.6
    scale = 4.2
    thr_x = bx + 0.6 * scale
    for i, yy in enumerate(ys):
        layer = i + 1
        sig = layer in SIGNIF
        reveal = ease(np.clip(p * 1.4 - (N_LAYERS - i) / N_LAYERS * 0.3, 0, 1))
        length = max(0.001, sep[i] * scale * reveal)
        col = GREEN if sig else GREY
        ax.add_patch(FancyBboxPatch((bx, yy - 0.055), length, 0.11,
                     boxstyle="round,pad=0.002,rounding_size=0.02",
                     fc=_fade(col, 1.0), ec="none", zorder=3))
        if sig and reveal > 0.9:
            ax.scatter(bx + length + 0.08, yy, s=16, color=GREEN, zorder=4)
    if p > 0.55:
        aa = ease((p - 0.55) / 0.45)
        ax.plot([thr_x, thr_x], [0.5, 4.95], color=_fade(RED, aa), lw=1.3,
                ls=(0, (5, 3)), zorder=5)
        ax.text(thr_x, 5.02, "significance threshold", ha="center", fontsize=7.2,
                color=_fade(RED, aa), zorder=6)
        # bracket over significant deep layers
        yhi, ylo = ys[27], ys[17]
        ax.annotate("", xy=(bx + 4.35, yhi), xytext=(bx + 4.35, ylo),
                    arrowprops=dict(arrowstyle="-", color=_fade(GREEN, aa), lw=1.4))
        ax.text(bx + 4.5, (yhi + ylo) / 2, "6 / 28 layers\nsignificant",
                fontsize=8.8, color=_fade("#256b25", aa), va="center",
                weight="bold")
    if p > 0.78:
        aa = ease((p - 0.78) / 0.22)
        _chip(ax, 7.55, 1.35, 4.0,
              "Holm–Bonferroni    min adj-p = 0.014", fc="#eaf4ea", ec=GREEN,
              tc="#256b25", a=aa, weight="bold", fs=9)
        _chip(ax, 7.55, 0.78, 4.0, "GPT-2 baseline:  0 layers survive  ✗",
              a=aa, fs=8.5, tc=RED, ec=LGREY)


# ---------------------------------------------------------------------------
# Stage 3 — Steer (alpha sweep drives the reveal).
# ---------------------------------------------------------------------------

def stage_steer(ax, p):
    draw_chrome(ax, 2, p)
    x0, w, ys = draw_stream(ax, 1.0, mode="steer")
    a = ease(p)
    _content_header(ax, "Steer  h' = h + αb  and read an independent channel", a)

    yb = ys[PEAK - 1]
    ax.add_patch(FancyArrowPatch((x0 + w + 0.55, yb), (x0 + w + 0.03, yb),
                 arrowstyle="-|>", mutation_scale=15, color=RED, lw=2.4, zorder=5))
    ax.text(x0 + w + 0.9, yb + 0.22, "+αb", fontsize=12, color=RED,
            weight="bold", ha="left")

    # response plot region
    px0, py0, pw, ph = 3.3, 1.35, 4.7, 3.5
    cx0 = px0 + pw / 2
    cy0 = py0 + ph / 2

    def X(al):
        return px0 + (al + 8) / 16.0 * pw

    def Y(r):
        return cy0 + r / 0.30 * (ph / 2 - 0.15)

    # axes
    ax.plot([px0, px0], [py0, py0 + ph], color=INK, lw=1.1, zorder=3)
    ax.plot([px0, px0 + pw], [cy0, cy0], color=LGREY, lw=1.0, zorder=3)
    ax.text(cx0, py0 - 0.28, "steering  α  (−8 … +8)", ha="center",
            fontsize=8.5, color=MUT)
    ax.text(px0 - 0.18, py0 + ph, "risk", fontsize=8, color=RED, ha="right", va="top")
    ax.text(px0 - 0.18, py0, "calm", fontsize=8, color=GREY, ha="right", va="bottom")
    ax.text(px0 - 0.42, cy0, "behavioral\nreadout", rotation=90, va="center",
            ha="center", fontsize=8, color=MUT)

    al_full = np.linspace(-8, 8, 60)
    lines = [
        ("belief b", BLUE, 2.8, "-", 0.0300),
        ("swap (other horizon)", PURPLE, 2.2, (0, (4, 3)), 0.0285),
        ("random", GREY, 1.6, "-", 0.0020),
        ("orthogonal", "#b6bec7", 1.6, (0, (1, 2)), 0.0015),
    ]
    cur_al = -8 + 16 * ease(p)
    ncur = int(np.clip((cur_al + 8) / 16 * 60, 1, 60))
    for lbl, col, lw, ls, slope in lines:
        r = slope * al_full
        ax.plot([X(v) for v in al_full[:ncur]], [Y(v) for v in r[:ncur]],
                color=col, lw=lw, ls=ls, zorder=4, solid_capstyle="round")
    # current markers on belief + swap
    ax.scatter(X(cur_al), Y(0.0300 * cur_al), s=34, color=BLUE, zorder=6,
               edgecolors="white", linewidths=0.8)
    # legend
    lx, ly = px0 + 0.15, py0 + ph - 0.18
    for j, (lbl, col, lw, ls, slope) in enumerate(lines):
        yy = ly - j * 0.28
        ax.plot([lx, lx + 0.35], [yy, yy], color=col, lw=lw, ls=ls)
        ax.text(lx + 0.45, yy, lbl, fontsize=7.6, color=INK, va="center")

    if p > 0.6:
        aa = ease((p - 0.6) / 0.4)
        ax.text(X(3.4), Y(0.0300 * 3.4) + 0.24, "belief ≈ swap", fontsize=7.8,
                color=_fade(PURPLE, aa), weight="bold", ha="right")
        ax.text(X(3.4), Y(0.0300 * 3.4) - 0.02, "(not horizon-specific)",
                fontsize=7.0, color=_fade(MUT, aa), ha="right")

    if p > 0.72:
        aa = ease((p - 0.72) / 0.28)
        _chip(ax, 8.35, 4.35, 3.3, "vs random     d = 47.9   ✓", fc="#eaf4ea",
              ec=GREEN, tc="#256b25", a=aa, fs=8.8, weight="bold")
        _chip(ax, 8.35, 3.78, 3.3, "vs orthogonal  d = 48.6   ✓", fc="#eaf4ea",
              ec=GREEN, tc="#256b25", a=aa, fs=8.8, weight="bold")
        _chip(ax, 8.35, 3.21, 3.3, "vs swap        n.s.  ✗", fc="#fdeceb",
              ec=RED, tc="#9e2420", a=aa, fs=8.8, weight="bold")
        ax.text(8.35, 2.72, "reversible  ±0.24 · readout ≠ steering axis",
                fontsize=7.6, color=_fade(GREEN, aa), weight="bold")
        ax.text(8.35, 2.42, "(non-circular by construction)", fontsize=7.0,
                color=_fade(MUT, aa))


# ---------------------------------------------------------------------------
# Stage 4 — Stress.
# ---------------------------------------------------------------------------

def stage_stress(ax, p):
    draw_chrome(ax, 3, p)
    draw_stream(ax, 0.32, mode="peak")
    a = ease(p)
    _content_header(ax, "Track belief-projection drift under a sustained regime shift", a)

    px0, py0, pw, ph = 3.4, 1.2, 5.6, 3.7
    ax.plot([px0, px0, px0 + pw], [py0 + ph, py0, py0], color=INK, lw=1.2)
    ax.text(px0 + pw / 2, py0 - 0.3, "reasoning step", ha="center", fontsize=8.8,
            color=MUT)
    ax.text(px0 - 0.28, py0 + ph / 2, "| projection |", rotation=90, va="center",
            ha="center", fontsize=8.8, color=MUT)

    n = 44
    xs = np.linspace(px0 + 0.12, px0 + pw - 0.12, n)
    tt = np.linspace(0, 1, n)
    stable = 0.32 + 0.02 * np.sin(tt * 9)
    drift = 0.32 + 0.02 * np.sin(tt * 9) + 1.85 * tt ** 2.4
    show = int(np.clip(ease(p) * n, 1, n))
    ys_s = py0 + stable * ph
    ys_d = py0 + np.clip(drift, None, 1.0) * ph
    ax.plot(xs[:show], ys_s[:show], color=GREY, lw=2.2)
    ax.plot(xs[:show], ys_d[:show], color=RED, lw=2.6, solid_capstyle="round")
    if show > 1:
        ax.scatter(xs[show - 1], ys_d[show - 1], s=30, color=RED, zorder=5)

    ax.text(px0 + 0.2, py0 + ph - 0.12, "— regime shift", color=RED,
            fontsize=8.5, va="top")
    ax.text(px0 + 0.2, py0 + ph - 0.42, "— stable", color=GREY, fontsize=8.5,
            va="top")

    if p > 0.7:
        aa = ease((p - 0.7) / 0.3)
        onset = int(0.55 * n)
        if show > onset:
            ax.plot([xs[onset], xs[onset]], [py0 + 0.1, py0 + ph - 0.1],
                    color=_fade(ORANGE, aa), lw=1.4, ls=(0, (4, 3)))
            ax.text(xs[onset], py0 + ph + 0.02, "drift onset", ha="center",
                    fontsize=8.2, color=_fade(ORANGE, aa), weight="bold")
        _chip(ax, 9.2, 3.5, 2.5, "drift ratio  1.44", fc="#eaf4ea", ec=GREEN,
              tc="#256b25", a=aa, weight="bold", fs=9.2)
        ax.text(9.2, 3.0, "shift / stable > 1", fontsize=7.6, color=_fade(MUT, aa))
        ax.text(9.2, 2.7, "visible before any\noutput error", fontsize=7.6,
                color=_fade(MUT, aa), va="top")


STAGES = [stage_identify, stage_localize, stage_steer, stage_stress]


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
    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "pipeline.gif")
    frames = _render_frames()
    from PIL import Image
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128)
            for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / FPS), loop=0, optimize=True)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({len(frames)} frames, {len(frames)/FPS:.1f}s, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
