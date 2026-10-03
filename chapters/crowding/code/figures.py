"""Figures for the chapter page, in the site's style: white image, dark ink, IBM Plex Sans,
fixation-dot red for the model. The page puts them on its mid-grey display.

Used by stimuli.py (the conditions figure) and analyze.py (the results figure).
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import stimuli

HERE = Path(__file__).parent
INK = "#1d1d1b"
SOFT = "#8a8780"
RED = "#ff4d5e"

font_manager.fontManager.addfont(str(HERE / "fonts" / "IBMPlexSans.ttf"))
plt.rcParams.update({
  "font.family": "IBM Plex Sans", "font.size": 10, "text.color": INK,
  "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
  "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
  "legend.frameon": False, "savefig.facecolor": "white",
})


def conditions_figure(path):
  """Top: the whole image at 0 and 10 deg, as the model gets it. Below: the six spacing
  conditions, cut around the target (they look the same at 0 and 10 deg)."""
  example = dict(target="k", spacing=4, inner_flanker="r", outer_flanker="m")
  ppd = stimuli.PX_PER_DEG
  rows = stimuli.SPACINGS + [None]
  fig = plt.figure(figsize=(7.2, 6.0))
  # row 2 is an empty spacer between the two halves
  grid = fig.add_gridspec(3 + len(rows), 1, height_ratios=[1.25, 1.25, 0.6] + [0.55] * len(rows), hspace=0.35)

  for i, ecc in enumerate(stimuli.ECCENTRICITIES):
    ax = fig.add_subplot(grid[i])
    ax.imshow(stimuli.draw_trial(dict(example, ecc_deg=ecc)), interpolation="nearest", extent=(-16, 16, -2, 2))
    ax.set_yticks([])
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    if i == 0:
      ax.set_xticklabels([])
    else:
      ax.set_xticklabels(["-15", "-10", "-5", "0", "5", "10", "15°"], fontsize=8)
    for side in ax.spines.values():
      side.set_visible(True); side.set_color(SOFT)
    ax.set_ylabel("target at %d°" % ecc, rotation=0, ha="right", va="center", fontsize=8.5)
    if i == 0:
      ax.set_title("Whole images (1024 × 128 px), flankers 4 slots away", fontsize=9, loc="left")

  half = 5.5
  for row, spacing in enumerate(rows):
    ax = fig.add_subplot(grid[3 + row])
    img = stimuli.draw_trial(dict(example, ecc_deg=0, spacing=spacing))
    x = img.width / 2
    crop = img.crop((round(x - half * ppd), round(img.height / 2 - 0.45 * ppd),
                     round(x + half * ppd), round(img.height / 2 + 0.45 * ppd)))
    ax.imshow(crop, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
      s.set_visible(True); s.set_color(SOFT); s.set_linewidth(0.6)
    if row == 0:
      ax.set_title("The six conditions, cut to 11° around the target", fontsize=9, loc="left", pad=8)
    label = "alone" if spacing is None else "%d slot%s, %.2f°" % (spacing, "" if spacing == 1 else "s", stimuli.spacing_deg(spacing))
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.5)
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
  plt.close(fig)


def results_figure(path, model, bouma):
  """Proportion correct against spacing. model: one row per trial (run_model.ipynb);
  bouma: bouma1970_from_coates2021.csv."""
  alone_x = 9.0          # where "alone" is drawn on the log axis
  fig, ax = plt.subplots(figsize=(6.4, 3.9))
  ax.axhline(1 / 25, color=SOFT, lw=0.8, ls=(0, (2, 2)))
  ax.text(0.27, 1 / 25 + 0.015, "guessing (1/25)", color=SOFT, fontsize=8)

  # Bouma (1970): his nearest tested eccentricities, edge-to-edge gaps turned into centre-to-centre
  for ecc, marker in [(1, "s"), (7, "^")]:
    d = bouma[bouma.ecc_deg == ecc]
    flanked = d.dropna(subset=["cc_spacing_deg"]).groupby("cc_spacing_deg").p_correct.mean()
    ax.plot(flanked.index, flanked.values, marker=marker, color=SOFT, mfc="white", ms=5.5, lw=1, ls=":",
            label="Bouma (1970), %d°, x flankers" % ecc)
    alone = d[d.cc_spacing_deg.isna()].p_correct
    if len(alone):
      ax.plot(alone_x, alone.mean(), marker=marker, color=SOFT, mfc="white", ms=5.5)

  # the model
  for ecc, colour, marker in [(0, INK, "o"), (10, RED, "o")]:
    d = model[model.ecc_deg == ecc]
    flanked = d.dropna(subset=["spacing"]).groupby("spacing").correct.mean()
    x = np.array([stimuli.spacing_deg(s) for s in flanked.index])
    ax.plot(x, flanked.values, marker=marker, color=colour, ms=5.5, lw=1.6, label="Model, %d°" % ecc)
    ax.plot(alone_x, d[d.spacing.isna()].correct.mean(), marker=marker, color=colour, ms=5.5)

  ax.set_xscale("log")
  ticks = [stimuli.spacing_deg(s) for s in stimuli.SPACINGS]
  ax.set_xticks(ticks + [alone_x])
  ax.set_xticklabels(["%.2f" % t for t in ticks] + ["alone"])
  ax.minorticks_off()
  ax.set_xlim(0.22, 12)
  ax.set_ylim(0, 1.04)
  ax.set_xlabel("Target–flanker spacing, centre to centre (degrees)")
  ax.set_ylabel("Proportion correct")
  ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.1))
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
  plt.close(fig)


PATCH_PX = 16      # Qwen3-VL cuts the image into 16 x 16 pixel squares
TOKEN_PX = 32      # and packs each 2 x 2 block of squares into one token


def grid_figure(path):
  """The model's grid drawn on our images: the whole image in tokens, then close-ups at 10 deg
  with tokens and 16-px squares."""
  example = dict(target="k", ecc_deg=10, inner_flanker="r", outer_flanker="m")
  W, H = stimuli.CANVAS_PX
  fig = plt.figure(figsize=(7.2, 3.3))
  grid = fig.add_gridspec(2, 3, height_ratios=[0.55, 1], hspace=0.28, wspace=0.08)

  ax = fig.add_subplot(grid[0, :])
  ax.imshow(stimuli.draw_trial(dict(example, spacing=2)), interpolation="nearest")
  for x in range(0, W + 1, TOKEN_PX):
    ax.axvline(x - 0.5, color=RED, lw=0.5, alpha=0.6)
  for y in range(0, H + 1, TOKEN_PX):
    ax.axhline(y - 0.5, color=RED, lw=0.5, alpha=0.6)
  ax.set_xticks([]); ax.set_yticks([])
  ax.set_title("The whole image: %d × %d = %d tokens, the same grid everywhere" % (W // TOKEN_PX, H // TOKEN_PX, (W // TOKEN_PX) * (H // TOKEN_PX)),
               fontsize=9, loc="left")

  x_target = W / 2 + 10 * stimuli.PX_PER_DEG
  x0 = round(x_target - 2 * TOKEN_PX)          # 4 tokens wide, target in the middle
  y0 = H // 2 - TOKEN_PX
  for col, spacing in enumerate([1, 2, 4]):
    ax = fig.add_subplot(grid[1, col])
    img = stimuli.draw_trial(dict(example, spacing=spacing))
    ax.imshow(img.crop((x0, y0, x0 + 4 * TOKEN_PX, y0 + 2 * TOKEN_PX)), interpolation="nearest",
              extent=(x0, x0 + 4 * TOKEN_PX, y0 + 2 * TOKEN_PX, y0))
    # grid lines at their real image positions
    for x in range(0, W + 1, PATCH_PX):
      if x0 <= x <= x0 + 4 * TOKEN_PX:
        ax.axvline(x, color=RED, lw=1.0 if x % TOKEN_PX == 0 else 0.6, ls="-" if x % TOKEN_PX == 0 else ":", alpha=0.55)
    for y in range(y0, y0 + 2 * TOKEN_PX + 1, PATCH_PX):
      ax.axhline(y, color=RED, lw=1.0 if y % TOKEN_PX == 0 else 0.6, ls="-" if y % TOKEN_PX == 0 else ":", alpha=0.55)
    ax.set_xlim(x0, x0 + 4 * TOKEN_PX)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%d slot%s (%.2f°)" % (spacing, "" if spacing == 1 else "s", stimuli.spacing_deg(spacing)), fontsize=9)
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
  plt.close(fig)
