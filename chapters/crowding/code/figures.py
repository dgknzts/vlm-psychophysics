"""Figures for the chapter page, in the site's style: white image, dark ink, IBM Plex Sans,
fixation-dot red for the model. The page puts them on its mid-grey display.

Used by stimuli.py (the stimulus and token figures) and analysis.ipynb (the results figures).
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
  """The model's grid drawn on our images: the whole image in tokens, then three close-ups
  that show what can happen to a letter. The tokens the target touches are shaded."""
  W, H = stimuli.CANVAS_PX
  fig = plt.figure(figsize=(7.2, 3.5))
  grid = fig.add_gridspec(2, 3, height_ratios=[0.55, 1], hspace=0.32, wspace=0.08)

  ax = fig.add_subplot(grid[0, :])
  ax.imshow(stimuli.draw_trial(dict(target="k", ecc_deg=10, spacing=2, inner_flanker="r", outer_flanker="m")),
            interpolation="nearest")
  for x in range(0, W + 1, TOKEN_PX):
    ax.axvline(x - 0.5, color=RED, lw=0.5, alpha=0.6)
  for y in range(0, H + 1, TOKEN_PX):
    ax.axhline(y - 0.5, color=RED, lw=0.5, alpha=0.6)
  ax.set_xticks([]); ax.set_yticks([])
  ax.set_title("The whole image: %d × %d = %d tokens, the same grid everywhere" % (W // TOKEN_PX, H // TOKEN_PX, (W // TOKEN_PX) * (H // TOKEN_PX)),
               fontsize=9, loc="left")

  cases = [(0, None, "0°, alone: token edges\ncut the target in four"),
           (10, 1, "10°, 1 slot: target and left\nflanker in the same token"),
           (10, 4, "10°, 4 slots: each letter\nin its own tokens")]
  for col, (ecc, spacing, title) in enumerate(cases):
    ax = fig.add_subplot(grid[1, col])
    trial = dict(target="k", ecc_deg=ecc, spacing=spacing, inner_flanker="r", outer_flanker="m")
    img = stimuli.draw_trial(trial)
    x_target = W / 2 + ecc * stimuli.PX_PER_DEG
    x0 = TOKEN_PX * round(x_target / TOKEN_PX) - 2 * TOKEN_PX     # 4 tokens wide, target near the middle
    y0 = H // 2 - TOKEN_PX
    ax.imshow(img.crop((x0, y0, x0 + 4 * TOKEN_PX, y0 + 2 * TOKEN_PX)), interpolation="nearest",
              extent=(x0, x0 + 4 * TOKEN_PX, y0 + 2 * TOKEN_PX, y0))
    # shade the tokens that hold some of the target's ink
    alone = np.array(stimuli.draw_trial(dict(trial, spacing=None)).convert("L")) < 128
    rows, cols = np.where(alone)
    for ty in range(rows.min() // TOKEN_PX, rows.max() // TOKEN_PX + 1):
      for tx in range(cols.min() // TOKEN_PX, cols.max() // TOKEN_PX + 1):
        ax.add_patch(plt.Rectangle((tx * TOKEN_PX, ty * TOKEN_PX), TOKEN_PX, TOKEN_PX, color=RED, alpha=0.12, lw=0))
    # grid lines at their real image positions
    for x in range(0, W + 1, PATCH_PX):
      if x0 <= x <= x0 + 4 * TOKEN_PX:
        ax.axvline(x, color=RED, lw=1.0 if x % TOKEN_PX == 0 else 0.6, ls="-" if x % TOKEN_PX == 0 else ":", alpha=0.55)
    for y in range(y0, y0 + 2 * TOKEN_PX + 1, PATCH_PX):
      ax.axhline(y, color=RED, lw=1.0 if y % TOKEN_PX == 0 else 0.6, ls="-" if y % TOKEN_PX == 0 else ":", alpha=0.55)
    ax.set_xlim(x0, x0 + 4 * TOKEN_PX)
    ax.set_ylim(y0 + 2 * TOKEN_PX, y0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
  plt.close(fig)


def errors_figure(path, model):
  """What the model answered at each spacing: the target, the right or left flanker, or
  any other letter. model: one row per trial (run_model.ipynb)."""
  kinds = [("target", "#d6d3cc"), ("right flanker", RED), ("left flanker", INK), ("another letter", "#4a7bd0")]
  fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
  for ax, ecc in zip(axes, stimuli.ECCENTRICITIES):
    d = model[(model.ecc_deg == ecc) & model.spacing.notna()]
    counts = {k: [] for k, _ in kinds}
    for spacing in stimuli.SPACINGS:
      t = d[d.spacing == spacing]
      counts["target"].append((t.answer == t.target).sum())
      counts["right flanker"].append((t.answer == t.outer_flanker).sum())
      counts["left flanker"].append((t.answer == t.inner_flanker).sum())
      counts["another letter"].append(len(t) - counts["target"][-1] - counts["right flanker"][-1] - counts["left flanker"][-1])
    x = np.arange(len(stimuli.SPACINGS))
    bottom = np.zeros(len(x))
    for kind, colour in kinds:
      ax.bar(x, counts[kind], bottom=bottom, color=colour, width=0.7, label=kind)
      bottom += counts[kind]
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % stimuli.spacing_deg(s) for s in stimuli.SPACINGS], fontsize=8)
    ax.set_xlabel("Spacing, centre to centre (°)")
    ax.set_title("Target at %d°" % ecc, fontsize=9, loc="left")
  axes[0].set_ylabel("Answers (of 100)")
  axes[0].set_ylim(0, 100)
  axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
  plt.close(fig)


def compare_figure(path, model, bouma, bouma_critical, cutoffs=(0.6, 0.9)):
  """Model next to Bouma (1970). Left: a letter on its own, by eccentricity. Right: critical
  spacing by eccentricity, as a range from the lower to the higher cutoff.
  bouma: bouma1970_from_coates2021.csv; bouma_critical: bouma1970_critical_spacing.csv."""
  import fit_bouma
  fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(wspace=0.35))

  alone = bouma[bouma.gap_slots.isna()]
  left.plot(alone.ecc_deg, alone.p_correct, marker="s", color=SOFT, mfc="white", ms=5.5, lw=1, ls=":", label="Bouma (1970)")
  model_alone = model[model.spacing.isna()].groupby("ecc_deg").correct.mean()
  left.plot(model_alone.index, model_alone.values, "o", color=RED, ms=6, label="Model")
  left.set_ylim(0, 1.04); left.set_xlim(-0.5, 11)
  left.set_xlabel("Eccentricity (°)"); left.set_ylabel("Proportion correct")
  left.set_title("A letter on its own", fontsize=9, loc="left")
  left.legend(fontsize=8, loc="lower left")

  ecc = np.array([0, 11])
  right.plot(ecc, 0.5 * ecc, color=SOFT, lw=0.8, ls=(0, (3, 2)))
  right.text(9.3, 4.15, "0.5 × eccentricity", color=SOFT, fontsize=8, ha="left", rotation=33)
  human = bouma_critical[bouma_critical.cutoff.isin(cutoffs)].pivot(index="ecc_deg", columns="cutoff", values="critical_spacing_deg")
  right.vlines(human.index, human[cutoffs[0]], human[cutoffs[1]], color=SOFT, lw=2.5, label="Bouma (1970)")
  for e, d in model.groupby("ecc_deg"):
    x = np.log(d.spacing.map(stimuli.spacing_deg).fillna(np.inf))
    counts = d.assign(x=x).groupby("x").correct.sum()
    A, m, w = fit_bouma.fit(counts.index.to_numpy(), counts.to_numpy()).x
    lo, hi = (fit_bouma.critical_spacing(m, w, c) for c in cutoffs)
    right.vlines(e, lo, hi, color=RED, lw=2.5, label="Model" if e == 0 else None)
  right.set_xlim(-0.5, 11); right.set_ylim(0, 6.2)
  for ax in (left, right):
    ax.set_xticks([0, 2, 4, 6, 8, 10])
  right.set_xlabel("Eccentricity (°)"); right.set_ylabel("Critical spacing (°)")
  right.set_title("Critical spacing, %d–%d%% cutoff" % (100 * cutoffs[0], 100 * cutoffs[1]), fontsize=9, loc="left")
  right.legend(fontsize=8, loc="upper left")
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
  plt.close(fig)


VISION_WIDTH = 1024   # numbers per 16-px square in Qwen3-VL's vision encoder (config.json, vision hidden_size)
TOKEN_WIDTH = 2048    # numbers per token, the language model's width (vision out_hidden_size)
VISION_LAYERS = 24    # vision encoder depth


def _step_trial():
  """The example for the four steps: target k at 10 deg, flankers r and m one slot away."""
  return np.array(stimuli.draw_trial(dict(target="k", ecc_deg=10, spacing=1, inner_flanker="r", outer_flanker="m")).convert("L"))


STEP_SQUARE = (816, 48)    # the square with most of the target in it (x, y of its top-left pixel)
STEP_TOKEN = (800, 32)     # the token that holds it
BLUE, GREY = "#4a7bd0", "#9a968d"


def step1_figure(path):
  """Step 1: the image cut into 16-px squares, around the target."""
  img = _step_trial()
  x0, y0, w, h = 752, 32, 160, 64
  sx, sy = STEP_SQUARE
  fig, ax = plt.subplots(figsize=(6.4, 2.6))
  ax.imshow(img[y0:y0 + h, x0:x0 + w], cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(x0, x0 + w, y0 + h, y0))
  for x in range(x0, x0 + w + 1, PATCH_PX):
    ax.axvline(x, color=RED, lw=0.7, alpha=0.6)
  for y in range(y0, y0 + h + 1, PATCH_PX):
    ax.axhline(y, color=RED, lw=0.7, alpha=0.6)
  ax.add_patch(plt.Rectangle((sx, sy), PATCH_PX, PATCH_PX, color=RED, alpha=0.2, lw=0))
  ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + h, y0)
  ax.set_xticks([]); ax.set_yticks([])
  ax.set_title("16 × 16-pixel squares around the target. The shaded one holds the top of the k.", fontsize=8.5, loc="left")
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
  plt.close(fig)


ANIM_PALE = "#f4f2ed"      # background of the animations, so white pixels show


def _ease(t):
  t = min(max(t, 0), 1)
  return t * t * (3 - 2 * t)


def _grab(fig):
  from PIL import Image
  fig.canvas.draw()
  frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
  plt.close(fig)
  return frame


SPEED = 1.4      # the animations play this much faster than their frame times


def save_video(path, frames, durations, fps=25):
  """An MP4 from frames with their own display times (ms): each frame is repeated for as
  long as it should stay on screen."""
  import imageio_ffmpeg
  w, h = frames[0].size
  w2, h2 = w + w % 2, h + h % 2          # the video format needs even sizes
  writer = imageio_ffmpeg.write_frames(str(path), (w2, h2), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                       macro_block_size=1, output_params=["-crf", "20", "-movflags", "+faststart"])
  writer.send(None)
  from PIL import Image
  for frame, ms in zip(frames, durations):
    if (w2, h2) != (w, h):
      padded = Image.new("RGB", (w2, h2), ANIM_PALE); padded.paste(frame, (0, 0)); frame = padded
    data = np.ascontiguousarray(np.asarray(frame.convert("RGB")))
    for _ in range(max(1, round(ms / SPEED / 1000 * fps))):
      writer.send(data)
  writer.close()


def _internals():
  """The model's real numbers for the example image (model_internals.py)."""
  import json
  z = np.load(HERE.parent / "data" / "model_internals.npz")
  info = json.load(open(HERE.parent / "data" / "model_internals.json", encoding="utf-8"))
  return z, info


ATTN_COLS = (43, 60)     # the part of the 64-square-wide grid shown around the target (column 51)


def layers_figure(path):
  """Where our square (top of the k) takes information from, in each of the 24 layers:
  its attention, averaged over the 16 heads, drawn on the image."""
  z, info = _internals()
  img = _step_trial()
  att = z["attention"]
  c0, c1 = ATTN_COLS
  sx, sy = STEP_SQUARE
  crop = img[:, c0 * PATCH_PX:c1 * PATCH_PX]
  fig, axes = plt.subplots(6, 4, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.35, wspace=0.06))
  for layer, ax in enumerate(axes.T.flat if False else axes.flat):
    a = att[layer][:, c0:c1]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(c0, c1, 8, 0))
    ax.imshow(a / a.max(), cmap="Reds", vmin=0, vmax=1, alpha=0.75, interpolation="nearest", extent=(c0, c1, 8, 0))
    ax.add_patch(plt.Rectangle((sx // PATCH_PX, sy // PATCH_PX), 1, 1, fill=False, ec=INK, lw=0.8))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
      s.set_color(SOFT); s.set_linewidth(0.5)
    ax.set_title("layer %d" % (layer + 1), fontsize=7.5, pad=2)
  fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
  plt.close(fig)


def _vector(ax, x, y, w, h, values, vertical=False, edge=INK, lw=0.7):
  """A list of real numbers as a strip of grey cells, light = high. Scaled to the list's own
  2nd-98th percentile, since a few numbers are much larger than the rest."""
  lo, hi = np.percentile(values, [2, 98])
  data = values.reshape(-1, 1) if vertical else values.reshape(1, -1)
  ax.imshow(data, cmap="gray", vmin=lo, vmax=hi, aspect="auto", interpolation="nearest", extent=(x, x + w, y, y + h))
  ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=edge, lw=lw))


def _frame(size=(6.4, 4.0)):
  fig, ax = plt.subplots(figsize=size, dpi=130, facecolor=ANIM_PALE)
  fig.subplots_adjust(0, 0, 1, 1)
  ax.set_aspect("equal")          # squares stay square
  return fig, ax


def step2_animation():
  """Step 2 with the model's real numbers, slowly: the square's rows slide into one list;
  the list is copied for red, green and blue, then once more (the code also reads two-frame
  video); the six copies are put end to end (1,536 numbers, -1 to 1); the first learned
  layer turns them into 1,024 numbers."""
  from matplotlib.colors import LinearSegmentedColormap
  z, _ = _internals()
  img = _step_trial()
  sx, sy = STEP_SQUARE
  square = img[sy:sy + PATCH_PX, sx:sx + PATCH_PX]
  flat = square.reshape(-1).astype(float)
  tints = {"red": "#ff8a94", "green": "#8fd18f", "blue": "#8fb0ff"}
  cmaps = {k: LinearSegmentedColormap.from_list(k, ["#1d1d1b", v]) for k, v in tints.items()}
  sq_x, sq_top, sq_cell = 4.6, 6.9, 0.175
  size = 16 * sq_cell
  depth = 0.28                                    # how far each colour layer sits behind the one in front
  L, W, H = 1.5, 10.0, 0.3                        # the lists: left edge, width, height
  base_y = 3.2
  rows = [("red", 1), ("green", 1), ("blue", 1), ("red", 2), ("green", 2), ("blue", 2)]
  row_y = [2.75 - i * 0.42 for i in range(6)]      # where each copy sits, under the grey list
  # in the model's list the order is red (frame 1, 2), green (1, 2), blue (1, 2)
  slot = {("red", 1): 0, ("red", 2): 1, ("green", 1): 2, ("green", 2): 3, ("blue", 1): 4, ("blue", 2): 5}
  cw = W / flat.size
  move, stagger = 10, 2
  unroll = stagger * (PATCH_PX - 1) + move
  phases = [("unroll", unroll, "The square's 256 pixels, row after row, become one list"),
            ("rgb", 16, "The model reads each pixel three times: for red, green and blue"),
            ("twice", 14, "and all of it twice, because the same code also reads video (two frames)"),
            ("join", 18, "Put end to end: 1,536 numbers, rescaled from −1 (black) to 1 (white)"),
            ("layer", 18, "Its first learned layer turns the 1,536 numbers into 1,024 (the real values)"),
            ("all", 1, "Each of the 512 squares gets 1,024 numbers like this, from its own pixels only")]

  def strip(x, y, w, values, cmap, vmin=0, vmax=255, alpha=1.0, edge=INK):
    ax.imshow(values.reshape(1, -1), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
              extent=(x, x + w, y, y + H), alpha=alpha)
    ax.add_patch(plt.Rectangle((x, y), w, H, fill=False, ec=edge, lw=0.6, alpha=alpha))

  def stack(x, shown, label="", alpha=1.0):
    """The square as a 3-D block: a red, a green and a blue layer, one behind the other.
    shown: how visible each layer is (red, green, blue)."""
    for j in (2, 1, 0):                               # back to front
      if shown[j] <= 0:
        continue
      colour = ("red", "green", "blue")[j]
      x0, y0 = x + j * depth, sq_top - size + j * depth
      ax.imshow(square, cmap=cmaps[colour], vmin=0, vmax=255, interpolation="nearest",
                extent=(x0, x0 + size, y0, y0 + size), alpha=shown[j] * alpha)
      ax.add_patch(plt.Rectangle((x0, y0), size, size, fill=False, ec=INK, lw=0.6, alpha=shown[j] * alpha))
      ax.text(x0 + size + 0.08, y0 + 0.12, colour, fontsize=6.5, color=SOFT, alpha=shown[j] * alpha)
    if label:
      ax.text(x + (size + 2 * depth) / 2, sq_top - size - 0.32, label, ha="center", fontsize=7.5, color=SOFT, alpha=alpha)

  frames, durations = [], []
  for name, n, caption in phases:
    for f in range(n):
      u = _ease((f + 1) / n)
      fig, ax = _frame((6.4, 4.4))
      ax.set_xlim(0, 12); ax.set_ylim(-0.1, 8.2); ax.axis("off")
      ax.text(6, 7.75, caption, ha="center", fontsize=9.5)
      if name == "unroll":
        ax.add_patch(plt.Rectangle((sq_x, sq_top - size), size, size, fill=False, ec=RED, lw=1.2, alpha=0.5))
      elif name == "rgb":
        stack(sq_x, [_ease((f - j * 4) / 6) for j in range(3)], "16 × 16 pixels × 3 colours")
      else:
        # two frames: the first block moves left, a copy appears on the right
        m = u if name == "twice" else 1
        left, right = sq_x - 2.3 * m, sq_x + 2.3
        stack(left, [1, 1, 1], "frame 1" if m == 1 else "")
        if m > 0.3:
          stack(right, [1, 1, 1], "frame 2", alpha=min(1, (m - 0.3) / 0.5))
        if m == 1:
          ax.text(6, sq_top - size - 0.32, "× 2", ha="center", fontsize=8.5, color=SOFT)

      if name == "unroll":
        ax.add_patch(plt.Rectangle((L, base_y), W, H, fill=False, ec=SOFT, lw=0.6))
        for r in range(PATCH_PX):
          t = _ease((f - r * stagger) / move)
          for c in range(PATCH_PX):
            x0, y0 = sq_x + c * sq_cell, sq_top - (r + 1) * sq_cell
            x1 = L + (r * PATCH_PX + c) * cw
            ax.add_patch(plt.Rectangle((x0 + (x1 - x0) * t, y0 + (base_y - y0) * t), sq_cell + (cw - sq_cell) * t,
                                       sq_cell + (H - sq_cell) * t, fc=plt.cm.gray(int(square[r, c]) / 255),
                                       ec="#cdc9c0", lw=0.4 * (1 - t)))
        if f == n - 1:
          ax.text(6, base_y - 0.35, "256 values, 0 (black) to 255 (white)", ha="center", fontsize=8.5, color=SOFT)

      elif name in ("rgb", "twice"):
        # the grey list stays; the copies fade in under it, one after another
        strip(L, base_y, W, flat, "gray", alpha=1 - 0.6 * (u if name == "rgb" else 1))
        count = 3 if name == "rgb" else 6
        first = 0 if name == "rgb" else 3
        for i in range(count):
          colour, frame = rows[i]
          t = 1 if i < first else _ease((f - (i - first) * 4) / 6)
          if t <= 0:
            continue
          y = row_y[i] + 0.2 * (1 - t)
          strip(L, y, W, flat, cmaps[colour], alpha=t)
          ax.text(L - 0.1, y + H / 2, colour + ("" if frame == 1 else " again"), ha="right", va="center",
                  fontsize=7.5, color=SOFT, alpha=t)

      if name in ("join", "layer", "all"):
        t = u if name == "join" else 1
        seg = W / 6
        for i, (colour, frame) in enumerate(rows):
          x = L + (L + slot[(colour, frame)] * seg - L) * t
          y = row_y[i] + (base_y - row_y[i]) * t
          w = W + (seg - W) * t
          strip(x, y, w, flat, cmaps[colour])
        if t == 1:
          for colour, frame in rows:
            s = slot[(colour, frame)]
            ax.text(L + (s + 0.5) * seg, base_y - 0.28, colour, ha="center", fontsize=7.5, color=SOFT)
          ax.text(6, base_y - 0.62, "1,536 numbers, from −1 (black) to 1 (white)", ha="center", fontsize=8.5, color=SOFT)

      if name in ("layer", "all"):
        f, n = (f, n) if name == "layer" else (17, 18)     # "all" shows the finished picture
        a = _ease((f + 1) / 6)
        ax.annotate("", xy=(6, 2.0), xytext=(6, 2.45), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, alpha=a))
        ax.add_patch(plt.Rectangle((4.0, 1.4), 4.0, 0.55, fc="white", ec=INK, lw=0.7, alpha=a))
        ax.text(6, 1.67, "learned layer: 1,536 in, 1,024 out", ha="center", va="center", fontsize=8.5, alpha=a)
        ax.annotate("", xy=(6, 0.95), xytext=(6, 1.35), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, alpha=a))
        shown = int(round(1024 * _ease((f - 5) / (n - 6))))
        if shown > 0:
          values = z["embed"]
          lo, hi = np.percentile(values, [2, 98])
          ax.imshow(values[:shown].reshape(1, -1), cmap="gray", vmin=lo, vmax=hi, aspect="auto", interpolation="nearest",
                    extent=(2.0, 2.0 + 8.0 * shown / 1024, 0.45, 0.9))
        ax.add_patch(plt.Rectangle((2.0, 0.45), 8.0, 0.45, fill=False, ec=RED, lw=1.4, alpha=a))
        ax.text(6, 0.05, "our square's 1,024 numbers", ha="center", fontsize=8.5, alpha=a)
      frames.append(_grab(fig)); durations.append(90)
    durations[-1] = 2800          # time to read each caption
  durations[0] = 1200
  durations[-1] = 4000
  return frames, durations


def step3_animation():
  """Step 3 with the model's real numbers: for each of the 24 layers, where our square takes
  information from (its attention, averaged over the 16 heads) and its 1,024 numbers after it."""
  z, _ = _internals()
  img = _step_trial()
  c0, c1 = ATTN_COLS
  sx, sy = STEP_SQUARE
  crop = img[:, c0 * PATCH_PX:c1 * PATCH_PX]
  X0, X1, Y0, Y1 = 0.5, 8.0, 1.75, 5.25          # where the image sits in the frame
  sqw = (X1 - X0) / (c1 - c0)
  box = (X0 + (sx // PATCH_PX - c0) * sqw, Y1 - (sy // PATCH_PX + 1) * sqw)
  frames, durations = [], []
  steps = [(None, "Where step 2 ended: our square (black box) knows only its own pixels")] + \
          [(L, "Layer %d: red shows where our square takes information from" % (L + 1)) for L in range(24)]
  for L, caption in steps:
    fig, ax = _frame()
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(6, 5.75, caption if (L is None or L < 3 or L == 23) else "Layer %d" % (L + 1), ha="center", fontsize=9.5)
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(X0, X1, Y0, Y1))
    if L is not None:
      a = z["attention"][L][:, c0:c1]
      ax.imshow(a / a.max(), cmap="Reds", vmin=0, vmax=1, alpha=0.75, interpolation="nearest", extent=(X0, X1, Y0, Y1))
    ax.add_patch(plt.Rectangle(box, sqw, sqw, fill=False, ec=INK, lw=1.2))
    ax.add_patch(plt.Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec=SOFT, lw=0.5))
    # the layer counter
    ax.text(10.0, 4.2, "layer", ha="center", fontsize=9, color=SOFT)
    ax.text(10.0, 3.2, "–" if L is None else str(L + 1), ha="center", fontsize=26, color=INK)
    ax.text(10.0, 2.65, "of %d" % VISION_LAYERS, ha="center", fontsize=9, color=SOFT)
    # our square's numbers after this layer
    values = z["embed"] if L is None else z["layers"][L]
    _vector(ax, 0.5, 0.75, 11.0, 0.45, values, edge=RED, lw=1.2)
    ax.text(0.5, 0.35, "our square's 1,024 numbers %s" % ("before layer 1" if L is None else "after layer %d" % (L + 1)),
            fontsize=8, color=SOFT)
    frames.append(_grab(fig))
    durations.append(2600 if L is None or L < 3 else 650)
  durations[-1] = 3500
  return frames, durations


def step4_animation():
  """Step 4 with the model's real numbers: the 2 x 2 squares of our token after the 24
  layers, put end to end, and the merger's output: one token of 2,048 numbers."""
  z, info = _internals()
  img = _step_trial()
  tx, ty = STEP_TOKEN
  t = info["our_token_index"]
  four = z["final"][t * 4:t * 4 + 4]                 # squares 1-4 of our token, in the model's order
  joined = four.reshape(-1)
  boxes = [(0.5, 3.0), (2.0, 3.0), (0.5, 1.5), (2.0, 1.5)]
  strips = [(4.5, 4.3), (4.5, 3.6), (4.5, 2.9), (4.5, 2.2)]
  sw, sh = 1.5, 0.35
  long_x, long_y, long_w = 6.3, 3.3, 5.4
  token_x, token_y, token_w = 7.65, 1.55, 2.7
  phases = [("intro", 8, "Our token is a block of 2 × 2 squares")]
  phases += [("square%d" % i, 8, "Square %d: its 1,024 numbers after the 24 layers" % (i + 1)) for i in range(4)]
  phases += [("join", 18, "The four lists are put end to end: 4,096 numbers"),
             ("shrink", 18, "A small network turns them into 2,048 numbers: one token"),
             ("end", 1, "This is our token, as the language model gets it (real values)")]
  frames, durations = [], []
  for name, n, caption in phases:
    for f in range(n):
      u = _ease((f + 1) / n)
      fig, ax = _frame((6.4, 3.2))
      ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
      ax.text(6, 5.45, caption, ha="center", fontsize=9.5)
      ax.imshow(img[ty:ty + TOKEN_PX, tx:tx + TOKEN_PX], cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                extent=(0.5, 3.5, 1.5, 4.5))
      ax.add_patch(plt.Rectangle((0.5, 1.5), 3.0, 3.0, fill=False, ec=RED, lw=1.8))
      ax.plot([2.0, 2.0], [1.5, 4.5], color=RED, lw=0.7); ax.plot([0.5, 3.5], [3.0, 3.0], color=RED, lw=0.7)
      for i, (bx, by) in enumerate(boxes):
        ax.text(bx + 0.1, by + 1.4, str(i + 1), fontsize=8, color=RED, va="top")
      if name.startswith("square") or name == "intro":
        done = int(name[-1]) if name.startswith("square") else -1
        for i in range(done + 1):
          x, y = strips[i]
          a = u if i == done else 1
          if i == done:
            bx, by = boxes[i]
            ax.add_patch(plt.Rectangle((bx, by), 1.5, 1.5, fill=False, ec=INK, lw=2))
            ax.plot([bx + 1.5, x], [by + 0.75, y + sh / 2], color=SOFT, lw=0.8, alpha=a)
          if a > 0.3:
            _vector(ax, x, y, sw, sh, four[i], lw=0.6)
            ax.text(x + sw + 0.1, y + sh / 2, "square %d" % (i + 1), fontsize=7.5, va="center")
      elif name == "join":
        for i, (x, y) in enumerate(strips):
          ex, ew = long_x + i * long_w / 4, long_w / 4
          _vector(ax, x + (ex - x) * u, y + (long_y - y) * u, sw + (ew - sw) * u, sh, four[i], lw=0.6)
        if f == n - 1:
          ax.text(long_x + long_w / 2, long_y - 0.35, "4,096 numbers", ha="center", fontsize=8.5, color=SOFT)
      else:
        # the 4,096 numbers stay; the small network and the token appear under them
        v = 1.0 if name == "end" else u
        cx = long_x + long_w / 2
        _vector(ax, long_x, long_y, long_w, sh, joined, lw=0.6)
        ax.text(cx, long_y - 0.35, "4,096 numbers", ha="center", fontsize=8.5, color=SOFT)
        a = min(1, v * 3)
        ax.annotate("", xy=(cx, 2.35), xytext=(cx, 2.85), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, alpha=a))
        ax.add_patch(plt.Rectangle((cx - 1.2, 1.75), 2.4, 0.55, fc="white", ec=INK, lw=0.7, alpha=a))
        ax.text(cx, 2.02, "small network", ha="center", va="center", fontsize=8, alpha=a)
        ax.annotate("", xy=(cx, 1.15), xytext=(cx, 1.7), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, alpha=a))
        shown = int(round(2048 * _ease((v - 0.35) / 0.65)))
        if shown > 0:
          values = z["token"]
          lo, hi = np.percentile(values, [2, 98])
          ax.imshow(values[:shown].reshape(1, -1), cmap="gray", vmin=lo, vmax=hi, aspect="auto", interpolation="nearest",
                    extent=(token_x, token_x + token_w * shown / 2048, 0.75, 0.75 + sh))
        ax.add_patch(plt.Rectangle((token_x, 0.75), token_w, sh, fill=False, ec=INK, lw=1.6, alpha=a))
        ax.text(cx, 0.35, "one token: 2,048 numbers", ha="center", fontsize=8.5, alpha=a)
      frames.append(_grab(fig)); durations.append(90)
    durations[-1] = 2200
  durations[-1] = 3500
  return frames, durations


def step_animations(img_dir):
  """The videos for steps 2 to 5 (step 1 is a still picture)."""
  save_video(Path(img_dir) / "vlm_step2.mp4", *step2_animation())
  save_video(Path(img_dir) / "vlm_step3.mp4", *step3_animation())
  save_video(Path(img_dir) / "vlm_step4.mp4", *step4_animation())
  save_video(Path(img_dir) / "vlm_step5.mp4", *step5_animation())


def step5_animation():
  """Step 5 with the model's real numbers: the image as 128 tokens, read in order; then a
  pan along the whole input (175 tokens), each token drawn with its 2,048 numbers; at the
  end, the model's answer."""
  z, info = _internals()
  img = _step_trial()
  seq = z["sequence"]
  before, after = info["text_before"], info["text_after"]
  n_img = info["n_image_tokens"]
  gw, gh = stimuli.CANVAS_PX[0] // TOKEN_PX, stimuli.CANVAS_PX[1] // TOKEN_PX      # 32 x 4 tokens
  lo, hi = np.percentile(seq, [2, 98])

  # the ribbon: every token with its x position and width
  items, x = [], 0.0
  def label(p):
    return p.replace("\n", "↵") if p.strip() == "" else p.strip()
  for p in before:
    w = 0.25 + 0.11 * len(label(p)); items.append(("text", label(p), x, w)); x += w + 0.08
  for k in range(n_img):
    items.append(("image", k, x, 0.5)); x += 0.5 + 0.06
  for p in after:
    w = 0.25 + 0.11 * len(label(p)); items.append(("text", label(p), x, w)); x += w + 0.08
  items.append(("answer", info["answer"], x + 0.3, 0.6))
  total = x + 1.2
  view = 11.6                                         # how much of the ribbon is in view
  RY, RH, SY, SH = 3.15, 0.5, 1.75, 1.2               # ribbon row, and the numbers under it
  i0, i1 = len(before), len(before) + n_img           # where the image tokens are in the ribbon

  def grid(ax, upto=0, current=None):
    """The image with its token grid, numbered up to `upto`, `current` outlined."""
    X0, Y0, Wd = 0.2, 5.0, 11.6
    Ht = Wd * gh / gw
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(X0, X0 + Wd, Y0, Y0 + Ht))
    cell = Wd / gw
    for c in range(gw + 1):
      ax.plot([X0 + c * cell] * 2, [Y0, Y0 + Ht], color=RED, lw=0.4, alpha=0.6)
    for r in range(gh + 1):
      ax.plot([X0, X0 + Wd], [Y0 + r * cell] * 2, color=RED, lw=0.4, alpha=0.6)
    for k in range(upto):
      r, c = divmod(k, gw)
      ax.text(X0 + (c + 0.5) * cell, Y0 + Ht - (r + 0.5) * cell, str(k + 1), ha="center", va="center", fontsize=3.8, color=SOFT)
    if current is not None:
      r, c = divmod(current, gw)
      ax.add_patch(plt.Rectangle((X0 + c * cell, Y0 + Ht - (r + 1) * cell), cell, cell, fill=False, ec=RED, lw=1.6))

  def token_pixels(k):
    r, c = divmod(k, gw)
    return img[r * TOKEN_PX:(r + 1) * TOKEN_PX, c * TOKEN_PX:(c + 1) * TOKEN_PX]

  frames, durations = [], []

  def frame(caption, upto=0, current=None, cam=None, show_answer=False, visible=None, flying=None,
            numbers=None, rows=(1, 1, 1)):
    """visible: how many tokens of the input row are shown so far; numbers: how many of them
    have their numbers drawn (default: all visible); flying: (image token, 0..1) on its way
    from the grid down into the row; rows: how visible the row, the numbers row and the
    overview bar are (0..1), so they can fade in."""
    if numbers is None:
      numbers = visible
    fig, ax = _frame((6.4, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.text(6, 7.05, caption, ha="center", fontsize=9.5)
    grid(ax, upto, current)
    if cam is not None:
      ra, na, ma = rows
      ax.text(0.2, SY + SH + RH + 0.55, "the input, token by token", fontsize=7.5, color=SOFT, alpha=ra)
      ax.text(0.2, SY - 0.3, "each token's 2,048 numbers", fontsize=7.5, color=SOFT, alpha=na)
      ax.add_patch(plt.Rectangle((0.2, RY - 0.08), view, RH + 0.16, fill=False, ec="#cfccc4", lw=0.8, ls=(0, (3, 2)), alpha=ra))
      ax.add_patch(plt.Rectangle((0.2, SY - 0.05), view, SH + 0.1, fill=False, ec="#cfccc4", lw=0.8, ls=(0, (3, 2)), alpha=na))
      for index, (kind, what, x, w) in enumerate(items):
        xs = x - cam + 0.2
        if xs + w < 0.2 or xs > 0.2 + view or (kind == "answer" and not show_answer):
          continue
        if kind != "answer" and visible is not None and index >= visible:
          continue
        if kind == "image":
          ax.imshow(token_pixels(what), cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(xs, xs + w, RY, RY + RH))
          ax.add_patch(plt.Rectangle((xs, RY), w, RH, fill=False, ec=RED, lw=0.6))
        elif kind == "text":
          special = what.startswith("<|")
          ax.add_patch(plt.Rectangle((xs, RY), w, RH, fc="#ebe8e1" if special else "white", ec=SOFT, lw=0.5))
          ax.text(xs + w / 2, RY + RH / 2, what, ha="center", va="center", fontsize=5.8, family="monospace",
                  color=SOFT if special else INK)
        else:
          ax.add_patch(plt.Rectangle((xs, RY - 0.1), w, RH + 0.2, fc="white", ec=RED, lw=1.6))
          ax.text(xs + w / 2, RY + RH / 2, what, ha="center", va="center", fontsize=10, family="monospace")
          ax.text(xs + w / 2, RY - 0.4, "answer", ha="center", fontsize=7, color=RED)
          continue
        if numbers is None or index < numbers:
          ax.imshow(seq[index][::16].reshape(-1, 1), cmap="gray", vmin=lo, vmax=hi, aspect="auto", interpolation="nearest",
                    extent=(xs, xs + w, SY, SY + SH))
      # cover what is outside the view
      ax.add_patch(plt.Rectangle((-1, SY - 0.05), 1.2, RY + RH - SY + 0.3, color=ANIM_PALE, lw=0, zorder=5))
      ax.add_patch(plt.Rectangle((0.2 + view, SY - 0.05), 2, RY + RH - SY + 0.3, color=ANIM_PALE, lw=0, zorder=5))
      # an image token on its way down from the grid
      if flying is not None:
        k, t = flying
        r, c = divmod(k, gw)
        cell = 11.6 / gw
        gx, gy = 0.2 + c * cell, 5.0 + 11.6 * gh / gw - (r + 1) * cell
        _, _, x, w = items[i0 + k]
        tx_, ty_ = x - cam + 0.2, RY
        px, py, pw, ph = gx + (tx_ - gx) * t, gy + (ty_ - gy) * t, cell + (w - cell) * t, cell + (RH - cell) * t
        ax.imshow(token_pixels(k), cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(px, px + pw, py, py + ph), zorder=8)
        ax.add_patch(plt.Rectangle((px, py), pw, ph, fill=False, ec=RED, lw=1.2, zorder=9))
      # the whole input in small, with the part in view
      MX, MW, MY = 0.2, 11.6, 0.45
      for a_, b_, colour in [(0, i0, "#cfccc4"), (i0, i1, "#f2a3ab"), (i1, len(items) - 1, "#cfccc4")]:
        x_a, x_b = items[a_][2], items[b_ - 1][2] + items[b_ - 1][3]
        ax.add_patch(plt.Rectangle((MX + x_a / total * MW, MY), (x_b - x_a) / total * MW, 0.25, color=colour, lw=0, zorder=6, alpha=ma))
      ax.add_patch(plt.Rectangle((MX + cam / total * MW, MY - 0.06), view / total * MW, 0.37, fill=False, ec=INK, lw=0.9, zorder=7, alpha=ma))
      ax.text(MX, MY - 0.35, "all %d tokens: grey = text, red = image" % info["n_input"], fontsize=7, color=SOFT, zorder=7, alpha=ma)
    frames.append(_grab(fig))

  # 1: the image as tokens, numbered in reading order
  frame("Our image is cut into %d × %d = %d tokens" % (gw, gh, n_img)); durations.append(2600)
  for upto in list(range(1, 33, 2)) + [64, 96, 128]:
    frame("The model reads them in order: left to right, then the next row, like lines of text", upto=upto)
    durations.append(110)
  durations[-1] = 2600

  # 2: the input row is built one token at a time; the view follows the newest token
  ours = info["our_token_index"]
  cam = 0.0
  def follow(index):
    nonlocal cam
    _, _, x, w = items[index]
    cam = max(cam, x + w + 0.6 - view)
    return cam

  # the empty rows fade in, one after another
  intro = [("Now the model builds its input: one long row of tokens", (t, 0, 0)) for t in (0.25, 0.5, 0.75, 1)]
  intro += [("Under each token go its 2,048 numbers", (1, t, 0)) for t in (0.25, 0.5, 0.75, 1)]
  intro += [("The bar at the bottom will show where we are in the whole row", (1, 1, t)) for t in (0.25, 0.5, 0.75, 1)]
  for caption, rows in intro:
    frame(caption, upto=128, cam=0, visible=0, rows=rows); durations.append(110)
    if rows[-1] == 1 or (rows[1] == 1 and rows[2] == 0) or (rows[0] == 1 and rows[1] == 0):
      durations[-1] = 2200
  for index in range(i0):
    caption = "First, %d text tokens start the message" % len(before)
    frame(caption, upto=128, cam=follow(index), visible=index + 1, numbers=index); durations.append(450)
    frame(caption, upto=128, cam=cam, visible=index + 1); durations.append(650)
  durations[-1] = 1800
  for k in range(n_img):
    index = i0 + k
    if k == 0:
      caption = "Then the image tokens, one by one, in reading order"
    elif k == ours:
      caption = "Token %d: our token from step 4, the top of the k" % (ours + 1)
    else:
      caption = "Then the image tokens, one by one, in reading order: token %d" % (k + 1)
    c = follow(index)
    # the first few and our token fly down slowly; the rest quickly
    slow = k < 3 or k == ours
    for t in ([0.33, 0.66] if slow else [0.5]):
      frame(caption, upto=128, current=k, cam=c, visible=index, flying=(k, t))
      durations.append(160 if slow else 60)
    if slow:
      frame(caption, upto=128, current=k, cam=c, visible=index + 1, numbers=index); durations.append(450)
    frame(caption, upto=128, current=k, cam=c, visible=index + 1)
    durations.append(2600 if k == ours else (900 if slow else 60))
  for j in range(len(after)):
    index = i1 + j
    frame("Then the %d tokens of the question" % len(after), upto=128, cam=follow(index), visible=index + 1)
    durations.append(260)
  durations[-1] = 1800
  cam = total - view
  frame("The model's answer: \"%s\" (p = %.2f). Every token, image or text, is a list of 2,048 numbers"
        % (info["answer"], info["p_answer"]), upto=128, cam=cam, show_answer=True)
  durations.append(4500)
  return frames, durations
