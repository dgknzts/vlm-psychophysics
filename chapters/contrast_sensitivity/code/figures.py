"""Figures for the chapter page, in the site's style: white image, dark ink, IBM Plex Sans,
fixation-dot red for the model, grey for humans. The page puts them on its mid-grey display.

Used by stimuli.py (the stimulus figure) and analysis.ipynb (the results figures).
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


def stimuli_figure(path, contrast=0.5):
  """Top: one whole display, as the model gets it. Below: the ten ModelFest Gabors."""
  fig = plt.figure(figsize=(7.2, 6.4))
  grid = fig.add_gridspec(3, 5, height_ratios=[2.1, 1, 1], hspace=0.45, wspace=0.08)

  ax = fig.add_subplot(grid[0, 1:4])
  trial = dict(frequency=4, contrast=contrast, phase=0.0)
  ax.imshow(stimuli.draw_trial(trial, "right"), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
  ax.set_xticks([]); ax.set_yticks([])
  for side in ax.spines.values():
    side.set_visible(True); side.set_color(SOFT)
  ax.axvline(stimuli.FIELD_PX - 0.5, color=SOFT, lw=0.6, ls=(0, (3, 3)))
  ax.set_title("One display: 512 × 256 px, the Gabor on the right (4 c/deg)", fontsize=9)

  for i, f in enumerate(stimuli.FREQUENCIES):
    ax = fig.add_subplot(grid[1 + i // 5, i % 5])
    field = stimuli.MEAN_GREY + 127 * contrast * stimuli.gabor(f)
    ax.imshow(field, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    ax.set_title("%g c/deg" % f, fontsize=8.5)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def contrast_figure(path):
  """What contrast and spatial frequency are: three gratings, and the light along one of them."""
  n = 240
  x = np.arange(n) / n                                  # one patch is 1 unit wide
  fig = plt.figure(figsize=(7.2, 3.6))
  grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.75], hspace=0.5, wspace=0.12)
  patches = [(3, 0.8, "high contrast"), (3, 0.15, "low contrast"), (9, 0.8, "higher frequency")]
  for i, (cycles, c, label) in enumerate(patches):
    ax = fig.add_subplot(grid[0, i])
    row = 128 + 127 * c * np.sin(2 * np.pi * cycles * x)
    ax.imshow(np.tile(row, (n, 1)), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
      side.set_visible(True); side.set_color(SOFT)
    ax.set_title(label, fontsize=9)

    ax = fig.add_subplot(grid[1, i])
    light = 1 + c * np.sin(2 * np.pi * cycles * x)
    ax.plot(x, light, color=INK, lw=1.2)
    ax.axhline(1, color=SOFT, lw=0.7, ls=(0, (3, 3)))
    ax.set_ylim(0, 2.3); ax.set_xlim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if i == 0:
      ax.annotate("", xy=(0.083, 1 + c), xytext=(0.083, 1), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1))
      ax.text(0.11, 2.1, "contrast", color=RED, fontsize=8.5, va="center")
      ax.text(0, -0.35, "light across the stripes", color=SOFT, fontsize=8, transform=ax.transData)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def chart_figure(path, width=1200, height=600):
  """A contrast sensitivity chart (after Campbell & Robson): the stripes get finer to the right
  and fainter towards the top. Where they seem to end is your own sensitivity curve."""
  x = np.arange(width) / width
  y = np.arange(height) / height                        # 0 at the top
  low, high = 4, 120                                    # cycles across the whole width, log-spaced
  k = np.log(high / low)
  phase = 2 * np.pi * low * (np.exp(k * x) - 1) / k     # so the local frequency rises smoothly
  contrast = 10 ** (-2.7 * (1 - y))                     # 0.2% at the top to 100% at the bottom
  img = 128 + 127 * contrast[:, None] * np.sin(phase)[None, :]
  fig, ax = plt.subplots(figsize=(7.2, 3.9))
  ax.imshow(np.round(img), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
  ax.set_xticks([]); ax.set_yticks([])
  for side in ax.spines.values():
    side.set_visible(False)
  ax.set_xlabel("finer stripes  →", fontsize=9, loc="right")
  ax.set_ylabel("lower contrast  →", fontsize=9, loc="top")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def modelfest_csf(path_csv=HERE.parent / "data" / "modelfest_baseline.csv"):
  """Sensitivity for ModelFest stimuli 1-10: one value per observer (mean of the 4 repeats in
  log units), and the mean over the 16 observers with its standard error."""
  import pandas as pd
  df = pd.read_csv(path_csv)
  df = df[df.stimulus <= 10]
  per_observer = df.groupby(["stimulus", "observer"]).log10_threshold.mean().unstack()   # 10 x 16
  logs = -per_observer.values                                                             # log10 sensitivity
  mean = logs.mean(axis=1)
  se = logs.std(axis=1, ddof=1) / np.sqrt(logs.shape[1])
  return np.array(stimuli.FREQUENCIES), logs, mean, se


def trial_figure(path):
  """A ModelFest trial: two short intervals, the Gabor in one of them; marks at the corners."""
  fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.2))
  field = stimuli.MEAN_GREY + 127 * 0.25 * stimuli.gabor(4)
  blank = np.full_like(field, stimuli.MEAN_GREY)
  for ax, img, label in zip(axs, [blank, field], ["interval 1", "interval 2"]):
    pad = 40
    big = np.full((field.shape[0] + 2 * pad, field.shape[1] + 2 * pad), float(stimuli.MEAN_GREY))
    big[pad:-pad, pad:-pad] = img
    ax.imshow(big, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    n = big.shape[0]
    for x0, y0, dx, dy in [(pad, pad, 1, 1), (n - pad, pad, -1, 1), (pad, n - pad, 1, -1), (n - pad, n - pad, -1, -1)]:
      ax.plot([x0 - dx * 12, x0 - dx * 12, x0 + dx * 14], [y0 + dy * 14, y0 - dy * 12, y0 - dy * 12], color=INK, lw=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
      side.set_visible(False)
    ax.set_title(label, fontsize=9)
  fig.text(0.5, 0.02, "Which interval had the stripes?", ha="center", fontsize=9.5)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def human_figure(path):
  """ModelFest contrast sensitivity: each observer (thin lines) and the mean of 16 (dots, +-2 SE)."""
  f, logs, mean, se = modelfest_csf()
  fig, ax = plt.subplots(figsize=(5.6, 4.0))
  for o in range(logs.shape[1]):
    ax.plot(f, 10 ** logs[:, o], color="#c9c6bf", lw=0.8)
  ax.errorbar(f, 10 ** mean, yerr=[10 ** mean - 10 ** (mean - 2 * se), 10 ** (mean + 2 * se) - 10 ** mean],
              fmt="o-", color="#55534e", ms=5, lw=1.4, capsize=0)
  ax.set_xscale("log"); ax.set_yscale("log")
  ax.set_xticks([1, 2, 4, 8, 16, 30]); ax.set_xticklabels(["1", "2", "4", "8", "16", "30"])
  ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["1", "10", "100", "1000"])
  ax.set_ylim(1, 1000)
  ax.set_xlabel("spatial frequency (c/deg)")
  ax.set_ylabel("contrast sensitivity (1 / threshold)")
  ax.text(0.98, 0.95, "16 observers (lines)\nmean ± 2 SE (dots)", transform=ax.transAxes, ha="right", va="top",
          fontsize=8.5, color=SOFT)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def model_table(results):
  """Proportion correct for each frequency x contrast, and the number of trials."""
  table = results.pivot_table(index="frequency", columns="contrast", values="correct", aggfunc="mean")
  count = results.pivot_table(index="frequency", columns="contrast", values="correct", aggfunc="count")
  return table, count


def model_thresholds(results, cutoff=0.75):
  """The threshold at each frequency, without any fitting: the faintest contrast at which the
  model is right in at least 75% of the trials (and stays so at all higher contrasts). The true
  threshold lies between that contrast and the one below it ("below")."""
  import pandas as pd
  rows = []
  for f, d in results.groupby("frequency"):
    p = d.groupby("contrast").correct.mean().sort_index()
    above = p.values >= cutoff
    k = len(p) - np.argmin(above[::-1]) if not above.all() else 0     # first contrast from which all are >= cutoff
    if k >= len(p):
      rows.append(dict(frequency=f, threshold=np.nan, below=np.nan))
      continue
    rows.append(dict(frequency=f, threshold=p.index[k], below=p.index[k - 1] if k > 0 else np.nan))
  return pd.DataFrame(rows)


def psychometric_figure(path, results, cutoff=0.75):
  """Proportion correct against contrast, one panel per frequency. The 75% line, and the
  threshold (circled): the faintest contrast with at least 75% correct."""
  freqs = sorted(results.frequency.unique())
  t = model_thresholds(results, cutoff).set_index("frequency")
  fig, axs = plt.subplots(2, 5, figsize=(7.6, 3.8), sharex=True, sharey=True)
  for ax, f in zip(axs.flat, freqs):
    p = results[results.frequency == f].groupby("contrast").correct.mean()
    ax.axhline(0.5, color=SOFT, lw=0.7, ls=(0, (3, 3)))
    ax.axhline(cutoff, color=SOFT, lw=0.7)
    ax.plot(p.index * 100, p.values, "-", color=RED, lw=0.8, alpha=0.5)
    ax.plot(p.index * 100, p.values, "o", color=RED, ms=3.5)
    c = t.loc[f, "threshold"]
    if np.isfinite(c):
      ax.plot(c * 100, p[c], "o", ms=9, mfc="none", mec=INK, mew=1.1)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_title("%g c/deg" % f, fontsize=8.5)
  for ax in axs[:, 0]:
    ax.set_ylabel("correct", fontsize=8.5)
    ax.set_yticks([0, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "50%", "75%", "100%"], fontsize=8)
  for ax in axs[-1]:
    ax.set_xlabel("contrast (%)", fontsize=8.5)
  axs[0, 0].set_ylim(-0.04, 1.06)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def csf_compare_figure(path, thresholds):
  """Model sensitivity (red: 1 / threshold; the bar reaches up to 1 / the contrast below, as
  the true threshold lies in between) next to ModelFest's 16 observers (grey). The dashed line
  is the most an 8-bit image could show (one grey level)."""
  f, logs, mean, se = modelfest_csf()
  fig, ax = plt.subplots(figsize=(5.6, 4.0))
  for o in range(logs.shape[1]):
    ax.plot(f, 10 ** logs[:, o], color="#dedbd4", lw=0.7)
  ax.plot(f, 10 ** mean, "o-", color="#55534e", ms=4.5, lw=1.3, label="humans (ModelFest, mean of 16)")
  t = thresholds.sort_values("frequency")
  ax.vlines(t.frequency, 1 / t.threshold, 1 / t.below, color=RED, lw=4, alpha=0.3)
  ax.plot(t.frequency, 1 / t.threshold, "o-", color=RED, ms=4.5, lw=1.3, label="Qwen3-VL-2B")
  ax.axhline(127, color=SOFT, lw=0.8, ls=(0, (4, 3)))
  ax.text(1.05, 140, "8-bit limit", fontsize=8, color=SOFT)
  ax.set_xscale("log"); ax.set_yscale("log")
  ax.set_xticks([1, 2, 4, 8, 16, 30]); ax.set_xticklabels(["1", "2", "4", "8", "16", "30"])
  ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["1", "10", "100", "1000"])
  ax.set_xlabel("spatial frequency (c/deg)")
  ax.set_ylabel("contrast sensitivity (1 / threshold)")
  ax.legend(fontsize=8, loc="lower left")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def _first_layer():
  return np.load(HERE.parent / "data" / "first_layer.npz")


def layer_csf_figure(path):
  """Step 3. (a) How much the first layer's 1,024 numbers change for 100% contrast stripes, at
  120 px/deg, next to the human curve; each scaled to its own peak. (b) The same layer in
  degrees at three pixel densities: the model's curve moves, because it only sees pixels."""
  z = _first_layer()
  cpp, size = z["frequencies"], z["response"]
  f, logs, mean, se = modelfest_csf()
  fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw=dict(wspace=0.3))
  a.plot(cpp * stimuli.PX_PER_DEG, size / size.max(), color=RED, lw=1.5, label="model's first layer")
  a.plot(f, 10 ** mean / (10 ** mean).max(), "o-", color="#55534e", ms=4, lw=1.2, label="humans (ModelFest)")
  for ax in (a, b):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 60); ax.set_ylim(0.02, 1.4)
    ax.set_xticks([1, 4, 16, 60]); ax.set_xticklabels(["1", "4", "16", "60"])
    ax.set_yticks([0.03, 0.1, 0.3, 1]); ax.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    ax.set_xlabel("spatial frequency (c/deg)")
  a.set_ylabel("response (1 = its peak)")
  a.set_title("a   At 120 px/deg (this chapter)", fontsize=9, loc="left")
  a.legend(fontsize=8, loc="lower center")
  for ppd, shade in [(32, 0.45), (64, 0.7), (120, 1.0)]:
    b.plot(cpp * ppd, size / size.max(), color=RED, alpha=shade, lw=1.4, label="%d px/deg" % ppd)
  b.legend(fontsize=8, loc="lower right")
  b.set_title("b   The same layer at three pixel densities", fontsize=9, loc="left")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def readout_drawing(path):
  """A drawing of the model, left to right, with a read-out at every layer (red dots)."""
  fig, ax = plt.subplots(figsize=(7.6, 2.3))
  ax.set_xlim(0, 15); ax.set_ylim(0, 4); ax.axis("off")
  trial = dict(frequency=4, contrast=0.5, phase=0.0)
  ax.imshow(stimuli.draw_trial(trial, "left"), cmap="gray", vmin=0, vmax=255, extent=(0.1, 2.3, 1.6, 2.7))
  ax.text(1.2, 1.25, "image", ha="center", fontsize=8.5)
  boxes = [(3.0, 4.4, "vision part", "24 layers"), (8.0, 1.1, "merger", ""), (9.7, 3.6, "language model", "28 layers")]
  for x, w, name, sub in boxes:
    ax.add_patch(plt.Rectangle((x, 1.4), w, 1.5, fc="white", ec=INK, lw=0.9))
    ax.text(x + w / 2, 2.35, name, ha="center", fontsize=8.5)
    if sub:
      ax.text(x + w / 2, 1.85, sub, ha="center", fontsize=8, color=SOFT)
  for x0, x1 in [(2.4, 2.95), (7.45, 7.95), (9.15, 9.65), (13.35, 13.9)]:
    ax.annotate("", xy=(x1, 2.15), xytext=(x0, 2.15), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
  ax.text(14.45, 2.15, "left /\nright", ha="center", va="center", fontsize=8.5)
  for x in list(np.linspace(3.2, 7.2, 8)) + [8.55] + list(np.linspace(9.9, 13.1, 7)):
    ax.plot([x, x], [1.4, 0.95], color=RED, lw=0.7)
    ax.plot(x, 0.9, "o", color=RED, ms=3.5)
  ax.text(8.3, 0.35, "a read-out at every layer: are the stripes left or right?", ha="center", fontsize=8.5, color=RED)
  ax.text(14.45, 1.1, "the model's\nanswer", ha="center", fontsize=8, color=SOFT)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


# ---- animations (same helpers as the crowding chapter) ----
ANIM_PALE = "#f4f2ed"      # background of the animations
SPEED = 1.4                # the animations play this much faster than their frame times


def _ease(t):
  t = min(max(t, 0), 1)
  return t * t * (3 - 2 * t)


def _grab(fig):
  from PIL import Image
  fig.canvas.draw()
  frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
  plt.close(fig)
  return frame


CONTROLS_PX = 100          # empty strip added under every frame: the video player's controls cover it


def save_video(path, frames, durations, fps=25):
  """An MP4 from frames with their own display times (ms). Each frame gets an empty strip at the
  bottom, so the player's controls never cover anything drawn."""
  import imageio_ffmpeg
  from PIL import Image
  def padded(frame):
    out = Image.new("RGB", (frame.width, frame.height + CONTROLS_PX), ANIM_PALE)
    out.paste(frame, (0, 0))
    return out
  frames = [padded(f) for f in frames]
  w, h = frames[0].size
  w2, h2 = w + w % 2, h + h % 2
  writer = imageio_ffmpeg.write_frames(str(path), (w2, h2), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                       macro_block_size=1, output_params=["-crf", "20", "-movflags", "+faststart"])
  writer.send(None)
  for frame, ms in zip(frames, durations):
    if (w2, h2) != (w, h):
      padded = Image.new("RGB", (w2, h2), ANIM_PALE); padded.paste(frame, (0, 0)); frame = padded
    data = np.ascontiguousarray(np.asarray(frame.convert("RGB")))
    for _ in range(max(1, round(ms / SPEED / 1000 * fps))):
      writer.send(data)
  writer.close()


def _frame(size=(6.4, 4.0)):
  fig, ax = plt.subplots(figsize=size, dpi=130, facecolor=ANIM_PALE)
  fig.subplots_adjust(0, 0, 1, 1)
  ax.set_aspect("equal")
  return fig, ax


def _internals():
  return np.load(HERE.parent / "data" / "model_internals.npz")


def _pct(share):
  """A share as a percentage: 190%, 27%, 0.15%."""
  v = 100 * share
  return ("%.0f%%" % v) if v >= 10 else ("%.2g%%" % v)


def _p_left(score):
  return 1 / (1 + np.exp(-np.asarray(score)))


def fade_animation():
  """Our Gabor fading from 50% to 0.1% contrast: the half with the Gabor and the half without,
  as two separate panels. (The layers are in timeline_figure.)"""
  frames, durations = [], []
  F = stimuli.FIELD_PX
  W, gap = 4.2, 0.8                                      # panel size and the space between them
  X0 = (12 - 2 * W - gap) / 2                            # centred
  Y0 = 2.2                                               # room at the bottom for the video controls

  def frame(caption, shown):
    fig, ax = _frame((6.4, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.8); ax.axis("off")
    ax.text(6, 7.25, caption, ha="center", fontsize=9.5)
    display = np.round(stimuli.draw_trial(dict(frequency=4, contrast=shown, phase=0.0), "left"))
    for k, (part, label) in enumerate([(display[:, :F], "Gabor here"), (display[:, F:], "no Gabor here")]):
      x = X0 + k * (W + gap)
      ax.imshow(part, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(x, x + W, Y0, Y0 + W))
      ax.text(x + W / 2, Y0 + W + 0.25, label, ha="center", fontsize=8.5, color=SOFT)
    frames.append(_grab(fig))

  frame("Our Gabor (4 c/deg), here at 50% contrast so you can see it", 0.5); durations.append(3000)
  for c in np.logspace(np.log10(0.5), -3, 20)[1:]:
    frame("Now at %.2g%% contrast" % (100 * c), c); durations.append(160)
  frame("At 0.1% it is too faint for you, and below the smallest grey step of a screen", 0.001); durations.append(3200)
  frame("The model gets the exact values, so for the model it is still there", 0.001); durations.append(3000)
  return frames, durations


def answer_animation():
  """The model's answer as the stripes get stronger: its chance of saying "left" (among left
  and right) for the Gabor on the left and on the right, and with no Gabor."""
  z = _internals()
  series, pl, pr = z["series"], _p_left(z["score_left"]), _p_left(z["score_right"])
  pb = float(_p_left(z["score_blank"]))
  frames, durations = [], []
  X0, X1 = 2.4, 10.6
  idx = np.arange(len(series))

  def frame(caption, k, show=(True, True)):
    fig, ax = _frame((6.4, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(6, 5.65, caption, ha="center", fontsize=9.5)
    c = np.exp(np.interp(k, idx, np.log(series)))
    left, right = np.interp(k, idx, pl), np.interp(k, idx, pr)
    ax.text(6, 4.85, "contrast %.3g%%" % (100 * c), ha="center", fontsize=10.5, color=RED)
    for y, p, label, on in [(3.4, left, "Gabor on the left", show[0]), (2.0, right, "Gabor on the right", show[1])]:
      ax.text(X0 - 0.2, y + 0.25, label, ha="right", va="center", fontsize=8.5, alpha=1 if on else 0.3)
      ax.add_patch(plt.Rectangle((X0, y), X1 - X0, 0.5, fc="white", ec=SOFT, lw=0.6))
      if on:
        ax.add_patch(plt.Rectangle((X0, y), (X1 - X0) * p, 0.5, fc=RED, ec="none", alpha=0.85))
        ax.text(X0 + (X1 - X0) * p + 0.1, y + 0.25, "%.0f%%" % (100 * p), va="center", fontsize=8.5)
    xb = X0 + (X1 - X0) * pb
    ax.plot([xb, xb], [1.7, 4.2], color=INK, lw=1, ls=(0, (3, 2)))
    ax.text(xb, 1.25, "no Gabor: %.0f%%" % (100 * pb), ha="center", fontsize=8, color=SOFT)
    ax.text(X0, 0.55, "0%", fontsize=8, color=SOFT)
    ax.text(X1, 0.55, "100% \"left\"", fontsize=8, color=SOFT, ha="right")
    frames.append(_grab(fig))

  frame("How often the model would say \"left\" (out of left and right)", 0, (False, False)); durations.append(3000)
  frame("With no stripes it already says \"left\" %.0f%% of the time: its lean" % (100 * pb), 0, (False, False))
  durations.append(3000)
  n = len(series) - 1
  for k in np.linspace(0, n, 12 * n + 1):
    c = np.exp(np.interp(k, idx, np.log(series)))
    left, right = np.interp(k, idx, pl), np.interp(k, idx, pr)
    caption = ("Faint stripes: almost no change" if abs(left - right) < 0.005 else
               "Here the stripes push the answer the wrong way" if left < right else
               "Stronger stripes: the answer follows them")
    frame(caption, k); durations.append(110)
    if min(abs(k - 4), abs(k - 5), abs(k - 8)) < 1e-9:
      durations[-1] = 2600
  durations[-1] = 4000
  return frames, durations


def tutorial_animations(img_dir):
  img_dir = Path(img_dir)
  save_video(img_dir / "vlm_travel_fade.mp4", *fade_animation())
  save_video(img_dir / "vlm_square.mp4", *square_animation())
  save_video(img_dir / "vlm_layers_vision.mp4", *layers_animation("vision"))
  save_video(img_dir / "vlm_layers_language.mp4", *layers_animation("language"))
  save_video(img_dir / "vlm_answer.mp4", *answer_animation())



def _strip(ax, x, y, w, h, values, cmap="gray", symmetric=False):
  """A list of real numbers as a strip of cells, scaled to the list's own range."""
  if symmetric:
    v = np.percentile(np.abs(values), 98) or 1
    lo, hi = -v, v
  else:
    lo, hi = np.percentile(values, [2, 98])
  ax.imshow(values.reshape(1, -1), cmap=cmap, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest",
            extent=(x, x + w, y, y + h))
  ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.6))


def _scale(ax, norm, x=2.2, y=0.35, w=7.6):
  """The colour scale, in words."""
  values = np.logspace(-4, 0, 200)
  ax.imshow(values.reshape(1, -1), cmap="Reds", norm=norm, aspect="auto", extent=(x, x + w, y, y + 0.28))
  ax.add_patch(plt.Rectangle((x, y), w, 0.28, fill=False, ec=SOFT, lw=0.5))
  ax.text(x, y + 0.45, "same numbers in both runs", fontsize=7.5, color=SOFT)
  ax.text(x + w, y + 0.45, "very different", fontsize=7.5, color=SOFT, ha="right")
  for v, label in [(1e-4, "0.01%"), (1e-2, "1%"), (1, "100%")]:
    ax.text(x + w * (np.log10(v) + 4) / 4, y - 0.25, label, fontsize=7, color=SOFT, ha="center")


def square_animation():
  """Step by step, for one square: the model runs twice (with and without the 0.1% Gabor); the
  square's 1,024 numbers after vision layer 1 in both runs; their difference; and how that
  difference becomes the square's colour. Real values."""
  from matplotlib.colors import LogNorm
  z = _internals()
  norm = LogNorm(1e-4, 1)
  blank, gabor = z["square_blank"][0], z["square_gabor"][0]
  diff = gabor - blank
  share = np.linalg.norm(diff) / np.linalg.norm(blank)
  r, c = z["square"]
  P = stimuli.PATCH
  rows, cols = stimuli.CANVAS_PX[1] // P, stimuli.CANVAS_PX[0] // P
  frames, durations = [], []
  W = 4.4
  H = W * rows / cols

  def frame(caption, step):
    fig, ax = _frame((6.4, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.8); ax.axis("off")
    ax.text(6, 7.25, caption, ha="center", fontsize=9.5)
    for k, label in enumerate(["run 1: with the Gabor (0.1%)", "run 2: no Gabor"]):
      x = 1.2 + k * (W + 0.8)
      ax.imshow(np.full((2, 2), 128), cmap="gray", vmin=0, vmax=255, extent=(x, x + W, 4.3, 4.3 + H))
      ax.add_patch(plt.Rectangle((x, 4.3), W, H, fill=False, ec=SOFT, lw=0.5))
      ax.plot([x + W / 2] * 2, [4.3, 4.3 + H], color=SOFT, lw=0.5, ls=(0, (3, 3)))
      ax.text(x + W / 2, 4.3 + H + 0.2, label, ha="center", fontsize=8.5, color=INK if k == 0 else SOFT)
      if step >= 1:
        cell = W / cols
        colour = plt.cm.Reds(norm(share)) if (step >= 4 and k == 0) else "none"
        ax.add_patch(plt.Rectangle((x + c * cell, 4.3 + H - (r + 1) * cell), cell, cell, fc=colour, ec=RED, lw=1.4))
    if step >= 2:
      ax.text(1.2, 3.55, "this square's 1,024 numbers after vision layer 1", fontsize=8, color=SOFT)
      ax.text(1.1, 3.05, "run 1", fontsize=8, ha="right", va="center")
      _strip(ax, 1.2, 2.85, 9.8, 0.4, gabor)
      ax.text(1.1, 2.45, "run 2", fontsize=8, ha="right", va="center")
      _strip(ax, 1.2, 2.25, 9.8, 0.4, blank)
    if step >= 3:
      ax.text(1.1, 1.65, "difference", fontsize=8, ha="right", va="center")
      _strip(ax, 1.2, 1.45, 9.8, 0.4, diff, cmap="RdBu_r", symmetric=True)
      ax.text(11.0, 1.15, "(enlarged %d times so it can be seen)" % round(np.percentile(np.abs(blank), 98) / np.percentile(np.abs(diff), 98), -1),
              fontsize=7.5, color=SOFT, ha="right")
    if step >= 4:
      ax.text(6, 0.5, "The difference is %s of the size of the numbers: that is the square's colour" % _pct(share),
              ha="center", fontsize=8.5, color=RED)
    frames.append(_grab(fig))

  frame("We run the model twice on the same display: once with the Gabor, once without", 0); durations.append(3200)
  frame("Take one square in the middle of the Gabor", 1); durations.append(2600)
  frame("After vision layer 1 it is a list of 1,024 numbers, in each run", 2); durations.append(3600)
  frame("The two lists look the same; subtract them to see the difference", 3); durations.append(3600)
  frame("The bigger the difference, the darker the red", 4); durations.append(4500)
  return frames, durations


def layers_animation(part):
  """The maps of Figure 6c (part "vision": layers 1-24 and the merger) and 6d (part
  "language": layers 1-28 and the place where the answer is written): every square or token
  coloured by how different its numbers are with and without the 0.1% Gabor. Real values."""
  from matplotlib.colors import LogNorm
  z = _internals()
  vision, merger, language, answer = z["vision"], z["merger"], z["language"], z["answer"]
  norm = LogNorm(1e-4, 1)
  W, gap, X0, Y0 = 3.9, 0.6, 0.6, 2.2
  frames, durations = [], []

  def frame(caption, grid, label, cell=None):
    fig, ax = _frame((6.4, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.8); ax.axis("off")
    ax.text(6, 7.25, caption, ha="center", fontsize=9.5)
    half = grid.shape[1] // 2
    for k, (part_grid, name) in enumerate([(grid[:, :half], "Gabor here"), (grid[:, half:], "no Gabor here")]):
      x = X0 + k * (W + gap)
      ax.imshow(np.clip(part_grid, 1e-4, 1), cmap="Reds", norm=norm, interpolation="nearest", extent=(x, x + W, Y0, Y0 + W))
      ax.add_patch(plt.Rectangle((x, Y0), W, W, fill=False, ec=SOFT, lw=0.5))
      ax.text(x + W / 2, Y0 + W + 0.25, name, ha="center", fontsize=8.5, color=SOFT)
    ax.text(X0, Y0 - 0.45, label, fontsize=9)
    bx = X0 + 2 * W + gap + 0.9
    if cell is not None:
      colour = "white" if isinstance(cell, str) else plt.cm.Reds(norm(np.clip(cell, 1e-4, 1)))
      ax.add_patch(plt.Rectangle((bx, Y0 + W / 2 - 0.5), 1.0, 1.0, fc=colour, ec=INK, lw=0.8))
      ax.text(bx + 0.5, Y0 + W / 2 + 0.75, "where the\nanswer is\nwritten", ha="center", fontsize=7.5, color=SOFT)
      if not isinstance(cell, str):
        ax.text(bx + 0.5, Y0 + W / 2 - 0.85, _pct(cell), ha="center", fontsize=8)
    _scale(ax, norm)
    frames.append(_grab(fig))

  if part == "vision":
    for i in range(24):
      caption = ("The first layers: only the squares of the Gabor differ" if i < 6 else
                 "Each layer mixes the squares: the difference spreads to the other half" if i < 23 else
                 "The last layer makes all numbers about 30 times bigger; a few squares differ a lot")
      frame(caption, vision[i], "vision layer %d of 24" % (i + 1))
      durations.append(3200 if i == 23 else 2400 if i in (0, 5, 6) else 500)
    frame("The merger brings every square back to the same size and packs 2 x 2 squares into a token",
          merger, "merger (tokens)"); durations.append(3500)

  if part == "language":
    frame("The language model gets the image tokens; its answer is written on the right", merger,
          "into the language model", cell="empty"); durations.append(3200)
    for i in range(28):
      frame("The difference stays in the image tokens, and reaches the place of the answer",
            language[i], "language layer %d of 28" % (i + 1), cell=answer[i])
      durations.append(2400 if i in (0, 27) else 420)
    pg, pb = _p_left(z["score_gabor"]), _p_left(z["score_blank"])
    frame("Yet the answer barely moves: %.1f%% \"left\" with the Gabor, %.1f%% without" % (100 * pg, 100 * pb),
          language[-1], "language layer 28 of 28", cell=answer[-1]); durations.append(4500)
  return frames, durations
