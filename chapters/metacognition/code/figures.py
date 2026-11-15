"""Figures and measures for the metacognition chapter, in the site's style: white image, dark ink,
IBM Plex Sans, fixation-dot red for the model, grey for humans. The page puts them on its mid-grey
display.

Used by analysis.ipynb.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
INK = "#1d1d1b"
SOFT = "#8a8780"
HUMAN = "#55534e"
RED = "#ff4d5e"

font_manager.fontManager.addfont(str(HERE / "fonts" / "IBMPlexSans.ttf"))
plt.rcParams.update({
  "font.family": "IBM Plex Sans", "font.size": 10, "text.color": INK,
  "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
  "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
  "legend.frameon": False, "savefig.facecolor": "white",
})


def _frame(ax):
  ax.set_xticks([]); ax.set_yticks([])
  for side in ax.spines.values():
    side.set_visible(True); side.set_color(SOFT)


def stimuli_figure(path):
  """Top: one whole display, as the model gets it. Below: one display at each difference."""
  import stimuli
  trials = stimuli.make_trials()
  diffs = stimuli.DIFFERENCES
  fig = plt.figure(figsize=(7.6, 3.9))
  grid = fig.add_gridspec(2, len(diffs), height_ratios=[2.6, 1], hspace=0.12, wspace=0.08)

  ax = fig.add_subplot(grid[0, 1:-1])
  big = next(t for t in trials if t["difference"] == 20 and t["fuller"] == "right")
  ax.imshow(stimuli.draw_trial(big), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
  _frame(ax)
  ax.set_title("One display: %d × %d px; %d dots left, %d right"
               % (*stimuli.CANVAS_PX, stimuli.STANDARD, stimuli.STANDARD + 20), fontsize=9)

  for i, d in enumerate(diffs):
    ax = fig.add_subplot(grid[1, i])
    t = next(t for t in trials if t["difference"] == d and t["fuller"] == "left")
    ax.imshow(stimuli.draw_trial(t), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    _frame(ax)
    ax.set_title("+%d on the left" % d, fontsize=8)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def human_trial_figure(path):
  """One trial of the human task, drawn from Rouault et al. (2018, Supplement): fixation cross,
  the dots for 300 ms, the empty boxes until the answer, then the confidence scale."""
  import stimuli
  t = dict(difference=40, fuller="left", seed=3)
  dots = stimuli.draw_trial(t)
  empty = dots.copy()
  y0 = (stimuli.CANVAS_PX[1] - stimuli.BOX_PX) // 2
  for half in range(2):
    x0 = half * stimuli.CANVAS_PX[0] // 2 + (stimuli.CANVAS_PX[0] // 2 - stimuli.BOX_PX) // 2
    empty[y0:y0 + stimuli.BOX_PX, x0:x0 + stimuli.BOX_PX] = stimuli.BLACK
  cross = np.full_like(dots, stimuli.GREY)
  h, w = cross.shape
  cross[h // 2 - 1:h // 2 + 2, w // 2 - 12:w // 2 + 12] = stimuli.WHITE
  cross[h // 2 - 12:h // 2 + 12, w // 2 - 1:w // 2 + 2] = stimuli.WHITE
  scale = np.full_like(dots, stimuli.GREY)

  screens = [(cross, "fixation, 1000 ms"), (dots, "dots, 300 ms"),
             (empty, "left or right?"), (scale, "how sure? (1 to 11)")]
  fig, axs = plt.subplots(1, 4, figsize=(7.6, 1.5), gridspec_kw=dict(wspace=0.12))
  for ax, (img, label) in zip(axs, screens):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    _frame(ax)
    ax.set_title(label, fontsize=8.5)
  ax = axs[-1]
  for k in range(1, 12):
    ax.text(46 + (k - 1) * 42, 112, str(k), color="white", fontsize=5.5, ha="center", va="center")
  ax.text(30, 165, "certainly\nwrong", color="white", fontsize=5, ha="left", va="center")
  ax.text(482, 165, "certainly\ncorrect", color="white", fontsize=5, ha="right", va="center")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def type2_auroc(confidence, correct):
  """Area under the type-2 ROC curve: how well confidence tells right answers from wrong ones.
  It equals the chance that a random right answer got higher confidence than a random wrong one
  (ties count half). 0.5 = no insight, 1 = perfect. NaN if there are no wrong (or no right) answers."""
  confidence = np.asarray(confidence, dtype=float)
  correct = np.asarray(correct).astype(bool)
  right, wrong = confidence[correct], confidence[~correct]
  if len(right) == 0 or len(wrong) == 0:
    return np.nan
  higher = (right[:, None] > wrong[None, :]).mean()
  same = (right[:, None] == wrong[None, :]).mean()
  return higher + 0.5 * same


def load_humans(path=DATA / "data_Rouault_2018_Expt1.csv"):
  """Rouault et al. (2018), experiment 1, from the Confidence Database: one row per trial."""
  return pd.read_csv(path)


def human_people(humans):
  """One row per person: accuracy, mean confidence and the type-2 ROC area."""
  rows = []
  for subject, d in humans.groupby("Subj_idx"):
    rows.append(dict(subject=subject, trials=len(d), accuracy=d.Accuracy.mean(), confidence=d.Confidence.mean(),
                     auroc=type2_auroc(d.Confidence, d.Accuracy)))
  return pd.DataFrame(rows)


def human_by_difference(humans, width=5):
  """Accuracy in bins of the dot difference (1-5, 6-10, ...): the mean over people, and its SE."""
  d = humans.assign(bin=(humans.DotDiff - 1) // width)
  per_person = d.groupby(["Subj_idx", "bin"]).Accuracy.mean().unstack()
  centre = per_person.columns * width + (width + 1) / 2
  return pd.DataFrame({"difference": centre, "accuracy": per_person.mean().values,
                       "se": (per_person.std() / np.sqrt(len(per_person))).values})


def human_by_confidence(humans):
  """Accuracy at each confidence level, over all trials of all people, and the number of trials."""
  return humans.groupby("Confidence").Accuracy.agg(accuracy="mean", trials="count").reset_index()


def human_figure(path, humans):
  """Left: accuracy against the dot difference. Right: accuracy at each confidence level."""
  by_diff = human_by_difference(humans)
  by_conf = human_by_confidence(humans)
  fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(wspace=0.35))

  a.axhline(0.5, color=SOFT, lw=0.7, ls=(0, (3, 3)))
  a.errorbar(by_diff.difference, by_diff.accuracy, yerr=by_diff.se, fmt="o-", color=HUMAN, ms=4, lw=1.2,
             capsize=0)
  a.set_xlabel("difference in dots")
  a.set_xlim(0, 72)

  b.axhline(0.5, color=SOFT, lw=0.7, ls=(0, (3, 3)))
  b.plot(by_conf.Confidence, by_conf.accuracy, "o-", color=HUMAN, ms=4, lw=1.2)
  b.set_xticks(range(1, 12))
  b.set_xlabel("confidence (1 = certainly wrong, 11 = certainly correct)", fontsize=9)

  for ax in (a, b):
    ax.set_ylim(0.25, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["25%", "50%", "75%", "100%"])
    ax.set_ylabel("correct")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


# ---- the model ----

def load_model(path=DATA / "model_results.csv"):
  return pd.read_csv(path)


def _percent_axis(ax, low=0.0):
  ax.set_ylim(low, 1.02)
  ticks = [t for t in [0, 0.25, 0.5, 0.75, 1] if t >= low]
  ax.set_yticks(ticks); ax.set_yticklabels(["%d%%" % (100 * t) for t in ticks])


def example_displays(results):
  """One easy and one hard display for the tutorial: the largest and the smallest difference,
  the first display of each (fuller box on the left)."""
  easy = results[(results.difference == results.difference.max()) & (results.fuller == "left")].iloc[0]
  hard = results[(results.difference == results.difference.min()) & (results.fuller == "left")].iloc[0]
  return [("easy", easy), ("hard", hard)]


def tutorial_answer_figure(path, results):
  """Step 1: the display, and the model's probabilities for "left" and "right" as its first word."""
  import stimuli
  examples = example_displays(results)
  fig, axs = plt.subplots(2, 2, figsize=(7.0, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1], hspace=0.5, wspace=0.25))
  for (name, row), (a, b) in zip(examples, axs):
    a.imshow(stimuli.draw_trial(row.to_dict()), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    _frame(a)
    a.set_title("%s: %d more dots on the left" % (name, row.difference), fontsize=8.5)
    total = row.p_left + row.p_right
    shares = [row.p_left / total, row.p_right / total]
    colours = [RED if w == row.answer else SOFT for w in ("left", "right")]
    b.bar([0, 1], shares, color=colours, width=0.6)
    for x, s in enumerate(shares):
      b.text(x, s + 0.03, "%.1f%%" % (100 * s) if s < 0.995 else ">99.5%", ha="center", fontsize=8)
    b.set_xticks([0, 1]); b.set_xticklabels(['"left"', '"right"'])
    _percent_axis(b)
    b.set_ylim(0, 1.15)
    b.set_title("first word", fontsize=8.5)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def tutorial_confidence_figure(path, results):
  """Step 2: after its answer, the model's probabilities for the digits 1 to 5."""
  examples = example_displays(results)
  fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True, gridspec_kw=dict(wspace=0.15))
  for (name, row), ax in zip(examples, axs):
    p = [row["p%d" % k] for k in range(1, 6)]
    best = int(np.argmax(p))
    ax.bar(range(1, 6), p, color=[RED if k == best else SOFT for k in range(5)], width=0.6)
    for k, v in enumerate(p):
      ax.text(k + 1, v + 0.03, "%.0f%%" % (100 * v) if v >= 0.005 else "<0.5%", ha="center", fontsize=7.5)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("1 = guessing, 5 = certain", fontsize=8.5)
    ax.set_title('%s display, after "%s"' % (name, row.reply), fontsize=8.5)
    _percent_axis(ax)
    ax.set_ylim(0, 1.15)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def right_wrong_figure(path, results):
  """Step 3: how each kind of confidence is spread over right answers (grey) and wrong answers
  (red). Left: stated confidence, the share of each group at each level. Right: hidden
  confidence, the share of each group in each bin."""
  right, wrong = results[results.correct == 1], results[results.correct == 0]
  fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.3))
  for ax, column, edges, ticks, labels, title in [
      (a, "stated", np.arange(0.5, 6), np.arange(1, 6), [str(k) for k in range(1, 6)], "stated confidence"),
      (b, "hidden", np.array(HIDDEN_BINS), np.arange(len(HIDDEN_LABELS)), HIDDEN_LABELS, "hidden confidence (%)")]:
    for shift, group, colour, label in [(-0.18, right, HUMAN, "right answers (%d)" % len(right)),
                                        (0.18, wrong, RED, "wrong answers (%d)" % len(wrong))]:
      counts, _ = np.histogram(group[column], edges)
      ax.bar(ticks + shift, counts / max(len(group), 1), width=0.36, color=colour, label=label)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8 if column == "hidden" else 9)
    ax.set_title(title, fontsize=8.5)
    _percent_axis(ax)
  a.set_ylabel("share of the answers")
  a.legend(fontsize=7.5, loc="upper left")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def model_by_difference(results):
  return results.groupby("difference").agg(correct=("correct", "mean"), trials=("correct", "size"),
                                           hidden=("hidden", "mean"), stated=("stated", "mean")).reset_index()


def accuracy_figure(path, results, humans):
  """Proportion correct against the difference in dots: humans (grey, each difference) and the
  model (red)."""
  h = human_by_difference(humans)
  m = model_by_difference(results)
  fig, ax = plt.subplots(figsize=(5.6, 3.6))
  ax.axhline(0.5, color=SOFT, lw=0.7, ls=(0, (3, 3)))
  ax.errorbar(h.difference, h.accuracy, yerr=h.se, fmt="o-", color=HUMAN, ms=3, lw=1.1, capsize=0,
              label="humans (498; bins of 5 dots)")
  ax.plot(m.difference, m.correct, "-", color=RED, lw=0.8, alpha=0.5)
  ax.plot(m.difference, m.correct, "o", color=RED, ms=5, label="Qwen3-VL-2B")
  ax.set_xscale("log")
  ticks = sorted(set([2, 5, 10, 20, 40, 70] + list(m.difference)))
  ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
  ax.minorticks_off()
  ax.set_xlabel("difference in dots (313 vs 313 + difference)")
  ax.set_ylabel("correct")
  _percent_axis(ax, 0.25)
  ax.legend(fontsize=8.5, loc="upper left")
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


HIDDEN_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
HIDDEN_LABELS = ["50-60", "60-70", "70-80", "80-90", "90-100"]


def by_level(results):
  """Proportion correct at each stated confidence level and each bin of hidden confidence."""
  stated = results.groupby("stated").correct.agg(correct="mean", trials="size").reset_index()
  b = pd.cut(results.hidden, HIDDEN_BINS, labels=HIDDEN_LABELS, right=False)
  hidden = results.groupby(b, observed=False).correct.agg(correct="mean", trials="size").reset_index()
  return stated, hidden


def confidence_figure(path, results, humans):
  """Proportion correct at each confidence level: humans (1-11), the model's stated confidence
  (1-5) and its hidden confidence (bins). The number of trials is written under each point."""
  hum = human_by_confidence(humans)
  stated, hidden = by_level(results)
  fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True,
                          gridspec_kw=dict(width_ratios=[11, 6, 6], wspace=0.12))
  panels = [(axs[0], hum.Confidence, hum.accuracy, hum.trials, HUMAN, "humans: stated (1-11)", range(1, 12)),
            (axs[1], stated.stated, stated.correct, stated.trials, RED, "model: stated (1-5)", range(1, 6)),
            (axs[2], np.arange(len(hidden)), hidden.correct, hidden.trials, RED, "model: hidden (%)",
             range(len(hidden)))]
  for ax, x, y, n, colour, title, ticks in panels:
    x, y, n = np.asarray(x), np.asarray(y, float), np.asarray(n)
    ax.axhline(0.5, color=SOFT, lw=0.7, ls=(0, (3, 3)))
    keep = n > 0
    ax.plot(x[keep], y[keep], "o-", color=colour, ms=4, lw=1.1)
    for xi, yi, ni in zip(x[keep], y[keep], n[keep]):
      ax.text(xi, 0.06, str(ni), ha="center", fontsize=5.5 if colour == HUMAN else 6.5, color=SOFT, rotation=90 if ni >= 1000 else 0)
    ax.set_xticks(list(ticks))
    ax.set_title(title, fontsize=8.5)
  axs[2].set_xticklabels(HIDDEN_LABELS, fontsize=7, rotation=30)
  axs[0].set_ylabel("correct")
  _percent_axis(axs[0], 0.0)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)
