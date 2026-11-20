"""Figures and measures for the visual search chapter, in the site's style: white image, dark ink,
IBM Plex Sans, fixation-dot red for the model, grey for humans. The page puts them on its mid-grey
display.

Used by stimuli.py and analysis.ipynb.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

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

NAMES = {"feature": "Red bar among green bars", "2v5": "2 among 5s"}
TASKS = ["feature", "2v5"]
SET_SIZES = [3, 6, 12, 18]


def _frame(ax):
  ax.set_xticks([]); ax.set_yticks([])
  for side in ax.spines.values():
    side.set_visible(True); side.set_color(SOFT)


def stimuli_figure(path):
  """Two rows, one per task: the same layout with the target and without it, at set sizes 6 and 18."""
  import stimuli
  trials = stimuli.make_trials()
  fig, axes = plt.subplots(2, 4, figsize=(8, 4.3), gridspec_kw=dict(wspace=0.06, hspace=0.18))
  for row, task in enumerate(stimuli.TASKS):
    col = 0
    for n in (6, 18):
      for present in (1, 0):
        t = next(t for t in trials if t["task"] == task and t["set_size"] == n and t["present"] == present)
        ax = axes[row, col]
        ax.imshow(stimuli.draw_trial(t), interpolation="nearest")
        _frame(ax)
        ax.set_title("%d items, target %s" % (n, "present" if present else "absent"), fontsize=8)
        col += 1
    axes[row, 0].set_ylabel({"feature": "Red bar among green", "2v5": "2 among 5s"}[task], fontsize=9)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


# ---------------------------------------------------------------- data

def load_humans():
  """Wolfe, Palmer & Horowitz (2010): one row per trial, both tasks, with their exclusions: practice
  trials out, RTs under 200 ms or over 4000 ms (feature) / 8000 ms (2 among 5s) out. The feature file
  has 10 observers; the paper analysed 9. The one left out is not named; "rm" is the one whose RTs are
  far from the rest (mean 774 ms, the others 310-510), and without rm the slopes match the paper's."""
  f = pd.read_csv(DATA / "wolfe2010_feature.csv", encoding="utf-8-sig")
  f = f[(f.BlockReport != "practice") & (f.sinit != "rm")]
  f = pd.DataFrame(dict(task="feature", subject=f.sinit, set_size=f.setsize, present=f["yn(stimulus)"],
                        error=f.error, rt=f.RT))
  s = pd.read_csv(DATA / "wolfe2010_2v5.csv", encoding="utf-8-sig")
  s = pd.DataFrame(dict(task="2v5", subject=s.Subject.astype(str), set_size=s.setsize,
                        present=s["yn(stimulus)"], error=s.Error, rt=s["RT(ms)"]))
  h = pd.concat([f[(f.rt >= 200) & (f.rt <= 4000)], s[(s.rt >= 200) & (s.rt <= 8000)]], ignore_index=True)
  h["correct"] = 1 - h.error
  return h


def load_model(path=DATA / "model_results.csv"):
  return pd.read_csv(path)


def human_rt(h, stat="median"):
  """RT of correct trials per person (median, or mean as in the paper), then the mean over people and its SE."""
  per = h[h.correct == 1].groupby(["task", "present", "set_size", "subject"]).rt.agg(stat).reset_index()
  return per.groupby(["task", "present", "set_size"]).rt.agg(["mean", "sem"]).reset_index()


def model_rt(m):
  """Median thinking tokens of correct displays."""
  return (m[m.correct == 1].groupby(["task", "present", "set_size"]).rt_tokens.median()
          .rename("median").reset_index())


def accuracy(d):
  """Proportion correct by task, target present or absent, and set size (all trials together)."""
  return d.groupby(["task", "present", "set_size"]).correct.agg(["mean", "count"]).reset_index()


def slopes(table, value):
  """Slope against set size, per task and target present (1) or absent (0)."""
  return {(t, p): float(np.polyfit(g.set_size, g[value], 1)[0]) for (t, p), g in table.groupby(["task", "present"])}


# ---------------------------------------------------------------- figures

def _set_size_axis(ax):
  ax.set_xticks(SET_SIZES); ax.set_xlim(0, 20); ax.set_xlabel("Number of items")


def _rt_panels(path, table, value, colour, ylabel, ymax, err=None):
  """One panel per task: target present solid, absent dashed."""
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
  for ax, task in zip(axes, TASKS):
    for present, style, label in [(1, "-", "target present"), (0, "--", "target absent")]:
      g = table[(table.task == task) & (table.present == present)].sort_values("set_size")
      ax.plot(g.set_size, g[value], style, color=colour, marker="o", ms=4.5, lw=1.6, label=label,
              mfc=colour if present else "white")
      if err is not None:
        ax.errorbar(g.set_size, g[value], g[err], fmt="none", color=colour, lw=1)
    ax.set_title(NAMES[task], fontsize=10)
    _set_size_axis(ax)
    ax.set_ylim(0, ymax)
  axes[0].set_ylabel(ylabel)
  axes[0].legend(loc="upper left", fontsize=8.5)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def human_rt_figure(path, h):
  _rt_panels(path, human_rt(h), "mean", HUMAN, "Reaction time (ms)", 2400, err="sem")


def model_rt_figure(path, m):
  _rt_panels(path, model_rt(m), "median", RED, "Thinking tokens", 100)


def accuracy_figure(path, m, h):
  """Proportion correct, model (red) and humans (grey), one panel per task."""
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
  for ax, task in zip(axes, TASKS):
    for d, colour in [(h, HUMAN), (m, RED)]:
      a = accuracy(d)
      for present, style in [(1, "-"), (0, "--")]:
        g = a[(a.task == task) & (a.present == present)].sort_values("set_size")
        ax.plot(g.set_size, g["mean"], style, color=colour, marker="o", ms=4.5, lw=1.6,
                mfc=colour if present else "white")
    ax.axhline(0.5, color=SOFT, lw=0.8, ls=":")
    ax.set_title(NAMES[task], fontsize=10)
    _set_size_axis(ax)
    ax.set_ylim(0.45, 1.03)
  axes[0].set_ylabel("Proportion correct")
  axes[0].legend([Line2D([], [], color=RED, lw=1.6), Line2D([], [], color=HUMAN, lw=1.6),
                  Line2D([], [], color=INK, lw=1.6), Line2D([], [], color=INK, lw=1.6, ls="--")],
                 ["Qwen3-VL-2B-Thinking", "humans", "target present", "target absent"],
                 loc="center left", fontsize=8.5)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


EXAMPLES = [658, 1438]   # display index: a red-bar reply and a 2-among-5s reply, both at their median length


def token_figure(path, m, examples=EXAMPLES):
  """Two real replies cut into the model's tokens. Each box is one token, written in one step of the
  model. The number of boxes before </think> is the RT."""
  from transformers import AutoTokenizer
  import observer
  tok = AutoTokenizer.from_pretrained(observer.MODEL)
  width, line_h, pad, gap = 7.6, 0.30, 0.025, 0.02

  probe = plt.figure()
  renderer = probe.canvas.get_renderer()
  def text_width(s):
    t = probe.text(0, 0, s, fontsize=8.5)
    w = t.get_window_extent(renderer).width / probe.dpi
    t.remove()
    return w

  items, y = [], 0.0
  for idx in examples:
    r = m[m["index"] == idx].iloc[0]
    ids = tok.encode(r.thinking, add_special_tokens=False)
    assert len(ids) == r.rt_tokens, (len(ids), r.rt_tokens)
    words = [tok.decode([i]).replace("\n", " ") for i in ids]
    items.append(("title", "%s, %d items, target %s" % (NAMES[r.task], r.set_size,
                                                        "present" if r.present else "absent"), 0, y))
    y -= line_h * 1.3
    x = 0.0
    for k, w in enumerate(words + ["</think>", r.reply]):
      w = w.strip() or "·"                           # a token that is only a space
      wdt = text_width(w) + 2 * pad
      if x + wdt > width:
        x, y = 0.0, y - line_h
      kind = "think" if k < len(words) else ("end" if k == len(words) else "answer")
      items.append((kind, (w, k, wdt), x, y))
      x += wdt + gap
    y -= line_h * 1.15
    items.append(("count", "%d tokens before </think>: RT = %d" % (len(words), len(words)), 0, y))
    y -= line_h * 1.9
  plt.close(probe)

  height = -y
  fig = plt.figure(figsize=(width, height + line_h))
  ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, width); ax.set_ylim(y, line_h); ax.axis("off")
  for kind, val, x0, yy in items:
    if kind == "title":
      ax.text(0, yy, val, fontsize=9.5, va="center")
    elif kind == "count":
      ax.text(0, yy, val, fontsize=9, color=RED, va="center")
    else:
      w, k, wdt = val
      face = {"think": "#ecebe7" if k % 2 == 0 else "#d4d2cc", "end": INK, "answer": RED}[kind]
      ax.add_patch(plt.Rectangle((x0, yy - line_h * 0.42), wdt, line_h * 0.84, color=face, lw=0))
      ax.text(x0 + pad, yy, w, fontsize=8.5, va="center", color=INK if kind == "think" else "white")
  fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
  plt.close(fig)
