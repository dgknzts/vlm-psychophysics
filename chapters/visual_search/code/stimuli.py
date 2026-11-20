"""The search displays shown to the model.

The human tasks (Wolfe, Palmer & Horowitz, 2010, Methods): a black display 22.5 deg square,
divided into an invisible 5 x 5 grid of 4.5-deg cells. Each item sits at a random place inside
its own cell. Set sizes 3, 6, 12 and 18; the target is there on half of the trials.
  feature search   a red vertical bar (1.0 x 3.5 deg) among green vertical bars
  2 among 5s       a white digital "2" among white digital "5"s (1.5 x 2.7 deg)
Humans saw the display until they pressed "present" or "absent". A white fixation cross stayed
in the middle; it is left out here, as in the crowding chapter (the model has no eyes to hold still).

Here: 32 px/deg, as in the crowding chapter, so the display is 720 px; an 8-px black margin makes
the image 736 x 736 px, a whole number of the model's 32-px squares, so it is not resized.
Colours: pure red and pure green. Their brightness ratio on a standard (sRGB) screen, 0.21 / 0.72
= 0.30, is close to Wolfe's 4.5 / 13 cd/m2 = 0.35. The stroke of the digits is not given in the
paper; 0.2 deg is read roughly from their Figure 1.

Each layout is used twice: once with the target, once with a distractor in its place, so present
and absent displays differ in one item only.
  2 tasks x 4 set sizes x PAIRS layouts x 2 (present, absent)

Run: python stimuli.py     (prints the numbers, checks the displays, saves check images in
../data/check/ and the stimulus figure ../img/vlm_stimuli.png)
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# ---- Wolfe et al. (2010) ----
DISPLAY_DEG = 22.5
GRID = 5                                   # 5 x 5 cells
CELL_DEG = DISPLAY_DEG / GRID              # 4.5
BAR_DEG = (1.0, 3.5)                       # width x height
DIGIT_DEG = (1.5, 2.7)
SET_SIZES = [3, 6, 12, 18]
TASKS = ["feature", "2v5"]

# ---- the image ----
PX_PER_DEG = 32
MARGIN = 8
CELL_PX = round(CELL_DEG * PX_PER_DEG)     # 144
DISPLAY_PX = GRID * CELL_PX                # 720
CANVAS_PX = DISPLAY_PX + 2 * MARGIN        # 736
BAR_PX = tuple(round(d * PX_PER_DEG) for d in BAR_DEG)       # 32 x 112
DIGIT_PX = tuple(round(d * PX_PER_DEG) for d in DIGIT_DEG)   # 48 x 86
STROKE_PX = round(0.2 * PX_PER_DEG)                          # 6
BLACK, RED, GREEN, WHITE = (0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 255)

# ---- the experiment ----
PAIRS = 100                                # layouts per task and set size: 100 displays per point, 1600 in all


def make_trials(tasks=TASKS, set_sizes=SET_SIZES, pairs=PAIRS, seed=0):
  """Every task x set size x layout, each layout twice: target present, then absent."""
  rng = np.random.default_rng(seed)
  trials = []
  for task in tasks:
    for n in set_sizes:
      for p in range(pairs):
        layout_seed = int(rng.integers(2 ** 31))
        for present in (1, 0):
          trials.append(dict(index=len(trials), task=task, set_size=n, pair=p, present=present,
                             seed=layout_seed))
  return trials


def layout(trial):
  """The items of a layout: (cell row, cell column, x, y of the item's top-left corner in px, is it
  the target slot). The same for both displays of a pair; in the absent one the target slot holds a
  distractor."""
  rng = np.random.default_rng(trial["seed"])
  w, h = BAR_PX if trial["task"] == "feature" else DIGIT_PX
  cells = rng.choice(GRID * GRID, trial["set_size"], replace=False)
  target = int(rng.integers(trial["set_size"]))
  items = []
  for k, c in enumerate(cells):
    r, col = divmod(int(c), GRID)
    x = MARGIN + col * CELL_PX + int(rng.integers(CELL_PX - w + 1))
    y = MARGIN + r * CELL_PX + int(rng.integers(CELL_PX - h + 1))
    items.append(dict(row=r, col=col, x=x, y=y, target=(k == target)))
  return items


# seven segments of a digital digit: a top, b top right, c bottom right, d bottom, e bottom left,
# f top left, g middle. A 2 is a,b,g,e,d; a 5 is a,f,g,c,d: the same parts, mirrored.
SEGMENTS = {"2": "abged", "5": "afgcd"}


def _digit(img, x, y, which):
  w, h, s = *DIGIT_PX, STROKE_PX
  mid = (h - s) // 2
  boxes = {"a": (0, 0, w, s), "d": (0, h - s, w, h), "g": (0, mid, w, mid + s),
           "f": (0, 0, s, mid + s), "b": (w - s, 0, w, mid + s),
           "e": (0, mid, s, h), "c": (w - s, mid, w, h)}
  for seg in SEGMENTS[which]:
    x0, y0, x1, y1 = boxes[seg]
    img[y + y0:y + y1, x + x0:x + x1] = WHITE


def draw_trial(trial):
  """The display as an 8-bit RGB array (rows x columns x 3)."""
  img = np.zeros((CANVAS_PX, CANVAS_PX, 3), np.uint8)
  for it in layout(trial):
    is_target = it["target"] and trial["present"]
    if trial["task"] == "feature":
      w, h = BAR_PX
      img[it["y"]:it["y"] + h, it["x"]:it["x"] + w] = RED if is_target else GREEN
    else:
      _digit(img, it["x"], it["y"], "2" if is_target else "5")
  return img


def to_image(rgb):
  """As a PIL image for the processor."""
  from PIL import Image
  return Image.fromarray(rgb)


def count_items(trial, img):
  """Count targets and distractors by looking at each item's place: a check that the drawing is right."""
  targets = distractors = 0
  for it in layout(trial):
    if trial["task"] == "feature":
      w, h = BAR_PX
      colour = tuple(img[it["y"] + h // 2, it["x"] + w // 2])
      targets += colour == RED
      distractors += colour == GREEN
    else:
      # a 2 has its top-right stroke (b) and bottom-left (e); a 5 has top-left (f) and bottom-right (c)
      w, h, s = *DIGIT_PX, STROKE_PX
      top_left = tuple(img[it["y"] + h // 4, it["x"] + s // 2])
      top_right = tuple(img[it["y"] + h // 4, it["x"] + w - 1 - s // 2])
      targets += top_right == WHITE and top_left == BLACK
      distractors += top_left == WHITE and top_right == BLACK
  return targets, distractors


def main():
  trials = make_trials()
  print("%d px/deg: display %d px (%.1f deg), cells %d px, image %d x %d px"
        % (PX_PER_DEG, DISPLAY_PX, DISPLAY_DEG, CELL_PX, CANVAS_PX, CANVAS_PX))
  print("bar %d x %d px, digit %d x %d px, stroke %d px" % (*BAR_PX, *DIGIT_PX, STROKE_PX))
  print("displays:", len(trials), "(%d per task x set size x present/absent)" % PAIRS)

  wrong = overlaps = 0
  for t in trials:
    img = draw_trial(t)
    targets, distractors = count_items(t, img)
    wrong += (targets, distractors) != (t["present"], t["set_size"] - t["present"])
    w, h = BAR_PX if t["task"] == "feature" else DIGIT_PX
    mask = np.zeros((CANVAS_PX, CANVAS_PX), int)
    for it in layout(t):
      mask[it["y"]:it["y"] + h, it["x"]:it["x"] + w] += 1
    overlaps += mask.max() > 1
  print("displays with the wrong items:", wrong, "| with touching items:", overlaps)

  check = HERE.parent / "data" / "check"
  check.mkdir(exist_ok=True)
  for task in TASKS:
    for n in SET_SIZES:
      for present in (1, 0):
        t = next(t for t in trials if t["task"] == task and t["set_size"] == n and t["present"] == present)
        to_image(draw_trial(t)).save(check / ("%s_%02d_%s.png" % (task, n, "present" if present else "absent")))
  print("check images:", check)

  import figures
  figures.stimuli_figure(HERE.parent / "img" / "vlm_stimuli.png")


if __name__ == "__main__":
  main()
