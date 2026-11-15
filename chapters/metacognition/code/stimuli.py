"""The dot displays shown to the model.

The human task (Rouault et al., 2018, experiment 1; Supplement, "Behavioral Task"): two black
boxes of randomly placed white dots. One box is always half full, 313 dots out of 625 positions;
the other has 1 to 70 dots more. The dots are shown for 300 ms, then the observer says which box
had more, and how sure they are. The size of the boxes on the screen is not given (it was a web
study), so here the 625 positions are a 25 x 25 grid of 8-px cells: a 200-px box.

The model sees one image, 512 x 256 px, mid-grey, a box in each half, and is asked "left or
right?". Each display is used twice, the second time with the two boxes swapped, so a habit of
saying "left" falls equally on both sides.
  DIFFERENCES x PAIRS pairs = DIFFERENCES x 2 x PAIRS images
  each pair: new random dot positions

A laptop pilot (pilot.py, 4 pairs at 5 to 200 dots) found the model near chance at 5 and 10 dots
and better from 20 up, levelling off near 75% correct: it often says "left" when the fuller box
is on the right. It stated confidence 5 on every display.

Run: python stimuli.py     (prints the numbers, checks the dot counts, saves the check images in
../data/check/ and the stimulus figure ../img/vlm_stimuli.png)
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# ---- Rouault et al. (2018) ----
GRID = 25                      # 25 x 25 = 625 positions per box
STANDARD = 313                 # dots in the half-full box

# ---- the image ----
CELL_PX = 8
BOX_PX = GRID * CELL_PX        # 200
CANVAS_PX = (512, 256)         # two 256-px halves, as in chapter 2
GREY, BLACK, WHITE = 127, 0, 255
DOT_RADIUS = 2.5               # px: a round dot about 5 px across, in an 8-px cell

# ---- the experiment ----
DIFFERENCES = [5, 10, 20, 40, 70]        # within the human range (1-70), chosen by pilot.py
PAIRS = 30                               # 30 pairs = 60 displays per difference, 300 in all


def make_trials(differences=DIFFERENCES, pairs=PAIRS, seed=0):
  """Every difference x pair, each pair twice: the fuller box on the left, then on the right."""
  rng = np.random.default_rng(seed)
  trials = []
  for d in differences:
    for p in range(pairs):
      dots_seed = int(rng.integers(2 ** 31))
      for fuller in ("left", "right"):
        trials.append(dict(index=len(trials), difference=d, pair=p, fuller=fuller, seed=dots_seed))
  return trials


def boxes(trial):
  """The two boxes of a pair: True where there is a dot. The same for both displays of the pair."""
  rng = np.random.default_rng(trial["seed"])
  def box(n):
    cells = np.zeros(GRID * GRID, bool)
    cells[rng.choice(GRID * GRID, n, replace=False)] = True
    return cells.reshape(GRID, GRID)
  return box(STANDARD + trial["difference"]), box(STANDARD)


def _dot(fine=8):
  """One cell with a round white dot in the middle, the edge smoothed: each pixel's grey is the
  share of it the dot covers (worked out on a grid 8 times finer)."""
  n = CELL_PX * fine
  y, x = (np.mgrid[:n, :n] + 0.5) / fine - CELL_PX / 2
  inside = (x ** 2 + y ** 2 <= DOT_RADIUS ** 2).reshape(CELL_PX, fine, CELL_PX, fine).mean(axis=(1, 3))
  return np.round(BLACK + (WHITE - BLACK) * inside).astype(np.uint8)


def draw_box(cells):
  """A black box with a white dot in every filled cell."""
  box = np.full((BOX_PX, BOX_PX), BLACK, np.uint8)
  tile = _dot()
  for r, c in zip(*np.nonzero(cells)):
    box[r * CELL_PX:(r + 1) * CELL_PX, c * CELL_PX:(c + 1) * CELL_PX] = tile
  return box


def draw_trial(trial):
  """The display as an 8-bit grey image (rows x columns): a box centred in each half."""
  fuller, standard = boxes(trial)
  left, right = (fuller, standard) if trial["fuller"] == "left" else (standard, fuller)
  w, h = CANVAS_PX
  img = np.full((h, w), GREY, np.uint8)
  y0 = (h - BOX_PX) // 2
  for half, cells in enumerate((left, right)):
    x0 = half * w // 2 + (w // 2 - BOX_PX) // 2
    img[y0:y0 + BOX_PX, x0:x0 + BOX_PX] = draw_box(cells)
  return img


def to_image(grey):
  """As an RGB image for the processor."""
  from PIL import Image
  return Image.fromarray(grey).convert("RGB")


def count_dots(img):
  """Count the dots in each half by looking at the cell centres: a check that the drawing is right."""
  w, h = CANVAS_PX
  y0 = (h - BOX_PX) // 2
  centre = CELL_PX // 2
  counts = []
  for half in range(2):
    x0 = half * w // 2 + (w // 2 - BOX_PX) // 2
    box = img[y0:y0 + BOX_PX, x0:x0 + BOX_PX]
    counts.append(int((box[centre::CELL_PX, centre::CELL_PX] == WHITE).sum()))
  return counts


def main():
  trials = make_trials()
  print("box: %d x %d cells of %d px = %d px; standard %d dots of %d; image %d x %d"
        % (GRID, GRID, CELL_PX, BOX_PX, STANDARD, GRID * GRID, *CANVAS_PX))
  print("differences:", DIFFERENCES, "| displays:", len(trials), "(%d per difference)" % (2 * PAIRS))

  wrong = 0
  for t in trials:
    left, right = count_dots(draw_trial(t))
    want = STANDARD + t["difference"]
    wrong += (left, right) != ((want, STANDARD) if t["fuller"] == "left" else (STANDARD, want))
  print("displays with the wrong dot counts:", wrong)

  check = HERE.parent / "data" / "check"
  check.mkdir(exist_ok=True)
  for d in DIFFERENCES:
    first = next(t for t in trials if t["difference"] == d)
    to_image(draw_trial(first)).save(check / ("difference_%02d.png" % d))
  print("check images:", check)

  import figures
  figures.stimuli_figure(HERE.parent / "img" / "vlm_stimuli.png")


if __name__ == "__main__":
  main()
