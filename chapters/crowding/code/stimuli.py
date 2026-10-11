"""The letter displays shown to the model.

Letter size and spacing follow Bouma (1970, Nature 226, p. 177): lowercase letters in a
Courier type with pronounced serifs, x-height 1.95 mm and letter pitch 2.55 mm, seen at
50 cm. That makes the x-height 13.4 arcmin and one letter "slot" 0.29 deg.

Bouma had paper, not pixels. We draw the display as a 24-inch, 1920-pixel monitor would
show it at his 50 cm: 31.5 pixels per degree, so the x-height is 7 pixels.

The experiment:
  target at 0 deg (image centre) or 10 deg (right of centre)
  one random flanker on each side, at 1, 2, 4, 8 or 16 slots centre to centre,
  or no flankers
  25 letters (a-z without x) x 4 = 100 trials per condition, 1,200 in all

There is no fixation cross: at 0 deg it would sit on the target, and the model does not
fixate anyway. The image centre plays the role of fixation.

Font: Courier Prime (SIL Open Font License, fonts/OFL-CourierPrime.txt).

Run: python stimuli.py     (prints the numbers, saves ../img/vlm_conditions.png, vlm_grid.png and Figure 6;
                            run model_internals.py first for Figure 6's real numbers)
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONT = HERE / "fonts" / "CourierPrime-Regular.ttf"

# ---- Bouma's numbers ----
VIEWING_MM = 500
X_HEIGHT_MM = 1.95
PITCH_MM = 2.55
MM_PER_DEG = 2 * VIEWING_MM * math.tan(math.radians(0.5))   # 8.73 mm
SLOT_DEG = PITCH_MM / MM_PER_DEG                             # 0.29 deg
X_HEIGHT_DEG = X_HEIGHT_MM / MM_PER_DEG                      # 13.4 arcmin

# ---- the experiment ----
PX_PER_MM = 1920 / (24 * 25.4 * 16 / math.hypot(16, 9))      # 24-inch 16:9 monitor, 1920 px wide
PX_PER_DEG = PX_PER_MM * MM_PER_DEG                          # 31.5
LETTERS = "abcdefghijklmnopqrstuvwyz"                        # 25 letters: a-z without x
ECCENTRICITIES = [0, 10]                                     # deg
SPACINGS = [1, 2, 4, 8, 16]                                  # slots, centre to centre; None = no flankers
TRIALS_PER_LETTER = 4                                        # 25 x 4 = 100 per condition
CANVAS_PX = (1024, 128)                                      # 32 x 4 deg; multiples of 32 (Qwen3-VL's blocks)


def spacing_deg(spacing):
  return spacing * SLOT_DEG


def conditions():
  """(eccentricity, spacing) pairs, spacing None = no flankers."""
  return [(ecc, s) for ecc in ECCENTRICITIES for s in SPACINGS + [None]]


def make_trials(seed=0):
  """Every letter TRIALS_PER_LETTER times per condition, with two random flankers that
  differ from the target and from each other."""
  rng = np.random.default_rng(seed)
  trials = []
  for ecc, spacing in conditions():
    for target in LETTERS:
      for rep in range(TRIALS_PER_LETTER):
        others = [l for l in LETTERS if l != target]
        inner, outer = (str(l) for l in rng.choice(others, size=2, replace=False))
        trials.append(dict(index=len(trials), target=target, ecc_deg=ecc, spacing=spacing,
                           inner_flanker=inner, outer_flanker=outer))
  return trials


_font = None

def font():
  """Courier Prime at the size whose x-height is Bouma's 13.4 arcmin."""
  global _font
  if _font is None:
    probe = ImageFont.truetype(str(FONT), 1000)
    _, top, _, bottom = probe.getbbox("x", anchor="ls")
    size = round(1000 * X_HEIGHT_DEG * PX_PER_DEG / (bottom - top))
    _font = ImageFont.truetype(str(FONT), size)
  return _font


def draw_trial(trial):
  """One display: black letters on white, the target trial["ecc_deg"] right of the centre.
  "inner" is the flanker on the centre side, "outer" the one further out."""
  img = Image.new("RGB", CANVAS_PX, "white")
  draw = ImageDraw.Draw(img)
  cx, cy = CANVAS_PX[0] / 2, CANVAS_PX[1] / 2
  baseline = cy + X_HEIGHT_DEG * PX_PER_DEG / 2     # centres the x-height on the middle line
  x = cx + trial["ecc_deg"] * PX_PER_DEG
  draw.text((x, baseline), trial["target"], fill="black", font=font(), anchor="ms")
  if trial["spacing"] is not None:
    d = spacing_deg(trial["spacing"]) * PX_PER_DEG
    draw.text((x - d, baseline), trial["inner_flanker"], fill="black", font=font(), anchor="ms")
    draw.text((x + d, baseline), trial["outer_flanker"], fill="black", font=font(), anchor="ms")
  return img


def main():
  print("%.1f px/deg; x-height %.1f arcmin = %.1f px; one slot %.3f deg = %.1f px"
        % (PX_PER_DEG, X_HEIGHT_DEG * 60, X_HEIGHT_DEG * PX_PER_DEG, SLOT_DEG, SLOT_DEG * PX_PER_DEG))
  print("spacings (deg):", [round(spacing_deg(s), 2) for s in SPACINGS])
  print("conditions:", len(conditions()), " trials:", len(make_trials()))
  far = 10 + spacing_deg(max(SPACINGS))
  print("farthest flanker at %.1f deg; canvas edge at %.1f deg" % (far, CANVAS_PX[0] / 2 / PX_PER_DEG))

  import figures
  figures.conditions_figure(HERE.parent / "img" / "vlm_conditions.png")
  figures.grid_figure(HERE.parent / "img" / "vlm_grid.png")
  figures.step1_figure(HERE.parent / "img" / "vlm_step1.png")
  # the rest of Figure 6 uses the model's real numbers, saved by model_internals.py
  figures.step_animations(HERE.parent / "img")
  figures.layers_figure(HERE.parent / "img" / "vlm_layers.png")


if __name__ == "__main__":
  main()
