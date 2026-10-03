"""Bouma (1970) letter displays for the model.

Bouma's stimuli (Nature 226, p. 177): lowercase letters in Courier 10 with pronounced
serifs, typed black on white paper, seen at 50 cm. x-height 1.95 mm, letter pitch 2.55 mm
(one "slot" = 0.29 deg). Target left or right of a fixation mark, alone or with an "x" on
each side, with n = 0, 1, 2, 5, 8, 11 or 15 empty slots between target and each x.

Bouma had no screen, so there are no pixels to copy. We draw his sheet at a chosen number of
pixels per mm; the geometry in degrees is the same at every resolution. Three resolutions:
    monitor  a 24-inch 1920-pixel screen at his 50 cm   (~32 px/deg, x-height ~7 px)
    150dpi   his sheet scanned at 150 dpi               (~52 px/deg, x-height ~11 px)
    300dpi   his sheet scanned at 300 dpi               (~103 px/deg, x-height ~23 px)

Font: Courier Prime (SIL Open Font License, fonts/OFL.txt), a Courier with clear serifs.
Letters are placed on Bouma's pitch directly; the font only draws them.

Run: python stimuli.py     (saves ../data/stimuli_examples.png)
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

# ---- our choices ----
LETTERS = "abcdefghijklmnopqrstuvwyz"    # 25 lowercase letters: a-z without the flanker x
ECCENTRICITIES = [2, 3, 5, 7]            # deg; the ones with a human fit (see fit_bouma.py)
GAPS = [0, 1, 2, 5, 8, 11, 15]           # empty slots between target and each flanker
TRIALS_PER_LETTER = 4                    # 25 x 4 = 100 per condition, as in Bouma; 2 left, 2 right
CANVAS_DEG = (25, 4)                     # width, height; the farthest flanker is at 11.6 deg
CROSS_DEG = 2 * X_HEIGHT_DEG             # fixation cross arm-to-arm (Bouma doesn't give its shape)

RESOLUTIONS = {                          # pixels per mm of Bouma's sheet
  "monitor": 1920 / (24 * 25.4 * 16 / math.hypot(16, 9)),
  "150dpi":  150 / 25.4,
  "300dpi":  300 / 25.4,
}


def pixels_per_degree(resolution):
  return RESOLUTIONS[resolution] * MM_PER_DEG


def canvas_size(resolution):
  """Canvas in pixels, rounded to a multiple of 32: Qwen3-VL cuts images into 32-pixel
  blocks and would otherwise resize them slightly."""
  ppd = pixels_per_degree(resolution)
  return tuple(32 * round(d * ppd / 32) for d in CANVAS_DEG)


def centre_spacing_deg(gap):
  """Target-to-flanker distance, centre to centre: the gap plus one letter."""
  return (gap + 1) * SLOT_DEG


def conditions():
  """(eccentricity, gap) pairs; gap None = no flankers.

  A spacing is used only if the inner flanker stays on the target's side of fixation and
  clear of the cross (centre at least 0.2 deg from it).
  """
  out = []
  for ecc in ECCENTRICITIES:
    for gap in GAPS:
      if ecc - centre_spacing_deg(gap) >= 0.2:
        out.append((ecc, gap))
    out.append((ecc, None))
  return out


def make_trials(flankers="x", seed=0):
  """All trials: every letter TRIALS_PER_LETTER times per condition, half left, half right.

  flankers="x": Bouma's design. flankers="random": two random letters from LETTERS,
  different from the target and from each other (the usual modern design).
  """
  rng = np.random.default_rng(seed)
  trials = []
  for ecc, gap in conditions():
    for target in LETTERS:
      for rep in range(TRIALS_PER_LETTER):
        side = "left" if rep % 2 == 0 else "right"
        if flankers == "x":
          pair = ("x", "x")
        else:
          others = [l for l in LETTERS if l != target]
          pair = tuple(str(l) for l in rng.choice(others, size=2, replace=False))
        trials.append(dict(index=len(trials), target=target, ecc_deg=ecc, gap=gap, side=side,
                           inner_flanker=pair[0], outer_flanker=pair[1], flankers=flankers))
  return trials


_fonts = {}

def font_for(resolution):
  """Courier Prime at the size whose x-height matches Bouma's 1.95 mm."""
  if resolution not in _fonts:
    x_height_px = X_HEIGHT_DEG * pixels_per_degree(resolution)
    probe = ImageFont.truetype(str(FONT), 1000)
    _, top, _, bottom = probe.getbbox("x", anchor="ls")
    size = round(1000 * x_height_px / (bottom - top))
    _fonts[resolution] = ImageFont.truetype(str(FONT), size)
  return _fonts[resolution]


TOKEN_PX = 32   # Qwen3-VL: 16-px patches, merged 2 x 2 into one token


def draw_trial(trial, resolution, align=None):
  """One display as a white RGB image, fixation cross in the centre.

  align=None: as is (the first run). Otherwise the whole display is shifted by up to half a
  token so the letter row sits in the middle of a row of tokens, and the target is
    "split":   cut in half by a token boundary (each token: one flanker + half the target)
    "centred": in the middle of one token
  Eccentricity and spacing in degrees do not change.
  """
  ppd = pixels_per_degree(resolution)
  width, height = canvas_size(resolution)
  img = Image.new("RGB", (width, height), "white")
  draw = ImageDraw.Draw(img)
  font = font_for(resolution)
  cx, cy = width / 2, height / 2
  if align is not None:
    sign = -1 if trial["side"] == "left" else 1
    target_x = cx + sign * trial["ecc_deg"] * ppd
    wanted = 0 if align == "split" else TOKEN_PX / 2
    cx += (wanted - target_x) % TOKEN_PX
    cy += (TOKEN_PX / 2 - cy) % TOKEN_PX
    if cx > width / 2 + TOKEN_PX / 2:
      cx -= TOKEN_PX
    if cy > height / 2 + TOKEN_PX / 2:
      cy -= TOKEN_PX

  # the cross is drawn with about the letters' stem width
  stroke = max(1, round(0.12 * X_HEIGHT_DEG * ppd))
  arm = CROSS_DEG * ppd / 2
  draw.line((cx - arm, cy, cx + arm, cy), fill="black", width=stroke)
  draw.line((cx, cy - arm, cx, cy + arm), fill="black", width=stroke)

  # letters sit on a baseline that centres their x-height on the fixation line
  baseline = cy + X_HEIGHT_DEG * ppd / 2
  sign = -1 if trial["side"] == "left" else 1
  target_x = cx + sign * trial["ecc_deg"] * ppd
  draw.text((target_x, baseline), trial["target"], fill="black", font=font, anchor="ms")
  if trial["gap"] is not None:
    d = centre_spacing_deg(trial["gap"]) * ppd
    draw.text((target_x - sign * d, baseline), trial["inner_flanker"], fill="black", font=font, anchor="ms")
    draw.text((target_x + sign * d, baseline), trial["outer_flanker"], fill="black", font=font, anchor="ms")
  return img


def main():
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  print("one slot = %.3f deg, x-height = %.1f arcmin" % (SLOT_DEG, X_HEIGHT_DEG * 60))
  for r in RESOLUTIONS:
    ppd = pixels_per_degree(r)
    font = font_for(r)
    widths = [font.getbbox(l)[2] - font.getbbox(l)[0] for l in LETTERS]
    print("%-8s %5.1f px/deg  x-height %4.1f px  slot %4.1f px  letter width %d-%d px (Bouma 1.4-2.5 mm = %.0f-%.0f px)  image %dx%d"
          % (r, ppd, X_HEIGHT_DEG * ppd, SLOT_DEG * ppd, min(widths), max(widths),
             1.4 * RESOLUTIONS[r], 2.5 * RESOLUTIONS[r], *canvas_size(r)))
  conds = conditions()
  print("%d conditions: %s" % (len(conds), conds))
  print("trials per resolution:", len(make_trials()))

  # examples: whole display at 5 deg with gap 2, then the target region at each resolution
  examples = [dict(target="k", ecc_deg=5, gap=2, side="right", inner_flanker="x", outer_flanker="x"),
              dict(target="k", ecc_deg=5, gap=None, side="left", inner_flanker="x", outer_flanker="x"),
              dict(target="k", ecc_deg=7, gap=0, side="right", inner_flanker="r", outer_flanker="m")]
  labels = ["gap 2, right", "alone, left", "random flankers, gap 0, right"]
  fig, axs = plt.subplots(len(RESOLUTIONS), 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})
  for i, r in enumerate(RESOLUTIONS):
    full = draw_trial(examples[0], r)
    axs[i, 0].imshow(full, interpolation="nearest")
    axs[i, 0].set_title("%s, %d x %d px: %s" % (r, full.width, full.height, labels[0]), fontsize=9)
    ppd = pixels_per_degree(r)
    tx = full.width / 2 + 5 * ppd
    zoom = full.crop((round(tx - 1.2 * ppd), round(full.height / 2 - 0.4 * ppd), round(tx + 1.2 * ppd), round(full.height / 2 + 0.4 * ppd)))
    cross = full.crop((round(full.width / 2 - 0.4 * ppd), round(full.height / 2 - 0.4 * ppd), round(full.width / 2 + 0.4 * ppd), round(full.height / 2 + 0.4 * ppd)))
    both = Image.new("RGB", (cross.width + zoom.width + 4, zoom.height), "lightgrey")
    both.paste(cross, (0, 0)); both.paste(zoom, (cross.width + 4, 0))
    axs[i, 1].imshow(both, interpolation="nearest")
    axs[i, 1].set_title("zoom: cross | target", fontsize=9)
  for ax in axs.flatten():
    ax.set_xticks([]); ax.set_yticks([])
  fig.tight_layout()
  fig.savefig(HERE.parent / "data" / "stimuli_examples.png", dpi=150)

  fig, axs = plt.subplots(2, 1, figsize=(14, 2.4))
  for ax, ex, lab in zip(axs, examples[1:], labels[1:]):
    ax.imshow(draw_trial(ex, "150dpi"), interpolation="nearest")
    ax.set_title("150dpi: " + lab, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
  fig.tight_layout()
  fig.savefig(HERE.parent / "data" / "stimuli_examples_more.png", dpi=150)


if __name__ == "__main__":
  main()
