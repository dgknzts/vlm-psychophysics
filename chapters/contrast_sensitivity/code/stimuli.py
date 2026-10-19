"""The Gabor displays shown to the model.

The Gabors are ModelFest stimuli 1-10 (Watson & Ahumada, 2005, J. Vision 5(9):6, Table 1):
"Gabor, fixed size", a horizontal-bar grating (modulated vertically) in cosine phase under a
round Gaussian window with a standard deviation of 0.5 deg, at 1.12 to 30 cycles/deg. Each
ModelFest image was 256 x 256 pixels at 120 pixels per degree (2.133 deg), on a grey equal to
the mean luminance:  grey level = 128 + 127 * contrast * gabor(x, y).

Humans saw the Gabor in one of two intervals and said which one (two-interval forced choice).
The model sees one image with two ModelFest fields side by side, the Gabor in one of them, and
is asked "left or right?". It leans towards "left", so each trial is a pair: the same Gabor once
on the left and once on the right. The trial is correct if the model says "left" more strongly
when the Gabor is on the left. Like the humans' task, it compares two displays.
  10 frequencies x 11 contrasts x 100 trials = 11,000 trials (22,000 images)
  each trial: a random phase of the stripes

A laptop pilot (4 pairs per point) found the model right at every frequency from 1% contrast
up, hence contrasts from 0.03% to 10%. Contrast below 0.8% is less than one grey level, so an 8-bit image cannot show it. The model gets
the exact values instead (pixel_values below), as on a lab display with finer grey steps.

Run: python stimuli.py     (prints the numbers, saves ../img/vlm_stimuli.png)
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# ---- ModelFest ----
PX_PER_DEG = 120
FIELD_PX = 256                                                   # 2.133 deg
SIGMA_DEG = 0.5
FREQUENCIES = [1.12, 2, 2.83, 4, 5.66, 8, 11.3, 16, 22.6, 30]    # cycles/deg, stimuli 1-10
MEAN_GREY = 128

# ---- the experiment ----
CONTRASTS = [round(float(c), 5) for c in np.logspace(-3.5, -1, 11)]    # 0.03% to 10%, 1/4 log unit apart
TRIALS = 100
CANVAS_PX = (2 * FIELD_PX, FIELD_PX)                             # 512 x 256: two fields side by side
PATCH, MERGE = 16, 2                                             # Qwen3-VL: 16-px squares, 2 x 2 per token


def make_trials(seed=0):
  """Every frequency x contrast, TRIALS times, each with a random phase."""
  rng = np.random.default_rng(seed)
  trials = []
  for f in FREQUENCIES:
    for c in CONTRASTS:
      for _ in range(TRIALS):
        trials.append(dict(index=len(trials), frequency=f, contrast=c,
                           phase=round(float(rng.uniform(0, 2 * np.pi)), 4)))
  return trials


def gabor(frequency, phase=0.0):
  """One ModelFest field, -1 to 1 around 0: horizontal bars under a Gaussian window."""
  x = (np.arange(FIELD_PX) - FIELD_PX / 2 + 0.5) / PX_PER_DEG
  X, Y = np.meshgrid(x, x)
  window = np.exp(-(X ** 2 + Y ** 2) / (2 * SIGMA_DEG ** 2))
  return window * np.cos(2 * np.pi * frequency * Y + phase)


def draw_trial(trial, side):
  """The display as exact grey levels (floats, 0-255): the Gabor on one side ("left" or
  "right"), plain grey on the other."""
  img = np.full((CANVAS_PX[1], CANVAS_PX[0]), float(MEAN_GREY))
  x0 = 0 if side == "left" else FIELD_PX
  img[:, x0:x0 + FIELD_PX] += 127 * trial["contrast"] * gabor(trial["frequency"], trial["phase"])
  return img


def to_image(grey):
  """The display rounded to 8 bits, as an ordinary image (for looking at, and for the processor)."""
  from PIL import Image
  return Image.fromarray(np.clip(np.round(grey), 0, 255).astype(np.uint8)).convert("RGB")


def pixel_values(grey):
  """What Qwen3-VL's processor makes of an image, but from the exact grey levels: each 16 x 16
  square becomes 3 colours x 2 frames x 256 numbers, rescaled from 0..255 to -1..1, and the
  squares come in 2 x 2 blocks, one block after another."""
  v = grey / 255 * 2 - 1
  h, w = v.shape[0] // PATCH, v.shape[1] // PATCH
  rows = []
  for bh in range(h // MERGE):
    for bw in range(w // MERGE):
      for mh in range(MERGE):
        for mw in range(MERGE):
          r, c = bh * MERGE + mh, bw * MERGE + mw
          square = v[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
          rows.append(np.broadcast_to(square, (3, 2, PATCH, PATCH)).reshape(-1))
  return np.stack(rows).astype(np.float32)


def main():
  print("%d px/deg; field %d px = %.3f deg; Gaussian sd %.1f deg = %d px"
        % (PX_PER_DEG, FIELD_PX, FIELD_PX / PX_PER_DEG, SIGMA_DEG, SIGMA_DEG * PX_PER_DEG))
  print("pixels per cycle:", [round(PX_PER_DEG / f, 1) for f in FREQUENCIES])
  print("contrasts (%):", [round(100 * c, 2) for c in CONTRASTS])
  print("one grey level is a contrast of %.2f%%" % (100 / 127))
  print("trials:", len(make_trials()))

  import figures
  figures.stimuli_figure(HERE.parent / "img" / "vlm_stimuli.png")


if __name__ == "__main__":
  main()
