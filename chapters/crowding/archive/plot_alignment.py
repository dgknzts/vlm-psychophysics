"""Draw the token-alignment conditions (run_alignment.ipynb) with the model's 32-px grid on top,
and the proportion correct from ../data/model_results_alignment.csv in each title.

Run: python plot_alignment.py     (saves ../data/alignment_conditions.png)
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stimuli

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"

CASES = [("monitor", 0), ("monitor", 1), ("150dpi", 0)]
ALIGNS = [None, "split", "centred"]
NAMES = {None: "none (as in the first run)", "split": "split: boundary through the target",
         "centred": "centred: target in one token"}


def main():
  results = pd.read_csv(DATA_DIR / "model_results_alignment.csv")
  results["align"] = results["align"].fillna("None")   # pandas reads the text "None" as empty

  example = dict(target="k", ecc_deg=5, side="right", inner_flanker="r", outer_flanker="m")
  fig, axs = plt.subplots(len(CASES), len(ALIGNS), figsize=(13, 8.5))
  for i, (resolution, gap) in enumerate(CASES):
    ppd = stimuli.pixels_per_degree(resolution)
    for j, align in enumerate(ALIGNS):
      img = stimuli.draw_trial({**example, "gap": gap}, resolution, align=align)

      # a 6 x 3 token window around the target, on the token grid
      x0 = int(img.width / 2 + 5 * ppd) // 32 * 32 - 64
      y0 = img.height // 2 // 32 * 32 - 32
      ax = axs[i, j]
      ax.imshow(img.crop((x0, y0, x0 + 192, y0 + 96)), extent=(0, 192, 96, 0), interpolation="nearest")
      for g in range(0, 193, 32):
        ax.axvline(g, color="red", lw=1)
      for g in range(0, 97, 32):
        ax.axhline(g, color="red", lw=1)
      for g in range(16, 192, 32):
        ax.axvline(g, color="red", lw=0.4, ls=":")
      for g in range(16, 96, 32):
        ax.axhline(g, color="red", lw=0.4, ls=":")
      ax.set_xticks([])
      ax.set_yticks([])

      sub = results[(results.resolution == resolution) & (results.gap == gap) & (results["align"] == str(align))]
      ax.set_title("%s\ncorrect %.2f   (n = %d)" % (NAMES[align], sub.correct.mean(), len(sub)), fontsize=9)
      if j == 0:
        ax.set_ylabel("%s, gap %d\n(%.0f px/deg)" % (resolution, gap, ppd), fontsize=10)

  fig.suptitle("Where the letters fall on Qwen3-VL's grid. Red: 32-px tokens; dotted: 16-px patches.\n"
               "Example display: target k at 5 deg right, random flankers r and m. "
               "Correct = all eccentricities (2, 3, 5, 7 deg) together.", fontsize=10)
  fig.tight_layout()
  fig.savefig(DATA_DIR / "alignment_conditions.png", dpi=130)


if __name__ == "__main__":
  main()
