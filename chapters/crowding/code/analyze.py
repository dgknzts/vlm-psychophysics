"""Results of run_model.ipynb: a printed summary and the results figure for the chapter.

Reads ../data/model_results.csv (one row per trial) and Bouma's data,
../data/bouma1970_from_coates2021.csv. Fits the same psychometric function as for Bouma's
data (fit_bouma.py) to get a critical spacing for the model.

Run: python analyze.py     (saves ../img/vlm_results.png)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import figures
import fit_bouma
import stimuli

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
IMG_DIR = HERE.parent / "img"


def main():
  model = pd.read_csv(DATA_DIR / "model_results.csv")
  bouma = pd.read_csv(DATA_DIR / "bouma1970_from_coates2021.csv")
  print("trials:", len(model))
  print(model.groupby(["ecc_deg", model.spacing.fillna(0)]).size().unstack().to_string())

  # proportion correct
  table = model.pivot_table(index="ecc_deg", columns=model.spacing.fillna(0).astype(int), values="correct")
  table.columns = ["%.2f deg" % stimuli.spacing_deg(c) if c else "alone" for c in table.columns]
  print("\nproportion correct (columns: centre-to-centre spacing)")
  print(table.round(2).to_string())

  # what the wrong answers were
  flanked = model[model.spacing.notna()].copy()
  wrong = flanked[~flanked.correct]
  print("\nwrong answers with flankers: %d of %d" % (len(wrong), len(flanked)))
  for ecc, w in wrong.groupby("ecc_deg"):
    inner = (w.answer == w.inner_flanker).sum()
    outer = (w.answer == w.outer_flanker).sum()
    print("  %2d deg: %d wrong; named a flanker %d (%.0f%%): inner/left %d, outer/right %d"
          % (ecc, len(w), inner + outer, 100 * (inner + outer) / max(len(w), 1), inner, outer))
  print("  (at 0 deg 'inner' is just the left flanker; at 10 deg it is the one nearer the centre)")
  print("  per spacing:", wrong.groupby(["ecc_deg", "spacing"]).size().to_dict())

  common = wrong.groupby(["target", "answer"]).size().sort_values(ascending=False).head(8)
  print("\nmost common mistakes (target -> answer):", ", ".join("%s->%s %d" % (t, a, n) for (t, a), n in common.items()))
  alone = model[model.spacing.isna()]
  print("letter alone, wrong:", ", ".join("%s->%s" % (t, a) for t, a in zip(alone[~alone.correct].target, alone[~alone.correct].answer)))

  # the same psychometric function as for Bouma's data
  print("\nlogistic fit (as fit_bouma.py): top A, critical spacing at 0.6 / 0.8 / 0.9 of the way up")
  for ecc, d in model.groupby("ecc_deg"):
    spacing_deg = d.spacing.map(lambda s: stimuli.spacing_deg(s) if pd.notna(s) else np.inf)
    counts = d.assign(x=np.log(spacing_deg)).groupby("x").correct.sum()
    r = fit_bouma.fit(counts.index.to_numpy(), counts.to_numpy())
    A, m, w = r.x
    cs = [fit_bouma.critical_spacing(m, w, c) for c in [0.6, 0.8, 0.9]]
    print("  %2d deg: A = %.2f; critical spacing %.2f / %.2f / %.2f deg (= %.1f / %.1f / %.1f slots)"
          % (ecc, A, *cs, *[c / stimuli.SLOT_DEG for c in cs]))

  figures.results_figure(IMG_DIR / "vlm_results.png", model, bouma)
  print("\nsaved", IMG_DIR / "vlm_results.png")


if __name__ == "__main__":
  main()
