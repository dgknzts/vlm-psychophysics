"""A small run on a laptop CPU, to choose the dot differences for the main run (run_model.ipynb).

1. Checks: the chat as the model sees it; a batch gives the same numbers as one display at a time.
2. The "left" habit: two empty boxes, and two boxes with the same number of dots.
3. 4 pairs (8 displays) at each of a wide range of differences, both questions.

Saves ../data/pilot.csv.  Run: python pilot.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import observer
import stimuli

HERE = Path(__file__).parent
DIFFERENCES = [5, 10, 20, 40, 70, 120, 200]
PAIRS = 4


def main():
  obs = observer.Observer()
  chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": observer.QUESTION}]},
          {"role": "assistant", "content": [{"type": "text", "text": "Left"}]},
          {"role": "user", "content": [{"type": "text", "text": observer.CONFIDENCE}]}]
  print(obs.proc.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

  trials = stimuli.make_trials(DIFFERENCES, PAIRS, seed=1)
  greys = [stimuli.draw_trial(t) for t in trials[:2]]
  together = obs.ask(greys)
  alone = [obs.ask([g])[0] for g in greys]
  print("batch vs one at a time, largest difference:",
        max(abs(a[k] - b[k]) for a, b in zip(together, alone) for k in ("hidden", "p1", "p5", "p_left")))

  empty = np.full((stimuli.CANVAS_PX[1], stimuli.CANVAS_PX[0]), stimuli.GREY, np.uint8)
  empty_boxes = stimuli.draw_trial(dict(difference=0, fuller="left", seed=0)).copy()
  y0 = (stimuli.CANVAS_PX[1] - stimuli.BOX_PX) // 2
  for half in range(2):
    x0 = half * stimuli.CANVAS_PX[0] // 2 + (stimuli.CANVAS_PX[0] // 2 - stimuli.BOX_PX) // 2
    empty_boxes[y0:y0 + stimuli.BOX_PX, x0:x0 + stimuli.BOX_PX] = stimuli.BLACK
  equal = [stimuli.draw_trial(dict(difference=0, fuller="left", seed=s)) for s in range(8)]
  for name, rows in [("grey image", obs.ask([empty])), ("two empty boxes", obs.ask([empty_boxes])),
                     ("313 vs 313 dots (8 displays)", obs.ask(equal))]:
    d = pd.DataFrame(rows)
    print("%-30s P(left) of left+right: %.2f | answer left: %d of %d | stated: %s"
          % (name, (d.p_left / (d.p_left + d.p_right)).mean(), (d.answer == "left").sum(), len(d),
             d.stated.tolist()))

  rows = []
  for b in range(0, len(trials), 4):
    batch = trials[b:b + 4]
    for t, r in zip(batch, obs.ask([stimuli.draw_trial(t) for t in batch])):
      rows.append({**t, **r, "correct": int(r["answer"] == t["fuller"])})
    print("%d / %d" % (len(rows), len(trials)), flush=True)
  df = pd.DataFrame(rows)
  df.to_csv(HERE.parent / "data" / "pilot.csv", index=False)

  print(df.groupby("difference").agg(correct=("correct", "mean"), hidden=("hidden", "mean"),
                                     stated=("stated", "mean"), said_left=("answer", lambda a: (a == "left").mean()),
                                     words=("p_left", lambda s: (s + df.loc[s.index, "p_right"]).mean()),
                                     digits=("p_digits", "mean")).round(2).to_string())
  print("stated confidence used:", df.stated.value_counts().sort_index().to_dict())


if __name__ == "__main__":
  main()
