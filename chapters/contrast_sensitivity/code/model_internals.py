"""What Qwen3-VL-2B does with a faint Gabor: the real numbers behind the tutorial's videos.

1. Following the stripes through the model. A 4 c/deg Gabor at 0.1% contrast on the left (a
   contrast where the model's answer is usually wrong), and the same display without it. At
   every layer, how much each place's numbers change because of the stripes: the length of
   the difference, divided by the typical length of that layer's numbers.
     vision layers 1-24   one value per 16 x 16 square (16 x 32 grid)
     merger               one value per image token (8 x 16 grid)
     language layers 1-28 one value per image token, and one for the place where the answer
                          is written
   and, for one square in the middle of the Gabor and one in the half with no Gabor, its 1,024
   numbers at every vision layer in both runs
2. The answer as the stripes get stronger. The model's score for "left" over "right" (log-odds
   of its first word) for the Gabor on the left and on the right, at contrasts from 0.01% to
   10%, and for the plain grey display.

Runs on a laptop CPU (a few minutes). Saves ../data/model_internals.npz.

Run: python model_internals.py
"""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import stimuli

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QUESTION = "Where is the striped pattern: left or right? Answer with one word."
FREQUENCY, CONTRAST, PHASE = 4, 0.001, 0.0
SERIES = [round(float(c), 6) for c in np.logspace(-4, -1, 13)]


def main():
  proc = AutoProcessor.from_pretrained(MODEL)
  model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
  vision = model.model.visual
  left_ids = [proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in ["left", "Left"]]
  right_ids = [proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in ["right", "Right"]]

  blank = np.full((stimuli.CANVAS_PX[1], stimuli.CANVAS_PX[0]), float(stimuli.MEAN_GREY))
  messages = [{"role": "user", "content": [{"type": "image", "image": stimuli.to_image(blank)},
                                           {"type": "text", "text": QUESTION}]}]
  base = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
  image_pos = (base["input_ids"][0] == model.config.image_token_id).nonzero()[:, 0].numpy()

  seen = {}
  def keep(name):
    def hook(module, args, out):
      seen[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
  for i, b in enumerate(vision.blocks):
    b.register_forward_hook(keep(i))
  vision.merger.register_forward_hook(keep("merger"))

  def run(grey):
    inputs = dict(base)
    inputs["pixel_values"] = torch.from_numpy(stimuli.pixel_values(grey))
    with torch.no_grad():
      out = model(**inputs, output_hidden_states=True, logits_to_keep=1)
    logp = torch.log_softmax(out.logits[0, -1], 0)
    score = float(torch.logsumexp(logp[left_ids], 0) - torch.logsumexp(logp[right_ids], 0))
    vis = [seen[i].clone() for i in range(len(vision.blocks))]
    return vis, seen["merger"].clone(), [h[0].clone() for h in out.hidden_states[1:]], score

  # 1. the faint Gabor against the plain display
  gabor = stimuli.draw_trial(dict(frequency=FREQUENCY, contrast=CONTRAST, phase=PHASE), "left")
  v0, m0, l0, s0 = run(blank)
  v1, m1, l1, s1 = run(gabor)

  # squares come in 2 x 2 blocks; put them back on the 16 x 32 grid
  rows, cols = stimuli.CANVAS_PX[1] // stimuli.PATCH, stimuli.CANVAS_PX[0] // stimuli.PATCH
  order = [(bh * 2 + mh, bw * 2 + mw) for bh in range(rows // 2) for bw in range(cols // 2) for mh in range(2) for mw in range(2)]
  def to_grid(values, shape, places):
    grid = np.zeros(shape)
    for v, (r, c) in zip(values, places):
      grid[r, c] = v
    return grid
  token_places = [(r, c) for r in range(rows // 2) for c in range(cols // 2)]
  def change(a, b):
    return ((b - a).norm(dim=-1) / a.norm(dim=-1).mean()).numpy()

  # one square in the middle of the Gabor and one in the half with no Gabor: their numbers in both runs
  ours, other = order.index((rows // 2, cols // 4)), order.index((rows // 2, 3 * cols // 4))
  square_blank = np.stack([v[ours].numpy() for v in v0])
  square_gabor = np.stack([v[ours].numpy() for v in v1])
  other_blank = np.stack([v[other].numpy() for v in v0])
  other_gabor = np.stack([v[other].numpy() for v in v1])

  vision_maps = np.stack([to_grid(change(a, b), (rows, cols), order) for a, b in zip(v0, v1)])
  merger_map = to_grid(change(m0, m1), (rows // 2, cols // 2), token_places)
  language_maps = np.stack([to_grid(change(a[image_pos], b[image_pos]), (rows // 2, cols // 2), token_places)
                            for a, b in zip(l0, l1)])
  answer_change = np.array([change(a[-1:], b[-1:])[0] for a, b in zip(l0, l1)])

  # 2. the answer as the stripes get stronger
  scores = {"left": [], "right": []}
  for c in SERIES:
    for side in ("left", "right"):
      scores[side].append(run(stimuli.draw_trial(dict(frequency=FREQUENCY, contrast=c, phase=PHASE), side))[3])
    print("contrast %.4f%%: left %.3f, right %.3f" % (100 * c, scores["left"][-1], scores["right"][-1]))

  np.savez(HERE.parent / "data" / "model_internals.npz", vision=vision_maps, merger=merger_map,
           language=language_maps, answer=answer_change, image=gabor, series=np.array(SERIES),
           score_left=np.array(scores["left"]), score_right=np.array(scores["right"]), score_blank=s0,
           score_gabor=s1, square=np.array([rows // 2, cols // 4]), other=np.array([rows // 2, 3 * cols // 4]),
           square_blank=square_blank, square_gabor=square_gabor, other_blank=other_blank, other_gabor=other_gabor)
  print("grey display: %.3f; faint Gabor on the left: %.3f" % (s0, s1))
  print("vision layer 1: biggest change on the left half %.2e, right half %.2e"
        % (vision_maps[0][:, :cols // 2].max(), vision_maps[0][:, cols // 2:].max()))
  print("answer position, last layer: %.2e" % answer_change[-1])


if __name__ == "__main__":
  main()
