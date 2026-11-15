"""The model as an observer: two questions about one display, in the same chat.

Turn 1  "Which box has more dots: left or right? Answer with one word."
        The model's probabilities for its first word. Its answer is "left" or "right", whichever
        is more likely (both spellings, "left" and "Left", count). Its hidden confidence is the
        probability of that answer out of the two: P(answer) / (P(left) + P(right)), 0.5 to 1.
Turn 2  The answer is written into the chat as the model's reply, then:
        "How sure are you that your answer is correct? Answer with one number from 1 (guessing)
        to 5 (certain)."
        The model's probabilities for the digits 1 to 5. Its stated confidence is the most
        likely digit; the probabilities are kept too.

Used by pilot.py (laptop CPU) and run_model.ipynb (Colab GPU).
"""
import numpy as np
import torch

import stimuli

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QUESTION = "Which box has more dots: left or right? Answer with one word."
CONFIDENCE = ("How sure are you that your answer is correct? "
              "Answer with one number from 1 (guessing) to 5 (certain).")
LEVELS = [1, 2, 3, 4, 5]


class Observer:
  def __init__(self, device="cpu", dtype=torch.float32):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    self.device = device
    self.proc = AutoProcessor.from_pretrained(MODEL)
    # on a GPU, load the weights straight onto it (Colab's RAM can run out otherwise)
    extra = dict(device_map=device) if device != "cpu" else {}
    self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=dtype, **extra).eval()
    tok = self.proc.tokenizer
    first = lambda w: tok.encode(w, add_special_tokens=False)[0]
    self.words = {"left": ["left", "Left"], "right": ["right", "Right"]}
    self.ids = {side: [first(w) for w in ws] for side, ws in self.words.items()}
    self.digits = [first(str(k)) for k in LEVELS]
    for ws in self.words.values():                  # each word must be one token
      assert all(len(tok.encode(w, add_special_tokens=False)) == 1 for w in ws)

  def _next_word(self, chats):
    """The model's probabilities for the next token, after each chat (a batch). All displays have
    the same size and both replies are one token, so the chats in a batch have the same length
    and need no padding."""
    inputs = self.proc.apply_chat_template(chats, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(self.device)
    with torch.no_grad():
      logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
    return torch.softmax(logits, dim=1).cpu()

  def ask(self, greys):
    """Both questions for a batch of displays (8-bit grey arrays). One dict per display."""
    images = [stimuli.to_image(g) for g in greys]
    first = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": QUESTION}]}]
             for im in images]
    p = self._next_word(first)
    rows, second = [], []
    for k in range(len(images)):
      p_left, p_right = float(p[k, self.ids["left"]].sum()), float(p[k, self.ids["right"]].sum())
      answer = "left" if p_left >= p_right else "right"
      # the reply written into the chat: the spelling the model itself prefers
      spelled = [(float(p[k, i]), w) for i, w in zip(self.ids[answer], self.words[answer])]
      reply = max(spelled)[1]
      rows.append(dict(p_left=p_left, p_right=p_right, answer=answer, reply=reply,
                       hidden=max(p_left, p_right) / (p_left + p_right)))
      second.append(first[k] + [{"role": "assistant", "content": [{"type": "text", "text": reply}]},
                                {"role": "user", "content": [{"type": "text", "text": CONFIDENCE}]}])
    q = self._next_word(second)
    for k, row in enumerate(rows):
      pc = q[k, self.digits].numpy().astype(float)
      row["p_digits"] = float(pc.sum())                   # how much of the next word is a digit 1-5
      for level, v in zip(LEVELS, pc / pc.sum()):
        row["p%d" % level] = float(v)
      row["stated"] = LEVELS[int(np.argmax(pc))]
    return rows
