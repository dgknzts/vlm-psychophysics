"""The model as an observer: one question about one display, answered after thinking.

The model is Qwen3-VL-2B-Thinking, the thinking version of the model in the other chapters. Its
reply has two parts: its thinking, written out as text and closed by a </think> token, then its
answer.
  question     "Is there a red bar in the image? Answer yes or no."  (feature search)
               "Is there a 2 in the image? Answer yes or no."        (2 among 5s)
  RT           the number of tokens it writes before </think>: the model's reaction time
  answer       "yes" or "no", the first of the two words after </think>
A reply that has not closed its thinking after MAX_TOKENS tokens is stopped and counted as no
answer, like a trial that runs out of time.

The reply is sampled with the settings Qwen recommends for this model (temperature 1, top-p 0.95,
top-k 20), so the same display can get a different RT each time, as with a person.

Used by pilot.ipynb and run_model.ipynb (Colab GPU).
"""
import re

import torch

import stimuli

MODEL = "Qwen/Qwen3-VL-2B-Thinking"
QUESTIONS = {"feature": "Is there a red bar in the image? Answer yes or no.",
             "2v5": "Is there a 2 in the image? Answer yes or no."}
MAX_TOKENS = 4096
SAMPLING = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20)


def parse(text):
  """The answer in a reply after </think>: the first "yes" or "no", or None."""
  m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
  return m.group(1).lower() if m else None


class Observer:
  def __init__(self, device="cpu", dtype=torch.float32):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    self.device = device
    self.proc = AutoProcessor.from_pretrained(MODEL)
    # on a GPU, load the weights straight onto it (Colab's RAM can run out otherwise)
    extra = dict(device_map=device) if device != "cpu" else {}
    self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=dtype, **extra).eval()
    tok = self.proc.tokenizer
    self.end_think = tok.convert_tokens_to_ids("</think>")
    self.stop = set(self.model.generation_config.eos_token_id) | {tok.pad_token_id}

  def chat(self, task, image=None):
    content = [{"type": "image", "image": image} if image is not None else {"type": "image"},
               {"type": "text", "text": QUESTIONS[task]}]
    return [{"role": "user", "content": content}]

  def ask(self, task, images, seed=None):
    """One task, a batch of displays (8-bit RGB arrays). One dict per display.
    The displays all have the same size and the question is the same, so the prompts in a batch have
    the same length and need no padding."""
    if seed is not None:
      torch.manual_seed(seed)
    chats = [self.chat(task, stimuli.to_image(im)) for im in images]
    inputs = self.proc.apply_chat_template(chats, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(self.device)
    with torch.no_grad():
      out = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, **SAMPLING)
    new = out[:, inputs["input_ids"].shape[1]:].cpu().tolist()
    rows = []
    for ids in new:
      while ids and ids[-1] in self.stop:            # padding after a reply that ended early
        ids = ids[:-1]
      if self.end_think in ids:
        k = ids.index(self.end_think)
        thinking, reply = ids[:k], ids[k + 1:]
      else:
        thinking, reply = ids, []
      reply_text = self.proc.tokenizer.decode(reply, skip_special_tokens=True).strip()
      answer = parse(reply_text) if self.end_think in ids else None
      rows.append(dict(rt_tokens=len(thinking), finished=int(self.end_think in ids),
                       answer=answer, reply=reply_text, reply_tokens=len(reply),
                       thinking=self.proc.tokenizer.decode(thinking, skip_special_tokens=True).strip()))
    return rows
