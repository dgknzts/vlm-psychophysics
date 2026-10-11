"""What Qwen3-VL-2B really computes for one of our images: the numbers behind Figure 6.

The image is the example of the chapter: target k at 10 deg, flankers r and m one slot away.
Runs on a laptop CPU (about a minute). Saves ../data/model_internals.npz and
../data/model_internals.json, which figures.py draws from.

What is saved, for the square that holds the top of the k ("our square") and its token:
  pixel_values   the numbers the model gets for our square (3 colours x 2 frames x 16 x 16 = 1,536)
  embed          our square after the first learned layer (1,024 numbers)
  layers         our square after each of the 24 layers (24 x 1,024)
  attention      how much our square takes from every square, at each layer (24 x 4 x 64),
                 averaged over the 16 attention heads; a row of the softmax inside attention
  token          our token after the merger (2,048 numbers)
  tokens         all 128 image tokens (128 x 2,048)
  final          every square after the 24 layers (512 x 1,024), in the model's order
  sequence       the whole input as the language model gets it: 175 tokens x 2,048 numbers
                 (text tokens from the word table, image tokens from the vision part)
  order          (row, col) of each square in that order
and in the .json: the full input as text tokens, the image tokens' place in it, and the
model's answer for this image.

Run: python model_internals.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import stimuli

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QUESTION = ("There is either one letter or a row of three letters. If there are three, name the middle one. "
            "If there is one, name it. Answer with a single lowercase letter.")
TRIAL = dict(target="k", ecc_deg=10, spacing=1, inner_flanker="r", outer_flanker="m")
SQUARE = (816, 48)     # top-left pixel of our square (same as figures.STEP_SQUARE)
PATCH, MERGE = 16, 2


def patch_order(grid_h, grid_w):
  """(row, col) of each patch in the order the model gets them: 2 x 2 blocks, one after another."""
  order = []
  for bh in range(grid_h // MERGE):
    for bw in range(grid_w // MERGE):
      for mh in range(MERGE):
        for mw in range(MERGE):
          order.append((bh * MERGE + mh, bw * MERGE + mw))
  return order


def main():
  torch.manual_seed(0)
  proc = AutoProcessor.from_pretrained(MODEL)
  model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
  vision = model.model.visual

  img = stimuli.draw_trial(TRIAL)
  messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": QUESTION}]}]
  inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
  _, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
  order = patch_order(grid_h, grid_w)
  ours = order.index((SQUARE[1] // PATCH, SQUARE[0] // PATCH))

  # check the patch order: our square's pixel values must match the image
  pv = inputs["pixel_values"][ours].reshape(3, 2, PATCH, PATCH).numpy()
  gray = np.asarray(img.convert("L"), dtype=float)[SQUARE[1]:SQUARE[1] + PATCH, SQUARE[0]:SQUARE[0] + PATCH]
  print("patch order check, largest difference:", np.abs((pv[0, 0] + 1) / 2 * 255 - gray).max().round(1))

  seen = {"layers": [], "attention": []}

  def before_attention(module, args, kwargs):
    # the input to attention (after the block's norm); recompute the softmax row of our square
    x = args[0] if args else kwargs["hidden_states"]
    n = x.shape[0]
    q, k, _ = module.qkv(x).reshape(n, 3, module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = kwargs["position_embeddings"]
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    scores = torch.einsum("hd,nhd->hn", q[ours], k) * module.scaling       # heads x squares
    seen["attention"].append(torch.softmax(scores, dim=-1).mean(0))

  hooks = [blk.attn.register_forward_pre_hook(before_attention, with_kwargs=True) for blk in vision.blocks]
  hooks += [blk.register_forward_hook(lambda m, a, out: seen["layers"].append(out[ours].clone())) for blk in vision.blocks]
  hooks.append(vision.patch_embed.register_forward_hook(lambda m, a, out: seen.__setitem__("embed", out[ours].clone())))

  seen_sequence = {}
  hooks.append(model.model.language_model.register_forward_pre_hook(
    lambda m, a, kw: seen_sequence.__setitem__("x", kw["inputs_embeds"][0].clone()), with_kwargs=True))
  with torch.no_grad():
    out = model(**inputs, logits_to_keep=1)
    image_out = vision(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
  for h in hooks:
    h.remove()

  # the hooks ran twice (full model, then vision alone); keep the first run
  layers = torch.stack(seen["layers"][:len(vision.blocks)]).numpy()
  attention = torch.stack(seen["attention"][:len(vision.blocks)]).numpy()
  # attention back on the image grid
  attention_grid = np.zeros((len(vision.blocks), grid_h, grid_w))
  for i, (r, c) in enumerate(order):
    attention_grid[:, r, c] = attention[:, i]
  tokens = image_out.pooler_output.numpy()
  token_index = ours // (MERGE * MERGE)

  # the answer
  letters = stimuli.LETTERS
  ids = [proc.tokenizer.encode(" " + L, add_special_tokens=False)[0] for L in letters]
  probs = torch.softmax(out.logits[0, -1, ids], dim=0).numpy()
  answer = letters[int(np.argmax(probs))]

  # the input as the language model gets it
  input_ids = inputs["input_ids"][0].tolist()
  pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
  pieces = [proc.tokenizer.decode([i]) for i in input_ids]
  info = dict(trial=TRIAL, question=QUESTION, grid=[grid_h, grid_w], our_square_index=ours, our_token_index=token_index,
              n_input=len(input_ids), n_image_tokens=input_ids.count(pad),
              text_before=pieces[:input_ids.index(pad)],
              text_after=pieces[len(input_ids) - input_ids[::-1].index(pad):],
              answer=answer, p_answer=float(probs.max()), p_target=float(probs[letters.index(TRIAL["target"])]))
  np.savez_compressed(HERE.parent / "data" / "model_internals.npz",
                      pixel_values=inputs["pixel_values"][ours].numpy(), embed=seen["embed"].numpy(), layers=layers,
                      attention=attention_grid, token=tokens[token_index], tokens=tokens,
                      final=image_out.last_hidden_state.numpy(), order=np.array(order),
                      sequence=seen_sequence["x"].numpy())
  json.dump(info, open(HERE.parent / "data" / "model_internals.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
  print("grid %d x %d squares, %d image tokens, %d input tokens in all" % (grid_h, grid_w, info["n_image_tokens"], info["n_input"]))
  print("answer:", answer, "p = %.2f" % info["p_answer"], "(target k: p = %.2f)" % info["p_target"])
  print("our square takes this much from itself, by layer:", attention_grid[:, SQUARE[1] // PATCH, SQUARE[0] // PATCH].round(2))


if __name__ == "__main__":
  main()
