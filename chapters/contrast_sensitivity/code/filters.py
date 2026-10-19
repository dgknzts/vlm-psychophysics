"""The first learned layer of Qwen3-VL-2B's vision part, as filters: the numbers behind Figure 7.

Each 16 x 16 square of the image becomes 1,024 numbers through one linear layer (the patch
embedding, crowding chapter Figure 6b). Its weights are 1,024 x 3 colours x 2 frames x 16 x 16.
A grey image is the same in all three colours and both frames, so for grey images each output
number is a plain weighted sum of the square's 256 pixels: one 16 x 16 filter per number.

For a grating of contrast c, the pixel values the model gets (-1..1) move by 254/255 * c * the
grating, so the 1,024 numbers move by c times a fixed amount for each frequency. We measure
that amount: the size (length) of the change in the 1,024 numbers for a grating of 100%
contrast, averaged over the grating's phase. Only the weights are needed, read from the model
file in the local Hugging Face cache; no model run.

Saves ../data/first_layer.npz:
  filters      1,024 x 16 x 16  the grey filters
  frequencies  cycles per pixel at which the response was measured
  response     size of the change in the 1,024 numbers, per frequency (horizontal bars)
  best         each filter's best frequency (cycles per pixel)

Run: python filters.py
"""
import glob
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PATCH = 16


def load_filters():
  import torch  # noqa: F401  (safetensors needs it for bfloat16)
  from safetensors import safe_open
  path = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/model.safetensors"))[0]
  with safe_open(path, "pt") as f:
    w = f.get_tensor("model.visual.patch_embed.proj.weight").double().numpy()   # 1024, 3, 2, 16, 16
  return w.sum(axis=(1, 2))                                                     # 1024, 16, 16


def responses(filters, frequencies):
  """For horizontal bars of 100% contrast at each frequency (cycles per pixel): per filter, the
  cosine and sine parts of its output, and the phase-averaged size of the whole change."""
  y = np.arange(PATCH) + 0.5
  scale = 254 / 255                                     # contrast -> change in the model's pixel values
  cos = np.array([(filters * np.cos(2 * np.pi * f * y)[None, :, None]).sum(axis=(1, 2)) for f in frequencies]) * scale
  sin = np.array([(filters * np.sin(2 * np.pi * f * y)[None, :, None]).sum(axis=(1, 2)) for f in frequencies]) * scale
  size = np.sqrt(((cos ** 2).sum(axis=1) + (sin ** 2).sum(axis=1)) / 2)          # averaged over phase
  per_filter = np.sqrt((cos ** 2 + sin ** 2) / 2)                                 # frequencies x 1024
  return size, per_filter


def main():
  filters = load_filters()
  frequencies = np.logspace(np.log10(1 / 256), np.log10(0.5), 80)   # cycles per pixel, up to 2 px per cycle
  size, per_filter = responses(filters, frequencies)
  best = frequencies[per_filter.argmax(axis=0)]
  np.savez(HERE.parent / "data" / "first_layer.npz", filters=filters, frequencies=frequencies,
           response=size, best=best)

  peak = frequencies[size.argmax()]
  print("1,024 filters of 16 x 16")
  print("the layer responds most at %.3f cycles per pixel (%.1f pixels per cycle)" % (peak, 1 / peak))
  print("  at 120 px/deg that is %.1f c/deg; at 32 px/deg, %.1f c/deg" % (peak * 120, peak * 32))
  low = size[np.argmin(np.abs(frequencies - 1.12 / 120))]
  print("response at 1.12 c/deg (120 px/deg) is %.2f of the peak" % (low / size.max()))
  print("filters whose best frequency is finer than 8 px per cycle: %.0f%%" % (100 * np.mean(best > 1 / 8)))


if __name__ == "__main__":
  main()
