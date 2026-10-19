"""Thresholds from proportion correct: a Weibull psychometric function for a two-choice task.

  p(c) = 1 - 0.5 * exp(-(c / alpha) ** beta)

starts at 0.5 (guessing) for c = 0 and rises to 1. Fitted by maximum likelihood to the
number of correct trials at each contrast. The threshold is the contrast where p reaches
the cutoff (0.75 by default):  c = alpha * (-ln(2 * (1 - cutoff))) ** (1 / beta).
"""
import numpy as np
from scipy.optimize import minimize


def weibull(c, alpha, beta):
  return 1 - 0.5 * np.exp(-(np.asarray(c, float) / alpha) ** beta)


def fit(contrasts, n_correct, n_trials):
  """Maximum-likelihood alpha, beta for one frequency."""
  c = np.asarray(contrasts, float)
  k = np.asarray(n_correct, float)
  n = np.asarray(n_trials, float)

  def cost(params):
    alpha, beta = np.exp(params)
    p = np.clip(weibull(c, alpha, beta), 1e-6, 1 - 1e-6)
    return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))

  # start from the contrast nearest 75% correct
  start = c[np.argmin(np.abs(k / n - 0.75))]
  best = min((minimize(cost, [np.log(start), np.log(b)], method="Nelder-Mead") for b in (1, 2, 4)),
             key=lambda r: r.fun)
  return tuple(np.exp(best.x))


def threshold(alpha, beta, cutoff=0.75):
  return alpha * (-np.log(2 * (1 - cutoff))) ** (1 / beta)
