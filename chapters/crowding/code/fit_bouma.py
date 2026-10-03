"""Fit crowding functions to Bouma (1970), as re-analysed by Coates et al. (2021).

For each eccentricity, proportion correct p against centre-to-centre spacing s (deg):
    p(s) = g + (A - g) / (1 + exp(-(ln s - m) / w)),   g = 1/25 (guessing among 25 letters)
A is the top of the curve (the unflanked level; the unflanked point counts as s = infinity).
Fit by maximum likelihood with binomial errors, 100 trials per point (Bouma 1970).

Critical spacing at cutoff k is where the curve has risen k of the way from g to A:
    s_k = exp(m + w * ln(k / (1 - k)))
(Coates et al. 2021, Fig. 2b-c). It depends only on m and w, not on A.

Left out: 7.5 and 10.5 deg, as in Coates Fig. 2 (7.5 has no unflanked point, 10.5 has
only the unflanked point), and 1 deg, which has just two points for three parameters, so
many curves fit it equally well.

Output (in ../data):
    bouma1970_fits.csv               A, m, w per eccentricity
    bouma1970_critical_spacing.csv   critical spacing per eccentricity and cutoff, with Coates' values
    bouma1970_fits_check.png         our curves and Bouma fractions next to Coates'

Run: python fit_bouma.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pymupdf
from scipy.optimize import minimize
from scipy.stats import binom

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
DATA = DATA_DIR / "bouma1970_from_coates2021.csv"
PDF = HERE.parent / "sources" / "coates2021.pdf"
GUESS = 1 / 25
N_TRIALS = 100
CUTOFFS = [0.6, 0.7, 0.8, 0.9]
LEFT_OUT = [1, 7.5, 10.5]


def curve(x, A, m, w):
    """Proportion correct at log spacing x (x = inf for unflanked)."""
    with np.errstate(over="ignore"):
        return np.where(np.isinf(x), A, GUESS + (A - GUESS) / (1 + np.exp(-(x - m) / w)))


def neg_log_lik(params, x, k):
    p = np.clip(curve(x, *params), 1e-9, 1 - 1e-9)
    return -binom.logpmf(k, N_TRIALS, p).sum()


def fit(x, k):
    """Maximum likelihood from several starting points; keeps the best."""
    bounds = [(GUESS + 0.01, 1.0), (np.log(0.01), np.log(50)), (0.05, 5.0)]
    best = None
    for m0 in np.log([0.2, 0.5, 1, 2, 4]):
        for w0 in [0.3, 0.7, 1.5]:
            r = minimize(neg_log_lik, [0.95, m0, w0], args=(x, k), bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
    return best


def critical_spacing(m, w, cutoff):
    return np.exp(m + w * np.log(cutoff / (1 - cutoff)))


def coates_fig2c():
    """Bouma fractions from Coates Fig. 2c (page 4), read from the vector markers."""
    page = pymupdf.open(PDF)[3]
    # axis ticks of panel c: x 1..7 deg, y 0..1
    to_ecc = np.polyfit([423.59, 505.67], [1, 7], 1)
    to_frac = np.polyfit([195.73, 76.21], [0, 1], 1)
    colours = {(0.0, 0.0, 0.02): 0.6, (0.32, 0.07, 0.49): 0.7, (0.72, 0.22, 0.47): 0.8, (0.99, 0.54, 0.38): 0.9}
    out = {}
    for d in page.get_drawings():
        r, f = d["rect"], d.get("fill")
        if f and len(d["items"]) == 8 and r.x0 > 420 and r.y1 < 190 and r.width < 1.5:
            cutoff = colours.get(tuple(round(v, 2) for v in f))
            if cutoff:
                ecc = round(float(np.polyval(to_ecc, (r.x0 + r.x1) / 2)), 2)
                out[(ecc, cutoff)] = round(float(np.polyval(to_frac, (r.y0 + r.y1) / 2)), 3)
    return out


def main():
    df = pd.read_csv(DATA)
    df = df[~df.ecc_deg.isin(LEFT_OUT)]
    coates = coates_fig2c()

    fits, spacings = [], []
    for ecc, d in df.groupby("ecc_deg"):
        x = np.log(d.cc_spacing_deg.fillna(np.inf).to_numpy())
        k = np.round(d.p_correct * N_TRIALS).to_numpy()
        r = fit(x, k)
        A, m, w = r.x
        fits.append(dict(ecc_deg=ecc, A=round(A, 4), m=round(m, 4), w=round(w, 4),
                         n_points=len(d), neg_log_lik=round(r.fun, 3)))
        for c in CUTOFFS:
            s = critical_spacing(m, w, c)
            spacings.append(dict(ecc_deg=ecc, cutoff=c, critical_spacing_deg=round(s, 3),
                                 bouma_fraction=round(s / ecc, 3),
                                 coates_bouma_fraction=coates.get((float(ecc), c))))
    fits = pd.DataFrame(fits)
    spacings = pd.DataFrame(spacings)
    fits.to_csv(DATA_DIR / "bouma1970_fits.csv", index=False)
    spacings.to_csv(DATA_DIR / "bouma1970_critical_spacing.csv", index=False)
    print(fits.to_string(index=False))
    print()
    print(spacings.pivot(index="ecc_deg", columns="cutoff", values=["bouma_fraction", "coates_bouma_fraction"]).to_string())

    # Check figure: left like Coates Fig. 2a, right like Fig. 2c (ours solid, Coates dashed).
    colours = {2: "tab:orange", 3: "tab:green", 5: "tab:red", 7: "tab:purple"}
    fig, (a, c) = plt.subplots(1, 2, figsize=(10, 4))
    inf_x = 30  # where unflanked points are drawn
    s = np.exp(np.linspace(np.log(0.05), np.log(15), 300))
    for _, f in fits.iterrows():
        col = colours[f.ecc_deg]
        a.plot(s, curve(np.log(s), f.A, f.m, f.w), color=col, label=f"{f.ecc_deg:g}°")
        d = df[df.ecc_deg == f.ecc_deg]
        a.plot(d.cc_spacing_deg.fillna(inf_x), d.p_correct, "o", color=col)
    a.set_xscale("log")
    a.set_xticks([0.1, 0.5, 1, 3, 10, inf_x], ["0.1", "0.5", "1", "3", "10", "∞"])
    a.set_ylim(0, 1.02)
    a.set_xlabel("Centre-to-centre spacing (deg)")
    a.set_ylabel("Proportion correct")
    a.legend(frameon=False)
    a.set_title("Fits (cf. Coates Fig. 2a)")
    shades = {0.6: "black", 0.7: "indigo", 0.8: "mediumvioletred", 0.9: "coral"}
    for cut, sub in spacings.groupby("cutoff"):
        c.plot(sub.ecc_deg, sub.bouma_fraction, "o-", color=shades[cut], label=f"{cut} ours")
        c.plot(sub.ecc_deg, sub.coates_bouma_fraction, "s--", color=shades[cut], alpha=0.5, mfc="none", label=f"{cut} Coates")
    c.set_ylim(0, 1)
    c.set_xlabel("Eccentricity (deg)")
    c.set_ylabel("Critical spacing / eccentricity")
    c.legend(frameon=False, ncol=2, fontsize=8)
    c.set_title("Bouma fraction (cf. Coates Fig. 2c)")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "bouma1970_fits_check.png", dpi=150)


if __name__ == "__main__":
    main()
