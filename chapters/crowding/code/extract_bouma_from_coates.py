"""Extract Bouma (1970) data from Coates, Ludowici & Chung (2021), Figure 1.

Figure 1 (page 3 of coates2021.pdf) is matplotlib vector output, so every marker is a
path in the PDF. Panel a gives eccentricity (x) and proportion correct (y), with the
marker shape coding flanker spacing. Panel b replots the same points against
centre-to-centre spacing, with colour coding eccentricity; we use it as a cross-check.

Output (in ../data):
    bouma1970_from_coates2021.csv      one row per marker in panel a
    bouma1970_extraction_check.png     extracted points drawn over the original figure

Run: python extract_bouma_from_coates.py
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
PDF = HERE.parent / "sources" / "coates2021.pdf"
PAGE = 2  # page 3
SLOT_DEG = 0.29
NOMINAL_ECC = [1, 2, 3, 5, 7, 7.5, 10.5]

# Plot areas (PDF points), read from the axes frames of the figure.
PANEL_A = pymupdf.Rect(136.3, 502.5, 279.3, 629.1)
PANEL_B = pymupdf.Rect(317.6, 502.5, 460.6, 629.1)
LEGEND_A_X0 = 246  # panel a legend box starts here; markers right of it are legend entries

# Tick labels are drawn as glyph outlines, not text, so their values are listed here
# in tick order (left to right, bottom to top), as printed in the figure.
A_XTICK_VALUES = [0, 2, 4, 6, 8, 10, 12]
A_YTICK_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
B_XTICK_VALUES = [1, 2, 3, 4, 5]  # the sixth tick is "infinity" (unflanked)
B_YTICK_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]


def ticks(drawings, panel, axis):
    """Positions of the short black tick marks along the x or y axis of a panel."""
    out = []
    for d in drawings:
        r, items = d["rect"], d["items"]
        if len(items) != 1 or items[0][0] != "l" or d.get("fill") != (0.0, 0.0, 0.0):
            continue
        if axis == "x" and r.width < 0.1 and abs(r.y0 - panel.y1) < 0.5 and panel.x0 - 1 < r.x0 < panel.x1 + 1:
            out.append(r.x0)
        if axis == "y" and r.height < 0.1 and abs(r.x1 - panel.x0) < 1 and panel.y0 - 1 < r.y0 < panel.y1 + 1:
            out.append(r.y0)
    return sorted(set(round(v, 2) for v in out))


def linmap(positions, values):
    slope, icpt = np.polyfit(positions, values, 1)
    resid = np.array(values) - (slope * np.array(positions) + icpt)
    return (lambda v: slope * v + icpt), float(np.abs(resid).max())


def points(path):
    pts = []
    for it in path["items"]:
        pts += [it[1], it[-1]] if it[0] in "lc" else []
    return pts


def classify(inner, outline):
    """Spacing label from a marker's inner path and its outline path."""
    ops_in = "".join(i[0] for i in inner["items"])
    ops_out = "".join(i[0] for i in outline["items"])
    white = inner.get("fill") == (1.0, 1.0, 1.0)
    if ops_in == "c" * 8:
        return "unflanked" if white else "0"
    if ops_in == "re":
        return "8"
    if ops_in == "ll":
        return "1" if ops_out.startswith("re") else "5"
    if ops_in == "lll":
        if ops_out.startswith("re"):
            return "11"  # tri_down marker
        ys = [p.y for p in points(inner)]
        top = sum(abs(y - min(ys)) < 0.3 for y in ys)
        bottom = sum(abs(y - max(ys)) < 0.3 for y in ys)
        return "2" if top < bottom else "15"  # apex up = 2x, apex down = 15x
    return None


def centre(inner, outline, label):
    if label == "11":  # tri_down: centre is the vertex shared by its three arms
        p, _ = Counter((round(q.x, 2), round(q.y, 2)) for q in points(inner)).most_common(1)[0]
        return p
    r = outline["rect"]
    return ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)


def markers(drawings, panel):
    """Inner path followed by its outline path, both marker-sized, inside the panel."""
    out = []
    small = lambda d: d["rect"].width < 5 and d["rect"].height < 5 and d["rect"].intersects(panel)
    for a, b in zip(drawings, drawings[1:]):
        if not (small(a) and small(b)):
            continue
        if abs(a["rect"].x0 - b["rect"].x0) > 0.6 or abs(a["rect"].y0 - b["rect"].y0) > 0.6:
            continue
        label = classify(a, b)
        if label is None:
            continue
        x, y = centre(a, b, label)
        out.append(dict(label=label, x=x, y=y, colour=b.get("fill")))
    return out


def main():
    page = pymupdf.open(PDF)[PAGE]
    drawings = page.get_drawings()

    ax_x, ex = linmap(ticks(drawings, PANEL_A, "x"), A_XTICK_VALUES)
    ax_y, ey = linmap(ticks(drawings, PANEL_A, "y")[::-1], A_YTICK_VALUES)
    bx_ticks = ticks(drawings, PANEL_B, "x")
    bx, ebx = linmap(bx_ticks[:5], B_XTICK_VALUES)
    by, eby = linmap(ticks(drawings, PANEL_B, "y")[::-1], B_YTICK_VALUES)
    print(f"axis fit max residuals: a.x {ex:.4f}  a.y {ey:.5f}  b.x {ebx:.4f}  b.y {eby:.5f}")
    b_inf_x = bx_ticks[5]

    # Colour -> eccentricity, from the small coloured bars on panel a's x axis.
    colour_ecc = {}
    for d in drawings:
        r, f = d["rect"], d.get("fill")
        if f and len(set(f)) > 1 and r.y1 > PANEL_A.y1 - 1 and r.height > 5 and PANEL_A.x0 < r.x0 < PANEL_A.x1:
            colour_ecc[tuple(round(c, 3) for c in f)] = round(float(ax_x((r.x0 + r.x1) / 2)), 2)
    print("colour -> ecc:", sorted(colour_ecc.values()))

    # Panel a: the data.
    rows = []
    for m in markers(drawings, PANEL_A):
        if m["x"] > LEGEND_A_X0 and m["y"] < 570:  # legend box
            continue
        ecc_raw = float(ax_x(m["x"]))
        near = min(NOMINAL_ECC, key=lambda e: abs(e - ecc_raw))
        flag = "" if abs(near - ecc_raw) < 0.05 else "ecc off-grid"
        gap = None if m["label"] == "unflanked" else int(m["label"])
        rows.append(dict(
            ecc_deg=near, ecc_raw=round(ecc_raw, 3), gap_slots=gap,
            edge_spacing_deg=None if gap is None else round(gap * SLOT_DEG, 3),
            cc_spacing_deg=None if gap is None else round((gap + 1) * SLOT_DEG, 3),
            p_correct=round(float(ax_y(m["y"])), 3), n_trials=100,
            source="Coates2021 Fig1a", flag=flag, _px=m["x"], _py=m["y"],
        ))
    df = pd.DataFrame(rows).sort_values(["ecc_deg", "gap_slots", "p_correct"], na_position="last")

    # Panel b: cross-check every panel a point.
    bpts = []
    for m in markers(drawings, PANEL_B):
        ecc = colour_ecc.get(tuple(round(c, 3) for c in m["colour"]))
        bpts.append(dict(label=m["label"], ecc=ecc, x=m["x"], p=float(by(m["y"])), px=m["x"], py=m["y"]))
    used, problems = set(), []
    for i, r in df.iterrows():
        label = "unflanked" if pd.isna(r.gap_slots) else str(int(r.gap_slots))
        xb = b_inf_x if label == "unflanked" else None
        cands = [(j, b) for j, b in enumerate(bpts) if j not in used and b["label"] == label and b["ecc"] == r.ecc_deg]
        if not cands:
            problems.append(f"no panel b match: ecc {r.ecc_deg}, gap {label}, p {r.p_correct}")
            continue
        j, b = min(cands, key=lambda jb: abs(jb[1]["p"] - r.p_correct))
        used.add(j)
        xb_val = b_inf_x if xb else float(bx(b["x"]))
        dx = 0 if xb else abs(xb_val - r.cc_spacing_deg)
        dp = abs(b["p"] - r.p_correct)
        if dp > 0.01 or dx > 0.03:
            problems.append(f"mismatch: ecc {r.ecc_deg}, gap {label}: a p={r.p_correct} b p={b['p']:.3f}, cc dx={dx:.3f}")
    unused = [b for j, b in enumerate(bpts) if j not in used]
    print(f"\npanel a points: {len(df)}   panel b points: {len(bpts)}   unmatched in b: {len(unused)}")
    for b in unused:
        print("  panel b only:", b["label"], b["ecc"], round(b["p"], 3))
    print("cross-check:", "all match" if not problems else "")
    for p in problems:
        print("  ", p)

    out = df.drop(columns=["_px", "_py"])
    out["gap_slots"] = out["gap_slots"].astype("Int64")
    out.to_csv(DATA_DIR / "bouma1970_from_coates2021.csv", index=False)
    print()
    print(out.drop(columns=["source", "n_trials"]).to_string(index=False))

    # Overlay: extracted points on the original figure.
    scale = 600 / 72
    clip = pymupdf.Rect(110, 495, 520, 650)
    pix = page.get_pixmap(dpi=600, clip=clip)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    fig, ax = plt.subplots(figsize=(pix.w / 300, pix.h / 300), dpi=300)
    ax.imshow(img)
    tx = lambda x: (x - clip.x0) * scale
    ty = lambda y: (y - clip.y0) * scale
    for _, r in df.iterrows():
        ax.plot(tx(r._px), ty(r._py), "o", ms=1.6, color="red", mec="none")
        ax.annotate(f"{r.p_correct:.2f}", (tx(r._px), ty(r._py)), xytext=(4, -2),
                    textcoords="offset points", fontsize=3, color="red")
    for b in bpts:
        ax.plot(tx(b["px"]), ty(b["py"]), "o", ms=1.6, color="red", mec="none")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(DATA_DIR / "bouma1970_extraction_check.png")


if __name__ == "__main__":
    main()
