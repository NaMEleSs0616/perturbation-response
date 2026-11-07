"""Norman 2019 pair analysis for the paper (exploratory, E3).

Per pair, the coefficient beta in  pair ≈ beta * (single A + single B)  is estimated two ways:
  ols: pair from evaluation controls, singles from the disjoint training controls
  iv:  instrumental variable; singles from cell half A instrument singles from cell half B,
       which removes attenuation from measurement error in the singles
Writes results/paper/pair_coefficients.csv.

    python paper/scripts/pair_analysis.py <processed Norman dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from biophysical.preprocessing.pipeline import load_response_data


def pair_coefficients(processed: str | Path) -> pd.DataFrame:
    d = load_response_data(processed)
    idx = {frozenset(g): i for i, g in enumerate(d.condition_genes())}
    rows = []
    for i, g in enumerate(d.condition_genes()):
        if len(g) != 2 or not all(frozenset([x]) in idx for x in g):
            continue
        a, b = (idx[frozenset([x])] for x in g)
        m = ~(d.self_mask[i] | d.self_mask[a] | d.self_mask[b])
        y = d.Y_eval[i][m]
        s_fit = (d.Y_fit[a] + d.Y_fit[b])[m]
        s_a = (d.Y_half_a[a] + d.Y_half_a[b])[m]
        s_b = (d.Y_half_b[a] + d.Y_half_b[b])[m]
        rows.append({
            "pair": d.conditions["condition"].iloc[i],
            "beta_ols": float(s_fit @ y / (s_fit @ s_fit)),
            "beta_iv": float(s_a @ y / (s_a @ s_b)),
            "singles_split_half_r": float(np.corrcoef(s_a, s_b)[0, 1]),
            "r_pair_vs_sum": float(np.corrcoef(s_fit, y)[0, 1]),
            "rms_sum_of_singles": float(np.sqrt((s_fit ** 2).mean())),
            "rms_pair": float(np.sqrt((y ** 2).mean())),
            "n_cells_pair": int(d.conditions["n_cells"].iloc[i]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path("results/paper")
    out.mkdir(parents=True, exist_ok=True)
    t = pair_coefficients(sys.argv[1])
    t.to_csv(out / "pair_coefficients.csv", index=False)
    print(t[["beta_ols", "beta_iv"]].describe().round(3).to_string())
