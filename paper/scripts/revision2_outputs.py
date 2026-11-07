"""Tables, figures and macros for the second-revision analyses (all post hoc). Writes paper/numbers_revision2.tex.

    python paper/scripts/revision2_outputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import BLUE, INK, MODEL_NAMES, MUTED, NAMES, ORANGE, Numbers, order, tex_name  # noqa: E402

R = Path("results/paper/revision2")
T = Path("paper/tables")
F = Path("paper/figures")
GENESET = {"primary": "Primary (HVG + targets)", "extended": "All detected genes", "outside_primary": "Outside primary set", "silent_in_controls": "Silent in controls ($<$1\\%)"}
METRIC = {"mse": "MSE", "pearson": "$r$ (raw)", "pearson_specific": "$r$ (specific)", "spearman_specific": "$\\rho$ (specific)", "topk_recall": "Top-20 recall"}


def tab(path: Path, header: str, cols: str, rows: list[str]) -> None:
    path.write_text("\n".join([rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def main() -> None:
    N = Numbers()
    ds_all = ["norman2019", "replogle2022_k562_essential", "replogle2022_rpe1", "tian2021_crispra", "tian2021_crispri"]
    # reproducibility limits
    lim = pd.read_csv(R / "reproducibility_limits.csv")
    rows = []
    for d in ds_all:
        g = lim[(lim["dataset_id"] == d) & (lim["kind"] == "batch")]
        c = lim[(lim["dataset_id"] == d) & (lim["kind"] == "cells")].iloc[0]
        k = tex_name(d)
        if len(g):
            p0 = g[g["partition"] == "Y_part0_A"].iloc[0]
            N.add(f"lim{k}", p0["limit"])
            N.add(f"limLo{k}", p0["limit_lo"])
            N.add(f"limHi{k}", p0["limit_hi"])
            N.add(f"limMin{k}", g["limit"].min())
            N.add(f"limMax{k}", g["limit"].max())
            n_distinct = int(g["limit"].round(6).nunique())
            rows.append(f"{NAMES[d]} & {int(p0['n_batch_levels'])} & {n_distinct} & {p0['limit']:.2f} ({p0['limit_lo']:.2f}, {p0['limit_hi']:.2f}) & {g['limit'].min():.2f}--{g['limit'].max():.2f} & {c['limit']:.2f} ({c['limit_lo']:.2f}, {c['limit_hi']:.2f}) & {100 * p0['frac_reproducible']:.0f}\\% \\\\")
        N.add(f"limCells{k}", c["limit"])
    tab(T / "limits.tex", "Dataset & Batches & Distinct partitions & Limit, primary partition (95\\% CI) & Range over partitions & Limit, cell halves (95\\% CI) & Reproducible", "lccccccc"[:7], rows)
    # discrimination with Wilson intervals
    dis = pd.read_csv(R / "discrimination.csv")
    rows = []
    for d in ds_all:
        for m in METRIC:
            a = dis[(dis["dataset_id"] == d) & (dis["metric"] == m) & (dis["uninformative"] == "mean")].iloc[0]
            b = dis[(dis["dataset_id"] == d) & (dis["metric"] == m) & (dis["uninformative"] == "zero")].iloc[0]
            rows.append(f"{NAMES[d] if m == 'mse' else ''} & {METRIC[m]} & {a.win_rate:.2f} ({a.lo:.2f}, {a.hi:.2f}) & {b.win_rate:.2f} ({b.lo:.2f}, {b.hi:.2f}) \\\\")
            N.add(f"winMean{tex_name(d)}{tex_name(m)}", a.win_rate)
            N.add(f"winMeanLo{tex_name(d)}{tex_name(m)}", a.lo)
            N.add(f"winMeanHi{tex_name(d)}{tex_name(m)}", a.hi)
            N.add(f"winZero{tex_name(d)}{tex_name(m)}", b.win_rate)
    tab(T / "discrimination.tex", "Dataset & Metric & Duplicate vs mean (95\\% CI) & Duplicate vs zero (95\\% CI)", "llcc", rows)
    rel = pd.read_csv(R / "discrimination_by_reliability.csv")
    rows = [f"{NAMES[r.dataset_id]} & {r.reproducibility_tertile} & {r.median_r:.2f} & {r.mse_win_rate:.2f} & {r.n} \\\\" for r in rel.itertuples()]
    tab(T / "discrimination_reliability.tex", "Dataset & Reproducibility third & Median $r$ & Duplicate beats mean on MSE & $n$", "llccr", rows)
    for d in ds_all:
        g = rel[rel["dataset_id"] == d].set_index("reproducibility_tertile")
        N.add(f"mseWinLow{tex_name(d)}", g.loc["low", "mse_win_rate"])
        N.add(f"mseWinHigh{tex_name(d)}", g.loc["high", "mse_win_rate"])
    # block bootstrap
    bb = pd.read_csv(R / "block_bootstrap.csv")
    rows = []
    for r in bb.itertuples():
        rows.append(f"{NAMES[r.dataset_id]} & {MODEL_NAMES[r.model]} & {r.n_clusters} & {r.skill:.3f} & ({r.skill_lo_perturbation:.3f}, {r.skill_hi_perturbation:.3f}) & ({r.skill_lo_block:.3f}, {r.skill_hi_block:.3f}) & ({r.dmse_lo_block:+.3f}, {r.dmse_hi_block:+.3f}) \\\\")
        k = f"{tex_name(r.dataset_id)}{tex_name(r.model)}"
        N.add(f"blockLo{k}", r.skill_lo_block, "{:.3f}")
        N.add(f"blockHi{k}", r.skill_hi_block, "{:.3f}")
        N.add(f"blockDmseHi{k}", r.dmse_hi_block, "{:+.3f}")
    tab(T / "block_bootstrap.tex", "Dataset & Model & Clusters & Skill & CI, perturbations & CI, clusters & $\\Delta$MSE$\\times10^3$ CI, clusters", "llrcccc", rows)
    # cell counts
    cc = pd.read_csv(R / "cell_count.csv")
    rows = []
    for (d, m), g in cc.groupby(["dataset_id", "model"], sort=False):
        g = g.set_index("stratum")
        rows.append(f"{NAMES[d]} & {MODEL_NAMES[m]} & " + " & ".join(f"{g.loc[s, 'skill']:.3f}" for s in ("fewest", "middle", "most", "cell-weighted", ">=100 cells")) + f" & {int(g.loc['>=100 cells', 'n'])} \\\\")
        k = f"{tex_name(d)}{tex_name(m)}"
        for s, key in (("fewest", "Fewest"), ("most", "Most"), ("cell-weighted", "Weighted"), (">=100 cells", "Hundred")):
            N.add(f"cc{key}{k}", g.loc[s, "skill"], "{:.3f}")
    tab(T / "cell_count.tex", "Dataset & Model & Fewest cells & Middle & Most cells & Cell-weighted & $\\ge$100 cells & $n$ ($\\ge$100)", "llcccccr", rows)
    # estimands for C1-C3
    est = pd.read_csv(R / "estimands.csv")
    rows = []
    for r in est.itertuples():
        fmt = "{:+.3f}"
        rows.append(f"{r.test} & {NAMES[r.dataset_id]} & {MODEL_NAMES[r.model]} & {fmt.format(r.mean)} ({fmt.format(r.mean_lo)}, {fmt.format(r.mean_hi)}) & {fmt.format(r.median)} ({fmt.format(r.median_lo)}, {fmt.format(r.median_hi)}) & {100 * r.frac_favouring_first:.0f}\\% & {r.p_holm if r.test != 'C2' else r.p_raw:.2g} \\\\")
        if r.test == "C3":
            k = f"{tex_name(r.dataset_id)}{tex_name(r.model)}"
            N.add(f"cThreeMean{k}", r.mean, fmt)
            N.add(f"cThreeMedian{k}", r.median, fmt)
            N.add(f"cThreeMedianLo{k}", r.median_lo, fmt)
            N.add(f"cThreeMedianHi{k}", r.median_hi, fmt)
            N.add(f"cThreeFrac{k}", f"{100 * r.frac_favouring_first:.0f}")
    tab(T / "estimands.tex", "Test & Dataset & Model & Mean (95\\% CI) & Median (95\\% CI) & Favouring & $p$", "lllcccr", rows)
    # gene sets
    gs = pd.read_csv(R / "gene_sets.csv")
    rows = []
    for d in order(gs["dataset_id"].unique().tolist()):
        for m in ("ridge", "mlp_ensemble", "knn"):
            g = gs[(gs["dataset_id"] == d) & (gs["model"] == m)].set_index("genes")
            if g.empty:
                continue
            cell = lambda s: "not available" if pd.isna(g.loc[s, "skill"]) else f"{g.loc[s, 'skill']:.3f} / {g.loc[s, 'r_specific']:.2f}"
            rows.append(f"{NAMES[d]} & {MODEL_NAMES[m]} & " + " & ".join(cell(s) for s in GENESET) + " \\\\")
            for s in GENESET:
                N.add(f"gs{tex_name(d)}{tex_name(m)}{tex_name(s)}", g.loc[s, "skill"], "{:.3f}")
                N.add(f"gsSpec{tex_name(d)}{tex_name(m)}{tex_name(s)}", g.loc[s, "r_specific"])
    tab(T / "gene_sets.tex", "Dataset & Model & " + " & ".join(GENESET.values()), "llcccc", rows)
    # variability over control splits and seeds
    var = pd.read_csv(R / "variability.csv")
    rows = []
    for (d, m), g in var.groupby(["dataset_id", "model"]):
        cs = g[g["run"].str.startswith("control_split") | (g["run"] == "primary_split_seed0")]["skill"]
        sd = g[g["run"].str.startswith("mlp_seed") | (g["run"] == "primary_split_seed0")]["skill"] if m == "mlp_ensemble" else pd.Series(dtype=float)
        rows.append(f"{NAMES[d]} & {MODEL_NAMES[m]} & {cs.min():.3f}--{cs.max():.3f} & " + (f"{sd.min():.3f}--{sd.max():.3f}" if len(sd) else "--") + " \\\\")
        k = f"{tex_name(d)}{tex_name(m)}"
        N.add(f"csMin{k}", cs.min(), "{:.3f}")
        N.add(f"csMax{k}", cs.max(), "{:.3f}")
        if len(sd):
            N.add(f"seedMin{k}", sd.min(), "{:.3f}")
            N.add(f"seedMax{k}", sd.max(), "{:.3f}")
    tab(T / "variability.tex", "Dataset & Model & Skill over 5 control splits & Skill over 5 MLP seeds", "llcc", rows)
    # intervals by selection rule
    iv = pd.read_csv(R / "intervals_selection.csv")
    rows = []
    for r in iv.itertuples():
        rows.append(f"{NAMES[r.dataset_id]} & {MODEL_NAMES[r.model]} & {r.cov_all:.2f} & {r.cov_top_predicted:.2f} & {r.cov_top_observed:.2f} \\\\")
        k = f"{tex_name(r.dataset_id)}{tex_name(r.model)}"
        N.add(f"ivAll{k}", r.cov_all)
        N.add(f"ivPred{k}", r.cov_top_predicted)
        N.add(f"ivObs{k}", r.cov_top_observed)
    tab(T / "intervals_selection.tex", "Dataset & Model & All genes & Top 20 by prediction & Top 20 by observation", "llccc", rows)
    # reproducibility subset selected independently
    ri = pd.read_csv(R / "reproducible_independent.csv")
    rows = []
    for r in ri.itertuples():
        if r.model not in ("ridge", "mlp_ensemble", "zero"):
            continue
        label = r.subset.replace(">=", r"$\ge$")
        rows.append(f"{NAMES[r.dataset_id]} & {MODEL_NAMES[r.model]} & {label} & {r.n} & {r.skill_setB:.3f} ({r.lo:.3f}, {r.hi:.3f}) & {r.r_specific_setB:.2f} \\\\")
        k = f"{tex_name(r.dataset_id)}{tex_name(r.model)}{'Rep' if r.subset.startswith('reproducible') else 'All'}"
        N.add(f"riSkill{k}", r.skill_setB, "{:.3f}")
        N.add(f"riLo{k}", r.lo, "{:.3f}")
        N.add(f"riHi{k}", r.hi, "{:.3f}")
        N.add(f"riSpec{k}", r.r_specific_setB)
        N.add(f"riN{k}", str(r.n))
    tab(T / "reproducible_independent.tex", "Dataset & Model & Perturbations & $n$ & Skill vs set-B target (95\\% CI) & $r$, specific", "lllrcc", rows)
    # ranking
    rk = pd.read_csv(R / "ranking.csv")
    rows = []
    for d in order(rk["dataset_id"].unique().tolist()):
        for name in ("ridge", "mlp_ensemble"):
            g = rk[(rk["dataset_id"] == d) & (rk["ranking"] == name)].set_index("top_fraction")
            rows.append(f"{NAMES[d]} & {MODEL_NAMES[name]} & " + " & ".join(f"{g.loc[f, 'precision']:.2f} ({g.loc[f, 'lo']:.2f}--{g.loc[f, 'hi']:.2f}); {g.loc[f, 'p_vs_random']:.3f}" for f in (0.1, 0.2, 0.3)) + " \\\\")
            for f in (0.1, 0.2, 0.3):
                k = f"{tex_name(d)}{tex_name(name)}{ {0.1: 'Ten', 0.2: 'Twenty', 0.3: 'Thirty'}[f] }"
                N.add(f"rk{k}", g.loc[f, "precision"])
                N.add(f"rkLo{k}", g.loc[f, "lo"])
                N.add(f"rkHi{k}", g.loc[f, "hi"])
                N.add(f"rkP{k}", g.loc[f, "p_vs_random"], "{:.3f}")
        seeds = rk[(rk["dataset_id"] == d) & rk["ranking"].str.startswith("mlp_seed") & (rk["top_fraction"] == 0.2)]["precision"]
        if len(seeds):
            N.add(f"rkSeedMin{tex_name(d)}", seeds.min())
            N.add(f"rkSeedMax{tex_name(d)}", seeds.max())
    tab(T / "ranking.tex", "Dataset & Model & Top 10\\%: precision (CI); $p$ & Top 20\\% & Top 30\\%", "llccc", rows)
    # figure: gene-set sensitivity in the large screens
    big = [d for d in order(gs["dataset_id"].unique().tolist()) if d.startswith("replogle")] or order(gs["dataset_id"].unique().tolist())
    fig, axes = plt.subplots(1, len(big), figsize=(3.6 * len(big), 2.4), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], big):
        for k, s in enumerate(GENESET):
            for j, (m, col) in enumerate((("ridge", MUTED), ("mlp_ensemble", BLUE))):
                r = gs[(gs["dataset_id"] == d) & (gs["model"] == m) & (gs["genes"] == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = -k + (0.15 if j == 0 else -0.15)
                ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=1.6)
                ax.plot(r["skill"], y, "o", color=col, ms=3.8, label=MODEL_NAMES[m] if k == 0 else None)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(NAMES[d], loc="left")
        ax.set_xlabel("Skill vs mean (held-out genes)")
    axes[0][0].set_yticks([-k for k in range(len(GENESET))], [v.replace("$<$", "<").replace("\\%", "%") for v in GENESET.values()])
    axes[0][0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "fig8_gene_sets.pdf")
    plt.close(fig)
    lines = ["% Second-revision (post hoc) numbers, generated by paper/scripts/revision2_outputs.py; do not edit."]
    lines += [f"\\newcommand{{\\num{tex_name(k)}}}{{{v}}}" for k, v in sorted(N.items.items())]
    Path("paper/numbers_revision2.tex").write_text("\n".join(lines) + "\n")
    print(len(N.items), "macros")


if __name__ == "__main__":
    main()
