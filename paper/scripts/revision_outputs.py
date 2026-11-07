"""Figures, tables and macros for the revision analyses (all post hoc). Appends to paper/numbers.tex.

    python paper/scripts/revision_outputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import BLUE, GRID, INK, MODEL_NAMES, MUTED, NAMES, ORANGE, Numbers, order, tex_name  # noqa: E402

from biophysical.evaluation.metrics import bootstrap_ci, paired_comparison
from biophysical.evaluation.study import holm

STUDY = Path("results/study/primary")
REV = Path("results/paper/revision")
PAPER = Path("paper")
FEATURE_NAMES = {"full": "Co-expression + baseline (primary)", "coexpression_only": "Co-expression only", "baseline_both": "Baseline mean + detection",
                 "baseline_mean_only": "Baseline mean only", "detection_only": "Detection rate only", "full_split0": "Primary, training-half controls"}


def c1_distribution(N: Numbers) -> None:
    f = pd.read_csv(STUDY / "per_perturbation_all.csv")
    f = f[(f["stratum"] == "single_unseen") & (f["split_seed"] == 0) & (f["variant"] == "coexpression")]
    ds = order(f["dataset_id"].unique().tolist())
    models = ["ridge", "gene_loading_ridge", "mlp_ensemble"]
    fig, axes = plt.subplots(1, len(ds), figsize=(2.3 * len(ds), 2.7), sharey=True)
    rows = []
    for ax, d in zip(axes, ds):
        g = f[f["dataset_id"] == d]
        base = g[g["model"] == "mean"].set_index("condition")["mse"]
        for k, m in enumerate(models):
            a = g[g["model"] == m].set_index("condition")["mse"]
            dd = (a - base.reindex(a.index)).to_numpy() * 1e3
            rng = np.random.default_rng(0)
            meds = [np.median(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(2000)]
            rows.append({"dataset": d, "model": m, "n": len(dd), "mean": dd.mean(), "median": np.median(dd), "med_lo": np.quantile(meds, 0.025),
                         "med_hi": np.quantile(meds, 0.975), "frac_better": (dd < 0).mean()})
            y = k + rng.uniform(-0.28, 0.28, size=len(dd))
            ax.scatter(dd, y, s=3, color=BLUE, alpha=0.25, linewidths=0)
            ax.plot([np.median(dd)] * 2, [k - 0.38, k + 0.38], color=ORANGE, lw=1.6)
            ax.plot([dd.mean()] * 2, [k - 0.38, k + 0.38], color=INK, lw=1.2, ls=(0, (2, 1.5)))
        ax.axvline(0, color=INK, lw=0.7)
        # each screen has its own scale; linear region spans the central half of the differences
        allv = np.concatenate([(g[g["model"] == m].set_index("condition")["mse"] - base).dropna().to_numpy() * 1e3 for m in models])
        lin = float(np.quantile(np.abs(allv), 0.5)) or 1e-6
        ax.set_xscale("symlog", linthresh=lin)
        lim = float(np.quantile(np.abs(allv), 0.995))
        ax.set_xlim(-lim, lim)
        ax.set_xticks([-lim / 1.2, 0, lim / 1.2], [f"{-lim / 1.2:.2g}", "0", f"{lim / 1.2:.2g}"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(NAMES[d].replace(" CRISPR", "\nCRISPR"), loc="left", fontsize=7.5)
        ax.set_xlabel("ΔMSE×10³ (model − mean)", fontsize=7)
    axes[0].set_yticks(range(len(models)), [MODEL_NAMES[m] for m in models])
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "figS1_c1_distribution.pdf")
    plt.close(fig)
    t = pd.DataFrame(rows)
    body = [f"{NAMES[r.dataset]} & {MODEL_NAMES[r.model]} & {r.mean:+.3f} & {r.median:+.3f} ({r.med_lo:+.3f}, {r.med_hi:+.3f}) & {100 * r.frac_better:.0f}\\% \\\\" for r in t.itertuples()]
    (PAPER / "tables" / "c1_medians.tex").write_text("\n".join([r"\begin{tabular}{llrcr}", r"\toprule", r"Dataset & Model & Mean $\Delta$MSE$\times10^3$ & Median (95\% CI) & Model better \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]))
    r = t[(t["dataset"] == "tian2021_crispra") & (t["model"] == "mlp_ensemble")].iloc[0]
    N.add("tianAMlpMedian", r["median"], "{:+.3f}")
    N.add("tianAMlpMedianLo", r["med_lo"], "{:+.3f}")
    N.add("tianAMlpMedianHi", r["med_hi"], "{:+.3f}")
    N.add("tianAMlpFracBetter", f"{100 * r['frac_better']:.0f}")
    N.add("tianAMlpMean", r["mean"], "{:+.3f}")


def c2_holm(N: Numbers) -> None:
    c = pd.read_csv(STUDY / "confirmatory_tests.csv")
    c2 = c[c["test"] == "C2"].copy()
    c2["p_holm_posthoc"] = holm(c2["p_raw"].tolist())
    rep = c2[c2["dataset_id"].str.startswith("replogle")]
    mx = float(rep["p_holm_posthoc"].max())
    e = int(np.floor(np.log10(mx)))
    N.add("cTwoMaxHolmReplogle", f"{mx / 10 ** e:.0f}\\times10^{{{e}}}")
    N.add("cTwoMinHolmTian", float(c2[c2["dataset_id"].str.startswith("tian")]["p_holm_posthoc"].min()), "{:.2f}")
    c2.to_csv(REV / "c2_holm_posthoc.csv", index=False)


def ablation(N: Numbers) -> None:
    a = pd.read_csv(REV / "feature_ablation.csv")
    ds = order(a["dataset_id"].unique().tolist())
    feats = [f for f in FEATURE_NAMES if f in a["features"].unique()]
    rows, plot = [], []
    for d in ds:
        for m in ("ridge", "mlp_ensemble"):
            g = a[(a["dataset_id"] == d) & (a["model"] == m)]
            ref = g[g["features"] == "full"].set_index("condition")
            for fset in feats:
                h = g[g["features"] == fset].set_index("condition")
                sk, ps = bootstrap_ci(h["skill_vs_mean"].to_numpy()), bootstrap_ci(h["pearson_specific"].to_numpy())
                pc = paired_comparison(h["skill_vs_mean"].to_numpy(), ref["skill_vs_mean"].reindex(h.index).to_numpy(), 1000) if fset != "full" else None
                rows.append({"dataset": d, "model": m, "features": fset, "skill": sk[0], "skill_lo": sk[1], "skill_hi": sk[2], "r_spec": ps[0],
                             "diff_vs_full": pc["mean_diff"] if pc else 0.0, "diff_lo": pc["ci_low"] if pc else 0.0, "diff_hi": pc["ci_high"] if pc else 0.0})
                k = f"{tex_name(d)}{tex_name(m)}{tex_name(fset)}"
                N.add(f"abl{k}", sk[0], "{:.3f}")
                N.add(f"ablSpec{k}", ps[0])
                if pc:
                    N.add(f"ablDiff{k}", pc["mean_diff"], "{:+.3f}")
                    N.add(f"ablDiffLo{k}", pc["ci_low"], "{:+.3f}")
                    N.add(f"ablDiffHi{k}", pc["ci_high"], "{:+.3f}")
    t = pd.DataFrame(rows)
    t.to_csv(REV / "feature_ablation_summary.csv", index=False)
    body, last = [], None
    for r in t.itertuples():
        lab = f"{NAMES[r.dataset]}, {MODEL_NAMES[r.model]}"
        diff = "--" if r.features == "full" else f"{r.diff_vs_full:+.3f} ({r.diff_lo:+.3f}, {r.diff_hi:+.3f})"
        body.append(f"{lab if lab != last else ''} & {FEATURE_NAMES[r.features]} & {r.skill:.3f} ({r.skill_lo:.3f}, {r.skill_hi:.3f}) & {r.r_spec:.3f} & {diff} \\\\")
        last = lab
    (PAPER / "tables" / "ablation.tex").write_text("\n".join([r"\begin{tabular}{lllcc}", r"\toprule", r"Dataset, model & Features & Skill (95\% CI) & $r$, specific & Skill $-$ primary (95\% CI) \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]))
    # figure: Replogle screens, skill by feature set
    big = [d for d in ds if d.startswith("replogle")]
    fig, axes = plt.subplots(1, len(big), figsize=(3.6 * len(big), 2.5), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], big):
        for k, fset in enumerate(feats):
            for j, (m, col) in enumerate((("ridge", MUTED), ("mlp_ensemble", BLUE))):
                r = t[(t["dataset"] == d) & (t["model"] == m) & (t["features"] == fset)].iloc[0]
                y = -k + (0.15 if j == 0 else -0.15)
                ax.plot([r.skill_lo, r.skill_hi], [y, y], color=col, lw=1.6)
                ax.plot(r.skill, y, "o", color=col, ms=3.8, label=MODEL_NAMES[m] if k == 0 else None)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(NAMES[d], loc="left")
        ax.set_xlabel("Skill vs mean (held-out genes)")
    axes[0][0].set_yticks([-k for k in range(len(feats))], [FEATURE_NAMES[f] for f in feats])
    axes[0][0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig7_feature_ablation.pdf")
    plt.close(fig)
    # stratified interval calibration (primary features only)
    cov = a[(a["features"] == "full") & a["top_std"].notna()]
    crow = []
    for d in ds:
        for m in ("ridge", "mlp_ensemble"):
            g = cov[(cov["dataset_id"] == d) & (cov["model"] == m)]
            vals = {k: float(g[k].mean()) for k in ("cov_std", "cov_str", "top_std", "top_str", "width_std_top", "width_str_top")}
            crow.append({"dataset": d, "model": m, **vals})
            k = f"{tex_name(d)}{tex_name(m)}"
            for key in ("cov_std", "cov_str", "top_std", "top_str"):
                N.add(f"{tex_name(key)}{k}", vals[key])
            N.add(f"widthRatio{k}", vals["width_str_top"] / vals["width_std_top"], "{:.1f}")
    ct = pd.DataFrame(crow)
    body = [f"{NAMES[r.dataset]} & {MODEL_NAMES[r.model]} & {r.cov_std:.2f} & {r.cov_str:.2f} & {r.top_std:.2f} & {r.top_str:.2f} & {r.width_str_top / r.width_std_top:.1f} \\\\" for r in ct.itertuples()]
    (PAPER / "tables" / "intervals.tex").write_text("\n".join([r"\begin{tabular}{llccccc}", r"\toprule",
        r" & & \multicolumn{2}{c}{All genes} & \multicolumn{2}{c}{Top-20 responders} & Width ratio, \\", r"Dataset & Model & Standard & Stratified & Standard & Stratified & top-20 \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]))


def main() -> None:
    N = Numbers()
    c1_distribution(N)
    c2_holm(N)
    if (REV / "feature_ablation.csv").exists():
        ablation(N)
    lines = ["% Revision (post hoc) numbers, generated by paper/scripts/revision_outputs.py; do not edit."]
    lines += [f"\\newcommand{{\\num{tex_name(k)}}}{{{v}}}" for k, v in sorted(N.items.items())]
    (PAPER / "numbers_revision.tex").write_text("\n".join(lines) + "\n")
    print(len(N.items), "macros")


if __name__ == "__main__":
    main()
