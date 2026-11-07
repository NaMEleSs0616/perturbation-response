"""Second-revision analyses (POST HOC) on the outputs of revision2_extended.py.

    python paper/scripts/revision2_evaluate.py models <dataset_id> [...]   # model fits (slow for Replogle)
    python paper/scripts/revision2_evaluate.py summaries                   # everything that needs no refitting

models:
  run1  extended output genes, primary control split; models mean, zero, knn, ridge, gene-loading ridge, MLP.
        Scored on gene subsets: primary genes, all extended genes, genes outside the primary set, and genes nearly
        silent in controls (<1% detected). Also: interval coverage on genes chosen by predicted deviation,
        and evaluation against batch set B, for selection-independent reproducibility analysis.
  run2  control-split replicates r = 1..4 (ridge, MLP; primary genes)
  run3  MLP initialisation seeds 1..4 (primary genes, primary control split)
summaries:
  technical reproducibility limits with bootstrap intervals and spread over batch partitions; metric discrimination
  with Wilson intervals (duplicate vs mean, duplicate vs zero, never a per-perturbation maximum); gene-module block
  bootstrap; cell-count stratification; consistent estimands for C1-C3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from biophysical.config import get_paths, load_yaml
from biophysical.evaluation.benchmark import model_factories
from biophysical.evaluation.metrics import _corr, bootstrap_ci, perturbation_metrics, response_magnitude
from biophysical.evaluation.study import holm
from biophysical.models.baselines import FitData, MeanModel
from biophysical.models.features import default_featurizer
from biophysical.models.mlp import MLPConfig, MLPEnsemble
from biophysical.preprocessing.pipeline import load_response_data
from biophysical.preprocessing.splits import unseen_gene_folds
from biophysical.uncertainty.calibration import GeneIntervalCalibrator

EXT = Path("results/paper/revision2")
STUDY = Path("results/study/primary")
CFG = load_yaml("configs/benchmark_norman.yaml")
DATASETS = ["norman2019", "replogle2022_k562_essential", "replogle2022_rpe1", "tian2021_crispra", "tian2021_crispri"]


def primary_data(d: str):
    m = next(x for x in json.loads((STUDY / "datasets.json").read_text()) if x["dataset_id"] == d)
    return load_response_data(get_paths().processed / d / m["processed_version"])


def loo_reference(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """For each condition, the mean response of all other conditions (masked entries excluded)."""
    Yr = np.where(M, np.nan, Y)
    tot, cnt = np.nansum(Yr, axis=0), np.sum(~np.isnan(Yr), axis=0)
    own, own_c = np.nan_to_num(Yr), (~np.isnan(Yr)).astype(int)
    return (tot - own) / np.maximum(cnt - own_c, 1)


def subset_metrics(y, p, m, ref, cols: np.ndarray) -> dict:
    """Metrics on a gene subset; an empty or fully masked subset gives NaN (e.g. no silent genes in a file
    that contains only expressed genes)."""
    if cols.size == 0 or m[cols].all():
        return {k: float("nan") for k in ("mse", "skill_vs_mean", "pearson_specific")}
    return perturbation_metrics(y[cols], p[cols], m[cols], reference=ref[cols])


def run_models(d: str) -> None:
    z = np.load(EXT / f"{d}.npz", allow_pickle=True)
    prim = primary_data(d)
    genes = list(z["genes"])
    gene_index = {g: i for i, g in enumerate(genes)}
    feat = default_featurizer(prim)
    cg = prim.condition_genes()
    ids = prim.conditions["condition"].to_numpy()
    in_primary = z["in_primary"]
    subsets = {"primary": in_primary, "extended": np.ones(len(genes), bool), "outside_primary": ~in_primary, "silent_in_controls": z["control_detection"] < 0.01}
    prim_cols = np.flatnonzero(in_primary)
    folds = unseen_gene_folds(cg, CFG["splits"]["n_folds"], CFG["splits"]["calibration_fraction"], 0)
    rows, preds = [], {}

    def fitdata(Y, cols=None):
        Yc, Mc = (Y, z["self_mask"]) if cols is None else (Y[:, cols], z["self_mask"][:, cols])
        gi = gene_index if cols is None else {genes[c]: k for k, c in enumerate(cols)}
        return FitData(cg, Yc, Mc, feat, gi)

    # run1: extended outputs
    fit = fitdata(z["Y_fit_0"])
    Y_eval, M = z["Y_eval_0"], z["self_mask"]
    Y_B = z["Y_part0_B"] if "Y_part0_B" in z else None
    fac = model_factories(CFG, "unseen_gene", 0)
    for fold in folds:
        train, calib = fit.subset(fold.train), fit.subset(fold.calib)
        tg, cgn = [cg[i] for i in fold.test], [cg[i] for i in fold.calib]
        mean_model = MeanModel().fit(train)
        Rt, Rc = mean_model.predict(tg), mean_model.predict(cgn)
        for name in ("mean", "zero", "knn", "ridge", "gene_loading_ridge", "mlp_ensemble"):
            model = fac[name]().fit(train, calib)
            Pt, Pc = model.predict(tg), model.predict(cgn)
            cal = GeneIntervalCalibrator(alpha=0.1).fit(Y_eval[fold.calib][:, prim_cols], Pc[:, prim_cols], M[fold.calib][:, prim_cols])
            hw = cal.half_width()
            for j, i in enumerate(fold.test):
                if fold.strata[int(i)] != "single_unseen":
                    continue
                row = {"dataset_id": d, "run": "extended", "model": name, "fold": fold.fold, "condition": ids[i], "n_cells": int(z["n_cells"][i])}
                for sname, cols in subsets.items():
                    mm = subset_metrics(Y_eval[i], Pt[j], M[i], Rt[j], np.flatnonzero(cols))
                    row.update({f"{sname}_{k}": mm[k] for k in ("mse", "skill_vs_mean", "pearson_specific")})
                y, p, r, mk = Y_eval[i][prim_cols], Pt[j][prim_cols], Rt[j][prim_cols], M[i][prim_cols]
                inside = np.abs(y - p) <= hw
                top_pred = np.argsort(-np.where(mk, -1, np.abs(p - r)), kind="stable")[:20]
                top_obs = np.argsort(-np.where(mk, -1, np.abs(y - r)), kind="stable")[:20]
                row.update({"cov_all": inside[~mk].mean(), "cov_top_predicted": inside[top_pred].mean(), "cov_top_observed": inside[top_obs].mean()})
                if Y_B is not None and np.isfinite(Y_B[i]).all():
                    mb = subset_metrics(Y_B[i], Pt[j], M[i], Rt[j], prim_cols)
                    row.update({"setB_mse": mb["mse"], "setB_skill_vs_mean": mb["skill_vs_mean"], "setB_pearson_specific": mb["pearson_specific"]})
                rows.append(row)
            preds.setdefault(name, {}).update({int(i): Pt[j][prim_cols] for j, i in enumerate(fold.test)})
        print(d, "run1 fold", fold.fold, flush=True)
    # run2: control-split replicates; run3: MLP seeds (primary genes)
    for r in range(1, 5):
        f2 = fitdata(z[f"Y_fit_{r}"], prim_cols)
        Ye = z[f"Y_eval_{r}"][:, prim_cols]
        for fold in folds:
            train, calib = f2.subset(fold.train), f2.subset(fold.calib)
            tg = [cg[i] for i in fold.test]
            Rt = MeanModel().fit(train).predict(tg)
            for name in ("ridge", "mlp_ensemble"):
                Pt = fac[name]().fit(train, calib).predict(tg)
                for j, i in enumerate(fold.test):
                    if fold.strata[int(i)] == "single_unseen":
                        mm = perturbation_metrics(Ye[i], Pt[j], f2.mask[i], reference=Rt[j])
                        rows.append({"dataset_id": d, "run": f"control_split_{r}", "model": name, "fold": fold.fold, "condition": ids[i], "primary_mse": mm["mse"],
                                     "primary_skill_vs_mean": mm["skill_vs_mean"], "primary_pearson_specific": mm["pearson_specific"]})
        print(d, "run2 split", r, flush=True)
    f3 = fitdata(z["Y_fit_0"], prim_cols)
    Ye = Y_eval[:, prim_cols]
    for s in range(1, 5):
        cfg = MLPConfig(**{**CFG["models"]["mlp"], "seed": 7919 * s})
        for fold in folds:
            train, calib = f3.subset(fold.train), f3.subset(fold.calib)
            tg = [cg[i] for i in fold.test]
            Rt = MeanModel().fit(train).predict(tg)
            Pt = MLPEnsemble(cfg).fit(train, calib).predict(tg)
            for j, i in enumerate(fold.test):
                if fold.strata[int(i)] == "single_unseen":
                    mm = perturbation_metrics(Ye[i], Pt[j], f3.mask[i], reference=Rt[j])
                    rows.append({"dataset_id": d, "run": f"mlp_seed_{s}", "model": "mlp_ensemble", "fold": fold.fold, "condition": ids[i], "primary_mse": mm["mse"],
                                 "primary_skill_vs_mean": mm["skill_vs_mean"], "primary_pearson_specific": mm["pearson_specific"]})
                    preds.setdefault(f"mlp_seed_{s}", {})[int(i)] = Pt[j]
        print(d, "run3 seed", s, flush=True)
    pd.DataFrame(rows).to_csv(EXT / f"models_{d}.csv", index=False)
    np.savez_compressed(EXT / f"preds_{d}.npz", **{k: np.stack([v[i] for i in sorted(v)]) for k, v in preds.items()}, order=np.array(sorted(preds["mean"])))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def summaries() -> None:
    out = {}
    # 1. technical reproducibility limits: bootstrap over perturbations; spread over batch partitions
    rows = []
    for d in DATASETS:
        z = np.load(EXT / f"{d}.npz", allow_pickle=True)
        prim = z["in_primary"]
        M = z["self_mask"][:, prim]
        ref = loo_reference(z["Y_fit_0"][:, prim], M)
        parts = sorted({k.split("_")[1] for k in z.files if k.startswith("Y_part")})
        meta = json.loads((EXT / f"{d}.json").read_text())
        for kind, pairs in (("batch", [(f"Y_{p}_A", f"Y_{p}_B") for p in parts]), ("cells", [("Y_half_a", "Y_half_b")])):
            for a, b in pairs:
                A, B = z[a][:, prim], z[b][:, prim]
                r = np.array([_corr((A[i] - ref[i])[~M[i]], (B[i] - ref[i])[~M[i]]) if np.isfinite(A[i]).all() and np.isfinite(B[i]).all() else np.nan for i in range(len(A))])
                sb = np.where(r > 0, 2 * r / (1 + r), 0.0)
                lim = np.sqrt(sb)
                m, lo, hi = bootstrap_ci(lim[np.isfinite(r)])
                rows.append({"dataset_id": d, "kind": kind, "partition": a, "mean_r": float(np.nanmean(r)), "limit": m, "limit_lo": lo, "limit_hi": hi,
                             "frac_reproducible": float(np.nanmean(r >= 0.5)), "n_batch_levels": meta["n_batch_levels"]})
    pd.DataFrame(rows).to_csv(EXT / "reproducibility_limits.csv", index=False)
    # 2. metric discrimination: duplicate vs mean and vs zero, separately, with Wilson intervals
    rows = []
    ph = pd.read_csv("results/paper/calibration_posthoc.csv")
    for d in DATASETS:
        cal = pd.read_csv(STUDY / d / "metric_calibration.csv")
        p = ph[ph["dataset_id"] == d]
        for m in ("mse", "pearson", "pearson_specific", "spearman_specific", "topk_recall"):
            for ref, col, frame in (("mean", f"{m}_duplicate_wins", cal), ("zero", f"{m}_beats_zero", p)):
                k, n = int(frame[col].sum()), int(frame[col].notna().sum())
                lo, hi = wilson(k, n)
                rows.append({"dataset_id": d, "metric": m, "uninformative": ref, "win_rate": k / n, "lo": lo, "hi": hi, "n": n})
    pd.DataFrame(rows).to_csv(EXT / "discrimination.csv", index=False)
    # discrimination tracks reliability: duplicate-vs-mean MSE win rate by reproducibility tertile
    rows = []
    for d in DATASETS:
        c = pd.read_csv(STUDY / d / "noise_ceiling.csv").groupby("condition")["specific_half_r"].mean()
        cal = pd.read_csv(STUDY / d / "metric_calibration.csv").set_index("condition")
        j = cal.join(c, how="inner")
        j["tertile"] = pd.qcut(j["specific_half_r"].rank(method="first"), 3, labels=["low", "middle", "high"])
        for t, g in j.groupby("tertile", observed=True):
            rows.append({"dataset_id": d, "reproducibility_tertile": t, "median_r": float(g["specific_half_r"].median()), "mse_win_rate": float(g["mse_duplicate_wins"].mean()), "n": len(g)})
    pd.DataFrame(rows).to_csv(EXT / "discrimination_by_reliability.csv", index=False)
    # 3. block bootstrap over co-expression modules of targets, and 4. cell-count stratification, 5. estimands
    f = pd.read_csv(STUDY / "per_perturbation_all.csv")
    f = f[(f["stratum"] == "single_unseen") & (f["split_seed"] == 0)]
    rows, cc_rows, est = [], [], []
    for d in DATASETS:
        prim = primary_data(d)
        pos = {g: i for i, g in enumerate(prim.genes)}
        tg = [g[0] for g in prim.condition_genes()]
        E = prim.gene_embedding[[pos[t] for t in tg]]
        k = int(max(5, round(np.sqrt(len(tg)))))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9))
        cluster = dict(zip(prim.conditions["condition"], labels))
        g = f[(f["dataset_id"] == d) & (f["variant"] == "coexpression")]
        base = g[g["model"] == "mean"].set_index("condition")["mse"]
        rng = np.random.default_rng(0)
        for m in ("ridge", "gene_loading_ridge", "mlp_ensemble", "knn"):
            a = g[g["model"] == m].set_index("condition")
            dd = (a["mse"] - base.reindex(a.index)) * 1e3
            sk = a["skill_vs_mean"]
            cl = np.array([cluster[c] for c in a.index])
            uniq = np.unique(cl)
            boots_d, boots_s = [], []
            for _ in range(2000):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([np.flatnonzero(cl == u) for u in pick])
                boots_d.append(dd.to_numpy()[idx].mean())
                boots_s.append(sk.to_numpy()[idx].mean())
            pb_d, pb_s = bootstrap_ci(dd.to_numpy()), bootstrap_ci(sk.to_numpy())
            rows.append({"dataset_id": d, "model": m, "n_clusters": k, "dmse": dd.mean(), "dmse_lo_perturbation": pb_d[1], "dmse_hi_perturbation": pb_d[2],
                         "dmse_lo_block": np.quantile(boots_d, 0.025), "dmse_hi_block": np.quantile(boots_d, 0.975), "skill": sk.mean(),
                         "skill_lo_perturbation": pb_s[1], "skill_hi_perturbation": pb_s[2], "skill_lo_block": np.quantile(boots_s, 0.025), "skill_hi_block": np.quantile(boots_s, 0.975)})
            # cell-count strata and weighting
            n = a["n_cells"]
            ter = pd.qcut(n.rank(method="first"), 3, labels=["fewest", "middle", "most"])
            for t in ("fewest", "middle", "most"):
                s_ = sk[ter == t]
                cc_rows.append({"dataset_id": d, "model": m, "stratum": t, "median_cells": float(n[ter == t].median()), "skill": s_.mean(), "n": int(len(s_))})
            se2 = None
            w = n / n.sum()
            cc_rows.append({"dataset_id": d, "model": m, "stratum": "cell-weighted", "median_cells": float(n.median()), "skill": float((sk * w).sum()), "n": int(len(sk))})
            big = sk[n >= 100]
            cc_rows.append({"dataset_id": d, "model": m, "stratum": ">=100 cells", "median_cells": float(n[n >= 100].median()) if len(big) else np.nan, "skill": big.mean(), "n": int(len(big))})
    pd.DataFrame(rows).to_csv(EXT / "block_bootstrap.csv", index=False)
    pd.DataFrame(cc_rows).to_csv(EXT / "cell_count.csv", index=False)
    # consistent estimands for C1-C3: mean and median differences with intervals, Wilcoxon, fraction favouring
    c = pd.read_csv(STUDY / "confirmatory_tests.csv")
    g_all = f.groupby(["dataset_id", "variant", "model", "condition"], as_index=False)[["mse", "skill_vs_mean"]].mean()
    rng = np.random.default_rng(0)
    for _, r in c.iterrows():
        g = g_all[g_all["dataset_id"] == r["dataset_id"]]
        if r["test"] == "C1":
            a = g[(g["variant"] == "coexpression") & (g["model"] == r["model"])].set_index("condition")["mse"] * 1e3
            b = g[(g["variant"] == "coexpression") & (g["model"] == "mean")].set_index("condition")["mse"].reindex(a.index) * 1e3
            unit = "dMSE x1e3"
        else:
            other = "shuffled_coexpression" if r["test"] == "C2" else "coexpression"
            first = "coexpression" if r["test"] == "C2" else "gene2vec"
            a = g[(g["variant"] == first) & (g["model"] == r["model"])].set_index("condition")["skill_vs_mean"]
            b = g[(g["variant"] == other) & (g["model"] == r["model"])].set_index("condition")["skill_vs_mean"].reindex(a.index)
            unit = "skill difference"
        dd = (a - b).dropna().to_numpy()
        meds = [np.median(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(2000)]
        mn = bootstrap_ci(dd)
        est.append({"test": r["test"], "dataset_id": r["dataset_id"], "model": r["model"], "unit": unit, "mean": mn[0], "mean_lo": mn[1], "mean_hi": mn[2],
                    "median": float(np.median(dd)), "median_lo": float(np.quantile(meds, 0.025)), "median_hi": float(np.quantile(meds, 0.975)),
                    "frac_favouring_first": float((dd < 0).mean() if r["test"] == "C1" else (dd > 0).mean()), "p_raw": r["p_raw"], "p_holm": r["p_holm"]})
    pd.DataFrame(est).to_csv(EXT / "estimands.csv", index=False)
    print("summaries written")


def model_summaries() -> None:
    """Aggregate run1-run3 across datasets: gene subsets, control splits, seeds, intervals, set-B reproducibility, ranking."""
    frames = [pd.read_csv(p) for p in sorted(EXT.glob("models_*.csv"))]
    if not frames:
        return
    m = pd.concat(frames, ignore_index=True)
    rows = []
    ext = m[m["run"] == "extended"]
    for (d, model), g in ext.groupby(["dataset_id", "model"]):
        for s in ("primary", "extended", "outside_primary", "silent_in_controls"):
            if g[f"{s}_skill_vs_mean"].notna().sum() == 0:
                rows.append({"dataset_id": d, "model": model, "genes": s, "skill": np.nan, "lo": np.nan, "hi": np.nan, "r_specific": np.nan})
                continue
            sk, ps = bootstrap_ci(g[f"{s}_skill_vs_mean"].to_numpy()), g[f"{s}_pearson_specific"].mean()
            rows.append({"dataset_id": d, "model": model, "genes": s, "skill": sk[0], "lo": sk[1], "hi": sk[2], "r_specific": ps})
    pd.DataFrame(rows).to_csv(EXT / "gene_sets.csv", index=False)
    var_rows = []
    for (d, model, run), g in m[m["run"] != "extended"].groupby(["dataset_id", "model", "run"]):
        var_rows.append({"dataset_id": d, "model": model, "run": run, "skill": g["primary_skill_vs_mean"].mean(), "r_specific": g["primary_pearson_specific"].mean()})
    for (d, model), g in ext.groupby(["dataset_id", "model"]):
        if model in ("ridge", "mlp_ensemble"):
            var_rows.append({"dataset_id": d, "model": model, "run": "primary_split_seed0", "skill": g["primary_skill_vs_mean"].mean(), "r_specific": g["primary_pearson_specific"].mean()})
    pd.DataFrame(var_rows).to_csv(EXT / "variability.csv", index=False)
    iv = ext[ext["model"].isin(["ridge", "mlp_ensemble"])].groupby(["dataset_id", "model"])[["cov_all", "cov_top_predicted", "cov_top_observed"]].mean().reset_index()
    iv.to_csv(EXT / "intervals_selection.csv", index=False)
    # selection-independent reproducible subset: select on set-A halves, evaluate on set B
    rep_rows = []
    for d in ext["dataset_id"].unique():
        z = np.load(EXT / f"{d}.npz", allow_pickle=True)
        if "Y_Ahalf_0" not in z or "setB_skill_vs_mean" not in ext:
            continue
        prim = z["in_primary"]
        M = z["self_mask"][:, prim]
        ref = loo_reference(z["Y_fit_0"][:, prim], M)
        A0, A1 = z["Y_Ahalf_0"][:, prim], z["Y_Ahalf_1"][:, prim]
        conds = json.loads((EXT / f"{d}.json").read_text())["conditions"]
        rA = pd.Series([_corr((A0[i] - ref[i])[~M[i]], (A1[i] - ref[i])[~M[i]]) if np.isfinite(A0[i]).all() and np.isfinite(A1[i]).all() else np.nan for i in range(len(A0))], index=conds)
        g = ext[ext["dataset_id"] == d].copy()
        g["rA"] = g["condition"].map(rA)
        for model, h in g.groupby("model"):
            for label, sel in (("all", h["rA"].notna()), ("reproducible in set A (r>=0.5)", h["rA"] >= 0.5)):
                hh = h[sel & h["setB_skill_vs_mean"].notna()]
                ci = bootstrap_ci(hh["setB_skill_vs_mean"].to_numpy())
                rep_rows.append({"dataset_id": d, "model": model, "subset": label, "n": len(hh), "skill_setB": ci[0], "lo": ci[1], "hi": ci[2], "r_specific_setB": hh["setB_pearson_specific"].mean()})
    pd.DataFrame(rep_rows).to_csv(EXT / "reproducible_independent.csv", index=False)
    # ranking: precision at several fractions with bootstrap intervals and a permutation test; MLP seeds
    rk = []
    for d in ext["dataset_id"].unique():
        z = np.load(EXT / f"{d}.npz", allow_pickle=True)
        P = np.load(EXT / f"preds_{d}.npz")
        prim = z["in_primary"]
        order = P["order"]
        M = z["self_mask"][order][:, prim]
        y = response_magnitude(z["Y_eval_0"][order][:, prim], M)
        singles = np.isin(order, [i for i, c in enumerate(json.loads((EXT / f"{d}.json").read_text())["conditions"])])
        rng = np.random.default_rng(0)
        for name in [k for k in P.files if k not in ("order",)]:
            s = response_magnitude(P[name], M)
            for frac in (0.1, 0.2, 0.3):
                k = max(1, int(round(frac * len(y))))
                true_top = set(np.argsort(-y)[:k])
                prec = len(true_top & set(np.argsort(-s)[:k])) / k
                null = [len(true_top & set(rng.permutation(len(y))[:k])) / k for _ in range(2000)]
                boots = []
                for _ in range(1000):
                    ix = rng.integers(0, len(y), len(y))
                    tt = set(np.argsort(-y[ix])[:k])
                    boots.append(len(tt & set(np.argsort(-s[ix])[:k])) / k)
                rk.append({"dataset_id": d, "ranking": name, "top_fraction": frac, "k": k, "precision": prec, "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975),
                           "enrichment": prec / frac, "p_vs_random": float((np.array(null) >= prec).mean() + 1 / 2001)})
    pd.DataFrame(rk).to_csv(EXT / "ranking.csv", index=False)
    print("model summaries written")


if __name__ == "__main__":
    if sys.argv[1] == "models":
        for d in sys.argv[2:]:
            run_models(d)
    elif sys.argv[1] == "summaries":
        summaries()
        model_summaries()
