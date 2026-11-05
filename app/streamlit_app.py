"""Local research interface. Run: streamlit run app/streamlit_app.py

Every action calls the real package code on real files; nothing on these pages is simulated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from biophysical.config import get_paths
from biophysical.data.registry import DatasetRegistry
from biophysical.visualization import plots

st.set_page_config(page_title="Perturbation response", layout="wide")
PATHS = get_paths()
DEFAULT_CFG = "configs/benchmark_norman.yaml"


def csv_button(df: pd.DataFrame, label: str, name: str, container=st) -> None:
    container.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv")


def list_runs() -> list[Path]:
    runs = [p.parent for p in PATHS.results.glob("*/*/run.json")]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource(show_spinner="Loading models")
def load_service(run_dir: str):
    from biophysical.api.service import PredictionService

    return PredictionService(run_dir)


st.title("Perturbation response prediction")
st.caption("Predicts pseudobulk gene-expression change after CRISPRa in K562 cells. It is not a cell simulator, and predictions have not been tested prospectively.")

runs = list_runs()
with st.sidebar:
    st.subheader("Benchmark run")
    if runs:
        labels = [f"{p.name}" for p in runs]
        run_dir = runs[st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i])]
        info = json.loads((run_dir / "run.json").read_text())
        if info.get("quick"):
            st.warning("This is a quick smoke-test run. Its results are not benchmark evidence.")
        st.caption(f"Data version {info['processed_version']}, seed {info['seed']}, {info['seconds']:.0f} s, peak {info['peak_rss_mb']:.0f} MB")
    else:
        run_dir, info = None, None
        st.info("No completed runs yet. Use the Train tab.")

tabs = st.tabs(["Datasets", "Train", "Benchmark", "Predict", "Rank experiments", "Observations"])

with tabs[0]:
    reg = DatasetRegistry()
    table = pd.DataFrame([{"id": e.dataset_id, "status": e.current_status(reg.paths), "size_MB": e.size_mb, "cell context": e.cell_context, "perturbation types": ", ".join(e.perturbation_types), "accession": e.accession, "role": e.role} for e in reg.list()])
    st.dataframe(table, use_container_width=True, hide_index=True)
    ds_id = st.selectbox("Dataset", table["id"].tolist())
    entry = reg.get(ds_id)
    with st.expander("Registry record"):
        st.json({k: getattr(entry, k) for k in ["name", "publication", "source_url", "accession", "doi", "license", "organism", "assay", "cell_context", "md5", "download_instructions"]})
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Validate"):
        from biophysical.data.validate import validate_dataset

        with st.spinner("Validating (includes checksum)"):
            rec = validate_dataset(entry, paths=reg.paths)
        st.write(f"Status: **{rec['status']}**")
        st.dataframe(pd.DataFrame(rec["checks"]), use_container_width=True, hide_index=True)
    if c2.button("Inspect schema"):
        from biophysical.data.inspect import inspect_dataset

        try:
            with st.spinner("Reading metadata"):
                rep = inspect_dataset(entry, paths=reg.paths)
            s = rep["schema"]
            st.write({k: s[k] for k in ["n_obs", "n_vars", "X_encoding", "X_nnz", "replicate_like_columns"]})
            st.json(s.get("perturbation_summary", {}))
            st.dataframe(pd.DataFrame([{"column": c, **{k: d.get(k) for k in ["dtype", "n_unique", "n_missing"]}} for c, d in s["obs_columns"].items()]), hide_index=True, use_container_width=True)
        except FileNotFoundError as exc:
            st.error(str(exc))
    allow_large = c3.checkbox(f"Allow over {reg.defaults.get('max_default_download_mb')} MB")
    if c3.button("Download"):
        from biophysical.data.download import DownloadError, download_dataset

        try:
            with st.spinner(f"Downloading {entry.size_mb} MB; interrupted downloads resume"):
                st.success(str(download_dataset(entry, allow_large=allow_large, paths=reg.paths)))
        except DownloadError as exc:
            st.error(str(exc))
    if c4.button("Preprocess"):
        from biophysical.preprocessing.pipeline import preprocess_dataset

        cfg = entry.preprocessing_config
        try:
            with st.spinner("Preprocessing"):
                st.success(str(preprocess_dataset(ds_id, config_path=cfg)))
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
    st.divider()
    st.subheader("Use a local file")
    local = st.text_input("Path to an H5AD file for this dataset id")
    if st.button("Register local path") and local:
        from biophysical.cli import main as cli_main

        code = cli_main(["datasets", "set-local", ds_id, local])
        st.success("Registered") if code == 0 else st.error("That path does not exist")

with tabs[1]:
    st.write("Runs the full pipeline: preprocessing, splits, all models, calibration, acquisition simulation, and a serving model.")
    cfg_path = st.text_input("Config", DEFAULT_CFG)
    quick = st.checkbox("Quick smoke test (not benchmark evidence)", value=True)
    if st.button("Run benchmark"):
        from biophysical.evaluation.benchmark import run_benchmark

        with st.spinner("Training and evaluating. The full run with ablations takes a few minutes."):
            out = run_benchmark(cfg_path, quick=quick)
        st.success(f"Finished: {out}")
        st.cache_resource.clear()
        st.rerun()

with tabs[2]:
    if run_dir is None:
        st.info("No run selected.")
    else:
        summary = pd.read_csv(run_dir / "summary.csv", keep_default_na=False, na_values=[""])
        frame = pd.read_csv(run_dir / "per_perturbation_metrics.csv")
        ceil = pd.read_csv(run_dir / "noise_ceiling.csv")
        c1, c2, c3 = st.columns(3)
        strata = sorted(summary["stratum"].unique(), key=lambda x: (x != "single_unseen", x != "combo_seen_singles", x))
        stratum = c1.selectbox("Test stratum", strata)
        subset = c2.selectbox("Perturbations", ["all", "reproducible"], format_func=lambda s: {"all": "All", "reproducible": "Reproducible specific response"}[s], help="Specific response correlates at r ≥ 0.5 between disjoint batch sets")
        metric = c3.selectbox("Metric", list(plots.LABELS), format_func=lambda m: plots.LABELS[m])
        s = summary[(summary["stratum"] == stratum) & (summary["subset"] == subset)]
        ceiling = None
        cc = ceil[ceil["stratum"] == stratum]
        if metric == "pearson_specific":
            ceiling = float(cc["specific_batch_max_r"].mean())
        elif metric == "pearson":
            ceiling = float(cc["max_r"].mean())
        st.altair_chart(plots.model_comparison(s, metric, ceiling), use_container_width=True)
        st.caption("Points are means over perturbations, with 95% bootstrap intervals. A model with an undefined metric is left out: the zero model has no raw correlation, and the mean model has no perturbation-specific correlation. The dashed line marks the estimated reproducibility limit where one applies.")
        with st.expander("Table"):
            view = s[s["metric"] == metric].sort_values("mean", ascending=False)
            st.dataframe(view, hide_index=True, use_container_width=True)
            csv_button(summary, "Download full summary", "summary.csv")
        st.subheader("Every perturbation")
        f = frame[frame["stratum"] == stratum]
        models = sorted(f["model"].unique())
        default_models = [m for m in models if m not in ("zero",)][:6]
        chosen = st.multiselect("Models", models, default=default_models)
        st.altair_chart(plots.per_perturbation_strip(f[f["model"].isin(chosen)], metric), use_container_width=True)
        csv_button(frame, "Download per-perturbation metrics", "per_perturbation_metrics.csv")
        with st.expander("Paired comparisons with baselines"):
            st.dataframe(pd.read_csv(run_dir / "paired_comparisons.csv").query("stratum == @stratum"), hide_index=True, use_container_width=True)
        st.subheader("Uncertainty")
        u = pd.read_csv(run_dir / "uncertainty.csv").query("stratum == @stratum")
        st.dataframe(u, hide_index=True, use_container_width=True)
        with_dis = f[f["disagreement"].notna()] if "disagreement" in f else f.iloc[0:0]
        if len(with_dis):
            ens = st.selectbox("Ensemble", sorted(with_dis["model"].unique()))
            st.altair_chart(plots.uncertainty_scatter(with_dis[with_dis["model"] == ens]), use_container_width=True)
        if (run_dir / "acquisition_curves.csv").exists():
            st.subheader("Retrospective experiment selection")
            curves = pd.read_csv(run_dir / "acquisition_curves.csv")
            acq = pd.read_csv(run_dir / "acquisition_summary.csv")
            c1, c2 = st.columns(2)
            pool = c1.selectbox("Pool", sorted(curves["pool"].unique()) if "pool" in curves else ["singles"])
            am = c2.selectbox("Acquisition metric", ["skill_vs_mean", "pearson_specific", "mse"], format_func=lambda m: plots.LABELS[m])
            cv = curves[curves["pool"] == pool] if "pool" in curves else curves
            st.altair_chart(plots.acquisition_curves(cv, am), use_container_width=True)
            st.dataframe(acq[(acq["pool"] == pool) & (acq["metric"] == am)] if "pool" in acq else acq, hide_index=True, use_container_width=True)
            if (run_dir / "prioritisation.csv").exists():
                st.subheader("Finding the top perturbations, out of sample")
                st.caption("Precision among the top-ranked 20%. Random selection would have an expected precision of 0.20.")
                st.dataframe(pd.read_csv(run_dir / "prioritisation.csv"), hide_index=True, use_container_width=True)
        if (run_dir / "ablations.csv").exists():
            with st.expander("Ablations"):
                st.dataframe(pd.read_csv(run_dir / "ablations.csv"), hide_index=True, use_container_width=True)

with tabs[3]:
    if run_dir is None:
        st.info("No run selected.")
    else:
        svc = load_service(str(run_dir))
        c1, c2, c3 = st.columns(3)
        text = c1.text_input("Target gene or pair (e.g. KLF1 or CEBPE_KLF1)", "KLF1")
        ptype = c2.text_input("Perturbation type", svc.perturbation_type)
        ctx = c3.text_input("Cell context", svc.cell_context)
        unvalidated = st.checkbox("Show predictions the benchmark found worse than baseline")
        res = svc.predict(text, ptype, ctx, top_k=25, allow_unvalidated=unvalidated)
        if res["status"] == "abstain":
            st.error("No prediction returned: " + "; ".join(res["warnings"]))
        else:
            for w in res["warnings"]:
                st.warning(w)
            st.write(f"**Support:** {res['support_level'].replace('_', ' ')}. **Nearest measured condition:** {res['nearest_measured_condition']}.")
            m1, m2 = st.columns(2)
            m1.metric("Predicted RMS response", f"{res['predicted_magnitude']:.3f}")
            m2.metric("Model disagreement", f"{res['uncertainty']['model_disagreement_rms']:.3f}")
            top = pd.DataFrame(res["top_genes"])
            st.altair_chart(plots.predicted_response(top), use_container_width=True)
            st.caption(f"Rules show {res['uncertainty']['interval_level']:.0%} intervals. {res['uncertainty']['interval_calibration']}. {res['uncertainty']['not_included']}")
            with st.expander("Table"):
                st.dataframe(top, hide_index=True, use_container_width=True)
            export = {k: v for k, v in res.items() if k != "predicted_response"}
            c1, c2 = st.columns(2)
            csv_button(res["predicted_response"], "Download all genes (CSV)", f"prediction_{text}.csv", c1)
            c2.download_button("Download record (JSON)", json.dumps(export, indent=2, default=str), f"prediction_{text}.json")
            with st.expander("Provenance and limitations"):
                st.json({k: res[k] for k in ["model_version", "dataset", "preprocessing_version", "source_md5", "input", "known_limitations"]})

with tabs[4]:
    if run_dir is None:
        st.info("No run selected.")
    else:
        svc = load_service(str(run_dir))
        st.write("Scores here are documented heuristics. See the retrospective comparison on the Benchmark tab before relying on any strategy.")
        text = st.text_area("Candidates, one per line. Leave empty to use every unmeasured pair of measured genes.", height=110)
        c1, c2 = st.columns(2)
        budget = c1.number_input("Budget", 1, 200, 10)
        strategy = c2.selectbox("Strategy", ["hybrid", "uncertainty", "diversity", "magnitude", "expected_improvement", "random"])
        if st.button("Rank"):
            cands = [l.strip() for l in text.splitlines() if l.strip()] or None
            with st.spinner("Scoring candidates"):
                t = svc.rank(cands, int(budget), strategy, record=True)
            st.dataframe(t, hide_index=True, use_container_width=True)
            csv_button(t, "Download ranking", "ranking.csv")
            st.caption("Recorded in recommendations.jsonl with the model version.")

with tabs[5]:
    if run_dir is None:
        st.info("No run selected.")
    else:
        svc = load_service(str(run_dir))
        st.write("Upload the results of completed experiments: a CSV with a `condition` column and one column per gene, on the same scale as the target (difference in mean log1p CP10k versus control). Benchmark results are never altered.")
        up = st.file_uploader("Observed responses", type="csv")
        source = st.text_input("Source of these data")
        if up is not None and source and st.button("Register observations"):
            try:
                st.success(svc.add_observations(pd.read_csv(up), source))
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Refit with registered observations"):
            try:
                with st.spinner("Refitting"):
                    new = svc.refit_with_observations()
                st.success(f"New model {new.model_version}. Its intervals are carried over, so run a new benchmark before relying on them.")
            except ValueError as exc:
                st.error(str(exc))
