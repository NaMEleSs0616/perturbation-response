"""Interactive result charts (Altair, which ships with Streamlit). Each chart is drawn from real run
outputs, carries hover tooltips, and sits beside a table view in the app."""
from __future__ import annotations

import altair as alt
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# mid-tone neutrals stay legible on both the light and dark Streamlit themes
INK, MUTED, GRID = "#8b8a85", "#8b8a85", "#8b8a85"
STRATEGY_COLORS = {"random": BLUE, "uncertainty": ORANGE, "diversity": AQUA, "hybrid": YELLOW}
LABELS = {
    "skill_vs_mean": "Skill vs mean baseline (1 − SSE/SSE mean)",
    "pearson_specific": "Pearson r, perturbation-specific response",
    "pearson_specific_top": "Pearson r, top 20 specific genes",
    "pearson_interaction": "Pearson r, non-additive component",
    "pearson_interaction_top": "Pearson r, top 20 non-additive genes",
    "skill_vs_additive": "Skill vs additive baseline",
    "pearson": "Pearson r, raw response",
    "pearson_top": "Pearson r, top 20 genes",
    "spearman": "Spearman r, raw response",
    "mae": "Mean absolute error",
    "mse": "Mean squared error",
    "mae_top": "MAE, top 20 genes",
    "sign_acc_top": "Direction accuracy, top 20 genes",
    "topk_recall": "Recall of top 20 genes",
    "r2_vs_zero": "R² vs zero response",
}


def _base(chart: alt.Chart) -> alt.Chart:
    return chart.configure_axis(gridColor=GRID, gridOpacity=0.2, domainColor=MUTED, tickColor=MUTED, labelColor=INK, titleColor=INK, labelLimit=240).configure_view(strokeWidth=0).configure_legend(labelColor=INK, titleColor=INK)


def model_comparison(summary: pd.DataFrame, metric: str, ceiling: float | None = None) -> alt.Chart:
    """Means with bootstrap intervals, one row per model. Undefined metrics are left out, not zeroed."""
    d = summary[(summary["metric"] == metric) & summary["mean"].notna()].sort_values("mean", ascending=metric in ("mae", "mse", "mae_top"))
    order = d["model"].tolist()
    y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False))
    tip = [alt.Tooltip("model:N", title="Model"), alt.Tooltip("mean:Q", format=".3f", title="Mean"), alt.Tooltip("ci_low:Q", format=".3f", title="95% CI low"),
           alt.Tooltip("ci_high:Q", format=".3f", title="95% CI high"), alt.Tooltip("median:Q", format=".3f", title="Median"), alt.Tooltip("n_perturbations:Q", title="Perturbations")]
    rule = alt.Chart(d).mark_rule(color=BLUE, strokeWidth=2).encode(x=alt.X("ci_low:Q", title=LABELS.get(metric, metric), scale=alt.Scale(zero=False)), x2="ci_high:Q", y=y, tooltip=tip)
    dot = alt.Chart(d).mark_circle(color=BLUE, size=110, opacity=1).encode(x="mean:Q", y=y, tooltip=tip)
    layers = [rule, dot]
    if ceiling is not None:
        c = pd.DataFrame({"x": [ceiling], "label": [f"reproducibility limit ≈ {ceiling:.2f}"]})
        layers.append(alt.Chart(c).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q", tooltip=[alt.Tooltip("label:N", title="Noise ceiling")]))
    return _base(alt.layer(*layers).properties(height=max(120, 30 * len(order)), width="container"))


def per_perturbation_strip(frame: pd.DataFrame, metric: str) -> alt.Chart:
    """Every perturbation's score, so aggregate means cannot hide failures."""
    d = frame.dropna(subset=[metric])
    order = d.groupby("model")[metric].mean().sort_values(ascending=metric in ("mae", "mse", "mae_top")).index.tolist()
    chart = alt.Chart(d).mark_circle(color=BLUE, size=70, opacity=0.55).encode(
        x=alt.X(f"{metric}:Q", title=LABELS.get(metric, metric), scale=alt.Scale(zero=False)),
        y=alt.Y("model:N", sort=order, title=None),
        yOffset=alt.YOffset("jitter:Q"),
        tooltip=[alt.Tooltip("condition:N", title="Perturbation"), alt.Tooltip("model:N", title="Model"), alt.Tooltip(f"{metric}:Q", format=".3f", title=LABELS.get(metric, metric)), alt.Tooltip("n_cells:Q", format=".0f", title="Cells")],
    ).transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())*5")
    return _base(chart.properties(height=max(150, 44 * len(order)), width="container"))


def predicted_response(top: pd.DataFrame) -> alt.Chart:
    """Top predicted genes. Position shows direction, and the rule shows the calibrated interval."""
    d = top.copy()
    order = d["gene"].tolist()
    y = alt.Y("gene:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False))
    tip = [alt.Tooltip("gene:N", title="Gene"), alt.Tooltip("predicted_response:Q", format=".3f", title="Predicted"), alt.Tooltip("interval_low:Q", format=".3f", title="Interval low"),
           alt.Tooltip("interval_high:Q", format=".3f", title="Interval high"), alt.Tooltip("ensemble_sd:Q", format=".3f", title="Ensemble SD")]
    if "observed_response" in d:
        tip.append(alt.Tooltip("observed_response:Q", format=".3f", title="Observed"))
    interval = alt.Chart(d).mark_rule(color=MUTED, strokeWidth=2).encode(x=alt.X("interval_low:Q", title="Change in log1p expression vs control"), x2="interval_high:Q", y=y, tooltip=tip)
    dot = alt.Chart(d).mark_circle(color=BLUE, size=110, opacity=1).encode(x="predicted_response:Q", y=y, tooltip=tip)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK).encode(x="x:Q")
    return _base(alt.layer(zero, interval, dot).properties(height=max(160, 24 * len(order)), width="container"))


def acquisition_curves(curves: pd.DataFrame, metric: str) -> alt.Chart:
    """Mean over repeats; the same seed set starts every strategy within a repeat."""
    d = curves.groupby(["strategy", "n_acquired"], as_index=False)[metric].mean()
    strategies = [s for s in STRATEGY_COLORS if s in set(d["strategy"])]
    color = alt.Color("strategy:N", title="Strategy", scale=alt.Scale(domain=strategies, range=[STRATEGY_COLORS[s] for s in strategies]))
    tip = [alt.Tooltip("strategy:N", title="Strategy"), alt.Tooltip("n_acquired:Q", title="Perturbations acquired"), alt.Tooltip(f"{metric}:Q", format=".3f", title=LABELS.get(metric, metric))]
    x = alt.X("n_acquired:Q", title="Perturbations acquired", scale=alt.Scale(zero=False))
    yy = alt.Y(f"{metric}:Q", title=LABELS.get(metric, metric), scale=alt.Scale(zero=False))
    line = alt.Chart(d).mark_line(strokeWidth=2).encode(x=x, y=yy, color=color)
    pts = alt.Chart(d).mark_circle(size=70, opacity=1).encode(x=x, y=yy, color=color, shape=alt.Shape("strategy:N", legend=None), tooltip=tip)
    return _base(alt.layer(line, pts).properties(height=320, width="container"))


def uncertainty_scatter(frame: pd.DataFrame) -> alt.Chart:
    d = frame.dropna(subset=["disagreement"])
    chart = alt.Chart(d).mark_circle(color=BLUE, size=70, opacity=0.6).encode(
        x=alt.X("disagreement:Q", title="Ensemble disagreement", scale=alt.Scale(zero=False)), y=alt.Y("rmse:Q", title="Observed RMSE", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("condition:N", title="Perturbation"), alt.Tooltip("disagreement:Q", format=".4f", title="Disagreement"), alt.Tooltip("rmse:Q", format=".4f", title="RMSE"), alt.Tooltip("coverage:Q", format=".2f", title="Interval coverage")])
    return _base(chart.properties(height=320, width="container"))
