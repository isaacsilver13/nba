"""
NBA research dashboard (historical research results only; not live picks).
Run with: streamlit run dashboard.py
Select one run bundle; every number comes from that run's manifest-declared artifacts.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.research_dashboard.catalog import list_runs, outputs_dir
from src.research_dashboard.manifest import ODDS_TIERS
from src.research_dashboard.views import (
    bet_summary,
    chart_gate_message,
    comparison_table,
    filter_tiers,
    reliability,
)

st.set_page_config(page_title="NBA Research Dashboard", page_icon="🏀", layout="wide")

BASE_DIR = Path(__file__).parent
OUTPUTS = outputs_dir(BASE_DIR)


@st.cache_data
def _runs(outputs: str, _stamp: float):
    return list_runs(Path(outputs))


def _stamp() -> float:
    ms = list(OUTPUTS.glob("research_runs/*/manifest.json"))
    return max((m.stat().st_mtime for m in ms), default=0.0)


@st.cache_data
def load_tables(run_dir: str, mtime: float) -> dict:
    from src.research_dashboard.manifest import load_manifest
    run = load_manifest(run_dir)
    return {n: run.read(n) for n in ("fold_metrics", "predictions", "bets", "odds_audit")}


@st.cache_data
def load_results() -> pd.DataFrame:
    """Experiment log (results.tsv). Defensive: empty file or malformed params must not crash."""
    try:
        df = pd.read_csv(BASE_DIR / "results.tsv", sep="\t", parse_dates=["timestamp"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    def parse(s):
        try:
            return json.loads(s)
        except (TypeError, ValueError):
            return {}

    params = df["params"].apply(parse).apply(pd.Series)
    df = pd.concat([df.drop(columns=["params"]), params], axis=1)
    if "models" in df.columns:
        df["models_str"] = df["models"].apply(lambda x: "+".join(x) if isinstance(x, list) else str(x))
    return df


# ── Global controls ───────────────────────────────────────────────────────────
st.title("🏀 NBA Research Dashboard")
st.caption("Historical research results. Not betting advice or live picks.")

if st.sidebar.button("Refresh"):
    st.cache_data.clear()

runs = _runs(str(OUTPUTS), _stamp())
t_health, t_compare, t_model, t_odds, t_bets, t_log = st.tabs([
    "Run Health", "Run Comparison", "Model Quality", "Odds Quality", "Betting Simulation", "Experiment Log"
])

run = None
if runs:
    run = st.sidebar.selectbox("Run", runs, format_func=lambda r: f"{r.run_id} [{r.status}]")
else:
    st.sidebar.info(f"No run bundles under {OUTPUTS / 'research_runs'}. See README (capture_run).")

ok = run is not None and run.status == "complete"
T = load_tables(str(run.run_dir), (run.run_dir / "manifest.json").stat().st_mtime) if ok else {}
gate_message = chart_gate_message(run.run_id if run else None, run.status if run else None)


# ── Run Health ────────────────────────────────────────────────────────────────
with t_health:
    if run is None:
        st.info("No run bundles found.")
    else:
        c = st.columns(3)
        c[0].metric("Status", run.status)
        c[1].metric("Created", run.data.get("created_at", "?"))
        c[2].metric("Folds", len(run.data.get("folds", [])))
        if run.status == "complete":
            st.success("Tables and charts are available in Model Quality, Odds Quality, and Betting Simulation.")
        for p in run.problems:
            st.warning(p)
        if run.data.get("folds"):
            st.subheader("Fold test windows")
            st.dataframe(pd.DataFrame(run.data["folds"]), use_container_width=True)
        st.subheader("Artifacts")
        st.dataframe(pd.DataFrame(run.data.get("artifacts", {})).T, use_container_width=True)
        st.download_button("Download manifest.json", json.dumps(run.data, indent=2), f"{run.run_id}_manifest.json")

with t_compare:
    complete_runs = [r for r in runs if r.status == "complete"]
    run_ids = [r.run_id for r in complete_runs]
    selected_run_ids = st.multiselect("Runs to compare", run_ids, default=run_ids)
    comparison_tiers = st.multiselect(
        "Odds quality for betting columns",
        ODDS_TIERS,
        default=["matched"],
        key="comparison_odds_tiers",
        help="Only columns and charts labeled selected-tier use this filter. Full-run metrics never change with it.",
    )
    st.caption(
        "Historical runs may span code, data, sign, calibration, or odds changes. Compare only runs with compatible "
        "provenance; duplicate and reproducibility runs remain visible."
    )
    if not selected_run_ids:
        st.info("Select at least one complete run.")
    elif not comparison_tiers:
        st.info("Select at least one odds-quality tier for betting comparisons.")
    else:
        selected_runs = [r for r in complete_runs if r.run_id in selected_run_ids]
        comparison = comparison_table(selected_runs, comparison_tiers)
        fingerprint_columns = ["source_fingerprint", "config_fingerprint", "data_fingerprint"]
        missing_fingerprints = {c: int(comparison[c].isna().sum()) for c in fingerprint_columns}
        if any(missing_fingerprints.values()):
            missing_text = ", ".join(
                f"{name.removesuffix('_fingerprint')}={count}"
                for name, count in missing_fingerprints.items()
                if count
            )
            st.warning(
                f"Incomplete provenance fingerprints among selected runs ({missing_text}). "
                "Cross-run ranking is exploratory unless source, config, and data fingerprints all match."
            )
        display_columns = [
            "run_id", "created_at", "experiment_id", "source_fingerprint", "config_fingerprint",
            "data_fingerprint", "run_model_rmse", "run_flat_roi", "run_flat_roi_se", "run_score",
            "run_compound_return", "run_bets", "run_max_drawdown", "tier_bets", "tier_win_rate",
            "tier_roi", "tier_coverage", "matched_odds_share", "folds", "data_rows",
        ]
        st.dataframe(
            comparison[display_columns].style.format({
                "run_model_rmse": "{:.3f}",
                "run_flat_roi": "{:.1%}",
                "run_flat_roi_se": "{:.1%}",
                "run_score": "{:.3f}",
                "run_compound_return": "{:.1%}",
                "run_max_drawdown": "{:.1%}",
                "tier_win_rate": "{:.1%}",
                "tier_roi": "{:.1%}",
                "tier_coverage": "{:.1%}",
                "matched_odds_share": "{:.1%}",
            }, na_rep="—"),
            width="stretch",
            hide_index=True,
        )

        rmse = comparison.dropna(subset=["run_model_rmse"])
        if not rmse.empty:
            st.subheader("Full-run model error (lower is better)")
            st.plotly_chart(
                px.bar(rmse, x="run_id", y="run_model_rmse", hover_data=["created_at", "folds", "data_rows"]),
                width="stretch",
            )

        flat = comparison.dropna(subset=["run_flat_roi", "run_flat_roi_se"])
        if flat.empty:
            st.info("These runs do not expose flat-stake ROI and standard error for an uncertainty chart.")
        else:
            st.subheader("Full-run flat-stake ROI with standard error")
            flat_fig = px.bar(
                flat,
                x="run_id",
                y="run_flat_roi",
                error_y="run_flat_roi_se",
                hover_data=["run_score", "run_bets"],
            )
            flat_fig.add_hline(y=0, line_dash="dash")
            flat_fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(flat_fig, width="stretch")

        tier_returns = comparison.dropna(subset=["tier_roi"])
        if not tier_returns.empty:
            st.subheader(f"Selected-tier ROI ({', '.join(comparison_tiers)})")
            tier_fig = px.bar(
                tier_returns,
                x="run_id",
                y="tier_roi",
                hover_data=["tier_bets", "tier_win_rate", "tier_coverage", "matched_odds_share"],
            )
            tier_fig.add_hline(y=0, line_dash="dash")
            tier_fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(tier_fig, width="stretch")

        tradeoff = comparison.dropna(subset=["run_compound_return", "run_max_drawdown"])
        if not tradeoff.empty:
            st.subheader("Full-run compound return versus max drawdown")
            tradeoff_fig = px.scatter(
                tradeoff,
                x="run_max_drawdown",
                y="run_compound_return",
                size="run_bets",
                color="run_model_rmse",
                hover_name="run_id",
                hover_data=["run_flat_roi", "run_score", "matched_odds_share"],
            )
            tradeoff_fig.update_xaxes(tickformat=".1%")
            tradeoff_fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(tradeoff_fig, width="stretch")

if gate_message:
    for tab in (t_model, t_odds, t_bets):
        with tab:
            st.warning(gate_message)
else:
    pred, folds, bets, audit = T["predictions"], T["fold_metrics"], T["bets"], T["odds_audit"]
    d0, d1 = pd.to_datetime(pred["game_date"]).min().date(), pd.to_datetime(pred["game_date"]).max().date()
    dr = st.sidebar.date_input("Game date range", (d0, d1), min_value=d0, max_value=d1)
    if len(dr) == 2:
        lo, hi = (pd.Timestamp(x) for x in dr)
        in_range = lambda df: df[pd.to_datetime(df["game_date"]).between(lo, hi)]
        pred, bets, audit = in_range(pred), in_range(bets), in_range(audit)
    tiers = st.sidebar.multiselect("Odds quality", ODDS_TIERS, default=["matched"],
                                   help="Mirrored/default odds are diagnostics, not market-realistic evidence.")
    bets_f = filter_tiers(bets, tiers)
    audit_f = filter_tiers(audit, tiers)
    pred_f = pred[pred["game_id"].isin(audit_f["game_id"])]

    with t_model:
        st.subheader("Fold metrics")
        st.dataframe(folds, use_container_width=True)
        st.plotly_chart(px.line(folds, x="fold", y=["rmse", "mae"], markers=True), use_container_width=True)
        st.subheader(f"Calibration reliability (all test games in selected odds tiers: {len(pred_f):,})")
        rel = reliability(pred_f)
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Perfect")])
        fig.add_trace(go.Scatter(x=rel["mean_pred"], y=rel["actual"], mode="markers+lines",
                                 text=rel["n"].map("n={}".format), name="Model"))
        fig.update_layout(xaxis_title="Predicted P(home covers)", yaxis_title="Observed home cover rate")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bins with small n are noisy; see bin counts in hover.")
        st.dataframe(rel, use_container_width=True)

    with t_odds:
        counts = pd.Series(run.data["odds_counts"]).rename("games").to_frame()
        counts["share"] = counts["games"] / counts["games"].sum()
        st.dataframe(counts.style.format({"share": "{:.1%}"}), use_container_width=True)
        st.caption("Tier filter in the sidebar applies to the other pages; default is matched-only.")
        by = bets.groupby("odds_tier").agg(bets=("pnl", "size"), pnl=("pnl", "sum")).reset_index()
        st.subheader("Bets by odds tier (unfiltered)")
        st.dataframe(by, use_container_width=True)

    with t_bets:
        if bets_f.empty:
            st.info("No bets in the selected range and odds tiers.")
        else:
            s = bet_summary(bets_f)
            all_tiers = bet_summary(bets)
            c = st.columns(5)
            c[0].metric("Bets", s["bets"])
            c[1].metric("Win rate", f"{s['win_rate']:.1%}")
            c[2].metric("ROI (bankroll-weighted)", f"{s['roi']:.1%}")
            c[3].metric("All-tier max drawdown", f"{all_tiers['max_drawdown']:.1%}")
            c[4].metric("Candidate games", len(audit_f))
            st.caption("Bankroll starts at 1.0; Kelly stake and P&L values are fractions of current bankroll. "
                       "ROI converts them to common bankroll units. Settled bets only. "
                       f"Coverage: {len(bets_f) / max(len(audit_f), 1):.1%} of candidate games bet.")
            b = bets_f.sort_values("game_date")
            st.plotly_chart(px.line(b, x="game_date", y="bankroll", title="Bankroll"), use_container_width=True)
            l, r = st.columns(2)
            l.plotly_chart(px.histogram(b, x="kelly_fraction", title="Kelly fraction distribution"),
                           use_container_width=True)
            exp = b.groupby("game_date")["kelly_fraction"].sum().reset_index()
            r.plotly_chart(px.bar(exp, x="game_date", y="kelly_fraction", title="Daily exposure"),
                           use_container_width=True)

with t_log:
    st.caption("Experiment log from results.tsv. These rows are NOT results of the selected run.")
    res = load_results()
    if res.empty:
        st.info("results.tsv is missing or empty.")
    else:
        st.plotly_chart(px.line(res.sort_values("timestamp"), x="timestamp", y="ensemble_rmse",
                                hover_data=["exp_id", "score", "roi"]), use_container_width=True)
        st.dataframe(res.drop(columns=["models"], errors="ignore").sort_values("timestamp", ascending=False),
                     use_container_width=True, height=400)
