"""
NBA research dashboard (historical research results only; not live picks).
Run with: streamlit run dashboard.py
Select one run bundle; every number comes from that run's manifest-declared artifacts.
"""

import json
import runpy
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.research_dashboard.catalog import legacy_entry, list_runs, outputs_dir
from src.research_dashboard.manifest import ODDS_TIERS, safe_path
from src.research_dashboard.views import bet_summary, filter_tiers, reliability

st.set_page_config(page_title="NBA Research Dashboard", page_icon="🏀", layout="wide")

BASE_DIR = Path(__file__).parent
OUTPUTS = outputs_dir(BASE_DIR)


@st.cache_data
def _runs(outputs: str, stamp: float):  # no underscore: Streamlit excludes "_"-prefixed args from the cache key
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


def artifact_table(run) -> pd.DataFrame:
    rows = {}
    for name, e in run.data.get("artifacts", {}).items():
        try:
            mtime = pd.Timestamp(safe_path(run.run_dir, e["path"]).stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        except (OSError, ValueError, KeyError):
            mtime = "missing"
        rows[name] = {"path": e.get("path"), "rows": e.get("rows"), "sha256": str(e.get("sha256"))[:12], "modified": mtime}
    return pd.DataFrame(rows).T


# ── Global controls ───────────────────────────────────────────────────────────
st.title("🏀 NBA Research Dashboard")
st.caption("Historical research results. Not betting advice or live picks.")

if st.sidebar.button("Refresh"):
    st.cache_data.clear()
include_legacy = st.sidebar.toggle("Include legacy / unbundled data", value=False)

runs = _runs(str(OUTPUTS), _stamp())
tabs = ["Run Health", "Model Quality", "Odds Quality", "Betting Simulation", "Artifacts", "Experiment Log"]
if include_legacy:
    tabs.append("Legacy / unbundled")
t_health, t_model, t_odds, t_bets, t_art, t_log, *t_legacy = st.tabs(tabs)

run = None
if runs:
    run = st.sidebar.selectbox("Run", runs, format_func=lambda r: f"{r.run_id} [{r.status}]")
else:
    st.sidebar.info(f"No run bundles under {OUTPUTS / 'research_runs'}. See README (capture_run).")

ok = run is not None and run.status == "complete"
T = load_tables(str(run.run_dir), (run.run_dir / "manifest.json").stat().st_mtime) if ok else {}


def need_complete():
    if run is None:
        st.info("No run selected.")
    elif not ok:
        st.warning(f"Run `{run.run_id}` is **{run.status}**; charts are disabled. See Run Health.")
    return ok


# ── Run Health ────────────────────────────────────────────────────────────────
with t_health:
    if run is None:
        st.info("No run bundles found.")
    else:
        c = st.columns(3)
        c[0].metric("Status", run.status)
        c[1].metric("Created", run.data.get("created_at", "?"))
        c[2].metric("Folds", len(run.data.get("folds", [])))
        for p in run.problems:
            st.warning(p)
        if run.data.get("folds"):
            st.subheader("Fold test windows")
            st.dataframe(pd.DataFrame(run.data["folds"]), use_container_width=True)
        if ok:
            g = T["predictions"]["game_date"]
            st.caption(f"Data coverage: {len(g):,} test games, {g.min()} to {g.max()}; "
                       f"{len(T['bets']):,} bets across all odds tiers.")
        st.subheader("Artifacts")
        st.dataframe(artifact_table(run), use_container_width=True)
        st.caption("Modified = file mtime on disk. A checksum mismatch makes the run partial.")
        st.download_button("Download manifest.json", json.dumps(run.data, indent=2), f"{run.run_id}_manifest.json")

if need_complete():
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
        fig.update_layout(xaxis_title="Predicted P(home win)", yaxis_title="Observed home win rate")
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
            c = st.columns(5)
            c[0].metric("Bets", s["bets"])
            c[1].metric("Win rate", f"{s['win_rate']:.1%}")
            c[2].metric("ROI (P&L / staked)", f"{s['roi']:.1%}")
            c[3].metric("Max drawdown", f"{s['max_drawdown']:.1%}")
            c[4].metric("Candidate games", len(audit_f))
            st.caption("Units: fraction of starting bankroll (1.0). Settled bets only. "
                       f"Coverage: {len(bets_f) / max(len(audit_f), 1):.1%} of candidate games bet.")
            b = bets_f.sort_values("game_date")
            st.plotly_chart(px.line(b, x="game_date", y="bankroll", title="Bankroll"), use_container_width=True)
            l, r = st.columns(2)
            l.plotly_chart(px.histogram(b, x="kelly_fraction", title="Kelly fraction distribution"),
                           use_container_width=True)
            exp = b.groupby("game_date")["kelly_fraction"].sum().reset_index()
            r.plotly_chart(px.bar(exp, x="game_date", y="kelly_fraction", title="Daily exposure"),
                           use_container_width=True)

with t_art:
    if need_complete():
        st.caption("Tables behind every number in this run. Downloads are the manifest-declared files, unfiltered.")
        for name, e in run.data["artifacts"].items():
            f = safe_path(run.run_dir, e["path"])
            st.download_button(f"Download {e['path']}", f.read_bytes(), f"{run.run_id}_{e['path']}", key=f"dl_{name}")

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

if t_legacy:
    with t_legacy[0]:
        st.warning("Legacy / unbundled: these files share no run ID and carry no per-bet odds tier. "
                   "Do not read them as one result.")
        st.json(legacy_entry(BASE_DIR))
        try:
            runpy.run_path(str(BASE_DIR / "legacy_dashboard.py"))
        except Exception as exc:
            st.error(f"Legacy view unavailable: {exc}")
