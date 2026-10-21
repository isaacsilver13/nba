"""Run a deterministic closed-loop process over multiple iterations.

Loop per iteration:
1) Run model generation.
2) Run enhanced analysis with current parameters.
3) Run feedback summarizer.
4) Compile deterministic actions from findings.
5) Apply updated parameters for next iteration.

Artifacts are written to outputs/closed_loop/<run_id>/iter_XX.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import time

import pandas as pd


def _log(msg: str) -> None:
    """Print a timestamped progress message and flush immediately."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@dataclass
class LoopConfig:
    tuning_models: Tuple[str, ...]
    default_profile: str
    high_vol_profile: str
    vol_quantiles: Tuple[float, float]
    vol_window: int
    vol_min_periods: int
    min_group_bets: int
    top_factor_levels: int
    risk_level: int = 1


@dataclass
class IterationResult:
    iteration: int
    analysis_dir: str
    feedback_dir: str
    score: float
    high_findings: int
    medium_findings: int
    actions_count: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _build_run_id() -> str:
    return datetime.now().strftime("run_%Y_%m_%d_%H%M%S")


def _find_new_feedback_dir(outputs_dir: Path, before: Sequence[Path]) -> Optional[Path]:
    before_set = {p.resolve() for p in before}
    candidates = sorted(outputs_dir.glob("copilot_feedback_v*"), key=lambda p: p.stat().st_mtime)
    new_dirs = [c for c in candidates if c.resolve() not in before_set]
    if not new_dirs:
        return None
    return new_dirs[-1]


def _to_float(value: object) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _run_command(
    command: List[str],
    cwd: Path,
    command_log_path: Path,
    dry_run: bool,
    label: str = "",
) -> None:
    cmd_str = " ".join(command)
    with command_log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"$ {cmd_str}\n")
    tag = f" ({label})" if label else ""
    _log(f"  [CMD]{tag} $ {Path(command[1]).name if len(command) > 1 else cmd_str}")
    if dry_run:
        _log("  [CMD] dry-run — skipped")
        return
    t0 = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(f"    {line}", flush=True)
        output_lines.append(line)
    proc.wait()
    elapsed = time.monotonic() - t0
    with command_log_path.open("a", encoding="utf-8") as fh:
        for ln in output_lines:
            fh.write(ln + "\n")
        fh.write(f"[exit_code={proc.returncode}]\n\n")
    _log(f"  [CMD]{tag} done in {elapsed:.1f}s (exit_code={proc.returncode})")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd_str}")


def _risk_to_profiles(risk_level: int) -> Tuple[str, str, Tuple[float, float]]:
    mapping = {
        0: ("R5_stability_tier_b", "R5_stability_tier_b", (0.40, 0.80)),
        1: ("R5_stability_tier_b", "R6_stability_tier_b_c", (0.33, 0.67)),
        2: ("R6_stability_tier_b_c", "R6_stability_tier_b_c", (0.25, 0.60)),
    }
    return mapping.get(risk_level, mapping[1])


def compile_actions_from_findings(findings_df: pd.DataFrame, current: LoopConfig) -> Tuple[LoopConfig, pd.DataFrame]:
    if findings_df is None or findings_df.empty:
        return current, pd.DataFrame([
            {
                "action_type": "no_op",
                "detail": "No findings; retain current configuration.",
                "from_value": "",
                "to_value": "",
            }
        ])

    next_cfg = LoopConfig(**asdict(current))
    actions: List[Dict[str, object]] = []

    risk_shift = 0
    for _, row in findings_df.iterrows():
        severity = str(row.get("severity", "")).lower()
        area = str(row.get("area", "")).lower()
        finding = str(row.get("finding", "")).lower()
        value = _to_float(row.get("value"))

        if severity == "high" and area in {"time_stability", "market_segmentation", "portfolio_health"}:
            risk_shift -= 1
        if area == "portfolio_health" and "win rate" in finding and value is not None and value < 0.50:
            risk_shift -= 1
        if area == "regime_behavior" and value is not None and value > 0:
            risk_shift += 1

    original_risk = next_cfg.risk_level
    next_cfg.risk_level = max(0, min(2, next_cfg.risk_level + risk_shift))
    if next_cfg.risk_level != original_risk:
        actions.append(
            {
                "action_type": "risk_level",
                "detail": "Adjusted risk level based on findings.",
                "from_value": original_risk,
                "to_value": next_cfg.risk_level,
            }
        )

    default_profile, high_profile, vol_quantiles = _risk_to_profiles(next_cfg.risk_level)
    if next_cfg.default_profile != default_profile:
        actions.append(
            {
                "action_type": "default_profile",
                "detail": "Profile adjusted by risk policy.",
                "from_value": next_cfg.default_profile,
                "to_value": default_profile,
            }
        )
        next_cfg.default_profile = default_profile

    if next_cfg.high_vol_profile != high_profile:
        actions.append(
            {
                "action_type": "high_vol_profile",
                "detail": "High-vol profile adjusted by risk policy.",
                "from_value": next_cfg.high_vol_profile,
                "to_value": high_profile,
            }
        )
        next_cfg.high_vol_profile = high_profile

    if next_cfg.vol_quantiles != vol_quantiles:
        actions.append(
            {
                "action_type": "vol_quantiles",
                "detail": "Volatility thresholds adjusted by risk policy.",
                "from_value": f"{current.vol_quantiles[0]},{current.vol_quantiles[1]}",
                "to_value": f"{vol_quantiles[0]},{vol_quantiles[1]}",
            }
        )
        next_cfg.vol_quantiles = vol_quantiles

    calib_risk = findings_df[
        (findings_df["area"].astype(str).str.lower() == "calibration")
        & (findings_df["severity"].astype(str).str.lower().isin(["high", "medium"]))
    ]
    baseline_models = ("linear_regression", "elasticnet")
    if not calib_risk.empty and tuple(next_cfg.tuning_models) != baseline_models:
        actions.append(
            {
                "action_type": "tuning_models",
                "detail": "Calibration risk present; reverting tuning models to baseline pair.",
                "from_value": ",".join(next_cfg.tuning_models),
                "to_value": ",".join(baseline_models),
            }
        )
        next_cfg.tuning_models = baseline_models

    if not actions:
        actions.append(
            {
                "action_type": "no_op",
                "detail": "No deterministic rule triggered a config change.",
                "from_value": "",
                "to_value": "",
            }
        )

    return next_cfg, pd.DataFrame(actions)


def score_iteration(findings_df: pd.DataFrame, model_stats_df: pd.DataFrame) -> Tuple[float, int, int]:
    if findings_df is None or findings_df.empty:
        high_count = 0
        med_count = 0
    else:
        severity = findings_df["severity"].astype(str).str.lower()
        high_count = int((severity == "high").sum())
        med_count = int((severity == "medium").sum())

    avg_roi = 0.0
    if model_stats_df is not None and not model_stats_df.empty and "roi" in model_stats_df.columns:
        avg_roi = float(pd.to_numeric(model_stats_df["roi"], errors="coerce").mean())

    score = (avg_roi * 100.0) - (10.0 * high_count) - (3.0 * med_count)
    return score, high_count, med_count


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def archive_iteration_artifacts(iter_dir: Path, analysis_dir: Path, feedback_dir: Path) -> None:
    analysis_dst = iter_dir / "analysis"
    feedback_dst = iter_dir / "feedback"
    analysis_dst.mkdir(parents=True, exist_ok=True)
    feedback_dst.mkdir(parents=True, exist_ok=True)

    analysis_keep = [
        "copilot_betting_summary.csv",
        "bet_outcomes_summary.csv",
        "monthly_summary.csv",
        "volatility_regime_summary.csv",
        "market_tier_split_summary.csv",
        "experiment_matrix_results.csv",
        "regime_to_profile_summary.csv",
    ]
    for name in analysis_keep:
        _copy_if_exists(analysis_dir / name, analysis_dst / name)

    feedback_keep = [
        "analytics_summary.csv",
        "feedback_findings.csv",
        "next_steps.csv",
        "model_statistics.csv",
        "feature_importance_proxy.csv",
        "run_manifest.csv",
        "brief.md",
    ]
    for name in feedback_keep:
        _copy_if_exists(feedback_dir / name, feedback_dst / name)


def run_closed_loop(args: argparse.Namespace) -> Path:
    project_root = _project_root()
    outputs_dir = project_root / "outputs"

    model_script = project_root / "src/ingest/models/Edge_LightGBM/model_ensemble_odds_v2026_02_19.py"
    analysis_script = project_root / "src/ingest/models/Edge_LightGBM/betting_simulation/analyze_copilot_outputs_enhanced.py"
    feedback_script = project_root / "src/ingest/models/Edge_LightGBM/betting_simulation/summarize_copilot_analysis_feedback.py"

    run_id = args.run_id or _build_run_id()
    run_dir = (project_root / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    command_log_path = run_dir / "commands.log"

    _log(f"=== Closed-loop run starting ===")
    _log(f"  run_id      : {run_id}")
    _log(f"  output_dir  : {run_dir.as_posix()}")
    _log(f"  iterations  : {args.iterations}")
    _log(f"  dry_run     : {args.dry_run}")
    _log(f"  risk_level  : {args.risk_level}")

    current_cfg = LoopConfig(
        tuning_models=tuple([m.strip() for m in args.tuning_models.split(",") if m.strip()]),
        default_profile=args.default_profile,
        high_vol_profile=args.high_vol_profile,
        vol_quantiles=(args.vol_q1, args.vol_q2),
        vol_window=args.vol_window,
        vol_min_periods=args.vol_min_periods,
        min_group_bets=args.min_group_bets,
        top_factor_levels=args.top_factor_levels,
        risk_level=args.risk_level,
    )

    (run_dir / "initial_config.json").write_text(json.dumps(asdict(current_cfg), indent=2), encoding="utf-8")

    loop_t0 = time.monotonic()
    iteration_results: List[IterationResult] = []
    for i in range(1, args.iterations + 1):
        iter_t0 = time.monotonic()
        _log(f"")
        _log(f"=== Iteration {i} / {args.iterations} ===")
        iter_name = f"iter_{i:02d}"
        iter_dir = run_dir / iter_name
        iter_dir.mkdir(parents=True, exist_ok=True)

        (iter_dir / "config_before.json").write_text(json.dumps(asdict(current_cfg), indent=2), encoding="utf-8")

        run_tag = f"{datetime.now().strftime('%Y_%m_%d')}_iter{i:02d}"
        analysis_dir = outputs_dir / f"copilot_analysis_odds_v{run_tag}"

        _log(f"  Step 1/3: Running model generation...")
        model_cmd = [sys.executable, str(model_script)]
        analysis_cmd = [
            sys.executable,
            str(analysis_script),
            "--outputs-dir",
            str(outputs_dir),
            "--run-date",
            run_tag,
            "--tuning-models",
            ",".join(current_cfg.tuning_models),
            "--default-profile",
            current_cfg.default_profile,
            "--high-vol-profile",
            current_cfg.high_vol_profile,
            "--vol-window",
            str(current_cfg.vol_window),
            "--vol-min-periods",
            str(current_cfg.vol_min_periods),
            "--vol-quantiles",
            f"{current_cfg.vol_quantiles[0]},{current_cfg.vol_quantiles[1]}",
            "--min-group-bets",
            str(current_cfg.min_group_bets),
            "--top-factor-levels",
            str(current_cfg.top_factor_levels),
        ]

        _log(f"  Step 2/3: Running enhanced analysis...")
        existing_feedback_dirs = list(outputs_dir.glob("copilot_feedback_v*"))
        feedback_cmd = [
            sys.executable,
            str(feedback_script),
            "--input-dir",
            str(analysis_dir),
            "--output-root",
            str(outputs_dir),
        ]

        _run_command(model_cmd, cwd=project_root, command_log_path=command_log_path, dry_run=args.dry_run, label="model")
        _run_command(analysis_cmd, cwd=project_root, command_log_path=command_log_path, dry_run=args.dry_run, label="analysis")
        _log(f"  Step 3/3: Running feedback summarizer...")
        _run_command(feedback_cmd, cwd=project_root, command_log_path=command_log_path, dry_run=args.dry_run, label="feedback")

        if args.dry_run:
            findings_df = pd.DataFrame()
            model_stats_df = pd.DataFrame()
            next_cfg, actions_df = compile_actions_from_findings(findings_df, current_cfg)
            score, high_count, med_count = score_iteration(findings_df, model_stats_df)

            actions_df.to_csv(iter_dir / "applied_actions.csv", index=False)
            (iter_dir / "analysis_dir.txt").write_text(str(analysis_dir), encoding="utf-8")
            (iter_dir / "feedback_dir.txt").write_text("dry_run_no_feedback_dir", encoding="utf-8")
        else:
            feedback_dir = _find_new_feedback_dir(outputs_dir, existing_feedback_dirs)
            if feedback_dir is None:
                raise RuntimeError("Could not locate newly created feedback directory.")

            findings_path = feedback_dir / "feedback_findings.csv"
            model_stats_path = feedback_dir / "model_statistics.csv"

            findings_df = pd.read_csv(findings_path) if findings_path.exists() else pd.DataFrame()
            model_stats_df = pd.read_csv(model_stats_path) if model_stats_path.exists() else pd.DataFrame()

            next_cfg, actions_df = compile_actions_from_findings(findings_df, current_cfg)
            score, high_count, med_count = score_iteration(findings_df, model_stats_df)

            actions_df.to_csv(iter_dir / "applied_actions.csv", index=False)
            archive_iteration_artifacts(iter_dir=iter_dir, analysis_dir=analysis_dir, feedback_dir=feedback_dir)
            (iter_dir / "analysis_dir.txt").write_text(str(analysis_dir), encoding="utf-8")
            (iter_dir / "feedback_dir.txt").write_text(str(feedback_dir), encoding="utf-8")

        (iter_dir / "config_after.json").write_text(json.dumps(asdict(next_cfg), indent=2), encoding="utf-8")

        iter_elapsed = time.monotonic() - iter_t0
        _log(f"  Findings: {high_count} high, {med_count} medium | Score: {score:.2f} | Actions: {len(actions_df)}")
        _log(f"  Iteration {i} complete in {iter_elapsed:.1f}s")

        iteration_results.append(
            IterationResult(
                iteration=i,
                analysis_dir=str(analysis_dir),
                feedback_dir="dry_run" if args.dry_run else (iter_dir / "feedback_dir.txt").read_text(encoding="utf-8").strip(),
                score=score,
                high_findings=high_count,
                medium_findings=med_count,
                actions_count=len(actions_df),
            )
        )

        current_cfg = next_cfg

    summary_df = pd.DataFrame([asdict(r) for r in iteration_results])
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)

    if not summary_df.empty:
        best_row = summary_df.sort_values("score", ascending=False).iloc[0]
        best_payload = {
            "best_iteration": int(best_row["iteration"]),
            "best_score": float(best_row["score"]),
            "selected_at": datetime.now().isoformat(timespec="seconds"),
        }
        (run_dir / "best_iteration.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
        _log(f"")
        _log(f"=== Run complete ===")
        _log(f"  Total time  : {time.monotonic() - loop_t0:.1f}s")
        _log(f"  Best iter   : {best_payload['best_iteration']} (score={best_payload['best_score']:.2f})")
        _log(f"  Artifacts   : {run_dir.as_posix()}")

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic 5-iteration closed-loop model/analysis/feedback workflow.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/closed_loop")
    parser.add_argument("--dry-run", action="store_true", default=False)

    parser.add_argument("--tuning-models", type=str, default="linear_regression,elasticnet")
    parser.add_argument("--default-profile", type=str, default="R5_stability_tier_b")
    parser.add_argument("--high-vol-profile", type=str, default="R6_stability_tier_b_c")
    parser.add_argument("--vol-q1", type=float, default=0.33)
    parser.add_argument("--vol-q2", type=float, default=0.67)
    parser.add_argument("--vol-window", type=int, default=28)
    parser.add_argument("--vol-min-periods", type=int, default=14)
    parser.add_argument("--min-group-bets", type=int, default=25)
    parser.add_argument("--top-factor-levels", type=int, default=10)
    parser.add_argument("--risk-level", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if not (0.0 < args.vol_q1 < args.vol_q2 < 1.0):
        raise ValueError("--vol-q1 and --vol-q2 must satisfy 0 < q1 < q2 < 1")

    run_dir = run_closed_loop(args)


if __name__ == "__main__":
    main()
