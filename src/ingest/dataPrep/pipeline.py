"""Orchestrates dataPrep pipeline steps."""

import os
import sys

try:
    from .config import get_config
    from .steps import (
        fetch_games, calculate_elo_and_rest, fetch_player_boxscores, aggregate_player_to_team, engineer_advanced_features,
        calculate_travel_fatigue as step5, engineer_rolling_features as step6, finalize_feature_selection as step6a,
    )
except ImportError:  # pragma: no cover
    PACKAGE_DIR = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from ingest.dataPrep.config import get_config
    from ingest.dataPrep.steps import (
        fetch_games, calculate_elo_and_rest, fetch_player_boxscores, aggregate_player_to_team, engineer_advanced_features,
        calculate_travel_fatigue as step5, engineer_rolling_features as step6, finalize_feature_selection as step6a,
    )


def run_pipeline(config=None, mode="full"):
    """
    Run the data preparation pipeline.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    mode : str
        Execution mode: "full", "ingest", or "features".
        - "full": Run all steps 1 through 6a.
        - "ingest": Run steps 1, 3, and 3a only.
        - "features": Run steps 2, 4, 5, 6, 6a only.

    Returns
    -------
    dict
        Dictionary of outputs from the executed steps.
    """
    cfg = get_config() if config is None else config
    outputs = {}

    print("=" * 72)
    print(f"Starting dataPrep pipeline | mode={mode}")
    print("=" * 72)

    if mode in ("full", "ingest"):
        print("\n[1/8] Step 1: fetch_games")
        outputs.update(fetch_games.run(cfg))
        print(f"[DONE] Step 1 complete | output keys: {list(outputs.keys())}")

        print("\n[2/8] Step 3: fetch_player_boxscores")
        outputs.update(fetch_player_boxscores.run(cfg, outputs))
        print(f"[DONE] Step 3 complete | output keys: {list(outputs.keys())}")

        print("\n[3/8] Step 3a: aggregate_player_to_team")
        outputs.update(aggregate_player_to_team.run(cfg, outputs))
        print(f"[DONE] Step 3a complete | output keys: {list(outputs.keys())}")

    if mode in ("full", "features"):
        print("\n[4/8] Step 2: calculate_elo_and_rest")
        outputs.update(calculate_elo_and_rest.run(cfg, outputs))
        print(f"[DONE] Step 2 complete | output keys: {list(outputs.keys())}")

        print("\n[5/8] Step 4: engineer_advanced_features")
        outputs.update(engineer_advanced_features.run(cfg, outputs))
        print(f"[DONE] Step 4 complete | output keys: {list(outputs.keys())}")

        print("\n[6/8] Step 5: calculate_travel_fatigue")
        outputs.update(step5.run(cfg, outputs))
        print(f"[DONE] Step 5 complete | output keys: {list(outputs.keys())}")

        print("\n[7/8] Step 6: engineer_rolling_features")
        outputs.update(step6.run(cfg, outputs))
        print(f"[DONE] Step 6 complete | output keys: {list(outputs.keys())}")

        print("\n[8/8] Step 6a: finalize_feature_selection")
        outputs.update(step6a.run(cfg, outputs))
        print(f"[DONE] Step 6a complete | output keys: {list(outputs.keys())}")

    print("\n" + "=" * 72)
    print(f"Pipeline finished | mode={mode} | total output keys={len(outputs)}")
    print("=" * 72)

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NBA data preparation pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "ingest", "features"],
        default="full",
        help="Pipeline mode: 'full' (all steps), 'ingest' (steps 1,3,3a), 'features' (steps 2,4,5,6,6a)"
    )
    args = parser.parse_args()

    run_pipeline(mode=args.mode)
