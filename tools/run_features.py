import sys, time, importlib.util, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ingest.dataPrep.config import get_config
from ingest.dataPrep.steps import aggregate_player_to_team as s3a, calculate_elo_and_rest as s2, engineer_advanced_features as s4

def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "src", "ingest", "dataPrep", "steps", fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run_features(cfg=None, with_team_stats=True):
    """Steps 3a (team/opponent sums), 2, 4, 5, 6, 6a. Step 1/3 (downloads, need nba_api) are not run."""
    cfg = cfg or get_config(); out = {}
    t = time.time()
    steps = [("2", s2), ("4", s4), ("5", _load("s5", "calculate_travel_fatigue.py")),
             ("6", _load("s6", "engineer_rolling_features.py")), ("6a", _load("s6a", "finalize_feature_selection.py"))]
    if with_team_stats:
        steps.insert(0, ("3a", s3a))
    for name, mod in steps:
        out.update(mod.run(cfg, out)); print(f"step {name} done {time.time()-t:.0f}s", flush=True)
    return out

if __name__ == "__main__":
    cfg = get_config(); cfg["cache"]["read_on_hit"] = False
    run_features(cfg)
