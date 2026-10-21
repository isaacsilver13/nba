"""
NBA Prediction Autoresearch Loop
==================================
Calls an LLM API to iteratively improve experiment.py.

Usage:
    python autorun.py                          # GitHub Models gpt-4o (50 req/day)
    python autorun.py --model gpt-4o-mini      # GitHub Models gpt-4o-mini (150 req/day)
    python autorun.py --groq                   # Groq free API (~1000 req/day)
    python autorun.py --groq --model llama-3.3-70b-versatile
    python autorun.py --max-iters 20           # stop after N experiments
    python autorun.py --dry-run                # validate loop without API

Requirements:
    pip install openai
    GitHub Models:  set GITHUB_TOKEN env var
    Groq:           set GROQ_API_KEY env var  (free at console.groq.com)
"""

import argparse
import ast

from config_guard import validate_config
import difflib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


def _load_token_file(path: str) -> str:
    """Read a token from a file (supports %USERPROFILE% expansion). Returns '' if missing."""
    try:
        return open(os.path.expandvars(path), "r").read().strip()
    except OSError:
        return ""


GITHUB_TOKEN = (
    os.environ.get("GITHUB_TOKEN")
    or ""
)
GITHUB_MODELS_BASE = "https://models.inference.ai.azure.com"

GROQ_API_KEY = (
    os.environ.get("GROQ_API_KEY")
    or ""
)
GROQ_API_BASE  = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Copilot API (parked — internal token exchange endpoint returns 404)
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_API_BASE  = "https://api.githubcopilot.com"

EXPERIMENT_FILE = "experiment.py"
RESULTS_TSV = "results.tsv"
EXPERIMENTS_DIR = "experiments"
PROGRAM_MD = "program.md"

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_FILE = REPO_ROOT / "experiment.py"
RESULTS_TSV = REPO_ROOT / "results.tsv"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PROGRAM_MD = REPO_ROOT / "program.md"
PYTHON = sys.executable


# ── Utilities ────────────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_best_score() -> float:
    if not os.path.exists(RESULTS_TSV):
        return float("-inf")
    try:
        import pandas as pd
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        if "score" in df.columns and len(df) > 0:
            return float(df["score"].max())
    except Exception:
        pass
    return float("-inf")


def get_last_n_results(n: int = 8) -> str:
    if not os.path.exists(RESULTS_TSV):
        return "No results yet — this is the first experiment."
    try:
        import pandas as pd
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        cols = [c for c in ["timestamp", "exp_id", "ensemble_rmse", "roi", "score"] if c in df.columns]
        tail = df.tail(n)[cols].to_string(index=False)
        return tail
    except Exception as e:
        return f"Error reading results: {e}"


def config_diff_lines(old_config: str, new_config: str) -> list:
    """Return unified-diff lines showing what changed between two config strings."""
    old_lines = old_config.splitlines()
    new_lines = new_config.splitlines()
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=0))
    # Skip the --- / +++ header lines
    return [l for l in diff if l.startswith("+") or l.startswith("-")]


def update_program_md_context(
    best_score: float,
    recent_results: str,
    tried_history=None,
) -> None:
    """Prepend a 'Current state' block to program.md so the agent sees live data."""
    content = read_file(PROGRAM_MD)
    # Remove previous injected block if present
    marker_start = "<!-- autorun:state:start -->"
    marker_end = "<!-- autorun:state:end -->"
    if marker_start in content:
        start = content.index(marker_start)
        end = content.index(marker_end) + len(marker_end)
        content = content[:start].rstrip() + "\n" + content[end:].lstrip()

    tried_section = ""
    if tried_history:
        lines = ["\n**Already tried this session — DO NOT repeat these changes:**"]
        for entry in tried_history[-12:]:
            result = f"score={entry['score']:.6f}" if entry["score"] is not None else "?"
            lines.append(f"  Iter {entry['iteration']}: {result}")
            for dl in entry["diff_lines"][:8]:
                lines.append(f"    {dl}")
        tried_section = "\n".join(lines) + "\n"

    state_block = (
        f"{marker_start}\n"
        f"## Current state (auto-updated by autorun.py)\n\n"
        f"**Best score so far:** `{best_score:.6f}`\n\n"
        f"**Recent experiment history:**\n```\n{recent_results}\n```\n"
        f"{tried_section}"
        f"{marker_end}\n\n"
    )
    write_file(PROGRAM_MD, state_block + content)


# ── Agent call ───────────────────────────────────────────────────────────────

FROZEN_MARKER = "# FROZEN"
CONFIG_MARKER = "# AGENT-EDITABLE CONFIG"

# Variables that must always be present in the returned config section
REQUIRED_CONFIG_VARS = [
    "TRAIN_SIZE", "TEST_SIZE", "EV_THRESHOLD", "FRACTIONAL_KELLY",
    "MAX_KELLY", "TOP_N_BETS_PER_DAY", "DAILY_MAX_EXPOSURE",
    "MODEL_SPECS", "ENSEMBLE_WEIGHTS", "EXTRA_FEATURE_EXCLUSIONS",
    "EXTRA_FEATURE_INCLUSIONS", "PYTH_EXPONENT", "FF_WEIGHTS",
]


def split_experiment(code: str) -> tuple:
    """Split experiment.py into (header_part, config_part, frozen_part).

    header_part  — docstring + imports (never sent to agent, never dropped)
    config_part  — AGENT-EDITABLE CONFIG block (sent to agent for modification)
    frozen_part  — everything from the FROZEN ═══ separator onwards
    """
    lines = code.splitlines(keepends=True)

    # Find FROZEN boundary (walk back over ═══ separator)
    frozen_start = len(lines)
    for i, line in enumerate(lines):
        if FROZEN_MARKER in line:
            j = i
            while j > 0 and ("\u2550" in lines[j - 1] or lines[j - 1].strip() == ""):
                j -= 1
            frozen_start = j
            break

    # Find config start (walk back over ═══ separator before AGENT-EDITABLE CONFIG)
    config_start = None
    for i, line in enumerate(lines[:frozen_start]):
        if CONFIG_MARKER in line:
            j = i
            while j > 0 and ("\u2550" in lines[j - 1] or lines[j - 1].strip() == ""):
                j -= 1
            config_start = j
            break

    if config_start is None:
        raise RuntimeError(
            f"Could not find CONFIG_MARKER ({CONFIG_MARKER!r}) in experiment.py "
            "before the FROZEN section. Refusing to proceed, since silently "
            "treating the whole header as agent-editable config would "
            "duplicate the docstring/imports on every iteration. Restore the "
            f"'{CONFIG_MARKER}' comment line above the config block."
        )

    header_part = "".join(lines[:config_start]).rstrip()
    config_part = "".join(lines[config_start:frozen_start]).rstrip()
    frozen_part = "\n" + "".join(lines[frozen_start:])
    return header_part, config_part, frozen_part


def call_agent(model_id: str, frozen_part: str, use_copilot: bool = False, use_groq: bool = False) -> str:
    """Call the configured LLM API; returns only the modified config section."""
    if use_groq:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "then set it with:  $env:GROQ_API_KEY='gsk_...'"
            )
        client = OpenAI(base_url=GROQ_API_BASE, api_key=GROQ_API_KEY)
    elif use_copilot:
        raise RuntimeError(
            "--copilot is not currently functional (token exchange returns 404).\n"
            "Use --groq (free ~1000 req/day) or --model gpt-4o-mini instead."
        )
    else:
        client = OpenAI(base_url=GITHUB_MODELS_BASE, api_key=GITHUB_TOKEN)

    program_md_content = read_file(PROGRAM_MD)
    experiment_code = read_file(EXPERIMENT_FILE)
    _, config_part, _ = split_experiment(experiment_code)

    system_prompt = (
        "You are an expert ML researcher optimizing an NBA spread prediction model. "
        "Read the instructions carefully, then propose and apply exactly ONE focused "
        "improvement to the AGENT-EDITABLE CONFIG section. "
        "CRITICAL RULES:\n"
        "1. Return the COMPLETE config section \u2014 every variable that appears in the input "
        "must also appear in your output. Do NOT drop any variables.\n"
        "2. Change only ONE variable or model spec.\n"
        "3. Output valid Python only \u2014 no markdown fences, no prose, no ellipses."
    )

    user_prompt = (
        f"## Instructions (program.md)\n\n{program_md_content}\n\n"
        f"## Current AGENT-EDITABLE CONFIG\n"
        f"(Return this block in full, with exactly one value changed)\n\n"
        f"{config_part}\n\n"
        "Return the complete config block above with ONE value changed. "
        "Include every variable. Do not include the FROZEN separator or anything after it."
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=1.0,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wrapped the output anyway
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = next(
            (i for i, l in enumerate(lines) if l.startswith("```python") or l == "```"), 0
        ) + 1
        end = next(
            (i for i in range(len(lines) - 1, 0, -1) if lines[i].startswith("```")),
            len(lines),
        )
        raw = "\n".join(lines[start:end])

    # Strip any FROZEN boundary the model included anyway
    frozen_idx = raw.find(FROZEN_MARKER)
    if frozen_idx != -1:
        raw = raw[:frozen_idx].rstrip()

    validate_config(raw)

    return raw.rstrip()


# ── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(timeout: int = 1800) -> tuple:
    """Run experiment.py; returns (success, score, output_tail)."""
    result = subprocess.run(
        [PYTHON, str(EXPERIMENT_FILE)],
        capture_output=True, text=True, timeout=timeout,
        cwd=REPO_ROOT,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    score = float("-inf")
    for line in (result.stdout or "").splitlines():
        if "[experiment] score=" in line:
            try:
                score = float(line.split("=", 1)[1].strip())
            except Exception:
                pass

    return result.returncode == 0, score, output


def save_snapshot(iteration: int) -> None:
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    dst = EXPERIMENTS_DIR / f"exp_{iteration:04d}.py"
    shutil.copy2(EXPERIMENT_FILE, dst)
    print(f"[autorun] snapshot saved → {dst}")


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Prediction Autoresearch Loop")
    parser.add_argument("--max-iters", type=int, default=10,
                         help="Maximum iterations (default: 10; must be positive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API call; run current experiment.py as baseline")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model ID (default: gpt-4o; use llama-3.3-70b-versatile with --groq)")
    parser.add_argument("--groq", action="store_true",
                        help="Use Groq free API (~1000 req/day) — requires GROQ_API_KEY env var")
    parser.add_argument("--copilot", action="store_true",
                        help="[PARKED] Copilot API — token exchange currently broken; use --groq")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-experiment timeout in seconds (default: 1800)")
    args = parser.parse_args()
    if args.max_iters <= 0 or not 1 <= args.timeout <= 1800:
        parser.error("max-iters must be positive and timeout must be 1..1800 seconds")

    global GITHUB_TOKEN, GROQ_API_KEY
    if not args.dry_run:
        if args.groq:
            GROQ_API_KEY = GROQ_API_KEY or _load_token_file(r"%USERPROFILE%\.github_token_groq.txt")
        else:
            GITHUB_TOKEN = GITHUB_TOKEN or _load_token_file(r"%USERPROFILE%\.github_token")

    if not GITHUB_TOKEN and not args.dry_run and not args.groq:
        print(
            "ERROR: GITHUB_TOKEN environment variable is not set.\n"
            "  Get a GitHub Personal Access Token and run:\n"
            "  $env:GITHUB_TOKEN='ghp_...'\n"
            "  Or use --groq with GROQ_API_KEY (free at console.groq.com)"
        )
        sys.exit(1)

    # Validate current source before running even a dry-run experiment.
    _, current_config, _ = split_experiment(read_file(EXPERIMENT_FILE))
    validate_config(current_config)

    # Save baseline snapshot (iteration 0)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    baseline_path = os.path.join(EXPERIMENTS_DIR, "exp_0000.py")
    if not os.path.exists(baseline_path):
        shutil.copy2(EXPERIMENT_FILE, baseline_path)
        print(f"[autorun] baseline saved → {baseline_path}")
    iteration = 0
    best_score = get_best_score()
    tried_history: list = []  # track all tried config diffs this session

    if args.groq and args.model == "gpt-4o":
        args.model = GROQ_DEFAULT_MODEL

    if args.copilot:
        print("[autorun] WARNING: --copilot is parked (token exchange broken). Falling back to GitHub Models.")
        args.copilot = False

    api_label = f"Groq ({GROQ_API_BASE})" if args.groq else f"GitHub Models ({GITHUB_MODELS_BASE})"
    print(f"[autorun] Starting. Current best score: {best_score:.6f}")
    print(f"[autorun] Model: {args.model}  |  API: {api_label}  |  Max iters: {args.max_iters or 'unlimited'}")

    while True:
        iteration += 1
        if args.max_iters > 0 and iteration > args.max_iters:
            print(f"\n[autorun] Reached max iterations ({args.max_iters}). Done.")
            break

        print(f"\n[autorun] ── Iteration {iteration} {'─'*50}")

        # Save unmodified code in case we need to revert
        original_code = read_file(EXPERIMENT_FILE)

        # The agent only ever returns the config block, so the header (docstring + imports) and the frozen
        # section are always the current file's own. (Taking the header from the exp_0000.py snapshot broke
        # every iteration once experiment.py gained imports the snapshot lacked.)
        header, _, frozen_part = split_experiment(original_code)

        tried_entry: dict = {"iteration": iteration, "score": None, "diff_lines": []}

        if not args.dry_run:
            # Update program.md with current best + history + tried diffs
            update_program_md_context(best_score, get_last_n_results(), tried_history)

            api_name = "Groq" if args.groq else "GitHub Models API"
            print(f"[autorun] Calling {args.model} via {api_name}...")
            try:
                new_config = call_agent(args.model, frozen_part, use_copilot=args.copilot, use_groq=args.groq)
            except RateLimitError as e:
                if args.copilot:
                    print(f"[autorun] Rate limit reached on Copilot API: {e}. Skipping.")
                    time.sleep(15)
                    continue
                import re as _re
                m = _re.search(r'wait (\d+) second', str(e))
                wait_h = f"{int(m.group(1)) // 3600}h {int(m.group(1)) % 3600 // 60}m" if m else "~24h"
                print(f"[autorun] Rate limit reached. Retry in {wait_h}. Stopping.")
                sys.exit(0)
            except Exception as e:
                print(f"[autorun] API call failed: {e}. Skipping iteration.")
                time.sleep(15)
                continue

            # Compute diff of what the agent changed vs current best
            _, current_config, _ = split_experiment(original_code)
            tried_entry["diff_lines"] = config_diff_lines(current_config, new_config)

            # Reconstruct: immutable header + new config + original frozen code
            validate_config(new_config)
            modified_code = header + "\n\n" + new_config + frozen_part
            ast.parse(modified_code)

            # Sanity-check: frozen boundary must still be present
            if FROZEN_MARKER not in modified_code:
                print("[autorun] Frozen boundary missing in reconstructed file. Reverting.")
                continue

            write_file(EXPERIMENT_FILE, modified_code)
            print("[autorun] experiment.py updated by agent.")

        # Run the experiment
        print(f"[autorun] Running experiment.py (timeout={args.timeout}s)...")
        try:
            success, score, output = run_experiment(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("[autorun] Experiment timed out. Reverting.")
            write_file(EXPERIMENT_FILE, original_code)
            continue
        except Exception as e:
            print(f"[autorun] Unexpected error running experiment: {e}. Reverting.")
            write_file(EXPERIMENT_FILE, original_code)
            continue

        # Show last 3000 chars of output
        print(output[-3000:])

        if not success:
            print("[autorun] Experiment FAILED (non-zero exit). Reverting experiment.py.")
            write_file(EXPERIMENT_FILE, original_code)
            continue

        save_snapshot(iteration)

        if score > best_score:
            delta = score - best_score
            print(f"[autorun] ✓  IMPROVED  {best_score:.6f} → {score:.6f}  (+{delta:.6f}). Keeping.")
            best_score = score
            tried_entry["score"] = score
        else:
            print(f"[autorun] ✗  No improvement  ({score:.6f} ≤ {best_score:.6f}). Reverting.")
            write_file(EXPERIMENT_FILE, original_code)
            tried_entry["score"] = score

        if not args.dry_run:
            tried_history.append(tried_entry)

        time.sleep(2)


if __name__ == "__main__":
    main()
