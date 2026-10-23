# Implement AI Development Orchestrator — MVP v0.1 (prompt v2)

Amends Codex's v0.1 prompt after a local check of Claude Code 2.1.304, the agent files and the
workspace. Changes from v1 are marked **[v2]**. Everything not marked is unchanged intent.

## Mission

Build the first working version of a custom AI Development Orchestrator that manages one software
task through planning, implementation, deterministic verification, independent review and human
approval. Build on the existing Claude Code infrastructure; do not replace it.

**Use the local environment as the source of truth. Do not assume undocumented features exist.**

**[v2] Input documents.** `orchestrator-infrastructure-audit.md` and its Oct 23 addendum v2 were
not found on disk. Before Step A, the user supplies their path. If they cannot be found, say so and
proceed from this prompt alone; do not invent their contents. Related existing files:
`nba_prediction/architecture-audit.md`, `nba_prediction/docs/plans/2026-10-19-audit-remediation.md`.

## 1. Approved decisions

- Python orchestration; Claude Code is the only provider; SQLite state; git worktree isolation.
- Planner = `architect`, builder = `coder`, reviewer = `tester` (existing agents; do not rewrite them
  or create replacements). `frontend-tester` is a later optional verifier.
- Deterministic verification before review. At most 2 repair rounds (3 implementation passes total).
- Human approval before any merge or deploy. Codex is a future adapter only.
- No OpenRouter, paid services, other agent frameworks, AIsaac integration, parallel execution,
  cloud workers, auto-merge, autonomous task generation, new personas.

## 2. Step A — verify assumptions (read-only first, then a small smoke test)

**[v2]** The CLI already exposes `--agent <name>`, `--agents`, `-p`, `--output-format`,
`--json-schema`, `--max-budget-usd`, `--no-session-persistence`, `--permission-mode`
(`dontAsk`), `--allowedTools/--disallowedTools`, `--tools`, `--settings`, `--add-dir`, `--bare`,
`--session-id`. A custom wrapper is only needed if the smoke tests below fail. Document each result.

Read-only:
1. Versions of Python, git, Claude Code. **Check Pydantic installs and imports on the local Python (3.14.x).**
2. Read `architect`, `coder`, `tester` definitions. **[v2] Find every location agents resolve from
   (`~/.claude/agents`, `<repo>/.claude/agents`); the two sets differ (e.g. tester). Decide and
   document which wins for a run, and record the resolved file's SHA-256 in the run state.**
3. Read project instructions, hooks, plugins, skills of the pilot repo (`repos/NFL_Confidence`).
4. Confirm the pilot test command passes on a clean checkout/worktree.
5. Existing worktrees and uncommitted changes (the workspace already has ~17 under
   `nba_prediction/.claude/worktrees`).

Smoke tests (a few real, cheap calls in a throwaway fixture repo; state the expected cost first):
- S1: `claude -p --agent <role> --output-format json` returns usable machine-readable output.
- S2: `--json-schema` round-trip: a reviewer reply validates against the ReviewVerdict schema.
- **S3: Hooks.** Interactive hooks/output styles (Learning style, ponytail, superpowers
  SessionStart) must not leak into headless runs. A Learning-style prompt asking the user to write
  code would stall an unattended coder. Test with and without `--bare`; pick the configuration that
  yields a clean, non-interactive agent and still loads the needed agents and CLAUDE.md.
- S4: `--disallowedTools` patterns actually block `git commit|push|checkout|switch|config|worktree`
  for the coder and tester.
- S5: `--max-budget-usd` behaviour under the current (subscription/OAuth) auth. If it is a no-op,
  rely on call and duration limits and say so.
- S6: Second brain is the read-only MCP server `second-brain` (`search_vault`, `read_vault_note`,
  `related_notes`). Confirm the planner can call it headlessly via `--allowedTools`.

## 3. Contracts (typed, schema-versioned, validated before use)

**[v2] Required:** `TaskSpec`, `AgentResult`, `ReviewVerdict`, `RunState`. Define `AgentRequest`,
`VerificationResult`, `ApprovalRecord` inline only where used. Pydantic emits JSON Schema on demand;
do not maintain separate `schemas/` files.

**[v2] The `events` table is the event stream** (append-only: id, run_id, type, ts, payload). No
separate event contract.

TaskSpec fields: `schema_version, task_id, repository, title, objective, acceptance_criteria,
verification` and execution limits. **[v2] Changes:**
- `verification.checks`: list of **check IDs** that must exist in `configs/repositories.yaml`
  (argv list, cwd, timeout, env allowlist; executed without a shell). The task file can never carry
  raw commands.
- `allowed_paths` / `forbidden_paths` (globs). Any changed file outside `allowed_paths` or inside
  `forbidden_paths` is a hard violation. Docs/changelog paths the coder is required by its own
  instructions to touch must be listed explicitly or the task will fail.
- `limits`: `max_fix_rounds` (default 2), `max_process_retries` (default 1), `max_agent_calls`,
  `max_duration_minutes`, `max_budget_usd` per call.
- `allow_network` / `allow_deployment` are **advisory only** (enforced solely through
  `--disallowedTools` patterns). Do not present them as a sandbox.
- `require_approval` is always true in v0.1 (not configurable).

ReviewVerdict: `verdict` in `approved | changes_requested | blocked`, `commit` (the head commit it
reviewed), `functional_findings[]`, `security_findings[]`, `acceptance_criteria_checked[]`, `summary`.
**[v2]** The existing `tester` emits free-form `FUNCTIONAL FINDINGS` / `SECURITY FINDINGS` prose.
Do not edit the agent. Supply the schema via `--json-schema` and an appended instruction block so
both mandatory passes appear as separate arrays. Reject output where either array is missing.

AgentResult: `status in completed|failed|blocked|timed_out`, `summary`, `structured_output`,
`error_message`, `usage` (calls/duration/cost if measurable).

Adapter interface: a small `Protocol` (`run(request) -> AgentResult`); `ClaudeAdapter` and
`FakeAdapter` only. No Codex adapter.

## 4. Engine

States: `QUEUED → PLANNING → IMPLEMENTING → VERIFYING → REVIEWING → AWAITING_APPROVAL`, with
`REPAIRING → VERIFYING` on failed checks or `changes_requested`, and terminals `BLOCKED`,
`NEEDS_HUMAN`, `FAILED`, `APPROVED`, `REJECTED`.

Rules:
1. Validate the task (including check IDs and path globs) before any agent call.
2. Create the worktree and branch `orch/<task_id>/<run_id>` under the orchestrator's own
   `runs/<run_id>/wt`, **not** under any repo's `.claude/worktrees`. **[v2]** Use only
   `git worktree remove`/`prune` for cleanup; never recursive-delete a worktree, never create
   junctions/symlinks into it (a previous removal deleted through a `node_modules` junction),
   refuse to operate when a reparse point is found inside it.
3. **[v2] The orchestrator makes all commits** (snapshot after implement and after each repair), so
   `head_commit` is meaningful. Agents are denied git write operations (S4).
4. **[v2] Before and after every agent call:** snapshot `git rev-parse HEAD`, `git status
   --porcelain` and `git for-each-ref`. If any ref other than the run branch changed, or HEAD moved
   without the orchestrator, the run goes to `FAILED` with evidence. This applies to the **tester
   too**, which has Bash.
5. Deterministic verification runs only the check IDs from config. Agents cannot mark verification
   passed. A reviewer `approved` never overrides a failed mandatory check.
6. **[v2] A verdict is bound to the commit it reviewed.** If HEAD or the tree differs at approval
   time, the verdict is void and review reruns.
7. Fix round = a new coder attempt after failed verification or `changes_requested`. Max 2.
   Exhaustion → `BLOCKED`. **[v2] Process retry** (crash, timeout, malformed output, max 1 per call)
   is separate and never counts as a fix round. **Before any process retry, hard-reset the worktree
   to the last recorded commit**, because a timed-out coder may have edited files.
8. Timeout, malformed output, or unavailable `claude` is never success. Missing CLI → stop with an
   actionable error before creating anything.
9. Material product/security ambiguity → `NEEDS_HUMAN`.
10. Per-repository run lock; no two runs on one repo concurrently.
11. Never send `.env`/secret files to agents; redact known secret patterns before persisting logs.
    Worktrees contain tracked files only, which helps, but do not claim a sandbox.

## 5. Persistence and recovery

SQLite tables: `runs`, `phases` (keyed `(run_id, phase, attempt)` with status and result JSON),
`events`, `verifications`, `approvals`. Artifacts (plan, transcripts, verification output, review
JSON, final approval package) in `runs/<run_id>/artifacts/`.

`resume`: reconcile recorded phase with actual `HEAD`/`status`; if a phase's result row exists,
do not redo it; if the mutation happened but was not recorded, adopt it after checking the diff
against `allowed_paths`; otherwise reset to the last recorded commit and rerun that phase once.

## 6. Second brain (minimal)

**[v2]** No Python integration. The planner agent may use the read-only `second-brain` MCP tools
(via `--allowedTools`). Proposed knowledge updates are a field in the final summary, saved as an
artifact for human review. No automatic writes.

## 7. CLI

`orchestrator validate|run|status|resume|approve|reject`. `approve` and `reject` record the decision
only; they never merge, push or deploy.

## 8. Tests (no paid model calls)

Fake adapters covering: valid and invalid task; unknown check ID; raw command in task rejected;
path allow/forbid violations; happy path; verification failure → repair; reviewer rejection →
repair; second repair; exhaustion → BLOCKED; process failure; timeout with reset-before-retry;
malformed structured output; **tester prose without both finding arrays rejected**; reviewer
approves but check fails; verdict void after HEAD change; **ref change by agent detected**; resume
after crash at each phase boundary; approval gate; missing CLI.

## 9. Execution order

A (verify + smoke) → B (contracts, storage, policies, engine, fake adapter) → C (tests green)
→ D (ClaudeAdapter) → E (one pilot) → F (report). Do not start real model calls beyond the Step A
smoke tests until C passes.

Pilot: `repos/NFL_Confidence`, a regression-test task for confidence-point scoring, with
`allowed_paths` limited to the backend test directory (plus explicitly listed doc files), check ID
for the existing pytest command, and run on a disposable branch. No production logic changes.

## 10. Location and deliverables

**[v2]** The orchestrator is its own repo at `repos/ai-orchestrator/`, not inside the
`nba_prediction` worktree/branch.

Deliver: source and tests; README (setup, usage, architecture, recovery, troubleshooting); an
example TaskSpec and `repositories.yaml`; Step A findings with the chosen invocation and hook
configuration; verification evidence; actual pilot results with model-call counts; known
limitations and unverified assumptions; prioritized v0.2 list. Do not report anything as
implemented unless it was tested.

**Principle: make the execution loop reliable and observable before making it more autonomous.**
