# STATUS (read first on every fresh session)

_Last updated: 2026-11-10 ~15:50 IST by the orchestrator/PI._

## Current stage
- **Pre-FEAS.** S0 is done, G-RP PASSED, and the reproductions have run and been audited.
- **G-RV is open and blocks S2 FEAS.**
- **Operational hold P001:** no expensive GPU run until the O4 soak localises the Xid-8 faults (7 events; 5 on 11-10, with CARLA off).

## Running
| What | How to resume | State |
|---|---|---|
| PI rulings → implement → O4 soak + VT-24r/22b/23b/13r/13p2 + R6 → G-RV auditor (**running in the ORIGINAL Claude Code process pid 10526**; this session's duplicate was stopped, P004) | `Workflow({scriptPath: "~/.claude/projects/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/workflows/scripts/pi-decisions-and-grv-wf_b92a39ce-f17.js", resumeFromRunId: "wf_b92a39ce-f17"})` | Decide is cached; Implement is running |
| Six specialist audits (design, literature, implementation, evaluation, adversarial, schedule), report-only | `Workflow({scriptPath: ".../workflows/scripts/continuation-specialist-audits-wf_f382f689-3e6.js", resumeFromRunId: "wf_f382f689-3e6"})` | running |
| Cost-bench session 1 report (agent; no GPU) | resend a message to the cost-bench agent, or re-derive from `experiments/bench/cost_session1/` | writing `reports/cost_benchmarks_session1.md` |

No detached GPU jobs are running. GPU jobs must go through `scripts/repro/gpu_lock.py`.

## Verified findings so far (validation only; no study results yet)
See `reports/results.md`.
- **PASSED:** G-RP, R1, R4, R5.
- **NOT PASSED as registered:** R2, R3 (VT-21 to VT-24) and R7 (Pythia fp32). Each is diagnosed as not a code defect. The routes chosen are in `pi_rulings_2026-11-10.md` §7 (residual risks) and §8 (new checks).

## Hypotheses
H1–H6 are untested. The predictions written down in advance are in protocol §2.

## Problems
- **Xid-8 / RC watchdog faults.** The display shares the GPU. O4 is pre-registered in `experiments/prereg/O4_soak.v2.yaml`.
- **CARLA restarted (~15:00):** about 7–11 GB and high utilisation, so the OOM margin is low and timing is invalid.
- **Unanswered user questions:** budget ceiling (default: Tier A, then ask); policy if all-full fails at 16K and 32K (default: pause and ask); CARLA schedule.

## Next actions
1. G-RV workflow completes.
2. The audits complete, then the PI consolidation meeting (decision_log P-rows), then fixes.
3. FEAS-SMOKE, then S2 FEAS if G-RV permits and O4 clears.

## Resume procedure (when the user says "continue")
0. **First check whether another Claude Code process is still running a workflow** (`ps -eo pid,lstart,cmd | grep 'claude code'`; look for recently modified agent-*.jsonl files under the workflow's transcript dir). Never resume a workflow that is still live elsewhere; that creates duplicate implementers.
1. `git add -A && git commit -m WIP` (large artefacts are git-ignored).
2. Resume each workflow above whose notification showed failed or stopped agents, using `resumeFromRunId`.
3. Check detached jobs and gpu_lock logs; restart any dead job with its launcher.
4. Update this file. Append history to `reports/status_log.md`.

## Memory files
`reports/decision_log.md` (PI) · `reports/decisions.md` (protocol rows) · `reports/results.md` · `reports/competing_explanations.md` · `experiments/registry.jsonl` and `.csv` · `knowledge/` · `reports/continuation_audit.md`
