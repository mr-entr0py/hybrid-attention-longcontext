# PI decision log

This log records PI-level decisions: direction, scheduling and scope. Protocol-level rows (D000–D091+, DEV-xx) live in `reports/decisions.md`, and rulings are detailed in `reports/pi_rulings_*.md`. Each entry records: date · decision · evidence · alternatives · reason · expected scientific impact · whether it changes the frozen protocol · what would trigger reconsideration.

| # | Date | Decision | Evidence | Alternatives | Reason | Impact | Protocol change? | Reconsider if |
|---|---|---|---|---|---|---|---|---|
| P001 | 2026-11-10 | No expensive GPU run until O4 localises the Xid-8 cause | 7 Xid-8 events (journalctl), 5 of them on 11-10 with CARLA off | Proceed and auto-resume on faults | Faults can corrupt runs and silently bias which runs complete | Delays FEAS by O4's duration | No (operational) | O4 PASS, or the cause is localised to a non-production path |
| P002 | 2026-11-10 | Run the six user-requested specialist audits in parallel with the G-RV workflow, report-only for src/ | User instruction 2026-11-10; the active workflow is editing src/ | Wait for the workflow to finish | Uses wall-clock time without code conflicts | None; fixes are applied afterwards | No | — |
