# O4 soak progress (auto-generated)
_updated 2026-11-24 16:03:39 IST_

| part | end reason | kernel events | duration | window |
|---|---|---|---|---|
| Z | timebox_equivalent(blocked_foreign_false_positive) | 6 | 2385.6s | ok |
| B | event_stop | 2 | 3535.1s | ok |
| A1 | completed | 0 | 35.2s | ok |
| S1 | timebox | 0 | 1802.7s | ok |
| S2 | timebox | 0 | 1803.7s | ok |
| S3 | timebox | 0 | 1803.4s | ok |
| S4 | timebox | 0 | 1803.1s | ok |
| A3 | completed | 0 | 30.2s | ok |

## Alerts
- Z: 6 kernel-log event(s) (Xid/RC watchdog/OOM) - see Z/Z.infra.json
- Z: CUDA error text in child log
- B: 2 kernel-log event(s) (Xid/RC watchdog/OOM) - see B/B.infra.json
- B: CUDA error text in child log

## O4-Z result
```json
{
 "E_Z": 6,
 "exposure_h": 0.663,
 "events": [
  {
   "line": "2026-11-24T11:45:53.811326+05:30 sedrica-lynx-MS-7E54 kernel: NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7",
   "ts": "2026-11-24T11:45:53.811326+05:30",
   "kind": "rc_watchdog",
   "class": "infra"
  },
  {
   "line": "2026-11-24T11:45:53.812307+05:30 sedrica-lynx-MS-7E54 kernel: NVRM: Xid (PCI:0000:02:00): 8, pid=923452, name=python, channel 0x00000003",
   "ts": "2026-11-24T11:45:53.812307+05:30",
   "kind": "xid",
   "xid": 8,
   "pid": "923452",
   "name": "python",
   "class": "infra"
  },
  {
   "line": "2026-11-24T11:50:12.883535+05:30 sedrica-lynx-MS-7E54 kernel: NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7",
   "ts": "2026-11-24T11:50:12.883535+05:30",
   "kind": "rc_watchdog",
   "class": "infra"
  },
  {
   "line": "2026-11-24T11:50:12.883611+05:30 sedrica-lynx-MS-7E54 kernel: NVRM: Xid (PCI:0000:02:00): 8, pid=925390, name=python, channel 0x00000003",
   "ts": "2026-11-24T11:50:12.883611+05:30",
   "kind": "xid",
   "xid": 8,
   "pid": "925390",
   "name": "python",
   "class": "infra"
  },
  {
   "line": "2026-11-24T12:11:15.475411+05:30 sedrica-lynx-MS-7E54 kernel: NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7",
   "ts": "2026-11-24T12:11:15.475411+05:30",
   "kind": "rc_watchdog",
   "class": "infra"
  },
  {
   "line": "2026-11-24T12:11:15.476319+05:30 sedrica-lynx-MS-7E54 kernel: NVRM: Xid (PCI:0000:02:00): 8, pid=931223, name=python, channel 0x00000003",
   "ts": "2026-11-24T12:11:15.476319+05:30",
   "kind": "xid",
   "xid": 8,
   "pid": "931223",
   "name": "python",
   "class": "infra"
  }
 ],
 "valid_window": true,
 "meaning": "GDN-workload hang persists on the exclusive GPU (E_Z >= 1): PI decision row needed; see prereg O4-Z.meaning"
}
```
## O4-A verdict
```json
{
 "verdict": "PASS",
 "P1_no_kernel_events": true,
 "P2_full_time_boxes": true,
 "P3_bitwise_determinism": true,
 "P4_no_oom": true,
 "probes": {
  "A1": {
   "n_steps": [
    50,
    50
   ],
   "trace_identical": true,
   "param_sha256": [
    "f3463b712a6148f83f7bcba562216ccdb5f720646763227181b37b6f2c07baed",
    "f3463b712a6148f83f7bcba562216ccdb5f720646763227181b37b6f2c07baed"
   ],
   "sha_identical": true
  },
  "A3": {
   "n_steps": [
    50,
    50
   ],
   "trace_identical": true,
   "param_sha256": [
    "f3463b712a6148f83f7bcba562216ccdb5f720646763227181b37b6f2c07baed",
    "f3463b712a6148f83f7bcba562216ccdb5f720646763227181b37b6f2c07baed"
   ],
   "sha_identical": true
  }
 },
 "segments": {
  "S1": {
   "reason": "timebox",
   "duration_s": 1802.7,
   "n_events": 0,
   "cuda_error_in_log": false
  },
  "S2": {
   "reason": "timebox",
   "duration_s": 1803.7,
   "n_events": 0,
   "cuda_error_in_log": false
  },
  "S3": {
   "reason": "timebox",
   "duration_s": 1803.4,
   "n_events": 0,
   "cuda_error_in_log": false
  },
  "S4": {
   "reason": "timebox",
   "duration_s": 1803.1,
   "n_events": 0,
   "cuda_error_in_log": false
  }
 }
}
```
