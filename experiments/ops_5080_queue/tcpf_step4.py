"""TC-PF step 4 (experiments/prereg/TRITON_prefeas.yaml v1): check W1 and W2, then record the hashes.

Writes experiments/repro/TC-PF/tc_pf.json and autotune_diff.json. Only if W1 and W2 both hold does it append the
registry note run_id 'pre-FEAS-triton-cache' to experiments/registry.jsonl and copy the archive to
~/outeropt_artifacts/. CPU only. Usage: python tcpf_step4.py [--dry-run]
"""
import hashlib, json, re, shutil, sys, tarfile, time
from pathlib import Path
REPO = Path.home() / "Downloads" / "outeropt"; sys.path.insert(0, str(REPO / "src"))
from hybridlc.training.compat import triton_cache_hash
from hybridlc.training.registry import Registry

DRY = "--dry-run" in sys.argv
OUT = REPO / "experiments/repro/TC-PF"; ARCH = REPO / "runs/_artifacts/triton_cache_prefeas.tar"
S0_ARCH = REPO / "runs/_artifacts/triton_cache.tar"; PROD = REPO / ".cache/triton"
log = (OUT / "warm.log").read_text()
drv = (OUT / "tcpf_driver.log").read_text()

# W1: exit 0, every configuration x scale listed, no error
CONFIGS = ["C0", "C3", "C12", "X-gc3", "X-E3", "O-gla3", "O-nc3"]
listed = {(s, c) for s, c in re.findall(r"\] (D384|D512) (\S+): \d+s, peak", log)}
missing = [f"{s} {c}" for s in ("D384", "D512") for c in CONFIGS if (s, c) not in listed]
rc_m = re.search(r"warm rc=(\d+)", drv); rc = int(rc_m.group(1)) if rc_m else None
errors = [ln for ln in log.splitlines() if re.search(r"Traceback|Error|error:|CUDA error|exception", ln, re.I)]
n_cells = re.search(r"; (\d+) eval cells", log)
W1 = dict(pass_=(rc == 0 and not missing and not errors), exit_code=rc, missing_config_scale=missing,
          error_lines=errors[:20], n_eval_cells=int(n_cells.group(1)) if n_cells else None,
          note="the log lists configs x scales; each config runs train (2x4K, 1x32K ckpt), every frozen eval cell family, and benchmark shapes (scripts/warm_triton.py)")

# W2: gpumon over the warm window: no other project CUDA process, no foreign compute beyond allowed clients
ALLOWED = ("gnome-remote-desktop", "Xorg", "gnome-shell", "Xwayland")
mons = sorted((REPO / "runs/_logs/gpumon_lock").glob("*_TC-PF_*.jsonl"))
viol, n_samp, holder_pids, foreign = [], 0, set(), set()
for m in mons:
    for ln in m.read_text().splitlines():
        r = json.loads(ln); n_samp += 1; holder_pids.add(r.get("holder_pid"))
        if r.get("violations"): viol.append(dict(ts=r["ts"], violations=r["violations"]))
        proj = [p for p in r["procs"] if p.get("project")]
        if len(proj) > 1: viol.append(dict(ts=r["ts"], violations=["more than one project CUDA process"], procs=proj))
        for p in proj:
            if p.get("lock_ok") is False: viol.append(dict(ts=r["ts"], violations=["lock_ok false"], proc=p))
        for p in r["procs"]:
            if not p.get("project"):
                foreign.add(p["name"])
                if not any(a in p["name"] for a in ALLOWED): viol.append(dict(ts=r["ts"], violations=["foreign compute"], proc=p))
W2 = dict(pass_=(n_samp > 0 and not viol), gpumon_files=[str(m.relative_to(REPO)) for m in mons], n_samples=n_samp,
          foreign_process_names=sorted(foreign), violations=viol[:50])

# hashes
th = triton_cache_hash(str(PROD))
arch_sha = hashlib.sha256(ARCH.read_bytes()).hexdigest() if ARCH.exists() else None
sha_file = Path(str(ARCH) + ".sha256").read_text().split()[0] if Path(str(ARCH) + ".sha256").exists() else None
n_files = sum(1 for p in PROD.rglob("*") if p.is_file())

# report-only: autotune result entries vs the S0 archive; files present in only one cache
def tar_members(path):
    out = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.isfile():
                out[m.name] = tf.extractfile(m).read() if m.name.endswith(".autotune.json") else m.size
    return out
s0 = tar_members(S0_ARCH); new = tar_members(ARCH) if ARCH.exists() else {}
at_s0 = {k: v for k, v in s0.items() if k.endswith(".autotune.json")}; at_new = {k: v for k, v in new.items() if k.endswith(".autotune.json")}
common = sorted(set(at_s0) & set(at_new))
def chosen(raw):
    """Triton autotuner choice: the config with the lexicographically smallest timing list."""
    d = json.loads(raw); ct = d.get("configs_timings") or []
    return json.dumps(min(ct, key=lambda c: c[1])[0], sort_keys=True) if ct else None
differ = [k for k in common if chosen(at_s0[k]) != chosen(at_new[k])]
autotune = dict(s0_archive=str(S0_ARCH.relative_to(REPO)), new_archive=str(ARCH.relative_to(REPO)),
                n_autotune_s0=len(at_s0), n_autotune_new=len(at_new), n_autotune_common=len(common),
                n_autotune_common_chosen_config_differs=len(differ), autotune_differs_examples=differ[:20],
                n_files_s0=len(s0), n_files_new=len(new), n_only_in_s0=len(set(s0) - set(new)), n_only_in_new=len(set(new) - set(s0)),
                note="report-only (prereg report_only): chosen config (min timing list) of *.autotune.json files with the same relative path, S0 archive vs new archive; plus files present in only one archive")

drv_commit = re.search(r"commit ([0-9a-f]{40})", drv).group(1)
pre = re.search(r"tar (\S+) sha256 ([0-9a-f]{64})", drv)
res = dict(id="TC-PF", prereg="experiments/prereg/TRITON_prefeas.yaml", prereg_sha256=re.search(r"prereg sha256 ([0-9a-f]{64})", drv).group(1),
           commit=drv_commit, ts=time.strftime("%FT%T%z"), W1=W1, W2=W2,
           W3="checked in FEAS-SMOKE (T35) and R2e-X (T36): 0 new files in fresh extractions",
           production_dir=str(PROD), production_trainer_style_hash=th, production_n_files=n_files,
           archive=str(ARCH.relative_to(REPO)), archive_sha256=arch_sha, archive_sha256_file=sha_file,
           pre_restore=dict(dir=".cache/triton-pre-restore-20261110", tar=pre.group(1) if pre else None, sha256=pre.group(2) if pre else None,
                            durable_copy=str(Path.home() / "outeropt_artifacts")),
           script_note_quirk="scripts/warm_triton.py also appends its own registry note under the legacy run_id 'S0-triton-cache' (registered command line kept unchanged); the authoritative note is run_id 'pre-FEAS-triton-cache'",
           verdict=("W1 and W2 hold" if W1["pass_"] and W2["pass_"] else "BLOCKED: W1 or W2 does not hold (prereg: empty the dir and repeat)"))
print(json.dumps({k: res[k] for k in ("verdict", "production_trainer_style_hash", "archive_sha256", "production_n_files")}, indent=1))
print("W1", W1["pass_"], "missing", missing, "errors", len(errors), "| W2", W2["pass_"], "samples", n_samp, "foreign", sorted(foreign), "violations", len(viol))
print("autotune", {k: autotune[k] for k in autotune if k.startswith("n_")})
if DRY:
    sys.exit(0)
(OUT / "tc_pf.json").write_text(json.dumps(res, indent=1)); (OUT / "autotune_diff.json").write_text(json.dumps(autotune, indent=1))
if W1["pass_"] and W2["pass_"] and arch_sha and arch_sha == sha_file:
    Registry(REPO / "experiments/registry.jsonl").append("pre-FEAS-triton-cache", "note", kind="triton_cache", triton_cache_hash=th,
        archive=str(ARCH.relative_to(REPO)), archive_sha256=arch_sha, commit=drv_commit, prereg="experiments/prereg/TRITON_prefeas.yaml",
        prereg_sha256=res["prereg_sha256"], n_files=n_files, tc_pf="experiments/repro/TC-PF/tc_pf.json")
    d = Path.home() / "outeropt_artifacts"; d.mkdir(exist_ok=True)
    shutil.copy2(ARCH, d / ARCH.name); shutil.copy2(str(ARCH) + ".sha256", d / (ARCH.name + ".sha256"))
    print("registry note pre-FEAS-triton-cache written; archive copied to", d)
else:
    print("NOT recording the note: W1/W2 or archive hash check failed")
