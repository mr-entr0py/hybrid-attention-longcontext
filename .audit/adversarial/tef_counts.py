"""Read-only, CPU-only: TEF block-plan counts (items per bucket x family x sub-type) for seed index 0, block 0.
Uses TEFStream.plan_counts / block_plan (no item generation, no GPU). Adversarial review 2026-11-10 (continuation)."""
import sys, os, collections
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/sedrica-lynx/Downloads/outeropt/src")
from hybridlc.data.tef import TEFStream
s = TEFStream(seed_index=0, steps=1000)
print("items_per_bucket():", s.items_per_bucket())
pc = s.plan_counts(0)
for k, v in pc.items():
    if isinstance(v, dict):
        print(k)
        for kk, vv in sorted(v.items(), key=lambda x: str(x[0])):
            print("   ", kk, vv)
    else:
        print(k, v)
