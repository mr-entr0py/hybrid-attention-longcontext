"""R7: item-by-item comparison of our scorer JSONL with the lm-eval diagnostic dump.

    python scripts/repro/r7_compare.py --scores <our .jsonl> --lmeval <lm-eval .jsonl> --data <lambada jsonl>
Checks: same doc order/text (lm-eval's dataset copy vs our pinned jsonl), EM_tf == is_greedy per
item, |logp_gold - ll| per item. Writes experiments/repro/R7/compare/<stem>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lmeval", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    ours = {r["context_id"]: r for r in map(json.loads, Path(a.scores).read_text().splitlines())}
    ref = {r["doc_id"]: r for r in map(json.loads, Path(a.lmeval).read_text().splitlines())}
    texts = [json.loads(x)["text"] for x in Path(a.data).read_text().splitlines() if x.strip()]
    assert set(ours) == set(ref) == set(range(len(texts))), "item sets differ"
    text_mismatch = [i for i in ref if ref[i]["text"] != texts[i]]
    flips = [i for i in ours if bool(ours[i]["EM_tf"]) != bool(ref[i]["is_greedy"])]
    dll = [abs(ours[i]["logp_gold"] - ref[i]["ll"]) for i in ours]
    res = dict(scores=a.scores, lmeval=a.lmeval, n=len(ours), text_mismatch=len(text_mismatch),
               em_flips=len(flips), flip_ids=flips[:50],
               acc_ours=sum(r["EM_tf"] for r in ours.values()) / len(ours),
               acc_lmeval=sum(r["is_greedy"] for r in ref.values()) / len(ref),
               max_abs_dll=max(dll), mean_abs_dll=sum(dll) / len(dll))
    out = ROOT / "experiments/repro/R7/compare" / (Path(a.scores).stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
