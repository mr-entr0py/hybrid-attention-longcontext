"""Auditor's independent re-evaluation of saved R4/R5 final weights on Zoology's MQAR test set.

Rebuilds each run's TrainConfig from the frozen run lists (our_mixers.r4_runs / r5_runs), replays
train()'s RNG order (set_determinism -> LanguageModel -> prepare_data), loads the saved state_dict
and runs Zoology's own Trainer.test(). Compares with the run JSON's final record.
"""
import json
import sys

sys.path.insert(0, "/home/sedrica-lynx/Downloads/outeropt/scripts/s1_zoology")
import torch  # noqa: E402

from zoology_compat import claim_gpu, setup_zoology  # noqa: E402

claim_gpu()
setup_zoology(micro_batch=None, ckpt_path=None)
import zoology.train as ztrain  # noqa: E402
from zoology.data.utils import prepare_data  # noqa: E402
from zoology.model import LanguageModel  # noqa: E402
from zoology.utils import set_determinism  # noqa: E402

import our_mixers as OM  # noqa: E402

ROOT = "/home/sedrica-lynx/Downloads/outeropt/experiments/repro/"


class NullLogger:
    def log(self, *a, **k):
        pass


cfgs = {c.run_id: c for c in OM.r4_runs() + OM.r5_runs()}
for spec in sys.argv[1:]:
    item, rid = spec.split(":")
    cfg = cfgs[rid]
    set_determinism(cfg.seed)
    model = LanguageModel(cfg.model)
    _, test_dl = prepare_data(cfg.data)
    sd = torch.load(f"{ROOT}{item}/weights/{rid}.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    model = model.cuda()
    tr = ztrain.Trainer(model=model, train_dataloader=None, test_dataloader=test_dl, input_type=cfg.input_type,
                        max_epochs=1, slice_keys=cfg.slice_keys, loss_type=cfg.loss_type, device="cuda",
                        logger=NullLogger())
    tr.loss_fn = torch.nn.CrossEntropyLoss()
    m = tr.test(0)
    ref = json.load(open(f"{ROOT}{item}/runs/{rid}.json"))
    out = {"run": rid, "reeval_overall": m["valid/accuracy"], "json_overall": ref["final_overall_acc"],
           "reeval_1024_256": m.get("valid/num_kv_pairs/accuracy-256"),
           "json_1024_256": ref["final_per_config_acc"]["T1024_kv256"],
           "reeval_512_128": m.get("valid/num_kv_pairs/accuracy-128"),
           "json_512_128": ref["final_per_config_acc"]["T512_kv128"]}
    out["abs_diff_overall"] = abs(out["reeval_overall"] - out["json_overall"])
    print(json.dumps(out), flush=True)
    del model, tr
    torch.cuda.empty_cache()
