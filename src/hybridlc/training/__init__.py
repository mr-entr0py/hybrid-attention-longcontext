"""Training infrastructure (protocol §7, §8, §13.3, §18).

Modules: ``config`` (run YAML → RunConfig, run_id), ``optim`` (AdamW groups), ``schedules`` (LR),
``trainer`` (``train(run_cfg, resume=True)``), ``checkpoint`` (atomic save/load), ``gdiv`` (G-div),
``registry`` (append-only registry.jsonl), ``queue`` (runner), ``gpumon`` (nvidia-smi monitor),
``genconfigs`` (configs/model/**), ``fakes`` (contract stand-ins for tests).
Submodules are imported lazily so that importing the package does not pull in torch.
"""
