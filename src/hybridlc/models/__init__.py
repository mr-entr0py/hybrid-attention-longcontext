"""hybridlc.models: the hybrid LM family of protocol §3 (see INTERFACES.md).

Heavy imports (fla, Triton) happen lazily in ``build``/``mixers``/``linear``; ``config``,
``patterns`` and ``analytic`` are importable without a GPU stack.
"""

from .config import ModelConfig  # noqa: F401
