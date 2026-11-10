"""Protocol-derived constants for the evaluation suite and the TEF (protocol §7.3, §9).

Everything here is copied from ``reports/experimental_protocol.md`` (v1.0, frozen) or
is an implementation decision that the protocol leaves open; the latter are marked
``[IMPL]`` and listed in the eval builder's report.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- lengths
K = 1024
#: Evaluation / TEF lengths in tokens ("1K" = 1,024 tokens).
LENGTHS = {"512": 512, "1K": 1 * K, "2K": 2 * K, "4K": 4 * K, "8K": 8 * K, "16K": 16 * K, "32K": 32 * K}

#: TEF length buckets (§7.3).
TEF_BUCKETS = (512, 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K)

# --------------------------------------------------------------------------- mechanics (§9.1)
#: Needles snap to a sentence boundary within +-32 tokens.
SNAP_WINDOW = 32
#: Number of queries per context (query branching, §9.1 / §7.3).
Q_PER_CONTEXT = 4
#: Values are uniform 4-digit strings, encoded as exactly 4 single-digit tokens (§6.2).
VALUE_DIGITS = 4
#: R-load: fact tokens <= 0.75 T (asserted, §9.3).
MAX_FACT_FRACTION = 0.75
#: Rank-accuracy candidate set: gold + <= 32 in-context values + 4 absent values (§9.2).
MAX_CONTEXT_CANDIDATES = 32
N_ABSENT_CANDIDATES = 4
#: Repetitive filler (R-dist only, and 10% of TEF items) (§9.1, RULER 'noise' haystack).
REPETITIVE_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
#: FineWeb-Edu documents are joined by this string inside one filler span (§9.1).
FWE_JOINER = "\n\n"
#: [IMPL] Every evaluation context and every TEF item starts with <|endoftext|> at position 0,
#: exactly like every training document (data layout: "each document begins with
#: <|endoftext|>"). It counts towards T. There is never a separator at positions >= 1 (D039).
BOS_AT_START = True

# --------------------------------------------------------------------------- A-freq / A-count
AFREQ_N_Y = 8  # n_X = round(8 * rho)
AFREQ_PAIRS = 4
AFREQ_MAX_BACKGROUND_OCC = 3
#: Windowed-count heuristic floors for A-freq (verification round 3, AFREQ-SHORT-WINDOW-FLOOR): a
#: counter that sees only the last W context tokens (ties = 1/2). Analysis/reporting only; the
#: frozen items are unchanged. AFREQ_WINDOW_FLOOR is the one reported in cell_summary.csv.
AFREQ_WINDOW_FLOORS = (256, 1024, 4096)
AFREQ_WINDOW_FLOOR = 1024
ACOUNT_RANGE = (1, 9)  # c in 1..9
ACOUNT_N_QUERIED = 4  # [IMPL] 4 queried animals (one per query) ...
ACOUNT_N_DISTRACTORS = 4  # ... plus 4 distractor animals (§9.4)

# --------------------------------------------------------------------------- scoring (§9.2)
#: Frozen evaluation batch sizes per length (B = 8 for T <= 4K, 2 for 8K-16K, 1 for 32K).
def frozen_batch_size(T: int) -> int:
    """Frozen evaluation batch size for context length ``T`` (§9.2)."""
    if T <= 4 * K:
        return 8
    if T <= 16 * K:
        return 2
    return 1


#: Fraction of items re-scored with a single-pass forward; EM flip rate must be < 1% (§9.2).
RESCORE_FRACTION = 0.05
RESCORE_MAX_FLIP_RATE = 0.01

# --------------------------------------------------------------------------- Tier 0 (§9.6)
LM1_SEQ = 32 * K
LM1_MAX_FWE_DOCS = 300
LM1_N_PG19_BOOKS = 100
#: Log-spaced position buckets [0,64) ... [16K,32K).
LM1_BUCKET_EDGES = (0, 64, 128, 256, 512, 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K)
LM2_SEQ = 4 * K
LM2_N_SEQ = 2000
LM3_NGRAM = 4
LM3_DELTA_EDGES = (64, 256, 1 * K, 4 * K, 16 * K, 32 * K)
LM3_TOP_FRACTION_EXCLUDED = 0.01
LM4_P_LEN = 256
LM4_GAPS = (0, 512, 2 * K, 8 * K, 30 * K)
LM4_N_PER_GAP = 200
LM4_RANDOM_VOCAB = 4096
LM4_SCORED_FROM = 8  # tokens 9..256 (1-based) of each copy of P
R0_N = 100

# --------------------------------------------------------------------------- TEF (§7.3)
TEF_STEPS_PER_BLOCK = 1000  # equal counts per bucket & family are exact per 1,000-step block
TEF_BATCH_TOKENS = 65536
TEF_SEQ_LEN = 32 * K
#: 90% synthetic tokens. With 1 K = 1,024 this gives floor(0.9 x 1,000 x 65,536 / 65,024) = 907
#: items per bucket per 1,000 steps; the protocol's "~928" [INF] used decimal K (65.5e6 / 63.5e3).
TEF_SYNTHETIC_SHARE = 0.90
TEF_FAMILY_MIX = {
    "retrieval": {"R1": 0.40, "R-load": 0.40, "R3": 0.10, "R-dist": 0.10},
    "multihop": {"M-hop:forward": 0.25, "M-hop:shuffled": 0.50, "M-hop:reversed": 0.25},
    "aggregation": {"A-freq": 0.70, "A-count": 0.30},
}
TEF_FAMILIES = tuple(TEF_FAMILY_MIX)
TEF_RANGES = dict(D=(0, 20), N=(4, 2048), h=(1, 6), rho=(1.1, 5.0), U=(1, 5))
TEF_REPETITIVE_SHARE = 0.10
TEF_TEMPLATES = ("T1", "T2", "T3")
TEF_ANSWER_WEIGHT = 0.5  # L = 1/2 mean CE(answer tokens) + 1/2 mean CE(other tokens) (D014)
#: Normalisation of the D014 means [IMPL, verification round 3, TEF-STEP-NORMALISED-ANSWER-WEIGHT-
#: LENGTH-SKEW; needs PI confirmation]. "block": every answer token of a 1,000-step block carries the
#: same weight (the 1/2 is spread over the block's answer tokens, likewise for the other tokens), so
#: the equal-count length buckets also get equal answer-loss weight (1/7 each). "step": the literal
#: per-step reading (answer tokens of each step share 1/2), which gives the 32K bucket ~0.29-0.30 of
#: the answer-loss weight because a step holding 32K items has few answer tokens.
TEF_ANSWER_NORM = "block"
TEF_ANSWER_NORMS = ("block", "step")
