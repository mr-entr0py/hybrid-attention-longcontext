# Idealised check of Review B's QK-norm dilution table (CPU only, numpy).
# Keys/query RMS-normalised to norm 8 (d_h=64, unit gains); logits = G * q.k / 8.
# Needle key == query (cos=1). D=3 same-format near-miss distractors with cos=0.5.
# Values v_i = mu + eps_i, eps ~ N(0, I_64). Since sum(w)=1, o - mu = sum_i w_i eps_i exactly.
# Readout: argmax over C candidate value-codes (true needle code, the 3 distractor codes,
# and C-4 random codes) of (o - mu) . eps_c   -> an *ideal linear* readout; ignores residual-stream interference.
import numpy as np, sys
rng = np.random.default_rng(20261109)
dh, C, D, trials = 64, 4096, 3, 400
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
print("G     n      needle_w  nearmiss_w(each)  ideal_readout_acc")
for G in (1.0, 1.24, 1.56):
    for n in (1024, 4096, 16384, 32768):
        acc = 0; wn_s = 0.0; wd_s = 0.0
        for t in range(trials):
            q = unit(rng.standard_normal(dh)) * 8
            K = unit(rng.standard_normal((n, dh))) * 8          # bulk keys
            K[0] = q                                            # needle
            for j in range(1, D + 1):                           # near-miss distractors, cos = 0.5
                r = unit(rng.standard_normal(dh)); r = unit(r - (r @ unit(q)) * unit(q))
                K[j] = (0.5 * unit(q) + np.sqrt(1 - 0.25) * r) * 8
            logits = G * (K @ q) / 8.0
            w = np.exp(logits - logits.max()); w /= w.sum()
            E = rng.standard_normal((n, dh))                    # value codes (mean removed exactly)
            o = w @ E
            cand = np.vstack([E[:D + 1], rng.standard_normal((C - D - 1, dh))])
            acc += int(np.argmax(cand @ o) == 0)
            wn_s += w[0]; wd_s += w[1:D + 1].mean()
        print(f"{G:<5} {n:<6} {wn_s/trials:8.3f}  {wd_s/trials:10.4f}        {acc/trials:6.3f}")
    sys.stdout.flush()
