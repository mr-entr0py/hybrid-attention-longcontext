# Review B's R2 rule: level N used at length T iff 12N <= 0.75T. Assume per-level endpoint gaps
# (S12 - S0) that are CONSTANT in T (no true length effect). Compute the ladder-mean gap per T
# and its OLS slope per doubling of T over 1K..32K.
import numpy as np
T = np.array([1024, 2048, 4096, 8192, 16384, 32768]); x = np.log2(T / 1024)
gapR2 = {8: 10.0, 32: 20.0, 128: 40.0, 512: 60.0}           # assumed, illustrative
gapR1 = {3: 30.0, 15: 30.0}                                  # R1 levels are length-independent
def slope(y): return np.polyfit(x, y, 1)[0]
r2_only, r1r2 = [], []
for t in T:
    lv = [n for n in gapR2 if 12 * n <= 0.75 * t]
    r2_only.append(np.mean([gapR2[n] for n in lv]))
    r1r2.append(np.mean([gapR2[n] for n in lv] + list(gapR1.values())))
    print(t, "R2 levels used:", lv, " R2-mean gap %.2f" % r2_only[-1], " R1+R2-mean gap %.2f" % r1r2[-1])
print("spurious slope, R2 only: %.2f pp/doubling" % slope(np.array(r2_only)))
print("spurious slope, R1+R2 pooled (equal weight per level): %.2f pp/doubling" % slope(np.array(r1r2)))
# Smaller, more conservative assumption: gaps 5/10/20/30 for N=8/32/128/512 and R1 gaps 15
gapR2b = {8: 5.0, 32: 10.0, 128: 20.0, 512: 30.0}; yb = []
for t in T:
    lv = [n for n in gapR2b if 12 * n <= 0.75 * t]; yb.append(np.mean([gapR2b[n] for n in lv] + [15.0, 15.0]))
print("halved gaps, R1+R2 pooled: %.2f pp/doubling" % slope(np.array(yb)))
