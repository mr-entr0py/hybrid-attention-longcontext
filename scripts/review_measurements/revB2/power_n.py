# exact one-sided paired-t power for L3 at true R3=1.0 (L=6 pp at G=30), sigma=3 pp per config, seed effects cancel
from scipy import stats
import math
sL = 3.0 * math.sqrt(1 + 0.04 + 0.64)
for n in (3, 4, 5, 6):
    ncp = 6.0 / (sL / math.sqrt(n)); tc = stats.t.ppf(0.95, n - 1)
    print(n, round(1 - stats.nct.cdf(tc, n - 1, ncp), 3))
print('range 99th pct n=3 (in sigma):', round(stats.studentized_range.ppf(0.99, 3, 1e6), 2) if hasattr(stats,'studentized_range') else 'n/a')
