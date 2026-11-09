import importlib.util, sys
spec = importlib.util.spec_from_file_location("h", "/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/pi/h1_stats_pi.py")
src = open(spec.origin).read().split("NT = 60000")[0]          # reuse definitions only
src = src.replace("12: 1.782288,", "8: 1.859548, 10: 1.812461, 12: 1.782288, 18: 1.734064,")
exec(src)
for sigma in (2.0, 3.0):
    for scn in ('homo', 'hetero', 'bimodal'):
        for R3 in (0.8, 1.0):
            a = adaptive(scn, R3, sigma, 5, NT=40000)
            print(f"sigma={sigma} {scn:8s} R3={R3}: ADAPT 5->7 P(support) {a[0]:.3f}  trigger {a[1]:.2f}  E[n] {a[2]:.2f}")
