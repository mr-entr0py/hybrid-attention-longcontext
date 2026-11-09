import math
z=1.959964
def wilson(p,n):
    c=(p+z*z/(2*n))/(1+z*z/n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h,c+h
print("Wilson 95% half-width (max over interval)/2:")
for n in [50,100,200,300,400,500,1000,2000]:
    row=[]
    for p in [0.05,0.1,0.25,0.5,0.9,0.95]:
        lo,hi=wilson(p,n); row.append(f"{(hi-lo)/2*100:5.1f}")
    print(n,row)
print("n for Wald half-width h at p=0.5:")
for h in [0.10,0.05,0.04,0.03,0.025,0.02]:
    print(h, math.ceil(z*z*0.25/h/h))
print("n for h at p=0.9:")
for h in [0.05,0.03,0.02]:
    print(h, math.ceil(z*z*0.09/h/h))
# unpaired two-proportion power
zb=0.841621
def n2(p1,p2): 
    pb=(p1+p2)/2
    return math.ceil((z*math.sqrt(2*pb*(1-pb))+zb*math.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p1-p2)**2)
print("unpaired n per arm, alpha .05 power .8:")
for a,b in [(0.5,0.6),(0.5,0.55),(0.8,0.9),(0.9,0.95),(0.2,0.3),(0.05,0.15)]:
    print(a,b,n2(a,b))
# paired McNemar: n = (z*sqrt(pd) + zb*sqrt(pd - d^2))^2 / d^2, pd = discordant prob
print("paired McNemar n (items) for diff d with discordance pd:")
for d,pd in [(0.10,0.2),(0.10,0.3),(0.05,0.1),(0.05,0.2)]:
    print(d,pd, math.ceil((z*math.sqrt(pd)+zb*math.sqrt(pd-d*d))**2/d/d))
# design effect for clustered queries
print("design effect 1+(m-1)ICC:")
for m in [4,8,16]:
    for icc in [0.05,0.2,0.5]:
        print(m,icc, round(1+(m-1)*icc,2), "eff n per 100 contexts:", round(100*m/(1+(m-1)*icc)))
