import os, sys, torch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.argv=['x']; exec(open('model_bench.py').read().split("print('params(k=0,12)")[0])   # reuse model definitions only
def grads(k, det):
    torch.use_deterministic_algorithms(det)
    torch.manual_seed(0); m=LM(k).to(dev)
    g=torch.Generator().manual_seed(0); lens=[]
    for _ in range(2): lens+=doc_lens(4096,g)
    cu=torch.tensor([0]+list(torch.cumsum(torch.tensor(lens),0)),device=dev,dtype=torch.int32); pos=torch.cat([torch.arange(n,device=dev) for n in lens])
    ids=torch.randint(0,V,(8192,),device=dev,generator=torch.Generator(device=dev).manual_seed(1)); tgt=torch.roll(ids,-1)
    outs=[]
    for _ in range(2):
        m.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=m(ids,tgt,cu,max(lens),pos,False)
        loss.backward(); outs.append([p.grad.detach().clone() for p in m.parameters()])
    torch.use_deterministic_algorithms(False)
    same=all(torch.equal(a,b) for a,b in zip(*outs))
    emb_same=torch.equal(outs[0][0],outs[1][0])
    mx=max((a.float()-b.float()).abs().max().item() for a,b in zip(*outs))
    return same, emb_same, mx
for k in (0,12):
    for det in (False, True):
        try:
            s,e,mx=grads(k,det); print(f"k={k:2d} deterministic_mode={det}: all param grads bitwise={s} | embedding grad bitwise={e} | max|dgrad|={mx:.2e}",flush=True)
        except Exception as ex: print(f"k={k} det={det}: RAISES {str(ex).splitlines()[0][:150]}",flush=True)
