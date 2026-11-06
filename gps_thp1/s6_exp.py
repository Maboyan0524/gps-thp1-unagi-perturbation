"""试验：全部 969 个 landmark 基因（不做可预测性筛选），同一测试集（57 个高质量化合物）公平比较。
方法：gps_mcf7 / gps_median4（不训练）；ft_mcf7（MCF7 起点微调）；ft_4cl（4 个网络分别以 GPS 4 细胞系权重起点）；scratch。
训练沿用 GPS 的 RCL（不变类下采样、遗忘率、模糊共享），评测：准确率与 macro-F1，取最后 5 个 epoch 平均。"""
import argparse, json, os, random, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from functools import reduce
from sklearn.metrics import f1_score
from rdkit import Chem, RDLogger; from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*"); sys.path.insert(0, os.environ["GPS_CODE"])
ap=argparse.ArgumentParser()
ap.add_argument("--method",required=True); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--lr",type=float,default=1e-3)
ap.add_argument("--forget_rate",type=float,default=0.2); ap.add_argument("--stop_fuzzy",type=int,default=3); ap.add_argument("--fuzzy_exponent",type=int,default=2)
ap.add_argument("--num_gradual",type=int,default=4); ap.add_argument("--n_epoch",type=int,default=25); ap.add_argument("--out",default="exp")
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); tag=f"{a.method}_lr{a.lr}_seed{a.seed}"
torch.manual_seed(a.seed); np.random.seed(a.seed); dev="cuda"
CL=["HEPG2_t0","MCF7_t1","PC3_t1","VCAP_t1"]
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(2131,128); s.fc2=nn.Linear(128,32); s.fc3=nn.Linear(32,3)
    def forward(s,x):
        h=F.dropout(F.leaky_relu(s.fc1(x),.01),.5,s.training); h=F.dropout(F.leaky_relu(s.fc2(h),.01),.5,s.training); return s.fc3(h)
def gps(cl,k="model0"):
    src=torch.load(f"{os.environ['GPS_CODE']}/results/{cl}/multi/model.pkl",map_location="cpu",weights_only=False)
    m=MLP().to(dev); m.load_state_dict(src[k].state_dict()); return m
# ---- 数据：同一划分（与 s3 相同：高质量 cor_mean>0.7 中随机 10%，种子 1），全部基因 ----
P=pd.read_csv("unique_drug_profile.csv"); V=pd.read_csv("unique_drug_value.csv")
o=np.argsort(P.SMILES.values); P,V=P.iloc[o].reset_index(drop=True),V.iloc[o].reset_index(drop=True)
hq=np.where(P.cor_mean>0.7)[0]; test=np.sort(np.random.RandomState(1).choice(hq,int(round(.1*len(hq))),replace=False))
train=np.setdiff1d(np.arange(len(P)),test)
assert set(P.SMILES[test])==set(pd.read_csv("test_compounds.csv").SMILES), "测试集与 s3 不一致"
GO=pd.read_csv("go_fingerprints_l4.csv").set_index("gene").loc[V.columns].to_numpy(np.float32)
FP=np.stack([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),3,1024),dtype=np.float32) for s in P.SMILES])
Y=np.digitize(V.to_numpy(),[-1.5,1.5],right=True)
FPt=torch.tensor(FP,device=dev); GOt=torch.tensor(GO,device=dev)
def X(ci,gi): return torch.cat([FPt[ci],GOt[gi]],1)
ng=GO.shape[0]
te_c=np.repeat(test,ng); te_g=np.tile(np.arange(ng),len(test)); yte=Y[test].ravel()
Xte=X(torch.tensor(te_c,device=dev),torch.tensor(te_g,device=dev))
def evaluate(ms,agg="mean"):
    with torch.no_grad():
        ps=[]
        for m in ms:
            m.eval(); ps.append(torch.cat([F.softmax(m(Xte[i:i+65536]),1) for i in range(0,len(Xte),65536)]))
        p=torch.stack(ps); p=(p.median(0).values if agg=="median" else p.mean(0)).cpu().numpy()
    pr=p.argmax(1); return 100*(pr==yte).mean(), f1_score(yte,pr,average="macro")
res=dict(tag=tag,log=[])
if a.method in ("gps_mcf7","gps_median4"):
    ms=[gps("MCF7_t1")] if a.method=="gps_mcf7" else [gps(c) for c in CL]
    acc,f1=evaluate(ms,"median"); res.update(final_acc=acc,final_f1=f1)
    json.dump(res,open(f"{a.out}/{tag}.json","w")); print(f"=== {tag} acc {acc:.2f} f1 {f1:.4f} ==="); raise SystemExit
# ---- 训练集（GPS 下采样：不变类保留 0.5×其余两类之和）----
tr_c=np.repeat(train,ng); tr_g=np.tile(np.arange(ng),len(train)); ytr=Y[train].ravel(); q=np.repeat(P.cor_mean.values[train],ng)
u,c=np.unique(ytr,return_counts=True); random.seed(0)
keep=np.concatenate([np.where(ytr==0)[0],random.sample(list(np.where(ytr==1)[0]),k=int((c.sum()-c.max())*.5)),np.where(ytr==2)[0]])
tr_c,tr_g,ytr=tr_c[keep],tr_g[keep],torch.tensor(ytr[keep],device=dev)
trc=torch.tensor(tr_c,device=dev); trg=torch.tensor(tr_g,device=dev)
print(tag,"训练样本",len(ytr),"测试",len(yte),flush=True)
if a.method=="ft_mcf7": ms=[gps("MCF7_t1",f"model{k}") for k in range(4)]
elif a.method=="ft_4cl": ms=[gps(cl) for cl in CL]
else: ms=[MLP().to(dev) for _ in range(4)]
opts=[torch.optim.Adam(m.parameters(),lr=a.lr) for m in ms]
rate=np.ones(a.n_epoch)*a.forget_rate; rate[1:a.num_gradual+1]=np.linspace(0,a.forget_rate,a.num_gradual)
fz=np.zeros(a.n_epoch); L=a.num_gradual*a.stop_fuzzy; fz[1:L+1]=1-(np.arange(L)/float(L))**a.fuzzy_exponent
def upd(u_,i_,fr):
    np.random.seed(1)
    if len(i_)==len(u_): return i_
    d=np.setdiff1d(u_,i_); return np.concatenate([i_,np.random.choice(d,int(np.floor(len(d)*fr)))])
acc,f1=evaluate(ms); res["log"].append(dict(epoch=0,acc=acc,f1=f1)); print(f"{tag} ep00 acc {acc:.2f} f1 {f1:.4f}",flush=True)
g=torch.Generator().manual_seed(a.seed); B=128; n_it=len(ytr)//B
for ep in range(1,a.n_epoch):
    for m in ms: m.train()
    perm=torch.randperm(len(ytr),generator=g).to(dev)
    for it in range(n_it):
        b=perm[it*B:(it+1)*B]; xb=X(trc[b],trg[b]); yb=ytr[b]; lg=[m(xb) for m in ms]
        kp=int((1-rate[ep])*B); inds=[torch.argsort(F.cross_entropy(l,yb,reduction="none")).cpu().numpy()[:kp] for l in lg]
        for i,(l,o_) in enumerate(zip(lg,opts)):
            rest=[inds[k] for k in range(4) if k!=i]; e=upd(reduce(np.union1d,rest),reduce(np.intersect1d,rest),fz[ep])
            loss=F.cross_entropy(l[e],yb[e])/len(e); o_.zero_grad(); loss.backward(); o_.step()
    acc,f1=evaluate(ms); res["log"].append(dict(epoch=ep,acc=acc,f1=f1)); print(f"{tag} ep{ep:02d} acc {acc:.2f} f1 {f1:.4f}",flush=True)
res["final_acc"]=float(np.mean([x["acc"] for x in res["log"][-5:]])); res["final_f1"]=float(np.mean([x["f1"] for x in res["log"][-5:]]))
torch.save({f"model{k}":m.state_dict() for k,m in enumerate(ms)},f"{a.out}/{tag}_model.pt"); json.dump(res,open(f"{a.out}/{tag}.json","w"))
print(f"=== {tag} FINAL acc {res['final_acc']:.2f} f1 {res['final_f1']:.4f} ===",flush=True)
