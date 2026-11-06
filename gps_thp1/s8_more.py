"""进一步提升的试验（验证集选参数/轮次，测试集只报告）：
  --mode train_freeze : 4 细胞系起点，冻结 fc1，普通交叉熵，早停
  --mode train_l2sp   : 4 细胞系起点，全部可训，普通交叉熵 + L2-SP（拉向初始权重），早停
  --mode evaluate     : 汇总评测（gene prior / kNN / GPS 原版 / 各微调），并对所有方法做同样的类别校准与 GPS 混合"""
import argparse, glob, json, os, random, sys, copy, itertools, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from rdkit import Chem, DataStructs, RDLogger; from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*"); sys.path.insert(0, os.environ["GPS_CODE"])
ap=argparse.ArgumentParser(); ap.add_argument("--mode",required=True); ap.add_argument("--seed",type=int,default=0)
ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--lam",type=float,default=1e-2); ap.add_argument("--n_epoch",type=int,default=15)
a=ap.parse_args(); dev="cuda"; torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs("exp2",exist_ok=True)
CL=["HEPG2_t0","MCF7_t1","PC3_t1","VCAP_t1"]
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(2131,128); s.fc2=nn.Linear(128,32); s.fc3=nn.Linear(32,3)
    def forward(s,x):
        h=F.dropout(F.leaky_relu(s.fc1(x),.01),.5,s.training); h=F.dropout(F.leaky_relu(s.fc2(h),.01),.5,s.training); return s.fc3(h)
def gps(cl):
    src=torch.load(f"{os.environ['GPS_CODE']}/results/{cl}/multi/model.pkl",map_location="cpu",weights_only=False)
    m=MLP().to(dev); m.load_state_dict(src["model0"].state_dict()); return m
# ---- 与 s7 完全相同的划分 ----
P=pd.read_csv("unique_drug_profile.csv"); V=pd.read_csv("unique_drug_value.csv")
o=np.argsort(P.SMILES.values); P,V=P.iloc[o].reset_index(drop=True),V.iloc[o].reset_index(drop=True)
hq=np.where(P.cor_mean>0.7)[0]; test=np.sort(np.random.RandomState(1).choice(hq,int(round(.1*len(hq))),replace=False))
train=np.setdiff1d(np.arange(len(P)),test); hq_tr=np.intersect1d(hq,train)
val=np.sort(np.random.RandomState(2).choice(hq_tr,int(round(.1*len(hq_tr))),replace=False)); train=np.setdiff1d(train,val)
GO=pd.read_csv("go_fingerprints_l4.csv").set_index("gene").loc[V.columns].to_numpy(np.float32); ng=GO.shape[0]
fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),3,1024) for s in P.SMILES]
FP=np.stack([np.array(f,dtype=np.float32) for f in fps]); Y=np.digitize(V.to_numpy(),[-1.5,1.5],right=True)
FPt=torch.tensor(FP,device=dev); GOt=torch.tensor(GO,device=dev)
def X(ci,gi): return torch.cat([FPt[ci],GOt[gi]],1)
def pairs(ix): return torch.tensor(np.repeat(ix,ng),device=dev),torch.tensor(np.tile(np.arange(ng),len(ix)),device=dev)
Xva=X(*pairs(val)); Xte=X(*pairs(test)); yva=Y[val].ravel(); yte=Y[test].ravel()
def probs(ms,XX,agg="mean"):
    with torch.no_grad():
        ps=[]
        for m in ms: m.eval(); ps.append(torch.cat([F.softmax(m(XX[i:i+65536]),1) for i in range(0,len(XX),65536)]))
        p=torch.stack(ps); return (p.median(0).values if agg=="median" else p.mean(0)).cpu().numpy()
f1=lambda y,p: f1_score(y,p.argmax(1),average="macro"); acc=lambda y,p: 100*(p.argmax(1)==y).mean()
if a.mode.startswith("train"):
    tr_c,tr_g=[t.cpu().numpy() for t in pairs(train)]; ytr=Y[train].ravel()
    u,c=np.unique(ytr,return_counts=True); random.seed(0)
    keep=np.concatenate([np.where(ytr==0)[0],random.sample(list(np.where(ytr==1)[0]),k=int((c.sum()-c.max())*.5)),np.where(ytr==2)[0]])
    trc=torch.tensor(tr_c[keep],device=dev); trg=torch.tensor(tr_g[keep],device=dev); yt=torch.tensor(ytr[keep],device=dev)
    ms=[gps(cl) for cl in CL]; init=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in ms]
    if a.mode=="train_freeze":
        for m in ms: m.fc1.weight.requires_grad_(False); m.fc1.bias.requires_grad_(False)
    opts=[torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=a.lr) for m in ms]
    g=torch.Generator().manual_seed(a.seed); B=128; n_it=len(yt)//B
    best=(f1(yva,probs(ms,Xva)),0,[copy.deepcopy(m.state_dict()) for m in ms]); log=[]
    for ep in range(1,a.n_epoch):
        for m in ms: m.train()
        perm=torch.randperm(len(yt),generator=g).to(dev)
        for it in range(n_it):
            b=perm[it*B:(it+1)*B]; xb=X(trc[b],trg[b])
            for m,o_,i0 in zip(ms,opts,init):
                loss=F.cross_entropy(m(xb),yt[b])
                if a.mode=="train_l2sp": loss=loss+a.lam*sum(((p-i0[n])**2).sum() for n,p in m.named_parameters())
                o_.zero_grad(); loss.backward(); o_.step()
        vf=f1(yva,probs(ms,Xva)); log.append(vf)
        if vf>best[0]: best=(vf,ep,[copy.deepcopy(m.state_dict()) for m in ms])
        print(f"{a.mode} seed{a.seed} ep{ep:02d} val {vf:.4f}",flush=True)
    torch.save({f"model{k}":s for k,s in enumerate(best[2])},f"exp2/{a.mode}_lr{a.lr}_seed{a.seed}_model.pt")
    print(f"=== {a.mode} seed{a.seed} best epoch {best[1]} val {best[0]:.4f} ===",flush=True); raise SystemExit
# ================= evaluate =================
def load(f):
    sd=torch.load(f,map_location=dev); ms=[]
    for k in sorted(sd):
        m=MLP().to(dev); m.load_state_dict(sd[k]); ms.append(m)
    return ms
C={}   # 方法 -> [(val_probs, test_probs), ...]（每个种子一份）
gm=[gps(cl) for cl in CL]; C["GPS 4-line median"]=[(probs(gm,Xva,"median"),probs(gm,Xte,"median"))]
C["GPS MCF7"]=[(probs([gps("MCF7_t1")],Xva),probs([gps("MCF7_t1")],Xte))]
for nm,pat in [("FT-RCL (4-line, ES)","exp/ES_ft_4cl_lr0.001_seed*_model.pt"),("FT-RCL (MCF7, ES)","exp/ES_ft_mcf7_lr0.001_seed*_model.pt"),
               ("Scratch-RCL (ES)","exp/ES_scratch_lr0.001_seed*_model.pt"),("FT freeze-fc1","exp2/train_freeze_*_model.pt"),("FT L2-SP","exp2/train_l2sp_*_model.pt")]:
    fs=sorted(glob.glob(pat))
    if fs: C[nm]=[(probs(load(f),Xva),probs(load(f),Xte)) for f in fs]
# 参照 1：只看基因（训练集中每个基因三类的频率）
fr=np.stack([(Y[train]==k).mean(0) for k in range(3)],1)
C["Gene prior (no compound)"]=[(np.tile(fr,(len(val),1)),np.tile(fr,(len(test),1)))]
# 参照 2：最近邻（Tanimoto 最相似的 5 个训练化合物的标签频率）
def knn(ix,k=5):
    out=[]
    for i in ix:
        s=np.array(DataStructs.BulkTanimotoSimilarity(fps[i],[fps[j] for j in train])); nb=train[np.argsort(-s)[:k]]
        out.append(np.stack([(Y[nb]==c).mean(0) for c in range(3)],1))
    return np.concatenate(out)
C["kNN (5 nearest compounds)"]=[(knn(val),knn(test))]
def calib(pv,yv):              # 在验证集上选三类的对数偏置，最大化 macro-F1
    best=(f1(yv,pv),(0,0))
    for b0,b2 in itertools.product(np.linspace(-2,2,21),repeat=2):
        s=f1(yv,np.log(pv+1e-9)+[b0,0,b2])
        if s>best[0]: best=(s,(b0,b2))
    return np.array([best[1][0],0,best[1][1]])
rows=[]
for nm,runs in C.items():
    for cal in (False,True):
        a_,f_=[],[]
        for pv,pt in runs:
            if cal: b=calib(pv,yva); pt=np.log(pt+1e-9)+b
            a_.append(acc(yte,pt)); f_.append(f1(yte,pt))
        rows.append(dict(method=nm,calibrated=cal,n=len(runs),acc=np.mean(a_),f1=np.mean(f_),f1_min=np.min(f_),f1_max=np.max(f_)))
# 混合：GPS 中位数与微调模型按 α 混合（α 在验证集上选），再校准
gv,gt=C["GPS 4-line median"][0]
for nm in [k for k in C if k.startswith("FT")]:
    a_,f_=[],[]
    for pv,pt in C[nm]:
        al=max(np.linspace(0,1,11),key=lambda w: f1(yva,np.log(w*gv+(1-w)*pv+1e-9)+calib(w*gv+(1-w)*pv,yva)))
        mv,mt=al*gv+(1-al)*pv,al*gt+(1-al)*pt; b=calib(mv,yva); q=np.log(mt+1e-9)+b
        a_.append(acc(yte,q)); f_.append(f1(yte,q))
    rows.append(dict(method=f"Blend GPS + {nm}",calibrated=True,n=len(C[nm]),acc=np.mean(a_),f1=np.mean(f_),f1_min=np.min(f_),f1_max=np.max(f_)))
R=pd.DataFrame(rows); R.to_csv("exp2/summary.csv",index=False); print(R.round(4).to_string(index=False)); print("=== EVAL DONE ===")
