"""= GPS RCL/code/main_multi.py + DrugGene.py + loss.py（Python 3 移植，算法与超参不变）
--init none     ：从头训练（RCL）
--init gps_mcf7 ：以 GPS 官方 MCF7 权重初始化后同样流程训练（微调对照）
--init eval_mcf7：不训练，直接用 GPS 官方 MCF7 权重在同一测试集上评测（原版对照）
评测同 GPS：测试集上 model0 的 accuracy 与 macro-F1；最终取最后 5 个 epoch 平均。
唯一改动：推理时关闭 dropout（原代码 F.dropout2d 默认 training=True，评测时 dropout 仍开着）。"""
import argparse, json, os, random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from functools import reduce
from sklearn.metrics import f1_score, confusion_matrix
from rdkit import Chem, RDLogger; from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=0.001); ap.add_argument("--forget_rate",type=float,default=0.2)
ap.add_argument("--num_gradual",type=int,default=4); ap.add_argument("--exponent",type=float,default=1)
ap.add_argument("--n_epoch",type=int,default=25); ap.add_argument("--seed",type=int,default=1)
ap.add_argument("--epoch_decay_start",type=int,default=40); ap.add_argument("--fuzzy_exponent",type=int,default=2)
ap.add_argument("--stop_fuzzy",type=int,default=3); ap.add_argument("--num_networks",type=int,default=4)
ap.add_argument("--init",default="none"); ap.add_argument("--out",default="runs")
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
tag=f"{a.init}_net{a.num_networks}_fr{a.forget_rate}_sf{a.stop_fuzzy}_fe{a.fuzzy_exponent}_seed{a.seed}"
torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed); torch.backends.cudnn.deterministic=True
dev="cuda"
class MLP(nn.Module):
    def __init__(s,input_size=2131,n_outputs=3,dropout_rate=0.5):
        super().__init__(); s.dropout_rate=dropout_rate
        s.fc1=nn.Linear(input_size,128); s.fc2=nn.Linear(128,32); s.fc3=nn.Linear(32,n_outputs)
    def forward(s,x):
        h=F.dropout(F.leaky_relu(s.fc1(x),.01),p=s.dropout_rate,training=s.training)
        h=F.dropout(F.leaky_relu(s.fc2(h),.01),p=s.dropout_rate,training=s.training)
        return s.fc3(h)
GO=pd.read_csv("go_fingerprints_l4.csv").set_index("gene")
_fp={}
def fp(s):
    if s not in _fp: _fp[s]=np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),radius=3,nBits=1024),dtype=np.float32)
    return _fp[s]
def feats(df): return np.concatenate([np.stack([fp(s) for s in df.smiles]),GO.loc[df.gene].to_numpy(np.float32)],1)
def down_sampling(y,seed=0):                         # = DrugGene.down_sampling
    u,c=np.unique(y,return_counts=True); mv=u[np.argmax(c)]; n=int((c.sum()-c.max())*0.5)
    random.seed(seed); sel=random.sample(list(np.where(y==mv)[0]),k=n)
    return np.concatenate([np.where(y==0)[0],sel,np.where(y==2)[0]])
te=pd.read_csv("data_test.csv"); Xte=torch.tensor(feats(te),device=dev); yte=te.label.to_numpy()
def evaluate(models):
    accs,f1s,probs=[],[],None
    for k,m in enumerate(models):
        m.eval()
        with torch.no_grad(): p=torch.cat([F.softmax(m(Xte[i:i+8192]),1) for i in range(0,len(Xte),8192)]).cpu().numpy()
        pr=p.argmax(1); accs.append(100*(pr==yte).mean()); f1s.append(f1_score(yte,pr,average="macro"))
        if k==0: probs=p
    return accs,f1s,probs
def load_mcf7(n):
    src=torch.load("mcf7_model.pkl",map_location="cpu",weights_only=False); keys=sorted(src)
    ms=[]
    for k in range(n):
        m=MLP().to(dev); m.load_state_dict(src[keys[k%len(keys)]].state_dict()); ms.append(m)
    return ms
log=[]
if a.init=="eval_mcf7":
    ms=load_mcf7(1); acc,f1,_=evaluate(ms)
    log.append(dict(epoch=0,test_acc=acc[0],test_f1=f1[0]))
    json.dump(dict(tag=tag,log=log,final_acc=acc[0],final_f1=f1[0]),open(f"{a.out}/{tag}.json","w"),indent=1)
    print(tag,acc[0],f1[0]); raise SystemExit
tr=pd.read_csv("data_train.csv"); ii=down_sampling(tr.label.to_numpy()); tr=tr.iloc[ii].reset_index(drop=True)
Xtr=torch.tensor(feats(tr),device=dev); ytr=torch.tensor(tr.label.to_numpy(),device=dev); quality=tr.quality.to_numpy()
print(tag,"train",Xtr.shape,np.bincount(tr.label),"test",Xte.shape,flush=True)
B=128; n_iter=len(ytr)//B
rate=np.ones(a.n_epoch)*a.forget_rate; rate[1:a.num_gradual+1]=np.linspace(0,a.forget_rate**a.exponent,a.num_gradual)
fuzzy=np.zeros(a.n_epoch); L=a.num_gradual*a.stop_fuzzy
fuzzy[1:L+1]=1-(np.arange(0,L,1)/float(L))**a.fuzzy_exponent
def update_index(u,i_,fr,seed=1):
    np.random.seed(seed)
    if len(i_)==len(u): return i_
    n_in=int(np.floor((len(u)-len(i_))*fr)); d=np.setdiff1d(u,i_)
    return np.concatenate([i_,np.random.choice(d,n_in)])
def loss_multi(logits,t,fr,fz):
    keep=int((1-fr)*len(t)); inds=[]
    for lg in logits: inds.append(torch.argsort(F.cross_entropy(lg,t,reduction="none")).cpu().numpy()[:keep])
    out=[]
    for i,lg in enumerate(logits):
        rest=[inds[k] for k in range(len(logits)) if k!=i]
        e=update_index(reduce(np.union1d,rest),reduce(np.intersect1d,rest),fz)
        out.append(F.cross_entropy(lg[e],t[e])/len(e))
    return out
ms=load_mcf7(a.num_networks) if a.init=="gps_mcf7" else [MLP().to(dev) for _ in range(a.num_networks)]
opts=[torch.optim.Adam(m.parameters(),lr=a.lr) for m in ms]
acc,f1,_=evaluate(ms); log.append(dict(epoch=0,test_acc=acc[0],test_f1=f1[0]))
g=torch.Generator(device="cpu").manual_seed(a.seed)
for ep in range(1,a.n_epoch):
    for m in ms: m.train()
    perm=torch.randperm(len(ytr),generator=g).to(dev)
    for it in range(n_iter):
        b=perm[it*B:(it+1)*B]; lg=[m(Xtr[b]) for m in ms]
        ls=loss_multi(lg,ytr[b],rate[ep],fuzzy[ep])
        for o,l in zip(opts,ls): o.zero_grad(); l.backward(); o.step()
    acc,f1,_=evaluate(ms); log.append(dict(epoch=ep,test_acc=acc[0],test_f1=f1[0]))
    print(f"{tag} ep{ep:02d} acc {acc[0]:.2f} f1 {f1[0]:.4f}",flush=True)
fa=float(np.mean([x["test_acc"] for x in log[-5:]])); ff=float(np.mean([x["test_f1"] for x in log[-5:]]))
torch.save({f"model{k}":m.state_dict() for k,m in enumerate(ms)},f"{a.out}/{tag}_model.pt")
json.dump(dict(tag=tag,log=log,final_acc=fa,final_f1=ff),open(f"{a.out}/{tag}.json","w"),indent=1)
print(f"=== {tag} FINAL acc {fa:.2f} f1 {ff:.4f} ===",flush=True)
