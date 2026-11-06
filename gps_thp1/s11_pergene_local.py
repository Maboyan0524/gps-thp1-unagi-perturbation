"""本机：每个基因上"预测得分 vs 实测 z"在 57 个测试化合物间的 Spearman ρ（GPS 4 细胞系中位数 vs 微调 5 种子集成），供 b 图累积分布曲线。"""
import os
import glob, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger; from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*"); sys.path.insert(0, os.environ["GPS_CODE"])
dev="cuda" if torch.cuda.is_available() else "cpu"
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(2131,128); s.fc2=nn.Linear(128,32); s.fc3=nn.Linear(32,3)
    def forward(s,x):
        h=F.dropout(F.leaky_relu(s.fc1(x),.01),.5,s.training); h=F.dropout(F.leaky_relu(s.fc2(h),.01),.5,s.training); return s.fc3(h)
P=pd.read_csv("unique_drug_profile.csv"); V=pd.read_csv("unique_drug_value.csv")
o=np.argsort(P.SMILES.values); P,V=P.iloc[o].reset_index(drop=True),V.iloc[o].reset_index(drop=True)
hq=np.where(P.cor_mean>0.7)[0]; test=np.sort(np.random.RandomState(1).choice(hq,int(round(.1*len(hq))),replace=False))
GO=torch.tensor(pd.read_csv("go_fingerprints_l4.csv").set_index("gene").loc[V.columns].to_numpy(np.float32),device=dev); ng=GO.shape[0]
FP=torch.tensor(np.stack([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),3,1024),dtype=np.float32) for s in P.SMILES[test]]),device=dev)
Xte=torch.cat([FP.repeat_interleave(ng,0),GO.repeat(len(test),1)],1)
def pr(ms,agg):
    with torch.no_grad():
        p=torch.stack([F.softmax(m.eval()(Xte),1) for m in ms]); p=p.median(0).values if agg=="median" else p.mean(0)
    p=p.cpu().numpy(); return (p[:,2]-p[:,0]).reshape(len(test),ng)
gm=[]
for cl in ["HEPG2_t0","MCF7_t1","PC3_t1","VCAP_t1"]:
    m=MLP().to(dev); m.load_state_dict(torch.load(f"{os.environ['GPS_CODE']}/results/{cl}/multi/model.pkl",map_location="cpu",weights_only=False)["model0"].state_dict()); gm.append(m)
ft=[]
for f in sorted(glob.glob(os.environ["FINETUNED_MODELS"] + "/ES_ft_4cl_*_model.pt")):
    sd=torch.load(f,map_location=dev)
    for k in sorted(sd): m=MLP().to(dev); m.load_state_dict(sd[k]); ft.append(m)
z=V.to_numpy()[test]; out={}
for nm,S in [("GPS (4 cell lines)",pr(gm,"median")),("GPS fine-tuned on THP-1",pr(ft,"mean"))]:
    out[nm]=[spearmanr(S[:,j],z[:,j])[0] for j in range(ng)]
R=pd.DataFrame(out,index=V.columns); R.to_csv(os.environ["PERGENE_OUT"])
print(R.describe().round(3)); print("ρ>0 比例",(R>0).mean().round(3).to_dict()); print("微调更高的基因比例",round((R.iloc[:,1]>R.iloc[:,0]).mean(),3))
from scipy.stats import wilcoxon; print("配对 Wilcoxon P",wilcoxon(R.iloc[:,1],R.iloc[:,0],nan_policy="omit").pvalue)
