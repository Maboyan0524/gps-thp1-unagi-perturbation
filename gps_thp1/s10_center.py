"""THP-1 共同部分：同一模型对 999 个训练化合物预测谱的逐基因平均；药物特有部分 = 预测谱 − 共同部分。"""
import glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem, RDLogger; from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(2131,128); s.fc2=nn.Linear(128,32); s.fc3=nn.Linear(32,3)
    def forward(s,x):
        h=F.dropout(F.leaky_relu(s.fc1(x),.01),.5,s.training); h=F.dropout(F.leaky_relu(s.fc2(h),.01),.5,s.training); return s.fc3(h)
S=pd.read_csv("THP1_GPSes_signed.csv",index_col=0); genes=S.index
GOall=pd.concat([pd.read_csv(f,index_col=0) for f in sorted(glob.glob("go2k/*.csv"))]+[pd.read_csv("go_fingerprints_l4.csv").set_index("gene")])
GOall=GOall[~GOall.index.duplicated()].loc[genes]; G=torch.tensor(GOall.to_numpy(np.float32),device="cuda")
ms=[]
for f in sorted(glob.glob("exp/ES_ft_4cl_lr0.001_seed*_model.pt")):
    sd=torch.load(f)
    for k in sorted(sd): m=MLP().cuda(); m.load_state_dict(sd[k]); m.eval(); ms.append(m)
P=pd.read_csv("unique_drug_profile.csv"); acc=np.zeros(len(genes)); n=0
for s in P.SMILES:
    f=torch.tensor(np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),3,1024),dtype=np.float32),device="cuda")
    x=torch.cat([f.repeat(len(G),1),G],1)
    with torch.no_grad(): p=torch.stack([F.softmax(m(x),1) for m in ms]).mean(0).cpu().numpy()
    acc+=p[:,2]-p[:,0]; n+=1
base=pd.Series(acc/n,index=genes); base.to_csv("THP1_GPSes_common.csv",header=["common"])
D=S.sub(base,axis=0); D.to_csv("THP1_GPSes_specific.csv")
M=["luteolin","quercetin","isorhamnetin","myricetin","kaempferol","liquiritigenin","Tetrandrine"]
for nm,T in [("原始谱",S),("药物特有部分",D)]:
    C=T[M].corr().values; off=C[~np.eye(7,dtype=bool)]; print(f"{nm}: 七药两两相关 中位 {np.median(off):.3f} 范围 {off.min():.3f}~{off.max():.3f}  幅度(SD) {T[M].std().mean():.3f}")
print("=== CENTER DONE ===")
