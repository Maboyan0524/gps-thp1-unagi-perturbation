"""= GPS4Drugs/code/apply_RCL.py 的预测步骤：每个（化合物, 基因）输出 up/none/down 三类概率。
GPS 原规则：某类概率 > 0.95 才判上调/下调（离散）；另外输出连续值 s = P_up − P_down 供 UNAGI 排序（我们的增进）。
5 个种子的 model0 概率取平均。"""
import glob, numpy as np, pandas as pd, torch, torch.nn.functional as F
from rdkit import Chem, RDLogger; from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
import importlib.util, sys
spec=importlib.util.spec_from_file_location("m","s4_rcl.py")
src=open("s4_rcl.py").read(); exec(src.split("GO=pd.read_csv")[0].split("ap=argparse")[0]+src[src.index("class MLP"):src.index("GO=pd.read_csv")])
GOall=pd.concat([pd.read_csv(f,index_col=0) for f in sorted(glob.glob("go2k/*.csv"))]+[pd.read_csv("go_fingerprints_l4.csv").set_index("gene")])
GOall=GOall[~GOall.index.duplicated()]
cm=pd.concat([pd.read_csv(f) for f in sorted(glob.glob("cmpd__*.csv"))]).drop_duplicates("ID")
ms=[]
for f in sorted(glob.glob("runs/none_*_model.pt")):
    m=MLP().cuda(); m.load_state_dict(torch.load(f)["model0"]); m.eval(); ms.append(m)
print("模型",len(ms),"基因",len(GOall),"化合物",len(cm),flush=True)
G=torch.tensor(GOall.to_numpy(np.float32),device="cuda")
up,dn,nn_=[],[],[]
for s in cm.SMILES:
    f=torch.tensor(np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),radius=3,nBits=1024),dtype=np.float32),device="cuda")
    x=torch.cat([f.repeat(len(G),1),G],1)
    with torch.no_grad(): p=torch.stack([F.softmax(m(x),1) for m in ms]).mean(0).cpu().numpy()
    dn.append(p[:,0]); nn_.append(p[:,1]); up.append(p[:,2])
U=pd.DataFrame(np.array(up).T,index=GOall.index,columns=cm.ID); Dn=pd.DataFrame(np.array(dn).T,index=GOall.index,columns=cm.ID)
(U-Dn).to_csv("THP1_GPSrcl_signed.csv"); U.to_csv("THP1_GPSrcl_Pup.csv"); Dn.to_csv("THP1_GPSrcl_Pdown.csv")
call=((U>0.95).astype(int)-(Dn>0.95).astype(int)); call.to_csv("THP1_GPSrcl_call095.csv")
print("0.95 规则下每药判定为上/下调的基因数:",(call!=0).sum().to_dict()); print("=== S5 DONE ===")
