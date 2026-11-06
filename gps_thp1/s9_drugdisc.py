"""药物区分能力：每个基因上，预测得分（P_up−P_down）与实测 z 在测试化合物间的 Spearman 相关，取所有基因的平均（不受基因先验影响）。"""
import glob, sys, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
sys.argv=["x","--mode","evaluate"]; src=open("s8_more.py").read().split("# ================= evaluate =================")[0]
exec(src)
zte=V.to_numpy()[test]                                  # 测试化合物 × 基因 的实测 z
def disc(pt):
    s=(pt[:,2]-pt[:,0]).reshape(len(test),ng); r=[spearmanr(s[:,j],zte[:,j])[0] for j in range(ng)]
    r=np.array(r); return np.nanmean(r), np.mean(r[~np.isnan(r)]>0)
def load(f):
    sd=torch.load(f,map_location=dev); ms=[]
    for k in sorted(sd):
        m=MLP().to(dev); m.load_state_dict(sd[k]); ms.append(m)
    return ms
gm=[gps(cl) for cl in CL]; out=[("GPS 4-line median",[disc(probs(gm,Xte,"median"))])]
for nm,pat in [("FT-RCL (4-line, ES)","exp/ES_ft_4cl_lr0.001_seed*_model.pt"),("FT freeze-fc1","exp2/train_freeze_*_model.pt"),("Scratch-RCL (ES)","exp/ES_scratch_lr0.001_seed*_model.pt")]:
    out.append((nm,[disc(probs(load(f),Xte)) for f in sorted(glob.glob(pat))]))
for nm,v in out:
    r=[x[0] for x in v]; p=[x[1] for x in v]
    print(f"{nm:22s} 基因内药物排序 Spearman 均值 {np.mean(r):+.4f}（{min(r):+.4f}~{max(r):+.4f}）  正相关基因占比 {np.mean(p)*100:.1f}%")
print("=== DISC DONE ===")
