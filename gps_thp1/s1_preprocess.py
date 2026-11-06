"""= GPS RCL/code/pre_process.R（Python 版，逻辑逐行对应）
按 SMILES 合并重复：每个重复与平均谱求相关 r_j，按 r_j 加权平均；cor_mean = mean(r_j)；只有 1 个重复的化合物去掉。"""
import numpy as np, pandas as pd
X=np.load("l4_X.npy"); rows=np.load("l4_rows.npy"); meta=pd.read_csv("l4_meta.csv")
gids=np.load("l4_gene_ids.npy",allow_pickle=True).astype(str)
gm=pd.read_csv("landmark_go_map.csv",dtype={"gene_id":str})
col={g:i for i,g in enumerate(gids)}; gcols=[col[g] for g in gm.gene_id]
pos={r:i for i,r in enumerate(rows)}
Z=X[[pos[r] for r in meta.row]][:,gcols]                     # 重复 × 969 landmark
ci=pd.read_csv("compoundinfo_beta.txt",sep="\t").drop_duplicates("pert_id").set_index("pert_id")
meta["SMILES"]=ci.reindex(meta.pert_id)["canonical_smiles"].values
ok=meta.SMILES.notna()&(meta.SMILES!="restricted")&(meta.SMILES.astype(str).str.len()>0)
meta,Z=meta[ok].reset_index(drop=True),Z[ok.values]
prof,vals=[],[]
for smi,g in meta.groupby("SMILES",sort=True):
    df=Z[g.index.values]
    if len(df)>1:
        m=df.mean(0); v=np.array([np.corrcoef(df[j],m)[0,1] for j in range(len(df))])
        vals.append(v@df/v.sum()); prof.append((smi,len(df),v.mean(),"|".join(sorted(set(g.pert_id)))))
    # 1 个重复：R 里 cor_mean=0 随后被删除
P=pd.DataFrame(prof,columns=["SMILES","counts","cor_mean","pert_id"])
V=pd.DataFrame(np.array(vals),columns=gm.symbol2020.values)
P.to_csv("unique_drug_profile.csv",index=False); V.to_csv("unique_drug_value.csv",index=False)
print(f"重复样本 {len(meta)} → 化合物 {len(P)}（≥2 个重复）；高质量(cor_mean>0.7) {int((P.cor_mean>0.7).sum())}")
print("=== S1 DONE ===")
