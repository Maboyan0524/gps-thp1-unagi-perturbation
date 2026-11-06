import os
import pandas as pd, numpy as np, collections
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
W = os.environ["WORK_DIR"].rstrip("/") + "/"
s = pd.read_csv(W+"thp1_siginfo.tsv", sep="\t", low_memory=False)
cp = s[(s["pert_type"]=="trt_cp") & (s["qc_pass"]==1)].copy()
print("qc 通过的化合物签名", len(cp), " 化合物", cp["pert_id"].nunique())
ci = pd.read_csv(W+"compoundinfo_beta.txt", sep="\t", low_memory=False).drop_duplicates("pert_id")
ci = ci.set_index("pert_id")
have = cp["pert_id"].isin(ci.index)
cp = cp[have]
cp["smiles"] = ci.loc[cp["pert_id"], "canonical_smiles"].values
cp["name"] = ci.loc[cp["pert_id"], "cmap_name"].values
cp = cp[cp["smiles"].notna() & (cp["smiles"].astype(str) != "-666") & (cp["smiles"].astype(str) != "")]
print("有 SMILES 的签名", len(cp), " 化合物", cp["pert_id"].nunique())
# 和表达矩阵对齐
sig_all = np.load(W+"thp1_sigids.npy", allow_pickle=True)
pos = {sid: i for i, sid in enumerate(sig_all)}
cp["row"] = [pos.get(x, -1) for x in cp["sig_id"]]
cp = cp[cp["row"] >= 0]
print("能对上表达矩阵的签名", len(cp), " 化合物", cp["pert_id"].nunique())
cp.to_csv(W+"thp1_train_index.csv", index=False)
# 化学空间
uniq = cp.drop_duplicates("pert_id")[["pert_id","name","smiles"]]
def fp(x):
    m = Chem.MolFromSmiles(x)
    return AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=1024) if m else None
fps, nms = [], []
for _, r in uniq.iterrows():
    f = fp(r["smiles"])
    if f is not None: fps.append(f); nms.append(str(r["name"]))
print("可用分子", len(fps))
KEY = ["quercetin","luteolin","kaempferol","myricetin","apigenin","genistein","daidzein",
       "naringenin","hesperetin","catechin","rutin","baicalein","chrysin","fisetin","morin",
       "taxifolin","flavon","flavan","isoflav","curcumin","resveratrol","tetrandrine","berberine",
       "liquiritigenin","isorhamnetin","galangin","wogonin","puerarin","silib","icariin","hesperidin"]
hit = [n for n in nms if any(k in n.lower() for k in KEY)]
print("天然产物/黄酮类:", len(hit), hit[:25])
OUR = {"槲皮素":"Oc1ccc(-c2oc3cc(O)cc(O)c3c(=O)c2O)cc1O",
       "木犀草素":"Oc1ccc(-c2cc(=O)c3c(O)cc(O)cc3o2)cc1O",
       "山奈酚":"Oc1ccc(-c2oc3cc(O)cc(O)c3c(=O)c2O)cc1",
       "杨梅素":"Oc1cc(O)c2c(c1)oc(-c1cc(O)c(O)c(O)c1)c(O)c2=O",
       "异鼠李素":"COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)ccc1O",
       "甘草素":"Oc1ccc(C2CC(=O)c3cc(O)ccc3O2)cc1",
       "汉防己甲素":"CN1CCC2=CC(=C3C=C2[C@@H]1CC4=CC=C(C=C4)OC5=C(C=CC(=C5)C[C@H]6C7=C(O3)C(=C(C=C7CCN6C)OC)OC)OC)OC"}
print()
print("七个成分与 THP-1 训练集的最大相似度:")
for k, v in OUR.items():
    f = fp(v)
    sims = np.array(DataStructs.BulkTanimotoSimilarity(f, fps))
    o = np.argsort(sims)[::-1][:3]
    print("  %-10s 最大 %.3f   %s" % (k, sims[o[0]], ", ".join("%s(%.2f)" % (nms[i], sims[i]) for i in o)))
