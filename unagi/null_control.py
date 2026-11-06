"""空白对照：不加任何扰动，把同一份数据重复编码，量测量噪声。

如果"同一份数据编码两次"的质心漂移和药物 ΔD 一个量级，
那 L_CS / L_CP / ΔD 这套距离指标在这个模型上就不能用。
再加一个对照：扰动一组随机基因但 fold-change 全设为 1（等于不改表达），
用来分离"编码不确定性"和"真实扰动效应"。
"""
import os
import sys, types, warnings, json
warnings.filterwarnings("ignore")
sys.modules.setdefault("turtle", types.ModuleType("turtle")); sys.modules["turtle"].distance = None
sys.path.insert(0, os.environ["UNAGI_SRC"])
import numpy as np, pandas as pd, scanpy as sc, torch
from scipy.sparse import issparse
from UNAGI import UNAGI

DATA = sys.argv[1]; TASK = sys.argv[2]
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 20
it = max(i for i in range(10) if __import__("os").path.exists(f"{DATA}/{i}/stagedata/dataset.h5ad"))
adata = sc.read_h5ad(f"{DATA}/{it}/stagedata/dataset.h5ad")
adata.obs["stage"] = adata.obs["stage"].astype(int)
X = (adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)).astype("float32")
stage = adata.obs["stage"].values; grp = adata.obs["group"].astype(str).values

model = UNAGI(random_seed=8848)
model.input_dim = adata.n_vars; model.data_folder = DATA; model.data_path = DATA
model.stage_key = "stage"; model.label_key = "celltype"; model.ns = 3
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
model.setup_training(task=TASK, dist="ziln", device=dev, GPU=torch.cuda.is_available(),
                     epoch_iter=10, epoch_initial=20, max_iter=3, BATCHSIZE=256, GCN=False)
tr = model.unagi_trainer
for _m in (getattr(model, "model", None), getattr(tr, "model", None)):
    if _m is not None: _m.to(dev)

def encode(Xp):
    a = adata.copy(); a.X = Xp
    return tr.get_latent_representation(a, it, DATA)[2]

m_ctrl = grp == "Control"
m_sil = {s: (stage == s) & (grp == "SIL") for s in (0, 1, 2)}

print(f">>> 重复编码同一份未扰动数据 {REP} 次 ...", flush=True)
Zs = [encode(X) for _ in range(REP)]
print(">>> 完成，开始统计", flush=True)

Z0 = Zs[0]
c_ctrl0 = Z0[m_ctrl].mean(0)
spread = float(np.linalg.norm(Z0 - Z0.mean(0), axis=1).mean())
print(f"\n细胞散布尺度 = {spread:.4f}")

print("\n=== 1. 逐元素：同一份数据两次编码的差异 ===")
d_all = np.array([np.linalg.norm(Zs[i] - Zs[0], axis=1).mean() for i in range(1, REP)])
print(f"  每个细胞的隐向量平均漂移 = {d_all.mean():.4f} ± {d_all.std():.4f}  "
      f"（占散布 {d_all.mean()/spread*100:.2f}%）")

print("\n=== 2. 关键：L_CS 本身在重复编码间抖多少 ===")
for s in (0, 1, 2):
    vals = np.array([float(np.linalg.norm(Z[m_sil[s]].mean(0) - Z[m_ctrl].mean(0))) for Z in Zs])
    print(f"  stage{s}  L_CS = {vals.mean():.4f} ± {vals.std():.4f}   "
          f"范围 {vals.min():.4f}~{vals.max():.4f}   变异系数 {vals.std()/vals.mean()*100:.1f}%")

print("\n=== 3. 把「编码两次」当成一次假扰动，算出来的 ΔD ===")
L_CS0 = {s: float(np.linalg.norm(Z0[m_sil[s]].mean(0) - c_ctrl0)) for s in (0, 1, 2)}
for s in (0, 1, 2):
    fake = np.array([(float(np.linalg.norm(Z[m_sil[s]].mean(0) - c_ctrl0)) - L_CS0[s]) / L_CS0[s] * 100
                     for Z in Zs[1:]])
    print(f"  stage{s}  假 ΔD = {fake.mean():+.2f}% ± {fake.std():.2f}%   "
          f"范围 {fake.min():+.2f}% ~ {fake.max():+.2f}%")

print("\n=== 4. 对照 vs 矽肺 在隐空间可分吗 ===")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
y = (grp == "SIL").astype(int)
for i, Z in enumerate(Zs[:3]):
    auc = cross_val_score(LogisticRegression(max_iter=3000), Z, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          scoring="roc_auc").mean()
    print(f"  第{i+1}次编码  AUC = {auc:.4f}")
print("=== NULL DONE ===", flush=True)
