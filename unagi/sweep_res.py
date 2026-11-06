"""复用已训练的 iter-0 checkpoint，离线扫 leiden 分辨率，看哪个能让 connectNodes 连通。

比每个分辨率重训一遍快得多。只读 checkpoint，不写任何训练产物。
诊断依据：connectNodes 要求 距离矩阵的 均值/标准差 > 1.645，否则 p<0.05 的阈值变成负数，
结构上零连接。聚类数越多，该比值通常越大。
"""
import os
import sys, types, json
from pathlib import Path
sys.modules.setdefault("turtle", types.ModuleType("turtle")); sys.modules["turtle"].distance = None
sys.path.insert(0, os.environ["UNAGI_SRC"])
import numpy as np, scanpy as sc, torch
from scipy.stats import norm
from sklearn.neighbors import kneighbors_graph
from UNAGI import UNAGI
from UNAGI.utils.attribute_utils import updateAttributes
from UNAGI.dynamic_graphs.buildGraph import nodesDistance, connectNodes

DATA = Path(sys.argv[1]); TASK = sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
RESOLUTIONS = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4
                                  else ["1.0", "1.5", "2.0", "2.5", "3.0", "4.0"])]
NB = int(sys.argv[5]) if len(sys.argv) > 5 else 30

_orig = sc.pp.neighbors
sc.pp.neighbors = lambda *a, **kw: _orig(*a, **{**kw, "transformer": "sklearn"})

model = UNAGI(random_seed=8848)
model.input_dim = sc.read_h5ad(DATA / "0.h5ad", backed="r").n_vars
model.data_folder = DATA; model.data_path = DATA
model.stage_key = "stage"; model.label_key = "celltype"; model.ns = NS
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
model.setup_training(task=TASK, dist="ziln", device=dev, GPU=torch.cuda.is_available(),
                     epoch_iter=10, epoch_initial=20, max_iter=3, BATCHSIZE=256, GCN=False)
tr = model.unagi_trainer
# 权重是 load_state_dict 填进去的，模型对象本身还在 CPU；不搬就会 cpu/cuda 混用报错
for _m in (getattr(model, "model", None), getattr(tr, "model", None)):
    if _m is not None:
        _m.to(dev)

print(f">>> checkpoint: {DATA/'model_save'/f'{TASK}_0.pth'}", flush=True)
print(f">>> 分辨率候选: {RESOLUTIONS}  近邻数: {NB}\n", flush=True)

# 每个 stage 的隐空间只算一次
base = []
for s in range(NS):
    a = sc.read_h5ad(DATA / "0" / "stagedata" / f"{s}.h5ad")
    zl, zs, emb = tr.get_latent_representation(a, 0, DATA)
    a.obsm["z"] = emb
    base.append((a, zl, zs))
    print(f"    stage {s}: {a.n_obs} 细胞，隐空间 {emb.shape}", flush=True)

rows = []
for res in RESOLUTIONS:
    reps, ncl = [], []
    for s in range(NS):
        a0, zl, zs = base[s]
        a = a0.copy()
        for k in ("connectivities", "distances"):
            a.obsp.pop(k, None)
        a.obsp["connectivities"] = kneighbors_graph(a.obsm["z"], NB, mode="connectivity",
                                                    include_self=True, n_jobs=8)
        a.obsp["distances"] = kneighbors_graph(a.obsm["z"], NB, mode="distance",
                                               include_self=True, n_jobs=8)
        sc.pp.neighbors(a, use_rep="z", n_neighbors=NB, method="umap")
        sc.tl.leiden(a, resolution=res, flavor="igraph", n_iterations=2, directed=False)
        a.obs["ident"] = "None"
        a.obs["leiden"] = a.obs["leiden"].astype(str)
        a, _av, rep = updateAttributes(a, [zl, zs])
        reps.append((rep, a.uns["topGene"]))
        ncl.append(len(rep))
    line = {"resolution": res, "neighbors": NB, "clusters": ncl}
    ok = True
    for s in range(NS - 1):
        dm = np.array(nodesDistance(reps[s][0], reps[s + 1][0],
                                    reps[s][1], reps[s + 1][1]), dtype=float)
        mu, sd = dm.mean(), dm.std()
        ne = len(connectNodes(dm.tolist()))
        line[f"s{s}->{s+1}"] = f"{dm.shape} 比值{mu/sd:.3f} 边{ne}"
        ok &= ne > 0
    line["全部连通"] = ok
    rows.append(line)
    print(f"  res={res:<4} 聚类{ncl}  " +
          "  ".join(f"{k}:{v}" for k, v in line.items() if "->" in k) +
          f"  {'✓ 连通' if ok else '✗'}", flush=True)

print("\n======== 汇总 ========")
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
good = [r for r in rows if r["全部连通"]]
print(f"\n可用分辨率: {[r['resolution'] for r in good] or '无'}")
print("=== SWEEP DONE ===", flush=True)
