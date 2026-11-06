"""UNAGI v8：stage 改为用户自己的四个亚群（按拟时序排序），不再用我自编的极化三分位。

v6 只有 3001 个高变基因，GPS 的 2198 个 landmark 只有 251 个能进来，
每个成分最后只剩 3~23 个基因，不足以支撑扰动分析。

v7 基因集 = 3000 高变 + 全部 GPS landmark + 四亚群标志 + 靶点 + M1/M2 极化基因
v7 的 stage 是 (M1_score − M2_score) 三等分 —— 那两个分数是脚本自己算的，
用户从未做过 M1/M2 极化；而且它横切四个亚群，方向还与用户拟时序相反
（stage 0→2 平均拟时序 18.34→12.98）。

v8 stage = Macro-CDKN1C → Macro-CLEC4E → Macro-SEPTIN7 → Macro-DEFB1
（拟时序 3.95 → 10.40 → 21.90 → 25.20，即用户已发表的轨迹顺序），
四期结构也与 UNAGI 原文 IPF 数据（control + 3 个疾病等级）一致。
基因集、res=2.0、ziln、三轮迭代与 v7 完全相同，保证可比。
"""
import os
from pathlib import Path
import argparse, json, sys, types
sys.modules.setdefault("turtle", types.ModuleType("turtle")); sys.modules["turtle"].distance = None
sys.path.insert(0, os.environ["UNAGI_SRC"])
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp, torch
from UNAGI import UNAGI

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--nhvg", type=int, default=3000)
ap.add_argument("--res", type=float, default=2.0)
ap.add_argument("--nb", type=int, default=30)
ap.add_argument("--edge-cutoff", type=float, default=0.10,
                help="connect_edges_cutoff；默认 0.05 时 stage0->1 最小 p=0.0504 差一点连不上")
args = ap.parse_args()
DATA = Path(args.outdir)
SOURCE = Path(os.environ["SIL_H5AD"])
PLOTDATA = os.environ["SIL_PLOTDATA"]
IDREM = os.environ["UNAGI_IDREM"]
if DATA.exists(): raise RuntimeError(f"已存在: {DATA}")

M1 = ["NOS2","CD86","IL1B","TNF","IL6","CXCL9","CXCL10","SOCS3"]
M2 = ["CD163","MRC1","ARG1","IL10","TGFB1"]
SUBMK = ["CLEC4E","SEPTIN7","CDKN1C","DEFB1","PLAC8","FABP4","SPP1","SIGLEC10"]
TARGETS = ["MAP2K1","PPARG"]
gps = pd.read_csv(os.environ["GPS_SELECTED_GENES"])
GPSLM = [str(x).upper() for x in gps.iloc[:, 0]]
print(f">>> GPS landmark {len(GPSLM)} 个", flush=True)

a = sc.read_h5ad(SOURCE)
P = pd.read_csv(PLOTDATA).set_index("cell")
assert set(a.obs_names) == set(P.index)
a.obs["pseudotime"] = P.loc[a.obs_names, "pseudotime"].values
a.obs["celltype"] = a.obs["celltype"].astype(str)
print(f">>> 输入 {a.n_obs} 细胞 x {a.n_vars} 基因", flush=True)

sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
m1e = [g for g in M1 if g in a.var_names]; m2e = [g for g in M2 if g in a.var_names]
sc.tl.score_genes(a, m1e, score_name="M1_score", random_state=0)
sc.tl.score_genes(a, m2e, score_name="M2_score", random_state=0)
a.obs["polar_score"] = a.obs["M1_score"] - a.obs["M2_score"]

sc.pp.highly_variable_genes(a, n_top_genes=args.nhvg)
force = set(GPSLM) | set(SUBMK) | set(TARGETS) | set(M1) | set(M2)
hit = [g for g in a.var_names if g.upper() in force]
a.var.loc[hit, "highly_variable"] = True
a = a[:, a.var["highly_variable"]].copy()
inset = {g.upper() for g in a.var_names}
print(f">>> 基因集 {a.n_vars} 个（高变 {args.nhvg} + 强制加入 {len(hit)}）", flush=True)
print(f"    GPS landmark 进来 {len(set(GPSLM)&inset)}/{len(GPSLM)}", flush=True)
for nm, gl in [("亚群标志", SUBMK), ("靶点", TARGETS), ("M1", M1), ("M2", M2)]:
    print(f"    {nm}: {sum(g in inset for g in gl)}/{len(gl)}  缺 {[g for g in gl if g not in inset]}", flush=True)
sc.tl.pca(a, n_comps=50, svd_solver="arpack")

# stage = 用户自己的四个亚群，按其拟时序均值排序（= 已发表的轨迹顺序）
names = ["Macro-CDKN1C", "Macro-CLEC4E", "Macro-SEPTIN7", "Macro-DEFB1"]
n_stage = len(names)
obs_ct = set(a.obs["celltype"].unique())
assert obs_ct == set(names), f"亚群对不上: {obs_ct} vs {set(names)}"
pt_order = a.obs.groupby("celltype", observed=True)["pseudotime"].mean().sort_values()
print("\n=== 四个亚群的拟时序均值（核对顺序）===\n" + pt_order.round(2).to_string(), flush=True)
assert list(pt_order.index) == names, f"拟时序顺序与预期不符: {list(pt_order.index)}"
a.obs["stage"] = a.obs["celltype"].map({c: i for i, c in enumerate(names)}).astype(int)
a.obs["stage_name"] = a.obs["celltype"].astype(str)
ct_g = pd.crosstab(a.obs["stage_name"], a.obs["group"]).reindex(names)
print("\n=== stage x 分组 ===\n" + ct_g.to_string(), flush=True)
print("\n=== stage x 拟时序 ===\n" +
      a.obs.groupby("stage", observed=True)["pseudotime"].agg(["mean","min","max"]).round(2).to_string(),
      flush=True)

DATA.mkdir(parents=True)
ct_g.to_csv(DATA / "stage_by_group.csv")
a.obs[["stage","stage_name","group","celltype","pseudotime","M1_score","M2_score","polar_score"]]\
 .to_csv(DATA / "cell_stage.csv")
for i in range(n_stage):
    sub = a[a.obs["stage"] == i].copy()
    if sp.issparse(sub.X): sub.X = sub.X.toarray()
    sub.write_h5ad(DATA / f"{i}.h5ad")
    print(f">>> stage {i} ({names[i]}): {sub.n_obs} 细胞", flush=True)
(DATA / "manifest.json").write_text(json.dumps({
    "version": "v8", "source": str(SOURCE), "n_genes": int(a.n_vars),
    "n_hvg": args.nhvg, "forced_in": len(hit),
    "gps_landmark_in": len(set(GPSLM) & inset),
    "stage_basis": "用户四个亚群按拟时序排序: Macro-CDKN1C -> CLEC4E -> SEPTIN7 -> DEFB1",
    "connect_edges_cutoff": args.edge_cutoff,
    "connect_edges_cutoff_note": "默认 0.05 时 stage0(CDKN1C)->1(CLEC4E) 最小 p=0.0504，刚好卡住；0.10 时三对各有 13/16/12 条边",
    "model": f"ziln, max_iter=3, epoch_initial=20, epoch_iter=10, BATCHSIZE=256, GCN=False, CPO=False, res={args.res}, nb={args.nb}",
    "why_v8": "v7 的 stage 是脚本自编的 M1/M2 极化三分位，横切四个亚群且与用户拟时序方向相反",
}, ensure_ascii=False, indent=2), encoding="utf-8")

_orig = sc.pp.neighbors
sc.pp.neighbors = lambda *a_, **kw: _orig(*a_, **{**kw, "transformer": "sklearn"})
import scipy.sparse as _sp
from UNAGI.utils import CPO_utils as _cpo
_og = _cpo.get_neighbors
def _gn(stagedata, *a_, **kw):
    for _ad in stagedata:
        for _k in ("distances", "connectivities"):
            if _k in _ad.obsp and not _sp.isspmatrix_csr(_ad.obsp[_k]):
                _ad.obsp[_k] = _sp.csr_matrix(_ad.obsp[_k])
    return _og(stagedata, *a_, **kw)
_cpo.get_neighbors = _gn
from UNAGI.train import runner as _rmod
_rmod.get_neighbors = _gn
def _plain(o):
    if isinstance(o, (list, tuple)): return [_plain(x) for x in o]
    return o.item() if isinstance(o, np.generic) else o
from UNAGI.dynamic_regulatory_networks import processIDREM as _pid
_or, _og2 = _pid.runIdrem, _pid.getClusterPaths
_pid.runIdrem = _rmod.runIdrem = lambda paths, *a_, **kw: _or(_plain(paths), *a_, **kw)
_pid.getClusterPaths = _rmod.getClusterPaths = lambda *a_, **kw: _plain(_og2(*a_, **kw))
_rmod.UNAGI_runner.resolutions = property(lambda s: [args.res]*s.total_stage, lambda s,v: None)
_rmod.UNAGI_runner.neighbor_parameters = property(lambda s: [args.nb]*s.total_stage, lambda s,v: None)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"\n>>> 开始训练 v7，device={dev}", flush=True)
model = UNAGI(random_seed=8848)
model.setup_data(DATA/"0.h5ad", total_stage=n_stage, stage_key="stage",
                 label_key="celltype", neighbors=25, threads=8, calculate_cell_graph=False)
model.setup_training(task="branch1_v8", dist="ziln", device=dev, GPU=torch.cuda.is_available(),
                     epoch_iter=10, epoch_initial=20, max_iter=3, BATCHSIZE=256, GCN=False)
model.run_UNAGI(idrem_dir=IDREM, CPO=False, connect_edges_cutoff=args.edge_cutoff)
print("=== V7 TRAIN DONE ===", flush=True)
