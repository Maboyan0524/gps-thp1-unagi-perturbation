"""给药逆转 v2 —— 三处修正后的版本。

和 v1 的差别：
  1. 倍数在**线性空间**算：L' = log1p(2^logFC * expm1(L))，不再直接乘 log 值
  2. 主指标换成论文的判据：给药后到对照的距离 vs 给药前到对照的距离
       gap_closed% = (L_CS - L_CP) / L_CS * 100   正 = 朝对照靠
  3. 显著性**对零检验**（按细胞 bootstrap 2000 次），回答"有没有往正常推"；
     随机基因集那套保留为次要的严格对照，回答"比乱改基因更厉害吗"

两个问题分开报，不混为一谈。
"""
import os, sys, json, types, warnings, argparse
warnings.filterwarnings("ignore")
sys.modules.setdefault("turtle", types.ModuleType("turtle")); sys.modules["turtle"].distance = None
sys.path.insert(0, os.environ["UNAGI_SRC"])
import numpy as np, pandas as pd, scanpy as sc, torch
from scipy.sparse import issparse
from UNAGI import UNAGI

np.random.seed(42)
ap = argparse.ArgumentParser()
ap.add_argument("--datadir", required=True); ap.add_argument("--task", required=True)
ap.add_argument("--genes", choices=["anti", "all"], default="all")
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--nrandom", type=int, default=500)
ap.add_argument("--outdir", required=True)
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
DATA, OUT = args.datadir, args.outdir

WANT = ["Isorhamnetin", "Kaempferol", "Myricetrin", "Quercetin",
        "Luteolin", "Liquiritigenin", "Tetrandrine"]
CN = {"Isorhamnetin": "异鼠李素", "Kaempferol": "山奈酚", "Myricetrin": "杨梅苷",
      "Quercetin": "槲皮素", "Luteolin": "木犀草素", "Liquiritigenin": "甘草素",
      "Tetrandrine": "汉防己甲素(阳性对照)"}

it = max(i for i in range(10) if os.path.exists(f"{DATA}/{i}/stagedata/dataset.h5ad"))
a = sc.read_h5ad(f"{DATA}/{it}/stagedata/dataset.h5ad")
X = (a.X.toarray() if issparse(a.X) else np.asarray(a.X)).astype("float32")
genes = a.var_names.tolist(); gidx = {g.upper(): i for i, g in enumerate(genes)}
grp = a.obs["group"].astype(str).values
sil = grp == "SIL"; ctl = ~sil
print(f">>> {a.shape}  矽肺 {sil.sum()} / 对照 {ctl.sum()}", flush=True)
print(f">>> X 范围 {X.min():.3f}~{X.max():.3f}（log1p 后）", flush=True)

model = UNAGI(random_seed=8848)
model.input_dim = a.n_vars; model.data_folder = DATA; model.data_path = DATA
model.stage_key = "stage"; model.label_key = "celltype"; model.ns = 3
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
model.setup_training(task=args.task, dist="ziln", device=dev, GPU=torch.cuda.is_available(),
                     epoch_iter=10, epoch_initial=20, max_iter=3, BATCHSIZE=256, GCN=False)
tr = model.unagi_trainer
for _m in (getattr(model, "model", None), getattr(tr, "model", None)):
    if _m is not None: _m.to(dev)
def encode(Xp):
    o = a.copy(); o.X = Xp
    return tr.get_latent_representation(o, it, DATA)[2]

print(">>> 编码原始 ...", flush=True)
Z = encode(X)
c_ctl = Z[ctl].mean(0)
L_CS = float(np.linalg.norm(Z[sil].mean(0) - c_ctl))
print(f">>> L_CS（矽肺质心 → 对照质心）= {L_CS:.4f}", flush=True)

# ── 药物表 ─────────────────────────────────────────────────
DIRS = [os.environ["ITCM_DIR"],
        os.environ["UNAGI_DRUG"], os.environ["PROJECT_ROOT"]]
def path_of(n):
    for d in DIRS:
        p = os.path.join(d, f"{n}_差异基因结果.csv")
        if os.path.exists(p): return p
    return None
tend = {}
if args.genes == "anti":
    for sub in sorted(os.listdir(f"{DATA}/{it}/idremResults")):
        dj = f"{DATA}/{it}/idremResults/{sub}/DREM.json"
        if not os.path.exists(dj): continue
        tt = json.loads(open(dj).read()[5:-2] + "]", strict=False)
        arr = np.array(tt[8])
        for g, t in zip([x.upper() for x in arr[1:, 0]],
                        arr[1:, -1].astype(float) - arr[1:, 1].astype(float)):
            tend.setdefault(g, []).append(t)
avg = {g: float(np.mean(v)) for g, v in tend.items()}
UP = {g for g, t in avg.items() if t > .1}; DN = {g for g, t in avg.items() if t < -.1}

def logfc_vec(name):
    df = pd.read_csv(path_of(name)); df["gene"] = df["gene"].str.upper()
    sig = df[(df["P.Value"] < 0.05) & df["gene"].isin(gidx) & (df["logFC"] != 0)]
    if args.genes == "anti":
        sig = sig[((sig["logFC"] < 0) & sig["gene"].isin(UP)) |
                  ((sig["logFC"] > 0) & sig["gene"].isin(DN))]
    v = np.zeros(len(genes), dtype="float32")
    for g, l in zip(sig["gene"], sig["logFC"]): v[gidx[g]] = l
    return v, len(sig)

def apply_fc(lfc):
    """修正点1：倍数在线性空间生效。L' = log1p(2^logFC * expm1(L))"""
    Xp = X.copy()
    f = np.power(2.0, lfc).astype("float32")
    sub = Xp[sil]
    Xp[sil] = np.log1p(np.expm1(np.clip(sub, 0, None)) * f[None, :]).astype("float32")
    return np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)

def measure(lfc):
    Zp = encode(apply_fc(lfc))
    L_CP = float(np.linalg.norm(Zp[sil].mean(0) - c_ctl))
    return L_CP, Zp

print(f"\n=== 主指标：缩短了到对照的距离多少（基因集 {args.genes}）===", flush=True)
rows, ZP = [], {}
for n in WANT:
    lfc, k = logfc_vec(n)
    if k == 0:
        print(f"  {n}: 无可用基因"); continue
    L_CP, Zp = measure(lfc); ZP[n] = Zp
    closed = (L_CS - L_CP) / L_CS * 100
    rows.append(dict(compound=n, cn=CN[n], n_genes=k, L_CS=L_CS, L_CP=L_CP,
                     gap_closed_pct=closed, reversal=bool(L_CP < L_CS)))
    print(f"  {n:16s}({CN[n]:12s}) n={k:4d}  L_CP={L_CP:.4f}  "
          f"缩短 {closed:+6.2f}%  {'逆转✓' if L_CP < L_CS else '未逆转'}", flush=True)
tab = pd.DataFrame(rows)

# ── 修正点3：按细胞 bootstrap，对零检验 ─────────────────────
print(f"\n=== bootstrap {args.nboot} 次（按矽肺细胞重抽），对零检验 ===", flush=True)
sidx = np.where(sil)[0]
rng = np.random.default_rng(42)
boot_idx = [rng.choice(len(sidx), size=len(sidx), replace=True) for _ in range(args.nboot)]
for n in tab["compound"]:
    Zp = ZP[n]
    vals = np.empty(args.nboot)
    for b, bi in enumerate(boot_idx):
        cs = np.linalg.norm(Z[sidx[bi]].mean(0) - c_ctl)
        cp = np.linalg.norm(Zp[sidx[bi]].mean(0) - c_ctl)
        vals[b] = (cs - cp) / cs * 100
    lo, hi = np.percentile(vals, [2.5, 97.5])
    pv = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    pv = max(pv, 1.0 / args.nboot)
    tab.loc[tab.compound == n, ["ci_lo", "ci_hi", "p_vs_zero"]] = lo, hi, pv
    print(f"  {n:16s} 缩短 {tab.loc[tab.compound==n,'gap_closed_pct'].iloc[0]:+6.2f}%  "
          f"95%CI [{lo:+.2f}, {hi:+.2f}]  P(对零) = {pv:.4f}", flush=True)

# ── 次要：随机基因集严格对照 ────────────────────────────────
ref = tab.loc[tab["n_genes"].idxmax(), "compound"]
lfc_ref, n_ref = logfc_vec(ref)
nz = lfc_ref[lfc_ref != 0]
print(f"\n=== 次要对照：{args.nrandom} 次随机基因集（n={n_ref}，以 {ref} 为准）===", flush=True)
bg = []
for r in range(args.nrandom):
    v = np.zeros(len(genes), dtype="float32")
    pick = rng.choice(len(genes), size=min(n_ref, len(genes)), replace=False)
    v[pick] = rng.permutation(nz)[:len(pick)]
    L_CP, _ = measure(v)
    bg.append((L_CS - L_CP) / L_CS * 100)
    if (r + 1) % 100 == 0: print(f"    {r+1}/{args.nrandom}", flush=True)
bg = np.array(bg)
print(f">>> 随机背景缩短 {bg.mean():+.3f}% ± {bg.std():.3f}", flush=True)
np.save(f"{OUT}/bg_v2_{args.genes}.npy", bg)
tab["z_vs_random"] = (tab["gap_closed_pct"] - bg.mean()) / bg.std()
tab["p_vs_random"] = [2 * min((bg >= v).mean(), (bg <= v).mean()) or 1/len(bg)
                      for v in tab["gap_closed_pct"]]
tab = tab.sort_values("gap_closed_pct", ascending=False).reset_index(drop=True)
tab.to_csv(f"{OUT}/reversal_v2_{args.genes}.csv", index=False)
print(f"\n======== 最终（{args.genes}）========")
print(tab[["cn", "n_genes", "gap_closed_pct", "ci_lo", "ci_hi", "p_vs_zero",
           "z_vs_random", "p_vs_random"]].round(4).to_string(index=False))
print("=== V2 DONE ===", flush=True)
