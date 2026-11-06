"""按细胞算：每个矽肺细胞各自朝对照靠了没有。

质心距离会把"一半推对、一半推错"抵消掉。这里逐细胞算：
  d_before = ||z_i - 对照质心||,  d_after = ||z'_i - 对照质心||
  推对 = d_after < d_before
报 推对细胞占比、按亚群拆分、以及和随机基因集背景的比较。
"""
import os, sys, json, types, warnings, argparse
warnings.filterwarnings("ignore")
sys.modules.setdefault("turtle", types.ModuleType("turtle")); sys.modules["turtle"].distance = None
sys.path.insert(0, os.environ["UNAGI_SRC"])
import numpy as np, pandas as pd, scanpy as sc, torch
from scipy.sparse import issparse
from scipy.stats import binomtest
from UNAGI import UNAGI

np.random.seed(42)
ap = argparse.ArgumentParser()
ap.add_argument("--datadir", required=True); ap.add_argument("--task", required=True)
ap.add_argument("--genes", choices=["anti", "all"], default="all")
ap.add_argument("--nrandom", type=int, default=200)
ap.add_argument("--outdir", required=True)
args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)
DATA, OUT = args.datadir, args.outdir

WANT = ["Isorhamnetin", "Kaempferol", "Myricetrin", "Quercetin",
        "Luteolin", "Liquiritigenin", "Tetrandrine"]
CN = {"Isorhamnetin": "异鼠李素", "Kaempferol": "山奈酚", "Myricetrin": "杨梅苷",
      "Quercetin": "槲皮素", "Luteolin": "木犀草素", "Liquiritigenin": "甘草素",
      "Tetrandrine": "汉防己甲素(阳性)"}
SUB = ["Macro-CDKN1C", "Macro-CLEC4E", "Macro-SEPTIN7", "Macro-DEFB1"]

it = max(i for i in range(10) if os.path.exists(f"{DATA}/{i}/stagedata/dataset.h5ad"))
a = sc.read_h5ad(f"{DATA}/{it}/stagedata/dataset.h5ad")
X = (a.X.toarray() if issparse(a.X) else np.asarray(a.X)).astype("float32")
genes = a.var_names.tolist(); gidx = {g.upper(): i for i, g in enumerate(genes)}
grp = a.obs["group"].astype(str).values; ct = a.obs["celltype"].astype(str).values
sil = grp == "SIL"; sidx = np.where(sil)[0]

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

Z = encode(X); c = Z[~sil].mean(0)
d0 = np.linalg.norm(Z[sidx] - c, axis=1)
print(f">>> {sil.sum()} 个矽肺细胞，到对照质心的距离 {d0.mean():.3f} ± {d0.std():.3f}", flush=True)

DIRS = [os.environ["ITCM_DIR"],
        os.environ["UNAGI_DRUG"], os.environ["PROJECT_ROOT"]]
tend = {}
if args.genes == "anti":
    for sub in sorted(os.listdir(f"{DATA}/{it}/idremResults")):
        dj = f"{DATA}/{it}/idremResults/{sub}/DREM.json"
        if os.path.exists(dj):
            tt = json.loads(open(dj).read()[5:-2] + "]", strict=False); arr = np.array(tt[8])
            for g, t in zip([x.upper() for x in arr[1:, 0]],
                            arr[1:, -1].astype(float) - arr[1:, 1].astype(float)):
                tend.setdefault(g, []).append(t)
avg = {g: float(np.mean(v)) for g, v in tend.items()}
UP = {g for g, t in avg.items() if t > .1}; DN = {g for g, t in avg.items() if t < -.1}

def lfc_of(n):
    p = next((os.path.join(d, f"{n}_差异基因结果.csv") for d in DIRS
              if os.path.exists(os.path.join(d, f"{n}_差异基因结果.csv"))), None)
    df = pd.read_csv(p); df["gene"] = df["gene"].str.upper()
    s = df[(df["P.Value"] < 0.05) & df["gene"].isin(gidx) & (df["logFC"] != 0)]
    if args.genes == "anti":
        s = s[((s["logFC"] < 0) & s["gene"].isin(UP)) | ((s["logFC"] > 0) & s["gene"].isin(DN))]
    v = np.zeros(len(genes), dtype="float32")
    for g, l in zip(s["gene"], s["logFC"]): v[gidx[g]] = l
    return v, len(s)

def frac_improved(lfc):
    Xp = X.copy()
    f = np.power(2.0, lfc).astype("float32")
    Xp[sil] = np.log1p(np.expm1(np.clip(Xp[sil], 0, None)) * f[None, :]).astype("float32")
    Zp = encode(np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0))
    d1 = np.linalg.norm(Zp[sidx] - c, axis=1)
    return (d1 < d0), d1

print(f"\n=== 推对的细胞占比（基因集 {args.genes}）===", flush=True)
rows = {}
for n in WANT:
    lfc, k = lfc_of(n)
    if k == 0: continue
    imp, d1 = frac_improved(lfc)
    fr = imp.mean()
    bt = binomtest(int(imp.sum()), len(imp), 0.5).pvalue
    per_sub = {s: float(imp[ct[sidx] == s].mean()) if (ct[sidx] == s).sum() else np.nan
               for s in SUB}
    rows[n] = dict(compound=n, cn=CN[n], n_genes=k, frac_improved=fr,
                   p_binom=bt, mean_rel_change=float(((d1 - d0) / d0).mean() * 100), **per_sub)
    print(f"  {CN[n]:16s} n={k:4d}  推对 {fr*100:5.1f}%  P={bt:.2e}  "
          f"平均距离变化 {rows[n]['mean_rel_change']:+.2f}%", flush=True)
    print(f"      按亚群: " + "  ".join(
        f"{s.replace('Macro-','')} {per_sub[s]*100:.0f}%" for s in SUB
        if not np.isnan(per_sub[s])), flush=True)

tab = pd.DataFrame(rows).T.reset_index(drop=True)
ref = tab.loc[tab["n_genes"].idxmax(), "compound"]
lfc_ref, n_ref = lfc_of(ref); nz = lfc_ref[lfc_ref != 0]
rng = np.random.default_rng(7)
print(f"\n=== 随机基因集 {args.nrandom} 次（n={n_ref}）===", flush=True)
bg = []
for r in range(args.nrandom):
    v = np.zeros(len(genes), dtype="float32")
    pk = rng.choice(len(genes), size=min(n_ref, len(genes)), replace=False)
    v[pk] = rng.permutation(nz)[:len(pk)]
    bg.append(frac_improved(v)[0].mean())
    if (r + 1) % 50 == 0: print(f"    {r+1}/{args.nrandom}", flush=True)
bg = np.array(bg)
print(f">>> 随机背景推对占比 {bg.mean()*100:.1f}% ± {bg.std()*100:.1f}%", flush=True)
tab["z_vs_random"] = (tab["frac_improved"] - bg.mean()) / bg.std()
tab["p_vs_random"] = [2 * min((bg >= v).mean(), (bg <= v).mean()) or 1/len(bg)
                      for v in tab["frac_improved"]]
tab = tab.sort_values("frac_improved", ascending=False)
tab.to_csv(f"{OUT}/percell_{args.genes}.csv", index=False)
np.save(f"{OUT}/percell_bg_{args.genes}.npy", bg)
print(f"\n======== 最终（{args.genes}）========")
cols = ["cn", "n_genes", "frac_improved", "mean_rel_change", "z_vs_random", "p_vs_random"] + SUB
print(tab[cols].round(4).to_string(index=False))
print("=== PERCELL DONE ===", flush=True)
