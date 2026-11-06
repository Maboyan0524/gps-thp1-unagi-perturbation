"""照搬 run_drug_perturbation_antiCOPD.py 的做法，换成矽肺 v7 + 7 个成分。

  疾病方向 ← iDREM tendency（末期 − 初期），阈值 0.1
  药物基因 ← P.Value<0.05 + 在模型基因里 + 方向与疾病相反（取交集）
  UNAGI 扰动 log2fc=0.5 + 基因数匹配的随机背景 + analysis
"""
import warnings; warnings.filterwarnings("ignore")
import sys,types,os,gc,json,ast,re,argparse
sys.modules.setdefault("turtle",types.ModuleType("turtle")); sys.modules["turtle"].distance=None
sys.path.insert(0,os.environ["UNAGI_SRC"])
import pandas as pd; pd.options.future.infer_string=False
import numpy as np, scanpy as sc, torch, pickle
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument("--source",choices=["itcm","gps","thp1"],default="itcm")
ap.add_argument("--tend",type=float,default=0.1)
ap.add_argument("--nrandom",type=int,default=100)
ap.add_argument("--datadir",default=os.environ["MODEL_DATA"])
ap.add_argument("--task",default="branch1_v7")
args=ap.parse_args()
DATA=Path(args.datadir)
TASK=args.task; OUT=str(DATA/"analysis_full"); os.makedirs(OUT,exist_ok=True)
ITCM=os.environ["ITCM_DIR"]
GPSDIR=os.environ["GPS_CONTINUOUS"]
ZH={"Isorhamnetin":"异鼠李素","Kaempferol":"山奈酚","Myricetin":"杨梅素","Quercetin":"槲皮素",
    "Luteolin":"木犀草素","Liquiritigenin":"甘草素","Tetrandrine":"汉防己甲素"}
POS="Tetrandrine"
it=max(i for i in range(10) if (DATA/str(i)/"stagedata"/"dataset.h5ad").exists())
ds=DATA/str(it)/"stagedata"/"dataset.h5ad"; idrem=DATA/str(it)/"idremResults"
adata=sc.read_h5ad(ds); hvg={g.upper() for g in adata.var_names}
edges=ast.literal_eval(re.sub(r"np\.\w+\(([-\d.eE+]+)\)",r"\1",
                              (DATA/str(it)/"edges.txt").read_text().strip()))
print(f">>> v7 第 {it} 轮  {adata.shape}  模型基因 {len(hvg)}",flush=True)
# —— 1. 疾病方向（iDREM tendency）——
tend={}
for name in os.listdir(idrem):
    if name.startswith("."): continue
    p=os.path.join(idrem,name,"DREM.json")
    if not os.path.exists(p): continue
    lf=open(p).read(); lf=lf[5:-2]+"]"
    tt=json.loads(lf,strict=False); temp=np.array(tt[8])
    gs=[g.upper() for g in temp[1:,0].tolist()]
    td=temp[1:,-1].astype(float)-temp[1:,1].astype(float)
    for g,t in zip(gs,td): tend.setdefault(g,[]).append(t)
avg={g:np.mean(v) for g,v in tend.items()}
dis_up={g for g,t in avg.items() if t> args.tend}
dis_dn={g for g,t in avg.items() if t<-args.tend}
print(f">>> iDREM {len(avg)} 基因；疾病上调 {len(dis_up)}，疾病下调 {len(dis_dn)}  (阈值 {args.tend})",flush=True)
# —— 2. 药物基因（与疾病方向相反的交集）——
drug_genes={}
if args.source=="thp1":
    import pickle as _pk
    dgraw=_pk.load(open(os.environ["THP1_DRUG_GENES"],"rb"))
    ZH2EN={"木犀草素":"Luteolin","槲皮素":"Quercetin","山奈酚":"Kaempferol",
           "杨梅素":"Myricetin","异鼠李素":"Isorhamnetin","甘草素":"Liquiritigenin",
           "汉防己甲素":"Tetrandrine"}
    for zh,items in dgraw.items():
        en=ZH2EN.get(zh,zh)
        drug_genes[en]=[x for x in items if x.partition(":")[0] in hvg]
    ZH.update({v:k for k,v in ZH2EN.items()})
elif args.source=="itcm":
    FILES={k:f"{ITCM}/{k}_差异基因结果.csv" for k in ZH if k!="Myricetin"}
    for k,f in FILES.items():
        df=pd.read_csv(f); df["gene"]=df["gene"].astype(str).str.upper()
        df=df[(df["P.Value"]<0.05)&df["gene"].isin(hvg)&(df["logFC"]!=0)]
        mk=((df["logFC"]<0)&df["gene"].isin(dis_up))|((df["logFC"]>0)&df["gene"].isin(dis_dn))
        df=df[mk]
        drug_genes[k]=[f"{r['gene']}:{'+' if r['logFC']>0 else '-'}" for _,r in df.iterrows()]
else:
    fl=pd.read_csv(f"{GPSDIR}/flavonoids_MEDIAN_signed.csv",index_col=0)
    ex=pd.read_csv(f"{GPSDIR}/expanded_MEDIAN_signed.csv",index_col=0)
    M=fl.join(ex[["Tetrandrine"]],how="outer"); M.index=[str(x).upper() for x in M.index]
    ren={"isorhamnetin":"Isorhamnetin","myricetin":"Myricetin","luteolin":"Luteolin",
         "liquiritigenin":"Liquiritigenin","quercetin":"Quercetin","kaempferol":"Kaempferol"}
    M=M.rename(columns=ren)
    for k in M.columns:
        s=M[k].dropna(); s=s[s.index.isin(hvg)]
        mk=((s<0)&s.index.isin(dis_up))|((s>0)&s.index.isin(dis_dn))
        s=s[mk]
        drug_genes[k]=[f"{g}:{'+' if v>0 else '-'}" for g,v in s.items()]
for k,v in drug_genes.items(): print(f"    {ZH[k]:10s} {len(v):5d} 个逆疾病方向基因",flush=True)
if any(len(v)==0 for v in drug_genes.values()):
    sys.exit("有成分交集为空，调 --tend")
med=int(np.median([len(v) for v in drug_genes.values()]))
print(f">>> 随机背景基因数取中位数 {med}",flush=True)
adata.uns={"edges":edges,"data_drug_overlap_genes":drug_genes,
           "drug-gene_len_dict":{k:len(v) for k,v in drug_genes.items()}}
# —— 3. 跑 UNAGI ——
from UNAGI.perturbations.perturbation import perturbation as UNAGIPerturbation
from UNAGI.perturbations import get_top_compounds
runner=UNAGIPerturbation(adata,str(DATA/"model_save"/f"{TASK}_{it}.pth"),str(idrem))
cuda=torch.cuda.is_available()
print("\n>>> 药物扰动 ...",flush=True); runner.run("drug",0.5,inplace=True,CUDA=cuda); gc.collect()
print(f">>> 随机背景 × {args.nrandom} ...",flush=True)
runner.run("random_drug_background",0.5,inplace=True,random_times=args.nrandom,
           random_genes=med,CUDA=cuda); gc.collect()
print(">>> analysis ...",flush=True)
runner.analysis("drug",0.5,"all",overall_perturbation_analysis=True)
# 把 UNAGI 自己存的东西原样导出，后续作图直接用，不重新实现
import pickle as _pk
_dump={}
for k in ("drug_perturbation_score","drug_perturbation_deltaD",
          "random_background_perturbation_deltaD"):
    if k in runner.adata.uns: _dump[k]=runner.adata.uns[k]
with open(f"{OUT}/anti_{args.source}_uns.pkl","wb") as _f: _pk.dump(_dump,_f)
print(">>> 已导出 uns:",list(_dump),flush=True)
_ps=_dump.get("drug_perturbation_score",{}).get("0.5",{})
print(">>> drug_perturbation_score['0.5'] 的 key:",list(_ps)[:25],flush=True)
if "overall" in _ps: print(">>> overall 的 key:",list(_ps["overall"]),flush=True)
top=get_top_compounds(runner.adata,perturb_change=0.5,top_n=len(drug_genes)).set_index("compound")
rows=[]
for k in drug_genes:
    rows.append(dict(compound=k,cn=ZH[k],n_genes=len(drug_genes[k]),
                     score=float(top.loc[k,"perturbation score"]),
                     p=float(top.loc[k,"pval_adjusted"])))
R=pd.DataFrame(rows).sort_values("score",ascending=False)
R["rank"]=range(1,len(R)+1)
R.to_csv(f"{OUT}/anti_{args.source}_scores.csv",index=False)
with open(f"{OUT}/anti_{args.source}_genes.pkl","wb") as f: pickle.dump(drug_genes,f)
print(f"\n{'排名':4s}{'成分':10s}{'得分':>9s}{'P':>10s}{'基因数':>7s}",flush=True)
for _,r in R.iterrows():
    mk="  ← 阳性对照" if r["compound"]==POS else ""
    s="***" if r["p"]<.001 else "**" if r["p"]<.01 else "*" if r["p"]<.05 else "n.s."
    print(f"{r['rank']:<4d}{r['cn']:10s}{r['score']:9.4f}{r['p']:10.4f}{r['n_genes']:7d}  {s}{mk}",flush=True)
print(f"\n=== ANTI {args.source.upper()} DONE ===",flush=True)
