"""严格照搬 $UNAGI_SRC/make_fig5c_paga.py 的视觉规范，换成矽肺 6 个成分。
约定：最近 N 邻居干净点云 / C,S,P 节点 / PAGA 弧边 / 实线 L_CS,L_CP / S→P 虚线弧箭头 / 无椭圆。
方向按官方教程 translate_direction：':+' → ×2.0 上调，':-' → ×0.5 下调。
"""
import warnings; warnings.filterwarnings("ignore")
import os,sys,types,pickle,ast,re,gc
sys.modules.setdefault("turtle",types.ModuleType("turtle")); sys.modules["turtle"].distance=None
sys.path.insert(0,os.environ["UNAGI_SRC"])
import pandas as pd; pd.options.future.infer_string=False
import numpy as np, scanpy as sc
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA as skPCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager as fm
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    try: fm.fontManager.addfont(f)
    except Exception: pass
_c=sorted({x.name for x in fm.fontManager.ttflist if "CJK" in x.name or "WenQuanYi" in x.name})
plt.rcParams.update({"font.family":"sans-serif",
                     "font.sans-serif":_c[:1]+["Arial","Helvetica","DejaVu Sans"],
                     "font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":0.9,"xtick.major.width":0.9,"ytick.major.width":0.9,
                     "pdf.fonttype":42,"axes.unicode_minus":False})
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument("--datadir",default=os.environ["MODEL_DATA"])
_ap.add_argument("--task",default="branch1_v8")
_ap.add_argument("--tag",default="v8")
_a=_ap.parse_args()
DATA=Path(_a.datadir)
A=str(DATA/"analysis_full"); OUT=A+"/figures"; it=2; FC=2.0
os.makedirs(OUT,exist_ok=True)
ZH={"Luteolin":"木犀草素","Isorhamnetin":"异鼠李素","Kaempferol":"山奈酚",
    "Tetrandrine":"汉防己甲素","Quercetin":"槲皮素","Liquiritigenin":"甘草素"}
POS="Tetrandrine"; LEAD="Luteolin"
CTRL_C="#2271B2"; DIS_C="#E3633A"; LINE_K="#111111"
PCOLS={"Luteolin":"#C00000","Isorhamnetin":"#1F8B4C","Kaempferol":"#7B4F9E",
       "Tetrandrine":"#4C9FC4","Quercetin":"#D98E04","Liquiritigenin":"#777777"}
N_NEAREST=300
SCORES=pd.read_csv(f"{A}/anti_itcm_scores.csv").set_index("compound")
dg=pickle.load(open(f"{A}/anti_itcm_genes.pkl","rb"))
adata=sc.read_h5ad(DATA/str(it)/"stagedata"/"dataset.h5ad")
edges=ast.literal_eval(re.sub(r"np\.\w+\(([-\d.eE+]+)\)",r"\1",
                              (DATA/str(it)/"edges.txt").read_text().strip()))
adata.uns={"edges":edges,"data_drug_overlap_genes":dg,
           "drug-gene_len_dict":{k:len(v) for k,v in dg.items()}}
from UNAGI.perturbations.perturbation import perturbation as UP
r=UP(adata,str(DATA/"model_save"/f"{_a.task}_{it}.pth"),str(DATA/str(it)/"idremResults"))
pb=r.pb; names=r.adata.obs_names
r.adata.obs["stage"]=r.adata.obs["stage"].astype(str)
sv=r.adata.obs["stage"].values
ctrl_mask=sv=="0"; dis_mask=sv!="0"
z=pb.data.obsm["embdedding"]
ctrl_z=z[ctrl_mask].mean(0); dis_z=z[dis_mask].mean(0)
d0=float(np.linalg.norm(dis_z-ctrl_z))
axis_n=(ctrl_z-dis_z)/d0; dis_proj=z[dis_mask]@axis_n
print(f">>> C(stage0) n={ctrl_mask.sum()}  S(stage1+2) n={dis_mask.sum()}  L_CS={d0:.4f}",flush=True)
print(">>> stage0 的组成:",dict(r.adata.obs.loc[ctrl_mask,"group"].value_counts()),flush=True)
pca=skPCA(n_components=3,random_state=42).fit(z); X=pca.transform(z)
ad2=r.adata.copy(); ad2.obsm["embdedding"]=z
sc.pp.neighbors(ad2,use_rep="embdedding",n_neighbors=15,metric="euclidean")
ad2.obs["stage"]=ad2.obs["stage"].astype("category")
sc.tl.paga(ad2,groups="stage")
conn=ad2.uns["paga"]["connectivities"].toarray()
cats=ad2.obs["stage"].cat.categories.tolist(); i0=cats.index("0")
score_cs=float(np.mean([conn[i0,cats.index(s)] for s in ("1","2")]))
print(f">>> PAGA C–S 连通度 {score_cs:.3f}",flush=True)
paga_lw=lambda s:0.8+s*(6.0-0.8)
def nearest_sub(c,cen,n=N_NEAREST):
    return c[np.argsort(np.linalg.norm(c-cen,axis=1))[:n]]
def node(ax,p,fc,s=90,zo=12,marker="o"):
    ax.scatter(*p,s=s,c=fc,edgecolors="white",linewidths=1.2,zorder=zo,marker=marker)
def lbl(ax,p,t,dy=0.,dx=0.,color=LINE_K,fs=13):
    ax.text(p[0]+dx,p[1]+dy,t,fontsize=fs,color=color,fontweight="bold",ha="center",
            va="center",zorder=22,bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=.85))
def dist_line(ax,a,b,label,sign=1,scale=1.0):
    ax.plot([a[0],b[0]],[a[1],b[1]],color=LINE_K,lw=1.8,zorder=6,solid_capstyle="round")
    mid=(a+b)/2; dx,dy=b[0]-a[0],b[1]-a[1]; ln=np.hypot(dx,dy)
    if ln>1e-6:
        nx,ny=-dy/ln,dx/ln
        ax.text(mid[0]+nx*sign*0.20*scale,mid[1]+ny*sign*0.20*scale,label,fontsize=8.5,
                color=LINE_K,fontstyle="italic",ha="center",va="center",zorder=16)
ctrl_cent=X[ctrl_mask].mean(0)[:2]; dis_cent=X[dis_mask].mean(0)[:2]
pts_c=nearest_sub(X[ctrl_mask,:2],ctrl_cent); pts_s=nearest_sub(X[dis_mask,:2],dis_cent)
ORD=SCORES.sort_values("score",ascending=False).index.tolist()
G={}
for c in ORD:
    gid={}
    for item in dg[c]:
        n,_,sg=item.partition(":"); gid[n]=FC if sg=="+" else 1.0/FC
    pb.perturb_input_data(names,gid,mode="direct")
    pz=pb.data.obsm["perturbed_embedding"].copy()
    del pb.data.obsm["perturbed_embedding"]; gc.collect()
    p2=pca.transform(pz); pcent=p2[dis_mask].mean(0)[:2]
    G[c]=dict(pz=pz,p2=p2,cent=pcent,pts=nearest_sub(p2[dis_mask,:2],pcent))
    da=float(np.linalg.norm(pz[dis_mask].mean(0)-ctrl_z))
    G[c]["pct"]=(d0-da)/d0*100; G[c]["da"]=da
    t_,p_=stats.ttest_rel(pz[dis_mask]@axis_n,dis_proj); G[c]["p"]=p_
    print(f"    {ZH[c]:10s} L_CP={da:.4f}  Δdist={G[c]['pct']:+.2f}%  p={p_:.2e}",flush=True)
allp=np.vstack([pts_c,pts_s]+[G[c]["pts"] for c in ORD])
xl=(allp[:,0].min()-.4,allp[:,0].max()+.4); yl=(allp[:,1].min()-.4,allp[:,1].max()+.4)
SCALE=max(np.ptp(xl),np.ptp(yl))/6.0
fig,axes=plt.subplots(2,3,figsize=(16.5,10.2))
for ax,c in zip(axes.ravel(),ORD):
    col=PCOLS[c]; g=G[c]
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.scatter(*pts_c.T,c=CTRL_C,s=16,alpha=.45,linewidths=0,zorder=2)
    ax.scatter(*pts_s.T,c=DIS_C,s=16,alpha=.45,linewidths=0,zorder=2)
    ax.scatter(*g["pts"].T,c=col,s=16,alpha=.55,linewidths=0,zorder=3)
    if score_cs>=0.05:
        ax.add_patch(FancyArrowPatch(tuple(ctrl_cent),tuple(dis_cent),arrowstyle="-",
                     connectionstyle="arc3,rad=0.22",color="#BBBBBB",
                     lw=paga_lw(score_cs),alpha=.80,zorder=4))
    dist_line(ax,dis_cent,ctrl_cent,"$L_{CS}$",sign=-1,scale=SCALE)
    dist_line(ax,g["cent"],ctrl_cent,"$L_{CP}$",sign=1,scale=SCALE)
    arc=FancyArrowPatch(posA=tuple(dis_cent),posB=tuple(g["cent"]),arrowstyle="->",
                        color=col,lw=2.0,mutation_scale=14,
                        connectionstyle="arc3,rad=-0.30",zorder=10)
    arc.set_linestyle("dashed"); ax.add_patch(arc)
    node(ax,ctrl_cent,CTRL_C,s=90,zo=11); node(ax,dis_cent,DIS_C,s=90,zo=11)
    node(ax,g["cent"],col,s=90,zo=13,marker="D")
    lbl(ax,ctrl_cent,"C",dy=.35*SCALE,color=CTRL_C)
    lbl(ax,dis_cent,"S",dy=-.40*SCALE,color=DIS_C)
    lbl(ax,g["cent"],"P",dy=.35*SCALE,color=col)
    p_=g["p"]; ptag=("p < 0.001" if p_<.001 else f"p = {p_:.3f}" if p_<.05 else f"p = {p_:.2f}")
    mark=" ★" if c==LEAD else " ▲" if c==POS else ""
    ax.set_title(f"{ZH[c]}{mark}  (score {SCORES.loc[c,'score']:.4f}, n={SCORES.loc[c,'n_genes']} genes)\n"
                 f"$L_{{CP}}$ vs $L_{{CS}}$: $\\Delta$dist = {g['pct']:+.1f}%   {ptag}",
                 fontsize=9.5,fontweight="bold",color=col,pad=6,linespacing=1.4)
    ax.set_xlabel("PC1",fontsize=8.5); ax.set_ylabel("PC2",fontsize=8.5)
    ax.tick_params(labelsize=7)
leg=[mlines.Line2D([0],[0],marker="o",color=CTRL_C,markersize=8,linestyle="None",
                   label="$S_0$ Macro-CDKN1C / 节点 (C)"),
     mlines.Line2D([0],[0],marker="o",color=DIS_C,markersize=8,linestyle="None",
                   label="$S_1$–$S_3$ 病变细胞 / 节点 (S)"),
     mlines.Line2D([0],[0],marker="D",color="#888888",markersize=7,linestyle="None",
                   label="给药后节点 (P)"),
     mlines.Line2D([0],[0],color="#BBBBBB",lw=4.0,label="PAGA 边 (C–S)"),
     mlines.Line2D([0],[0],color=LINE_K,lw=1.8,label="$L_{CS}$ / $L_{CP}$ 距离"),
     mlines.Line2D([0],[0],color="#888888",lw=2.0,ls="--",label="S → P 扰动")]
fig.legend(handles=leg,ncol=len(leg),loc="lower center",bbox_to_anchor=(.5,-.035),
           fontsize=8.5,frameon=False,handlelength=1.8,columnspacing=1.3)
fig.suptitle("矽肺促纤维化轨迹的虚拟药物扰动（按 UNAGI 原文 Fig. 5c/5d 规范）\n"
             "stage = CDKN1C→CLEC4E→SEPTIN7→DEFB1（用户已发表轨迹顺序）\n"
             f"每格取各质心最近 {N_NEAREST} 个细胞；PAGA 边宽 ∝ 连通度；"
             f"$L_{{CS}}$/$L_{{CP}}$ = 到 $S_0$(Macro-CDKN1C，839/840 为对照) 的距离"
             f"（64 维潜空间，$L_{{CS}}$ = {d0:.3f}）；"
             "药物谱 ∩ 疾病方向基因，ITCM 实测谱",
             fontsize=11.5,fontweight="bold",y=1.03)
fig.tight_layout()
for e in ("png","pdf"):
    fig.savefig(f"{OUT}/Fig5c_silicosis_{_a.tag}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
pd.DataFrame([dict(compound=c,cn=ZH[c],L_CS=d0,L_CP=G[c]["da"],delta_pct=G[c]["pct"],
                   p_paired_t=G[c]["p"],score=SCORES.loc[c,"score"]) for c in ORD]
             ).to_csv(f"{OUT}/Fig5c_silicosis_{_a.tag}.csv",index=False)
print(f"\n>>> 已写 {OUT}/Fig5c_silicosis.png")
print("=== F5C DONE ===",flush=True)
