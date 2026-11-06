"""= GPS RCL/code/train_test_split.R：P_MSE<0.05 选基因；cor_mean>0.7 为高质量，随机抽其 10% 作测试集；±1.5 三分类"""
import numpy as np, pandas as pd
P=pd.read_csv("unique_drug_profile.csv"); V=pd.read_csv("unique_drug_value.csv"); R=pd.read_csv("Predictabilities.csv")
sel=sorted(R.Gene[R.P_MSE<0.05]); V=V[sel]
o=np.argsort(P.SMILES.values); P,V=P.iloc[o].reset_index(drop=True),V.iloc[o].reset_index(drop=True)
hq=np.where(P.cor_mean>0.7)[0]; n_test=int(round(0.1*len(hq)))
rng=np.random.RandomState(1); test=np.sort(rng.choice(hq,n_test,replace=False)); train=np.setdiff1d(np.arange(len(P)),test)
def flat(ix):
    v=V.iloc[ix].to_numpy().ravel(); lab=np.digitize(v,[-1.5,1.5],right=True)   # (-inf,-1.5]→0, (-1.5,1.5]→1, >1.5→2，同 R 的 cut
    return pd.DataFrame(dict(smiles=np.repeat(P.SMILES.values[ix],len(sel)),gene=np.tile(sel,len(ix)),target=v,label=lab,
                             quality=np.repeat(P.cor_mean.values[ix],len(sel))))
flat(train).to_csv("data_train.csv",index=False); flat(test).to_csv("data_test.csv",index=False)
pd.Series(sel).to_csv("selected_genes.csv",index=False,header=False)
P.iloc[test][["SMILES","pert_id","cor_mean"]].to_csv("test_compounds.csv",index=False)
print(f"选中基因 {len(sel)}；化合物 训练 {len(train)} / 测试 {len(test)}（高质量 {len(hq)}）"); print("=== S3 DONE ===")
