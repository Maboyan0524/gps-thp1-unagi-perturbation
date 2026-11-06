"""= GPS RCL/code/cal_predictability.py：每个基因用 Morgan(r=3,1024) 做随机森林回归，10 折，真实 vs 打乱标签，配对 t 检验"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, ttest_rel
from rdkit import Chem, RDLogger; from rdkit.Chem import AllChem
from joblib import Parallel, delayed
RDLogger.DisableLog("rdApp.*")
def fp(s): return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),radius=3,nBits=1024),dtype=np.float32)
P=pd.read_csv("unique_drug_profile.csv"); V=pd.read_csv("unique_drug_value.csv")
X=np.stack([fp(s) for s in P.SMILES])
def one(gene, run_times=1, cv_fold=10):
    np.random.seed(1); y=V[gene].to_numpy(); mt,mr,rt_,rr=[],[],[],[]
    for rt in range(run_times):
        tmp=np.arange(len(y)); np.random.shuffle(tmp); yr=y[tmp]
        for tr,ts in KFold(n_splits=cv_fold,shuffle=True,random_state=rt).split(X):
            m=RandomForestRegressor(random_state=1).fit(X[tr],y[tr]); p=m.predict(X[ts])
            mt.append(mean_squared_error(y[ts],p)); rt_.append(pearsonr(y[ts],p)[0])
            m=RandomForestRegressor(random_state=1).fit(X[tr],yr[tr]); p=m.predict(X[ts])
            mr.append(mean_squared_error(yr[ts],p)); rr.append(pearsonr(yr[ts],p)[0])
    _,mp=ttest_rel(mt,mr); _,rp=ttest_rel(rt_,rr)
    return [gene,mp,np.mean(mt),np.mean(mt)-np.mean(mr),rp,np.mean(rt_),np.mean(rt_)-np.mean(rr)]
res=Parallel(n_jobs=120,verbose=5)(delayed(one)(g) for g in V.columns)
R=pd.DataFrame(res,columns=["Gene","P_MSE","Average_MSE","Delta_MSE","P_PearsonR","Average_PearsonR","Delta_PearsonR"])
R.to_csv("Predictabilities.csv",index=False)
print("可预测基因 (P_MSE<0.05):",int((R.P_MSE<0.05).sum()),"/",len(R)); print("=== S2 DONE ===")
