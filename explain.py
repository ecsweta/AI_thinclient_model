"""Matched-case SHAP/LIME comparison on original features; no identifier exports."""
from common import *
import time
import itertools
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import figure_save

def jaccard(a,b,k=3):
    aa=set(np.argsort(np.abs(a))[-k:]); bb=set(np.argsort(np.abs(b))[-k:])
    return len(aa&bb)/len(aa|bb)

def main():
    d=cohort(); split=np.load(private_dir()/'split_indices.npz'); selection=json.loads((OUT/'selection.json').read_text())
    train=d.iloc[split['train']][FEATURES].copy(); test=d.iloc[split['test']][FEATURES].copy()
    cats=[FEATURES.index(c) for c in CATEGORICAL]
    mappings={c:sorted(train[c].dropna().unique()) for c in CATEGORICAL}
    # Ordinal values are only an adapter: model still consumes categorical strings,
    # and both explainers treat these columns as categories, not ordered predictors.
    def encode(frame):
        out=frame.copy()
        for c in CATEGORICAL:
            out[c]=out[c].map({v:i for i,v in enumerate(mappings[c])}).fillna(-1)
        return out.to_numpy(dtype=float)
    def decode(a):
        out=pd.DataFrame(np.asarray(a),columns=FEATURES)
        for c in CATEGORICAL:
            out[c]=out[c].round().astype(int).map(dict(enumerate(mappings[c]))).fillna('__unknown__')
        # Gaussian LIME perturbations can be implausible; predictions remain defined,
        # and this is measured/disclosed rather than silently clipping the neighborhood.
        return out
    tr=encode(train); ts=encode(test)
    rng=np.random.default_rng(SEED+60); background_idx=rng.choice(len(tr),min(100,len(tr)),replace=False)
    background=tr[background_idx]
    best=selection['explanation_model']; best_model=joblib.load(OUT/'models'/(best+'.joblib'))
    order=np.argsort(best_model.predict_proba(test)[:,1],kind='stable')
    cases=np.unique(order[np.linspace(0,len(order)-1,12).astype(int)])
    # Same 12 score-quantile cases for LR and RF, sampled without their labels.
    np.savez(private_dir()/'explanation_sampling.npz',test_positions=cases,train_background_positions=background_idx)
    rows=[]; weights={}; summary={}
    for name in ['logistic','random_forest']:
        model=joblib.load(OUT/'models'/(name+'.joblib'))
        calls={'rows':0}
        def predict(a):
            calls['rows']+=len(a)
            return model.predict_proba(decode(a))
        def p1(a): return predict(a)[:,1]
        # Warm the prediction adapter before timing either method.
        predict(background[:2])
        vals_shap=[]; vals_lime=[]
        for case_no,idx in enumerate(cases):
            v=ts[idx]; ss=[]; ll=[]
            for repeat in range(3):
                seed=SEED+1000+case_no*10+repeat
                # Independent audit neighborhood shared across methods, not the LIME fit sample.
                nrng=np.random.default_rng(seed+10000)
                donor=tr[nrng.choice(len(tr),300,replace=True)]
                mask=nrng.random((300,len(FEATURES)))<.8
                neighborhood=np.where(mask,v,donor); actual=p1(neighborhood)
                start=time.perf_counter(); calls['rows']=0
                ex=shap.PermutationExplainer(p1,shap.maskers.Independent(background,max_samples=100),seed=seed)
                sv=ex(v[None,:],max_evals=(2*len(FEATURES)+1)*20,silent=True)
                elapsed=time.perf_counter()-start; evaluations=calls['rows']; matched_budget=evaluations
                phi=sv.values[0]; base=float(sv.base_values[0]); ss.append(phi)
                approx=base+mask@phi
                rows.append({'model':name,'case':int(case_no),'repeat':repeat,'method':'SHAP',
                    'seconds':elapsed,'predicted_rows':evaluations,
                    'audit_neighborhood_r2':float(r2_score(actual,approx)),
                    'audit_neighborhood_rmse':float(np.sqrt(np.mean((actual-approx)**2))),
                    'point_reconstruction_error':float(abs(base+phi.sum()-p1(v[None,:])[0])),
                    'fit_neighborhood_r2':None})
                start=time.perf_counter(); calls['rows']=0
                lime=LimeTabularExplainer(tr,feature_names=FEATURES,categorical_features=cats,
                    class_names=['non_fraud','fraud'],discretize_continuous=False,random_state=seed,
                    sample_around_instance=True,mode='classification',feature_selection='none')
                le=lime.explain_instance(v,predict,labels=(1,),num_features=len(FEATURES),num_samples=matched_budget)
                elapsed=time.perf_counter()-start; evaluations=calls['rows']
                coef=np.zeros(len(FEATURES))
                for j,w in le.local_exp[1]: coef[j]=w
                ll.append(coef)
                z=(neighborhood-lime.scaler.mean_)/lime.scaler.scale_
                z[:,cats]=(neighborhood[:,cats]==v[cats]).astype(float)
                approx=float(le.intercept[1])+z@coef
                zv=(v-lime.scaler.mean_)/lime.scaler.scale_; zv[cats]=1
                rows.append({'model':name,'case':int(case_no),'repeat':repeat,'method':'LIME',
                    'seconds':elapsed,'predicted_rows':evaluations,
                    'audit_neighborhood_r2':float(r2_score(actual,approx)),
                    'audit_neighborhood_rmse':float(np.sqrt(np.mean((actual-approx)**2))),
                    'point_reconstruction_error':float(abs(float(le.intercept[1])+zv@coef-p1(v[None,:])[0])),
                    'fit_neighborhood_r2':float(le.score)})
            vals_shap.append(ss); vals_lime.append(ll)
        s=np.asarray(vals_shap); l=np.asarray(vals_lime)
        weights[name+'_shap']=s; weights[name+'_lime']=l
        subset=[r for r in rows if r['model']==name]
        summary[name]={}
        for method,arr in [('SHAP',s),('LIME',l)]:
            rr=[r for r in subset if r['method']==method]
            summary[name][method]={k:float(np.median([r[k] for r in rr])) for k in
                ['seconds','predicted_rows','audit_neighborhood_r2','audit_neighborhood_rmse','point_reconstruction_error']}
            summary[name][method]['mean_repeat_top3_jaccard']=float(np.mean([jaccard(a[i],a[j]) for a in arr for i,j in itertools.combinations(range(3),2)]))
            summary[name][method]['mean_feature_weight_sd']=float(np.std(arr,axis=1).mean())
        summary[name]['between_method_mean_top3_jaccard']=float(np.mean([jaccard(a,b) for a,b in zip(s.mean(axis=1),l.mean(axis=1))]))
        print(name+': matched-case explanations complete.')
    save_json(OUT/'explanation_runs.json',rows)
    save_json(OUT/'explanation_summary.json',{'models':summary,'settings':{
        'cases':len(cases),'case_selection':'12 score quantiles of validation-selected nontrivial model; not population-random',
        'repeats':3,'background_rows':100,'shap_permutations':20,'lime_samples':'matched per case/run to SHAP predicted-row count',
        'audit_neighborhood_rows':300,'mask_keep_probability':.8,'units':'fraud probability; LIME continuous coefficients per training SD',
        'limitations':'Predicted-row budgets matched; algorithm overhead and perturbation semantics differ. SHAP masking and LIME Gaussian perturbations have different semantics. Audit R2 is descriptive, not a native SHAP objective. Negative R2 is possible. Perturbations can violate feature dependence. Top-feature agreement is not correctness. Repeat stability measures stochastic estimation only, not stability to real-world changes.'}})
    np.savez(private_dir()/'explanation_weights.npz',**weights)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,name in zip(axs,['logistic','random_forest']):
        s=np.abs(weights[name+'_shap']).mean(axis=(0,1)); order=np.argsort(s)
        ax.barh(np.array(FEATURES)[order],s[order]); ax.set(title=name,xlabel='Mean absolute SHAP value (12 selected cases)')
    fig.suptitle('Synthetic fraud explanations: descriptive case sample'); fig.tight_layout(); figure_save(fig,'explanation_importance')

if __name__=='__main__': main()
