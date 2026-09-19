"""Evaluate frozen models once; cluster-bootstrap uncertainty is conditional on fitted models."""
from common import *
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score,
                             f1_score, confusion_matrix, brier_score_loss, log_loss,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve

def metrics(y,p,t):
    pred = p >= t
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'pr_auc_average_precision':float(average_precision_score(y,p)) if np.sum(y)>0 else None,
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)), 'f1':float(f1_score(y,pred,zero_division=0)),
            'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
            'fpr':float(fp/(fp+tn)) if fp+tn else None, 'fnr':float(fn/(fn+tp)) if fn+tp else None,
            'alert_rate':float(np.mean(pred)), 'prevalence':float(np.mean(y)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp), 'threshold':float(t)}

def cluster_samples(groups, n=500):
    rng = np.random.default_rng(SEED+30)
    ids = np.unique(groups)
    lookup = [np.flatnonzero(groups==u) for u in ids]
    for _ in range(n):
        yield np.concatenate([lookup[k] for k in rng.integers(0,len(ids),len(ids))])

def intervals(y,p,t,groups,n=500):
    keys=['roc_auc','pr_auc_average_precision','precision','recall','f1','brier','fpr','fnr','alert_rate','prevalence']
    vals={k:[] for k in keys}
    for idx in cluster_samples(groups,n):
        m=metrics(y[idx],p[idx],t)
        for k in keys:
            if m[k] is not None: vals[k].append(m[k])
    return {k:{'lower':float(np.quantile(v,.025)), 'upper':float(np.quantile(v,.975)), 'valid_replicates':len(v)} if v else None for k,v in vals.items()}

def figure_save(fig,name):
    dest=OUT/'figures'
    dest.mkdir(exist_ok=True)
    fig.savefig(dest/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(dest/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def main():
    selection=json.loads((OUT/'selection.json').read_text())
    assert digest(RAW)==selection['raw_sha256']
    d=cohort()
    split=np.load(private_dir()/'split_indices.npz')
    assert np.array_equal(d.index,split['source_rows'])
    te=split['test']; x=d.iloc[te][FEATURES]; y=d.iloc[te].is_fraud.to_numpy(); users=d.iloc[te].user_id.to_numpy()
    result={}; predictions={}; models={}
    for name, spec in selection['models'].items():
        m=joblib.load(OUT/'models'/(name+'.joblib')); p=m.predict_proba(x)[:,1]
        models[name]=m; predictions[name]=p
        result[name]={'point':metrics(y,p,spec['threshold']), 'ci_95_user_cluster_bootstrap':intervals(y,p,spec['threshold'],users)}
    np.savez(private_dir()/'test_predictions.npz', **predictions, y=y, source_row=d.iloc[te].index.to_numpy())
    save_json(OUT/'metrics.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(15,4.2))
    for name in ['prior_baseline','logistic','random_forest']:
        p=predictions[name]; a,b,_=roc_curve(y,p); axs[0].plot(a,b,label=name)
        a,b,_=precision_recall_curve(y,p); axs[1].step(b,a,where='post',label=name)
        observed,estimated=calibration_curve(y,p,n_bins=8,strategy='quantile'); axs[2].plot(estimated,observed,'o-',label=name)
    axs[0].plot([0,1],[0,1],'k:',lw=1); axs[1].axhline(y.mean(),color='k',ls=':',lw=1); axs[2].plot([0,1],[0,1],'k:',lw=1)
    axs[2].set_xlim(0,.08); axs[2].set_ylim(0,.08)
    for ax,title,xlabel,ylabel in zip(axs,['ROC','Precision–recall','Calibration (quantile bins)'],['False positive rate','Recall','Mean predicted probability'],['True positive rate','Precision','Observed fraud fraction']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.legend(fontsize=8)
    fig.suptitle('Synthetic post-transaction fraud: unseen-user test set'); fig.tight_layout(); figure_save(fig,'performance')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,name in zip(axs,['prior_baseline','logistic','random_forest']):
        cm=confusion_matrix(y,predictions[name]>=selection['models'][name]['threshold'],labels=[0,1]); ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(title=name,xlabel='Predicted fraud flag',ylabel='Actual synthetic fraud',xticks=[0,1],yticks=[0,1])
    fig.suptitle('Validation-selected F1 thresholds; synthetic data'); fig.tight_layout(); figure_save(fig,'confusion_matrices')
    # Frozen, raw-RF ablations and paired cluster-bootstrap differences.
    base=predictions['ablation_reference']; ablations={}
    for group in GROUPS:
        p=predictions['without_'+group]
        diffs=[]
        for idx in cluster_samples(users):
            if len(np.unique(y[idx]))==2:
                diffs.append(average_precision_score(y[idx],base[idx])-average_precision_score(y[idx],p[idx]))
        ablations[group]={'ap_full_minus_removed':float(average_precision_score(y,base)-average_precision_score(y,p)),
                          'ci_95':[float(v) for v in np.quantile(diffs,[.025,.975])]}
    save_json(OUT/'ablations.json',ablations)
    # Permute original columns together within each group, preserving within-group dependence.
    rng=np.random.default_rng(SEED+40); perm={}; m=models['ablation_reference']; base_ap=average_precision_score(y,base)
    for group,columns in {**GROUPS, **{'feature:'+c:[c] for c in FEATURES}}.items():
        vals=[]
        for _ in range(10):
            xp=x.copy(); idx=rng.permutation(len(x)); xp[columns]=x.iloc[idx][columns].to_numpy()
            vals.append(float(base_ap-average_precision_score(y,m.predict_proba(xp)[:,1])))
        perm[group]={'mean_ap_drop':float(np.mean(vals)),'permutation_sd':float(np.std(vals,ddof=1)),'repeats':vals}
    save_json(OUT/'permutation_importance.json',perm)
    fig,ax=plt.subplots(figsize=(8,4))
    keys=list(GROUPS); ax.barh(keys,[perm[g]['mean_ap_drop'] for g in keys],xerr=[perm[g]['permutation_sd'] for g in keys])
    ax.axvline(0,color='k',lw=.8); ax.set(xlabel='Test average-precision decrease (mean ± permutation SD)',title='Synthetic data: joint signal-group permutation, raw random forest')
    fig.tight_layout(); figure_save(fig,'group_importance')
    # Save label availability, cohort, and paired differences without raw identifiers.
    save_json(OUT/'evaluation_manifest.json',{'selection_sha256':digest(OUT/'selection.json'),'test_rows':len(te),
        'test_users':len(np.unique(users)),'bootstrap_replicates':500,'interval_scope':'conditional on fitted models and this split; synthetic population only',
        'raw_sources_unchanged':{RAW.name:digest(RAW),PAPER.name:digest(PAPER)}})
    print('Frozen test evaluation, cluster intervals, ablations and figures saved.')

if __name__=='__main__': main()
