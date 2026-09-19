"""Check saved output consistency, grouping, source integrity and fitted preprocessing."""
from common import *
import platform
import subprocess
import joblib
from evaluate import metrics

def main():
    audit=json.loads((OUT/'audit.json').read_text()); selection=json.loads((OUT/'selection.json').read_text())
    assert digest(RAW)==audit['source_sha256'][RAW.name]==selection['raw_sha256']
    assert digest(PAPER)==audit['source_sha256'][PAPER.name]
    d=cohort(); s=np.load(private_dir()/'split_indices.npz'); p=np.load(private_dir()/'test_predictions.npz')
    all_idx=np.concatenate([s[k] for k in ['train','validation','test']])
    assert len(all_idx)==len(d) and len(np.unique(all_idx))==len(d)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert set(d.iloc[s[a]].user_id).isdisjoint(d.iloc[s[b]].user_id)
    assert not {'is_fraud','transaction_id','user_id','receiver_id','status','balance_after_transaction','user_city_tier','user_kyc_status'} & set(FEATURES)
    saved=json.loads((OUT/'metrics.json').read_text())
    for name,record in selection['models'].items():
        m=joblib.load(OUT/'models'/(name+'.joblib'))
        prob=m.predict_proba(d.iloc[s['test']][FEATURES])[:,1]
        assert np.allclose(prob,p[name],rtol=0,atol=1e-12)
        result=metrics(p['y'],prob,record['threshold'])
        for key,value in result.items():
            assert value is None and saved[name]['point'][key] is None or np.isclose(value,saved[name]['point'][key])
        assert result['tn']+result['fp']+result['fn']+result['tp']==len(s['test'])
    # Independent check on fitted scaler statistics of uncalibrated reference.
    ref=joblib.load(OUT/'models'/'ablation_reference.joblib')
    scaler=ref.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
    assert np.allclose(scaler.mean_,d.iloc[s['train']][NUMERIC].mean().to_numpy())
    sample=d.iloc[s['test'][:3]][FEATURES].copy()
    sample.loc[sample.index[0],'payment_app']='__unseen_app__'
    sample.loc[sample.index[1],'amount']=np.nan
    assert np.isfinite(ref.predict_proba(sample)).all()
    ex=json.loads((OUT/'explanation_runs.json').read_text())
    for model in ['logistic','random_forest']:
        rr=[r for r in ex if r['model']==model]
        for case in range(12):
            for rep in range(3):
                pair=[r for r in rr if r['case']==case and r['repeat']==rep]
                assert len(pair)==2 and pair[0]['predicted_rows']==pair[1]['predicted_rows']
    from PIL import Image
    for fig in (OUT/'figures').glob('*.png'):
        with Image.open(fig) as im: im.verify()
    save_json(OUT/'verification.json',{'passed':True,'checks':['original SHA256 unchanged','complete disjoint user splits',
        'forbidden features excluded','saved predictions reproduce','metrics reconcile','train-only scaler statistics',
        'unseen-category and missing-amount inference','matched SHAP/LIME budgets','figure files readable'],
        'python':platform.python_version(),'platform':platform.platform(),
        'script_sha256':{f.name:digest(f) for f in (ROOT/'scripts').glob('*.py')},
        'requirements_sha256':digest(ROOT/'requirements.lock.txt')})
    print('Verification passed; original workbook and PDF hashes unchanged.')

if __name__=='__main__': main()
