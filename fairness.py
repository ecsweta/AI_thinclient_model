"""Descriptive observed-group diagnostics, not the unavailable demographic fairness study."""
from common import *
from evaluate import metrics, intervals

def main():
    d=cohort(); split=np.load(private_dir()/'split_indices.npz'); te=split['test']; dt=d.iloc[te]
    selection=json.loads((OUT/'selection.json').read_text()); chosen=selection['selected_model']
    pred=np.load(private_dir()/'test_predictions.npz'); p=pred[chosen]; y=pred['y']; t=selection['models'][chosen]['threshold']
    out={'status':'requested demographic fairness and interventions unsupported',
         'missing':['documented rural/urban classification','gender','income tier'],
         'decision':'1 means post-transaction review flag; approval rates are not defined',
         'interventions':'not run: protected groups absent; city-tier/KYC parity would substitute an unrequested objective',
         'model':chosen,'descriptive_groups':{}}
    for col in ['user_city_tier','user_kyc_status']:
        rows=[]
        for val in sorted(dt[col].unique()):
            mask=(dt[col]==val).to_numpy(); yy=y[mask]; pp=p[mask]; gg=dt.loc[mask,'user_id'].to_numpy()
            rows.append({'group':str(val),'rows':int(mask.sum()),'users':len(np.unique(gg)),
                         'fraud_cases':int(yy.sum()), 'small_sample_warning':bool(yy.sum()<30 or (len(yy)-yy.sum())<30 or len(np.unique(gg))<30),
                         'metrics':metrics(yy,pp,t),'ci_95_user_cluster_bootstrap':intervals(yy,pp,t,gg,300)})
        # Joint cluster bootstrap preserves cross-group user dependence when computing max-min spread.
        from evaluate import cluster_samples
        gaps=[]
        for idx in cluster_samples(dt.user_id.to_numpy(),300):
            rates=[]
            for val in sorted(dt[col].unique()):
                mask=(dt.iloc[idx][col]==val).to_numpy()
                if mask.any(): rates.append(float(np.mean(p[idx][mask]>=t)))
            if len(rates)==len(rows): gaps.append(max(rates)-min(rates))
        out['descriptive_groups'][col]={'groups':rows,
            'users_with_inconsistent_values_in_full_cohort':int((d.groupby('user_id')[col].nunique()>1).sum()),
            'alert_rate_max_minus_min':max(r['metrics']['alert_rate'] for r in rows)-min(r['metrics']['alert_rate'] for r in rows),
            'gap_ci_95':[float(v) for v in np.quantile(gaps,[.025,.975])],
            'interpretation':'synthetic transaction-row categories; no demographic fairness inference'}
    save_json(OUT/'fairness_availability_and_diagnostics.json',out)
    print('Unsupported fairness analyses documented; observed-category diagnostics saved.')

if __name__=='__main__': main()
