"""Select models without evaluating the final test partition."""
from common import *
import joblib
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

def pipeline(features, model):
    num = [c for c in NUMERIC if c in features]
    cat = [c for c in CATEGORICAL if c in features]
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat),
    ])
    return Pipeline([('preprocess', prep), ('model', model)])

def choose_threshold(y, p):
    prec, rec, thresholds = precision_recall_curve(y, p)
    f = 2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1], 1e-15)
    # Highest threshold wins ties, reducing alerts when validation F1 ties.
    k = np.flatnonzero(np.isclose(f, f.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[k])

def main():
    d = cohort()
    x, y = d[FEATURES], d.is_fraud
    splits = np.empty(len(d), dtype=int)
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(sg.split(x, y, d.user_id)):
        splits[idx] = k
    train, val, test = np.flatnonzero(splits >= 2), np.flatnonzero(splits == 1), np.flatnonzero(splits == 0)
    for a,b in [(train,val),(train,test),(val,test)]:
        assert set(d.iloc[a].user_id).isdisjoint(d.iloc[b].user_id)
    pdir = private_dir()
    np.savez(pdir/'split_indices.npz', train=train, validation=val, test=test, source_rows=d.index.to_numpy())
    split_info = {name: {'rows': len(idx), 'users': d.iloc[idx].user_id.nunique(),
                         'date_min': d.iloc[idx].timestamp.min(), 'date_max': d.iloc[idx].timestamp.max()}
                  for name, idx in [('train',train),('validation',val),('test',test)]}
    save_json(OUT/'split_summary.json', split_info)
    xt, yt, xv, yv = x.iloc[train], y.iloc[train], x.iloc[val], y.iloc[val]
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED+1).split(xt, yt, d.iloc[train].user_id))
    for a,b in cv:
        assert set(d.iloc[train[a]].user_id).isdisjoint(d.iloc[train[b]].user_id)
    models_dir = OUT/'models'
    models_dir.mkdir(exist_ok=True)
    specs = {
        'logistic': (LogisticRegression(max_iter=3000, random_state=SEED), {'model__C': [0.01,0.1,1,10]}),
        'random_forest': (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                          {'model__max_depth': [6,None], 'model__min_samples_leaf': [5,20]}),
    }
    records, artifacts, tuning = {}, {}, {}
    for name, (estimator, grid) in specs.items():
        search = GridSearchCV(pipeline(FEATURES, estimator), grid, scoring='average_precision', cv=cv, n_jobs=1, refit=True)
        search.fit(xt, yt)
        raw = search.best_estimator_
        calibrated = CalibratedClassifierCV(estimator=raw, method='sigmoid', cv=cv, ensemble=True, n_jobs=1)
        calibrated.fit(xt, yt)
        options = {'raw':raw, 'sigmoid':calibrated}
        calibration = min(options, key=lambda k: brier_score_loss(yv, options[k].predict_proba(xv)[:,1]))
        fitted = options[calibration]
        p = fitted.predict_proba(xv)[:,1]
        records[name] = {'validation_ap': average_precision_score(yv,p), 'validation_brier': brier_score_loss(yv,p),
                         'threshold': choose_threshold(yv,p), 'calibration':calibration,
                         'parameters':search.best_params_, 'features':FEATURES}
        tuning[name] = [{'parameters':param, 'cv_mean_ap':float(mean), 'cv_std_ap':float(std)} for param,mean,std in
                        zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score'])]
        artifacts[name] = fitted
        print(f'{name}: training CV and validation selection complete.')
    baseline = pipeline(FEATURES, DummyClassifier(strategy='prior')).fit(xt,yt)
    bp = baseline.predict_proba(xv)[:,1]
    # Prior threshold is selected by the same rule: may flag everyone. Report this openly.
    records['prior_baseline'] = {'validation_ap':average_precision_score(yv,bp), 'validation_brier':brier_score_loss(yv,bp),
                                 'threshold':choose_threshold(yv,bp), 'calibration':'none', 'features':FEATURES}
    artifacts['prior_baseline'] = baseline
    winner = max(records, key=lambda k: records[k]['validation_ap'])
    explanation_model = max(specs, key=lambda k: records[k]['validation_ap'])
    # Freeze ablation specification before opening test outcomes; fixed best training-CV RF parameters.
    rf_params = {k.replace('model__',''):v for k,v in records['random_forest']['parameters'].items()}
    for group in GROUPS:
        name = 'without_'+group
        features = [c for c in FEATURES if c not in GROUPS[group]]
        m = pipeline(features, RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1, **rf_params)).fit(xt,yt)
        p = m.predict_proba(xv)[:,1]
        records[name] = {'validation_ap':average_precision_score(yv,p), 'validation_brier':brier_score_loss(yv,p),
                         'threshold':choose_threshold(yv,p), 'calibration':'raw', 'features':features}
        artifacts[name] = m
    # Raw reference makes ablation comparison like-for-like even if primary RF uses calibration.
    raw_rf = pipeline(FEATURES, RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1, **rf_params)).fit(xt,yt)
    p = raw_rf.predict_proba(xv)[:,1]
    artifacts['ablation_reference'] = raw_rf
    records['ablation_reference'] = {'validation_ap':average_precision_score(yv,p), 'validation_brier':brier_score_loss(yv,p),
                                    'threshold':choose_threshold(yv,p), 'calibration':'raw', 'features':FEATURES}
    for name,m in artifacts.items():
        joblib.dump(m, models_dir/(name+'.joblib'))
    save_json(OUT/'tuning.json', tuning)
    save_json(OUT/'selection.json', {'seed':SEED,'raw_sha256':digest(RAW),'models':records,
                                    'selected_model':winner,'explanation_model':explanation_model,
                                    'selection_metric':'validation average precision; calibration by validation Brier',
                                    'threshold_rule':'maximum validation F1, highest threshold on ties',
                                    'test_evaluated':False})
    print('Frozen model/threshold selection saved. No test performance evaluated.')

if __name__ == '__main__':
    main()
