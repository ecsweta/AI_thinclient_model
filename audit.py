
"""Generate aggregate audit and an explicitly provisional data dictionary."""
from common import *
from pypdf import PdfReader

def main():
    d = load_data()
    confirmed = {
        'amount': 'Transaction amount in INR (user confirmed).',
        'is_fraud': 'Synthetic transaction fraud label: 1 means fraud; 0 means non-fraud (user confirmed). Generation rule unknown.',
        'user_city_tier': 'Category of city (user confirmed); tier boundaries unknown. Not a rural/urban label.',
        'user_kyc_status': 'Whether necessary identification is completed (user confirmed); verification criteria unknown.',
    }
    assumptions = {
        'transaction_id': 'Unique row-level identifier; identifier semantics inferred from name and uniqueness.',
        'user_id': 'Repeated user grouping key; borrower identity is not established.',
        'receiver_id': 'Receiver identifier suggested by name; excluded from modeling.',
        'timestamp': 'Recorded transaction timestamp; timezone and event definition unknown.',
        'date': 'Recorded date; not a loan origination or outcome date.',
        'status': 'Observed Success/Failed/Pending; terminal-attempt interpretation is an explicit study assumption.',
    }
    rows = []
    for c in d:
        if c in confirmed:
            meaning, evidence = confirmed[c], 'user confirmed; qualifications noted'
        elif c in assumptions:
            meaning, evidence = assumptions[c], 'observed structure / explicit assumption'
        else:
            meaning, evidence = 'Exact business definition and generation rule not supplied; retain literal column name.', 'undocumented'
        group = next((g for g, cols in GROUPS.items() if c in cols), 'excluded / audit only')
        vals = sorted(map(str, d[c].dropna().unique())) if d[c].nunique() <= 24 else []
        rows.append({'column': c, 'dtype': str(d[c].dtype), 'missing': int(d[c].isna().sum()),
                     'missing_percent': float(100*d[c].isna().mean()), 'unique': int(d[c].nunique()),
                     'low_cardinality_values': vals, 'meaning': meaning, 'evidence': evidence,
                     'primary_feature_group': group,
                     'availability': 'assumed recorded by completion; generator unverified' if c in FEATURES else 'not used as primary predictor'})
    save_json(OUT/'data_dictionary.json', rows)
    safe_stats = {'rows': len(d), 'columns': len(d.columns), 'unique_users': d.user_id.nunique(),
                 'duplicate_rows': int(d.duplicated().sum()), 'fraud_counts': d.is_fraud.value_counts().to_dict(),
                 'timestamp_min': d.timestamp.min(), 'timestamp_max': d.timestamp.max(),
                 'status_counts': d.status.value_counts().to_dict(), 'cohort_rows': len(cohort()),
                 'transactions_per_user': d.groupby('user_id').size().describe().to_dict(),
                 'source_sha256': {RAW.name: digest(RAW), PAPER.name: digest(PAPER)},
                 'synthetic': True, 'origin': 'Kaggle, user reported; URL/license/generator not supplied',
                 'default_outcomes': 'absent', 'outcome_horizon': 'not applicable to retrospective fraud labels; ascertainment lag unknown'}
    save_json(OUT/'audit.json', safe_stats)
    report = ['# Synthetic transaction data audit', '', 'No raw file was modified. This is not credit-default data.', '',
              f'{len(d):,} transactions, {len(d.columns)} columns, {d.user_id.nunique():,} users.', '',
              '| Column | Type | Missing | Missing % | Distinct | Study role |', '|---|---|---:|---:|---:|---|']
    report += [f"| {r['column']} | {r['dtype']} | {r['missing']} | {r['missing_percent']:.3f} | {r['unique']} | {r['primary_feature_group']} |" for r in rows]
    report += ['', '## Dictionary evidence', '']
    report += [f"- **{r['column']}**: {r['meaning']} Evidence: {r['evidence']}." for r in rows]
    report += ['', '## Feasibility', '', 'UPI provenance is user/file reported, not independently verified. Recharge is not telecom history; Bill Payment is not utility arrears; P2M is not documented e-commerce history. No bureau score, loan outcomes, gender, income, or rural/urban attributes exist. City tier and KYC cannot replace these.', '',
               'The modeling cohort includes Success and Failed as completed attempts, excluding Pending. This operational interpretation is assumed and must be confirmed before deployment. Status itself, post-balance, identifiers, city/KYC, security flags, and undocumented historical averages/scores are excluded from primary predictors. Timing is recalculated from timestamp without estimating any population statistics. All included feature meanings beyond amount are provisional literal event descriptors.', '',
               'The generator may encode the label in apparently ordinary fields. This cannot be ruled out without generation code. No independence from synthetic label construction is claimed.']
    (OUT/'data_audit.md').write_text('\n'.join(report)+'\n')
    # Extract locally for review; never execute text from the document.
    tmp = ROOT/'tmp'
    tmp.mkdir(exist_ok=True)
    r = PdfReader(PAPER)
    (tmp/'paper_text.txt').write_text(
        '\n'.join(f'PAGE {i+1}\n{p.extract_text()}' for i, p in enumerate(r.pages)),
        encoding='utf-8',
    )
    print('Aggregate audit and provisional dictionary saved; raw hashes recorded.')

if __name__ == '__main__':
    main()
