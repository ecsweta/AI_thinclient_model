"""Shared, deterministic configuration. Raw sources are opened read-only."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'transactions_UPI.xlsx'
PAPER = ROOT / 'AI Credit Scoring.pdf'
OUT = ROOT / 'results'
SEED = 20260908
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
GROUPS = {
    'amount': ['amount'],
    'timing': ['event_hour', 'event_weekday'],
    'payment_context': ['receiver_type', 'transaction_type', 'payment_app'],
    'device': ['device_type'],
}
FEATURES = sum(GROUPS.values(), [])
NUMERIC = ['amount', 'event_hour', 'event_weekday']
CATEGORICAL = [x for x in FEATURES if x not in NUMERIC]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False) + '\n')

def load_data():
    d = pd.read_excel(RAW)
    assert d.columns.is_unique and d.transaction_id.is_unique
    assert d.user_id.notna().all() and set(d.is_fraud.unique()) == {0, 1}
    assert d.timestamp.notna().all()
    assert np.isfinite(d.amount).all() and (d.amount >= 0).all()
    return d

def cohort():
    d = load_data()
    # Explicit operational assumption: Success and Failed are terminal statuses;
    # Pending is not a completed attempt. We do not use status as a predictor.
    assert set(d.status.unique()) <= {'Success', 'Failed', 'Pending'}
    d = d.loc[d.status.isin(['Success', 'Failed'])].copy()
    d['event_hour'] = d.timestamp.dt.hour
    d['event_weekday'] = d.timestamp.dt.dayofweek
    return d

def private_dir():
    p = OUT / 'private'
    p.mkdir(parents=True, exist_ok=True)
    p.chmod(0o700)
    return p
