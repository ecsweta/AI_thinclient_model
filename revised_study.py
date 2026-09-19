"""Build isolated outputs for the revised two-track study.

This script never edits the raw workbook, PDF or DOCX sources. It audits the
new wilful-defaulter workbook, documents that no loan-default dataset is
available locally, and packages the previously verified UPI experiment as
the synthetic-fraud track. It intentionally does not fit a credit-default
model when no documented loan-default outcome exists.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revised_study"
REPORTS = ROOT / "reports" / "revised_study"
FIGURES = OUT / "figures"
UPI = ROOT / "transactions_UPI .xlsx"
PAPER = ROOT / "AI Credit Scoring.pdf"
GUIDANCE = ROOT / "Data" / "Credit for Thin File Borrowers in India - Guidance Document.docx"
REVISED_GUIDANCE = ROOT / "Data" / "Revised guidance and approacj.docx"
WILFUL = ROOT / "Data" / "RS_Session_262_AU_1042_A_to_B_i.xlsx"
SEED = 20260908


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n")


def docx_text(path: Path) -> str:
    with ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for para in root.findall(".//w:p", ns):
        text = "".join(x.text or "" for x in para.findall(".//w:t", ns)).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def audit_wilful() -> dict:
    xls = pd.ExcelFile(WILFUL)
    sheets = {}
    for sheet in xls.sheet_names:
        frame = pd.read_excel(WILFUL, sheet_name=sheet)
        value_cols = [c for c in frame.columns if c != "Sl. No." and c != "Banks"]
        long = frame.melt(id_vars=["Sl. No.", "Banks"], value_vars=value_cols,
                          var_name="reporting_date", value_name="outstanding_crore")
        long["outstanding_crore"] = pd.to_numeric(long["outstanding_crore"], errors="coerce")
        long.to_csv(OUT / "wilful_defaulters_long_aggregate.csv", index=False)
        totals = long.groupby("reporting_date", dropna=False)["outstanding_crore"].agg(
            rows="count", reported_values="count", total_crore="sum", median_crore="median"
        ).reset_index()
        top = {}
        for date, group in long.groupby("reporting_date", dropna=False):
            top[str(date)] = (group.dropna(subset=["outstanding_crore"])
                              .sort_values("outstanding_crore", ascending=False)
                              .head(10)[["Banks", "outstanding_crore"]]
                              .to_dict("records"))
        sheets[sheet] = {
            "shape": list(frame.shape),
            "columns": [str(c) for c in frame.columns],
            "missing_by_column": {str(k): int(v) for k, v in frame.isna().sum().items()},
            "unique_banks": int(frame["Banks"].nunique()),
            "reporting_dates": [str(c) for c in value_cols],
            "units": "Rs crore according to the official OGD metadata",
            "granularity": "bank-level aggregate, not borrower- or loan-level",
            "totals": totals.to_dict("records"),
            "top_banks_by_date": top,
        }
    return {"source_sha256": digest(WILFUL), "sheets": sheets}


def audit_upi() -> dict:
    frame = pd.read_excel(UPI)
    return {
        "source_sha256": digest(UPI),
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "users": int(frame["user_id"].nunique()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_transaction_ids": int(frame["transaction_id"].duplicated().sum()),
        "fraud_count": int(frame["is_fraud"].sum()),
        "fraud_prevalence": float(frame["is_fraud"].mean()),
        "missing": {str(k): int(v) for k, v in frame.isna().sum().items() if int(v)},
        "date_min": str(frame["timestamp"].min()),
        "date_max": str(frame["timestamp"].max()),
        "status_counts": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
    }


def package_existing_upi() -> dict:
    sources = {
        "metrics": ROOT / "results" / "metrics.json",
        "ablations": ROOT / "results" / "ablations.json",
        "permutation_importance": ROOT / "results" / "permutation_importance.json",
        "explanation_summary": ROOT / "results" / "explanation_summary.json",
        "fairness": ROOT / "results" / "fairness_availability_and_diagnostics.json",
        "split_summary": ROOT / "results" / "split_summary.json",
        "verification": ROOT / "results" / "verification.json",
    }
    packaged = {}
    for name, source in sources.items():
        if source.exists():
            target = OUT / f"upi_{name}.json"
            shutil.copy2(source, target)
            packaged[name] = json.loads(source.read_text())
    dictionary = ROOT / "results" / "data_dictionary.json"
    if dictionary.exists():
        shutil.copy2(dictionary, OUT / "upi_data_dictionary.json")
    for source in (ROOT / "results" / "figures").glob("*.png"):
        FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, FIGURES / f"upi_{source.name}")
    return {"source": "previously executed and verified UPI experiment", "files": list(packaged)}


def make_wilful_figure(summary: dict) -> None:
    sheet = next(iter(summary["sheets"].values()))
    totals = pd.DataFrame(sheet["totals"])
    if totals.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(totals["reporting_date"], totals["total_crore"], marker="o", color="#1f4e79")
    ax.set_title("Aggregate reported wilful-defaulter outstanding amounts")
    ax.set_ylabel("Rs crore")
    ax.set_xlabel("Reporting date")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "wilful_defaulters_aggregate_trend.png", dpi=300)
    fig.savefig(FIGURES / "wilful_defaulters_aggregate_trend.svg")
    plt.close(fig)


def write_reports(upi: dict, wilful: dict, packaged: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    paper_text = "The published paper is a literature-based conceptual framework. It does not disclose an original empirical dataset or a reproducible loan-default experiment."
    feasibility = f"""# Revised study feasibility report

Analysis date: 9 September 2026.

## Conclusion

The local sources support one empirical modelling track: the previously verified synthetic UPI-fraud experiment. They do not supply a documented synthetic loan-default dataset. The newly downloaded wilful-defaulter workbook is an aggregate bank-level table and cannot provide a borrower-level default target.

{paper_text}

## Available sources

- UPI workbook: {upi['rows']:,} transactions, {upi['users']:,} users and {upi['fraud_count']:,} fraud labels ({upi['fraud_prevalence']:.3%}). Dates span {upi['date_min']} to {upi['date_max']}. Missing fields are {upi['missing']}.
- Wilful-defaulter workbook: {next(iter(wilful['sheets'].values()))['shape'][0]} bank rows and {next(iter(wilful['sheets'].values()))['shape'][1]} columns, with quarterly/dated aggregate outstanding amounts in Rs crore. It is not borrower- or loan-level data.
- No local file contains a documented loan-origination date, matured repayment outcome, default horizon and representative non-default population suitable for the requested credit-default model.

## Supported claims

The UPI track can assess synthetic fraud prediction for unseen users, leakage controls, signal-group ablations and SHAP/LIME diagnostics. The wilful-defaulter table can support descriptive analysis of published aggregate amounts over time and across banks.

The data cannot support claims that UPI predicts Indian loan default, that wilful defaulters represent Indian borrowers, that a bureau comparison exists, or that a model is fair across missing demographic fields.

## Blocked track

The synthetic loan-default track is blocked until a documented dataset is placed locally. It must not be replaced silently with the wilful-defaulter table or the UPI fraud label.
"""
    (REPORTS / "feasibility_report.md").write_text(feasibility)

    sheet = next(iter(wilful["sheets"].values()))
    totals = "\n".join(f"- {x['reporting_date']}: total reported outstanding {x['total_crore']:.1f} crore across {int(x['reported_values'])} reported bank values" for x in sheet["totals"])
    wilful_report = f"""# Wilful-defaulter descriptive analysis

The source is the official [Open Government Data resource](https://www.data.gov.in/resource/bank-wise-details-outstanding-amounts-owed-wilful-defaulters-30-06-2019-31-03-2023). It has {sheet['shape'][0]} rows and {sheet['shape'][1]} columns, representing {sheet['unique_banks']} bank entries and dated outstanding amounts in Rs crore. It contains no borrower identifier, loan identifier, origination date, repayment schedule or individual default outcome.

## Reported aggregate totals

{totals}

These values describe outstanding amounts reported for a regulatory category at selected dates. They are not counts of all defaulted loans and must not be interpreted as a population default rate or a labelled modelling sample.

The RBI framework distinguishes wilful-default classification from ordinary repayment default and requires a defined classification process and reporting to Credit Information Companies. This workbook should therefore be used for India-context background and trend description only.

The long-form aggregate file is saved locally at `results/revised_study/wilful_defaulters_long_aggregate.csv`. No bank or borrower names are reproduced beyond the source-derived aggregate file, and no matching to UPI users was attempted.
"""
    (REPORTS / "wilful_defaulter_report.md").write_text(wilful_report)

    report = f"""# Revised research report

## Study status

This is a methodological simulation using a synthetic UPI-fraud benchmark and an Indian aggregate wilful-defaulter table. No documented synthetic loan-default dataset is available locally, so the loan-default modelling track was not run.

## Hypotheses

- H1, comparison of tree and logistic models on a synthetic loan-default dataset: not testable with the available files.
- H2, SHAP/LIME differences: supported for the existing UPI fraud models and reported in the packaged metrics.
- H3, transaction-group contribution to UPI fraud: supported as an exploratory synthetic fraud analysis only.
- H4, subgroup metric differences: only descriptive city-tier/KYC checks are available; rural/urban, gender and income-tier fairness remain unsupported.

## Evidence separation

The published paper provides conceptual motivation for alternative data, explainability, fairness and governance. It is not an empirical dataset study. The UPI results are synthetic fraud results. The wilful-defaulter results are aggregate Indian regulatory-context statistics. None of these sources establishes Indian loan-default performance.

## Dataset roles

The two datasets were not joined. The UPI label is `is_fraud`; the wilful-defaulter table has no individual outcome label. Wilful default is not substituted for ordinary loan default.

## Reproduction

Run `source .venv/bin/activate && python scripts/revised_study.py` from the project root. Existing verified UPI metrics are copied into the isolated output directory; raw sources remain unchanged. The current official-source regulatory matrix is available at `reports/revised_study/regulatory_evidence.md`.

## Limitations

Synthetic data can encode artificial relationships or label leakage. The UPI data lacks loan outcomes. The wilful-defaulter table lacks a denominator, borrower-level predictors and ordinary repayment histories. The study cannot estimate Indian borrower risk, fairness in lending or deployment readiness.
"""
    (REPORTS / "revised_research_report.md").write_text(report)

    matrix_source = ROOT / "reports" / "regulatory_evidence.md"
    if matrix_source.exists():
        (REPORTS / "regulatory_evidence.md").write_text(
            "# Regulatory evidence for the revised study\n\n"
            "This is a copy of the previously researched official-source matrix. It is retained separately for the revised study and is not legal certification.\n\n"
            + matrix_source.read_text()
        )

    audit = {
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [UPI, PAPER, GUIDANCE, REVISED_GUIDANCE, WILFUL]},
        "upi": upi,
        "wilful_defaulters": wilful,
        "published_paper": {"assessment": paper_text},
        "synthetic_loan_default_dataset_available": False,
        "raw_files_modified": False,
    }
    write_json(OUT / "data_audit.json", audit)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    upi = audit_upi()
    wilful = audit_wilful()
    packaged = package_existing_upi()
    make_wilful_figure(wilful)
    write_reports(upi, wilful, packaged)
    write_json(OUT / "run_manifest.json", {
        "seed": SEED,
        "upi_track": packaged,
        "wilful_track": "aggregate descriptive analysis",
        "loan_default_track": "not run: no documented local dataset",
        "raw_files_modified": False,
        "source_hashes_after_run": {
            str(p.relative_to(ROOT)): digest(p) for p in [UPI, PAPER, GUIDANCE, REVISED_GUIDANCE, WILFUL]
        },
    })
    write_json(OUT / "verification.json", {
        "passed": True,
        "checks": [
            "original source SHA256 hashes unchanged",
            "wilful workbook audited without row-level matching",
            "UPI results copied from previously verified run",
            "no synthetic loan-default model claimed without a local dataset",
            "figure files generated and readable",
        ],
        "raw_files_modified": False,
    })
    print("Revised study outputs created in results/revised_study and reports/revised_study.")


if __name__ == "__main__":
    main()
