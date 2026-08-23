# SQLShield: Leakage-Resistant SQL Injection Detection

Reproducibility repository for the manuscript:

> **SQLShield: Leakage-Resistant SQL Injection Detection with Fine-Tuned CodeBERT, Classical Baselines, and Cross-Source Evaluation**

Release **v1.4.0** corresponds to `paper/SQLShield_Paper_IJIES_final.docx`. This release synchronizes the reproducibility artifact with the final manuscript while retaining the fully verified evidence bundle: Wilson confidence intervals, direct three-error analysis, scoped accuracy-cost reporting, near-duplicate sensitivity caveats, **all eight aligned per-sample fixed-test prediction files, plus cross-source and multi-seed prediction evidence**, and a strict prediction-based verifier that recomputes the reported paired tests from those predictions. No headline experimental result is changed by the v1.4.0 evidence-completion release.

## Fixed-test findings (`n = 5,654`)

| Model | Accuracy (95% Wilson CI) | F1 | Errors |
|---|---:|---:|---:|
| CodeBERT (seed 42) | **99.95% [99.84%, 99.98%]** | **0.9993** | **3** |
| BERT-base (seed 42) | 99.89% [99.77%, 99.95%] | 0.9986 | 6 |
| Char TF-IDF (3-5) + LinearSVC | **99.84% [99.70%, 99.92%]** | **0.9980** | **9** |
| Random Forest | 99.72% [99.54%, 99.83%] | 0.9964 | 16 |
| Char Logistic Regression | 99.65% [99.45%, 99.77%] | 0.9954 | 20 |
| XGBoost | 99.63% [99.43%, 99.76%] | 0.9952 | 21 |
| Word LinearSVC | 99.56% [99.35%, 99.70%] | 0.9943 | 25 |
| Word Logistic Regression | 99.08% [98.80%, 99.30%] | 0.9881 | 52 |

Wilson intervals are reported for **accuracy only**. They should not be transferred mechanically to F1 or AUC metrics.

Across five training seeds on the same fixed split, CodeBERT mean F1 is `0.999229 +/- 0.000124`; BERT-base mean F1 is `0.998411 +/- 0.000454`.

## Exact McNemar and error complementarity

The seven-comparison exact McNemar family shows:

- CodeBERT vs BERT-base: not significant (`p = 0.453125`).
- CodeBERT vs Char LinearSVC: not significant (`p = 0.109375`; Holm `p = 0.218750`).
- The other five CodeBERT comparisons remain significant after Holm and Bonferroni correction.

For CodeBERT vs Char LinearSVC, the paired discordant counts are `2` and `8`. Combined with total errors `3` and `9`, this implies exactly:

- **2 CodeBERT-only errors**;
- **8 Char-LinearSVC-only errors**;
- **1 shared error**;
- error union = `11`, error-set Jaccard = `1/11 = 9.1%`.

This is an important practical result: the two best-performing model classes make largely different mistakes even though their aggregate performance difference is not statistically significant.

## Logged computational cost: use with the correct scope

| Logged run | F1 / range | Errors | Wall-clock | Path |
|---|---:|---:|---:|---|
| CodeBERT fuzzy retraining, mean of tau 0.9/0.8/0.7 | 0.9981-0.9989 | 5-9 | **4,040.5 s (67.3 min)** | NVIDIA T4 |
| Char LinearSVC fixed-split job | 0.9980 | 9 | **5.91 s** | CPU path |
| Char Logistic Regression fixed-split job | 0.9954 | 20 | **7.54 s** | CPU path |

**Do not interpret this table as an inference-latency ratio.** The character timings cover TF-IDF/classifier fitting plus fixed-test prediction/scoring, whereas CodeBERT timing covers four-epoch retraining, validation, and final test evaluation. A controlled batch-1/batch-32 serving benchmark remains future work.

## Near-duplicate sensitivity

The primary split isolates exact `normalized_text` groups. The fuzzy sensitivity experiment uses 5-character shingles, MinHash-LSH candidate generation, **exact Jaccard verification**, connected components, and group-disjoint splitting.

| Jaccard threshold | Fuzzy families | Test n | F1 | Errors |
|---:|---:|---:|---:|---:|
| 1.0 | 45,915 | 5,654 | 0.9993 | 3 |
| 0.9 | 43,675 | 5,652 | 0.9989 | 5 |
| 0.8 | 40,705 | 5,754 | 0.9981 | 9 |
| 0.7 | 38,526 | 5,525 | 0.9988 | 5 |

Important caveats:

- `tau=1.0` uses the **original stratified normalized-group split**, while `tau<1.0` uses row-level `StratifiedGroupKFold`. Therefore the baseline-to-fuzzy comparison confounds **grouping strength** with **split construction**.
- The threshold-specific test sets differ; do not run paired McNemar tests across tau values.
- Only CodeBERT seed 42 is retrained at each fuzzy threshold.
- `max_exact_groups_in_cluster=400` and `max_rows_in_cluster=700` occur at both tau 0.8 and 0.7. These are observed outputs, **not hard-coded caps**; the clustering implementation imposes no maximum component size. The summary audit alone does not prove the same component produces both maxima.
- MinHash-LSH is an approximate candidate generator; accepted edges are exact-Jaccard verified, but the method is not exhaustive semantic deduplication.

## Cross-source evaluation

- A -> B: ROC-AUC `0.990252`, PR-AUC `0.839063`, balanced accuracy `0.960329`.
- B -> A: ROC-AUC `0.703293`, PR-AUC `0.034568`, balanced accuracy `0.503939`, MCC `0.007797`.
- Benign-only source identification: ROC-AUC `0.998904`.

Do not interpret the mixed-source score as universal source-independent portability.

## Prediction-level reproducibility

All **eight** aligned 5,654-row fixed-test prediction files are included under `results/predictions/`: CodeBERT seed 42, BERT-base seed 42, four word-level classical baselines, and the two character-level baselines. This closes the paired-evidence gap for the paper's most important exact McNemar result.

Run:

```bash
python scripts/verify_reported_results.py
python scripts/analyze_fixed_test_errors.py
```

The strict verifier independently reloads the eight prediction CSVs, checks identical text/order and true labels, recomputes confusion metrics, ROC-AUC/PR-AUC where scores are available, and recomputes all seven two-sided exact McNemar tests plus Holm and Bonferroni adjustments. In this release it returns:

```text
STRICT PREDICTION VERIFICATION: PASSED
Recomputed fixed-test confusion metrics and all seven exact McNemar comparisons from 5,654 aligned per-sample predictions.
```

The error-analysis script writes `results/error_analysis/codebert_three_errors.csv` and the full 11-row CodeBERT-vs-Char-LinearSVC error union. The three CodeBERT errors are: `5739-5738` (shared false negative), `sort` (CodeBERT-only false positive), and `hi or a = a` (CodeBERT-only false negative).

`export_saved_predictions.py` is retained as a convenience for rebuilding the same evidence directory from a restored Kaggle workspace; it is no longer required for this packaged release.

For a limited audit of the aggregate tables only:

```bash
python scripts/verify_reported_results.py --aggregate-only
```

That mode explicitly labels itself **aggregate consistency only**, not independent prediction-level verification.

## Repository layout

```text
sqlshield-leakage-resistant-sqli/
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── requirements-verification.txt
├── pyproject.toml
├── CITATION.cff
├── .github/workflows/verification.yml
├── sqlshield/
├── scripts/
│   ├── run_*.py
│   ├── compute_wilson_ci.py
│   ├── export_saved_predictions.py
│   ├── analyze_fixed_test_errors.py
│   └── verify_reported_results.py
├── LICENSE
├── figures/
│   ├── figure1_source.py
│   └── figure1_protocol.png
├── results/
│   ├── paper_reported/
│   ├── predictions/          # 12 prediction files (8 aligned fixed-test + 4 auxiliary)
│   ├── histories/            # per-epoch training curves (6 runs)
│   ├── error_analysis/
│   └── extension_raw/         # char n-gram + per-threshold fuzzy predictions
├── paper/
│   └── SQLShield_Paper_IJIES_final.docx
├── notebooks/
│   ├── kaggle_provenance_notebook.ipynb
│   └── kaggle_extensions_notebook.ipynb
└── docs/
```

## Installation

Python 3.10-3.12 is recommended. `requirements.txt` and `requirements-lock.txt` use exact version pins.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

**Environment provenance caveat:** the exact package versions of the historical Kaggle image were not captured in the exported experiment logs. The pinned file is therefore a **frozen compatibility/reproduction environment**, not a claim that those exact versions were the original Kaggle package versions. This distinction is documented in `docs/PROVENANCE.md`.

### Local dataset/output paths

The canonical code reads local paths from environment variables and otherwise falls back to the original Kaggle locations. For Linux/macOS:

```bash
export SQLSHIELD_A_PATH=/path/to/Modified_SQL_Dataset.csv
export SQLSHIELD_B_PATH=/path/to/sqliv2.csv
export SQLSHIELD_OUT=/path/to/sqlshield_corrected_validation
```

For Windows PowerShell:

```powershell
$env:SQLSHIELD_A_PATH="C:\path\to\Modified_SQL_Dataset.csv"
$env:SQLSHIELD_B_PATH="C:\path\to\sqliv2.csv"
$env:SQLSHIELD_OUT="C:\path\to\sqlshield_corrected_validation"
```

Raw datasets are intentionally not redistributed; see `data/README.md`.

## Reproduce the experiments

```bash
python scripts/run_classical.py
python scripts/run_transformers.py --models all --seeds 7 21 42 84 126
python scripts/run_char_ngram_baselines.py --base-dir /path/to/sqlshield_corrected_validation
python scripts/run_exact_mcnemar.py
python scripts/run_fuzzy_sensitivity.py --base-dir /path/to/sqlshield_corrected_validation
python scripts/run_cross_source.py
python scripts/run_source_shift_diagnostics.py
python scripts/run_threshold_transfer.py
```

GPU is strongly recommended for transformer and fuzzy-sensitivity retraining. `--cluster-only` on the fuzzy script performs the clustering/split audit without CodeBERT training.

## Machine-readable reporting artifacts

`results/paper_reported/` additionally contains:

- `fixed_test_accuracy_wilson95.csv`
- `error_overlap_summary.csv`
- `computational_cost_summary.csv`
- `prediction_recomputed_fixed_test_FINAL.csv`
- `prediction_recomputed_exact_mcnemar_FINAL.csv`

`results/error_analysis/` contains the prediction-derived three-error table and complete CodeBERT-vs-Char-LinearSVC error union.

## Provenance

The archived base experiment is Kaggle notebook `mohammadalkhazaleh/notebook691d0eee69`. Character and fuzzy-sensitivity extensions were executed afterward. See `docs/PROVENANCE.md`.

## Manuscript

```text
paper/SQLShield_Paper_IJIES_final.docx
```

## Continuous verification and citation

`.github/workflows/verification.yml` uses `requirements-verification.txt` to run the lightweight regression tests, strict prediction-level verifier, error-overlap audit, and repository SHA-256 check on GitHub Actions without installing the full transformer training stack. `CITATION.cff` supplies manuscript/repository citation metadata. After the public GitHub repository is created, update its commented `repository-code` field with the final repository URL.

## License

Source code (`sqlshield/`, `scripts/`, `tests/`, `figures/`) is released under the MIT License; see `LICENSE`.
Author-generated content under `results/` (predictions, scores, labels, statistics, cluster ids, training histories, derived summaries) is released under CC BY 4.0. Source-query text reproduced inside those files is not author-generated and is not relicensed; it remains subject to the original Kaggle publishers' terms.
The two source corpora are **not** redistributed here and remain subject to their original Kaggle terms; pretrained weights (`microsoft/codebert-base`, `bert-base-uncased`) remain under their upstream licenses.
