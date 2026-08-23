# Per-sample fixed-test prediction evidence

This directory contains the **eight** aligned fixed-test prediction files used for the paper's primary 5,654-row paired comparisons:

- CodeBERT seed 42
- BERT-base seed 42
- Random Forest
- XGBoost
- Word LinearSVC (`classical__SVM.csv`)
- Word Logistic Regression
- Character LinearSVC
- Character Logistic Regression

Every file contains `text`, `source`, `true_label`, `pred_label`, a model score/probability column, and `correct`. The strict verifier checks that all models use the same text/order and true labels before recomputing the reported metrics and McNemar tests.

```bash
python scripts/verify_reported_results.py
python scripts/analyze_fixed_test_errors.py
```

Expected strict result for this release:

```text
STRICT PREDICTION VERIFICATION: PASSED
Recomputed fixed-test confusion metrics and all seven exact McNemar comparisons from 5,654 aligned per-sample predictions.
```

`prediction_manifest.csv` records the expected filenames.

## Additional prediction evidence (v1.4.0)

Four further prediction files are released. They are not part of the strict eight-model McNemar family,
so the verifier does not consume them, but they make additional manuscript results recomputable:

| File | Rows | Backs |
| --- | --- | --- |
| `cross_source_A_sajid576_TO_B_sqliv2__CodeBERT__seed42.csv` | 15,182 | Table 9, Figure 7 (A->B) |
| `cross_source_B_sqliv2_TO_A_sajid576__CodeBERT__seed42.csv` | 12,408 | Table 9, Figure 7 (B->A), threshold transfer |
| `corrected_group_split_multiseed__CodeBERT__seed21.csv` | 5,654 | Multi-seed stability analysis (Section 4.4) and Figure 5 |
| `corrected_group_split_multiseed__BERT-base__seed126.csv` | 5,654 | Multi-seed stability analysis (Section 4.4) and Figure 5 |

Recompute the cross-source numbers directly:

```python
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, matthews_corrcoef

d = pd.read_csv("results/predictions/cross_source_B_sqliv2_TO_A_sajid576__CodeBERT__seed42.csv")
tn, fp, fn, tp = confusion_matrix(d.true_label, d.pred_label).ravel()
print(tn, fp, fn, tp)                                   # 97 12216 0 95
print(balanced_accuracy_score(d.true_label, d.pred_label))  # 0.503939
print(matthews_corrcoef(d.true_label, d.pred_label))        # 0.007797
```

## Scope limit: multi-seed coverage

Per-sample predictions exist for four of the ten multi-seed runs (CodeBERT seeds 21 and 42;
BERT-base seeds 42 and 126). For the remaining six — CodeBERT seeds 7, 84, 126 and BERT-base
seeds 7, 21, 84 — per-sample predictions were **not retained upstream**; only run-level metrics
survive, and those rows are marked `recovered_from_notebook_output` in
`results/paper_reported/transformer_multiseed_runs_FINAL.csv`. Their precision, recall, TN, and TP
are derived arithmetically from the logged FP/FN and the fixed test class counts (3,450 benign;
2,204 SQLi). The multi-seed stability analysis reported in Section 4.4 and plotted in Figure 5 is
therefore only partially recomputable from raw predictions. (Note that Table 3 in the manuscript is
the transformer training configuration, not the multi-seed results; the seed-level numbers appear in
the Section 4.4 text and in Figure 5.)

## Related evidence outside this directory

Per-threshold near-duplicate predictions for Table 8 are in `results/extension_raw/fuzzy/`
(`predictions_tau0p90.csv`, `predictions_tau0p80.csv`, `predictions_tau0p70.csv`). They are not
part of the aligned fixed-test family — each threshold induces a different test partition of
5,652 / 5,754 / 5,525 rows — and they carry an extra `fuzzy_cluster_id` column that lets the
family-isolation claim be checked directly.
