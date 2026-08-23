# Paper-reported machine-readable results

These tables support the IJIES-format manuscript (`paper/SQLShield_Paper_IJIES_final.docx`) and repository release v1.4.0.

Key reporting extensions:

- `fixed_test_accuracy_wilson95.csv`: 95% Wilson CIs for fixed-test accuracy.
- `error_overlap_summary.csv`: CodeBERT vs Char LinearSVC paired-error overlap (2 CodeBERT-only, 8 Char-only, 1 shared).
- `computational_cost_summary.csv`: logged wall-clock times with scope/hardware caveats.
- `prediction_recomputed_fixed_test_FINAL.csv`: fixed-test metrics independently recomputed from the eight shipped 5,654-row prediction files.
- `prediction_recomputed_exact_mcnemar_FINAL.csv`: all seven exact McNemar comparisons independently recomputed from the shipped predictions.

The original aggregate tables remain for provenance. Prediction-level evidence is under `../predictions/`, and `python scripts/verify_reported_results.py` passes in strict mode on this release.
