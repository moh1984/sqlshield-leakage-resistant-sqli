# Results guide — v1.4

## Fixed-test uncertainty

`results/paper_reported/fixed_test_accuracy_wilson95.csv` contains two-sided 95% Wilson score confidence intervals for the eight fixed-test accuracy estimates. Wilson intervals are used for accuracy only, not F1 or AUC.

## Error complementarity

`results/paper_reported/error_overlap_summary.csv` records the paired CodeBERT-vs-Char-LinearSVC error counts implied by the final McNemar table: 2 CodeBERT-only, 8 Char-only, 1 shared, union 11, Jaccard 1/11.

The row identities are now included directly in `results/predictions/`. Run `scripts/analyze_fixed_test_errors.py` to regenerate `results/error_analysis/codebert_three_errors.csv` and the complete 11-row CodeBERT-vs-Char-LinearSVC error union.

## Computational cost

`computational_cost_summary.csv` preserves the timing scope. Do not compare the logged seconds as if they were serving latency: the character jobs are fit+test CPU-path timings, while CodeBERT fuzzy timings are four-epoch train+validation+test T4 timings.

## Fuzzy sensitivity

`tau=1.0` is the original stratified normalized-group split. `tau<1.0` uses StratifiedGroupKFold. This is a sensitivity stress test and confounds grouping strength with split construction. The repeated maxima 400 groups / 700 rows at tau 0.8 and 0.7 are observed outputs, not implementation caps.

## Verification levels

- `python scripts/verify_reported_results.py`: strict per-sample prediction verification; this release includes all eight required prediction files and the check passes.
- `results/paper_reported/prediction_recomputed_fixed_test_FINAL.csv`: metrics recomputed directly from the shipped predictions.
- `results/paper_reported/prediction_recomputed_exact_mcnemar_FINAL.csv`: all seven McNemar comparisons recomputed from the shipped predictions.
- `results/error_analysis/codebert_three_errors.csv`: the three actual CodeBERT seed-42 errors.
- `results/error_analysis/codebert_vs_char_linearsvc_error_rows.csv`: the 11 distinct error rows in the union of the two strongest models.
- `python scripts/verify_reported_results.py --aggregate-only`: aggregate-table consistency audit only; not independent verification.
