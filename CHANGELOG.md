# Changelog

## 1.4.0 — evidence completion and licensing

- Added four per-sample prediction files that were previously described but not shipped:
  `cross_source_A_sajid576_TO_B_sqliv2__CodeBERT__seed42.csv` (15,182 rows),
  `cross_source_B_sqliv2_TO_A_sajid576__CodeBERT__seed42.csv` (12,408 rows),
  `corrected_group_split_multiseed__CodeBERT__seed21.csv`, and
  `corrected_group_split_multiseed__BERT-base__seed126.csv`.
  Table 9, Figure 7, and the B->A threshold-transfer claim are now recomputable from raw predictions.
- Added `results/histories/` with per-epoch training curves for six runs, making Figure 2 and the
  validation-F1 checkpoint-selection policy independently checkable.
- Added `LICENSE` (MIT for code, CC BY 4.0 for released results) and GitHub-compatible citation metadata in `CITATION.cff`.
  Earlier releases asserted no license, which left the artifacts formally all-rights-reserved.
- Added `figures/figure1_source.py` and `figures/figure1_protocol.png` so Figure 1 is regenerable.
- Added `notebooks/kaggle_extensions_notebook.ipynb`, which covers the character n-gram and
  near-duplicate sensitivity experiments absent from the archived base notebook.
- Extended `results/predictions/prediction_manifest.csv` with a `supports` column mapping each file to
  the manuscript table or figure it backs, and with explicit rows for the six multi-seed runs whose
  per-sample predictions were not retained upstream.
- Replaced `paper/` with the final IJIES-format manuscript.
- Added per-threshold near-duplicate evidence to `results/extension_raw/fuzzy/`: predictions, training
  histories, and result JSON for tau = 0.9, 0.8, and 0.7 (5,652 / 5,754 / 5,525 rows). Table 8 is now
  recomputable from raw predictions, and the `fuzzy_cluster_id` column makes family isolation directly
  checkable (0 mixed-label clusters at every threshold). Checkpoints and per-threshold split CSVs remain
  omitted.
- Regenerated `SHA256SUMS.txt` over the full file set.

## 1.3.0 — final manuscript synchronization and GitHub-release hardening

- Synchronized the repository with the final manuscript `paper/SQLShield_Paper_v17.docx`; the machine-readable results and verified prediction evidence are unchanged.
- Updated repository/version metadata from v1.2.1 to v1.3.0 across README, provenance, result guides, and upload instructions.
- Added explicit local `SQLSHIELD_A_PATH`, `SQLSHIELD_B_PATH`, and `SQLSHIELD_OUT` setup examples for Linux/macOS and Windows PowerShell.
- Added `CITATION.cff` with the manuscript title and four authors, without fabricating a journal, DOI, or repository URL.
- Added `requirements-verification.txt` and a lightweight GitHub Actions workflow that runs regression tests, strict prediction-level verification, error analysis, and SHA-256 integrity checks without the full transformer training stack.
- Regenerated repository and paper-reported SHA-256 manifests after synchronization.
- No scientific result, per-sample prediction, or reported statistical conclusion was changed in this release.

## 1.2.1 — complete paired-prediction verification

- Added the six original Kaggle-exported fixed-test prediction CSVs for CodeBERT seed 42, BERT-base seed 42, Random Forest, XGBoost, word LinearSVC, and word Logistic Regression; together with the two character files, all eight paired prediction artifacts are now public.
- Strict verification now passes after independently recomputing fixed-test metrics and all seven exact McNemar comparisons from 5,654 aligned rows.
- Added prediction-derived result tables `prediction_recomputed_fixed_test_FINAL.csv` and `prediction_recomputed_exact_mcnemar_FINAL.csv`.
- Added `results/error_analysis/codebert_three_errors.csv` and the complete CodeBERT-vs-Char-LinearSVC error union.
- Updated the manuscript with the exact three CodeBERT errors and a three-row error-analysis table.
- Updated the manuscript path to `SQLShield_Paper_v12_FINAL_verified_predictions.docx`.

## 1.2.0 — statistical uncertainty and prediction-evidence strengthening

- Added 95% Wilson confidence intervals for fixed-test accuracy and reduced misleading decimal precision in the manuscript.
- Added CodeBERT-vs-Char-LinearSVC error complementarity analysis: 2 CodeBERT-only, 8 Char-only, 1 shared error.
- Added a scoped computational cost table; explicitly distinguishes logged end-to-end training/evaluation wall-clock from inference latency.
- Explicitly documented the tau=1.0 vs tau<1.0 split-procedure confound.
- Documented that repeated 400-group/700-row fuzzy maxima are observed outputs, not implementation caps.
- Added character per-sample predictions, prediction exporter, error-analysis tool, Wilson calculator, and strict prediction-based verifier.
- Replaced `>=` runtime dependency specifications with exact pinned versions and documented that the pins are a frozen compatibility environment, not the exact historical Kaggle image.
- Updated manuscript to `SQLShield_Paper_v12_FINAL.docx`.

## 1.1.0 — 2026-08-17

- Added character 3–5-gram TF-IDF + LinearSVC and Logistic Regression baselines.
- Expanded the fixed-test benchmark from six to eight total models.
- Expanded the exact McNemar family from five to seven CodeBERT comparisons; CodeBERT vs character LinearSVC is non-significant.
- Added the 5-character-shingle fuzzy/near-duplicate sensitivity protocol at Jaccard thresholds 0.9, 0.8, and 0.7.
- Added final fuzzy sensitivity and cluster-audit machine-readable results.
- Added `datasketch` dependency.
- Updated verification script, documentation, README, manuscript, and provenance for the final extension.

## 1.0.0 — 2026-08-17

- Replaced concat-first text-column handling with source-specific parsing.
- Changed source-internal deduplication to `(text, label)` so contradictory labels remain detectable.
- Removed the single contradictory normalized group before merged and cross-source evaluation.
- Added normalized-group-aware 80/10/10 fixed split with zero-overlap assertions.
- Recomputed four word-level classical baselines.
- Added five-seed CodeBERT and BERT-base stability analysis.
- Recomputed exact McNemar tests with Holm and Bonferroni corrections.
- Added bidirectional normalized-overlap-free cross-source evaluation.
- Added validation-only threshold transfer, source-conditioned fixed-test metrics, and benign source-fingerprint diagnostics.
