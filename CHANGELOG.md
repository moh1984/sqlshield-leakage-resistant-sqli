# Changelog

# v1.5.0 — reviewer-requested experiments (IJIES revision)

Nothing in `sqlshield/` changed. The published results of v1.4.0 remain
reproducible exactly as before, and were reproduced during this revision.

### Added
- `scripts/run_cross_source_all_models.py` — bidirectional overlap-free
  cross-source evaluation for all six classical models and, with
  `--transformers`, for CodeBERT and BERT-base. Classical models are fitted on
  the training partition alone so that no model sees more data than the
  transformers.
- `scripts/run_baseline_tuning.py` — validation-only grid over n-gram range,
  vocabulary size, regularisation and class weighting; each distinct vectorizer
  is fitted once and its matrices reused across classifiers.
- `scripts/run_lsh_recall_validation.py` — MinHash-LSH candidate recall against
  exhaustive pairwise Jaccard on a sampled subset.
- `scripts/run_multi_split.py` — repeated independent group-aware partitions,
  with `--classical-splits` allowing the cheap family to be run on more
  partitions than the transformers.
- `scripts/run_inference_benchmark.py` — batch-1 latency, throughput, parameter
  count, model size and peak device memory on one documented device.
- `results/review_2026/` — outputs of the above.

### Corrected in the manuscript, not in the code
- Exact cross-source deduplication removes 7,680 rows, not 7,678. The 30,764
  figure for Source A already excludes the two label-conflicted rows, which must
  therefore not be subtracted a second time.
- Table 7 became a controlled inference benchmark; the three logged fuzzy-family
  runtimes (4,030.7 s, 4,032.6 s, 4,058.1 s) moved into Section 7.2, and the
  fixed-split run is recorded as never having been timed separately.

### Fixed after internal review
- `run_lsh_recall_validation.py` used 128 MinHash permutations and no seed,
  while `run_fuzzy_sensitivity.py` uses 256 permutations and seed 12345. The
  validation therefore measured a different candidate generator from the one
  that produced Table 8. It now matches the published parameters, including the
  `max(0.10, tau - margin)` candidate floor and `update_batch` hashing.
- `run_inference_benchmark.py` measured classical throughput on all 1,000
  queries in one call while the transformers were measured at batch 32. The
  classical path is now chunked at the same batch size, so the two families are
  comparable.
- Superseded outputs moved to `results/review_2026/superseded/` with a note
  explaining why each was replaced.

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
