# Provenance

## Base corrected experiment

The corrected base experiments were audited and rerun in Kaggle notebook:

```text
mohammadalkhazaleh/notebook691d0eee69
```

The archived notebook in this package corresponds to saved **Version 9** and documents the main corrected preprocessing, fixed split, transformer multi-seed, cross-source, and diagnostic work. It contains exploratory/recovery cells and is therefore not the canonical clean implementation.

## Character-baseline extension

On 2026-08-17, the fixed split was extended with character 3–5-gram TF-IDF + LinearSVC and character 3–5-gram TF-IDF + Logistic Regression. The downloaded original small result files are retained under `results/extension_raw/character/`. The final McNemar family was expanded from five to seven comparisons.

## Fuzzy/near-duplicate sensitivity extension

On 2026-08-17, a completed CodeBERT sensitivity run used verified 5-character-shingle Jaccard family clustering at thresholds 0.9, 0.8, and 0.7. The run generated a full archive of roughly 1.4 GB because it contained threshold-specific datasets, predictions, training histories, and checkpoints. That large archive is intentionally not included here. The final printed metrics and cluster-audit values were captured and transcribed into `results/paper_reported/fuzzy_*_FINAL.*`; the complete canonical reproduction code is `scripts/run_fuzzy_sensitivity.py`.

## Canonical implementation

Use the clean repository code as authoritative implementation:

- `sqlshield/preprocessing.py`
- `sqlshield/pipeline.py`
- `scripts/*.py`

Two corrections must not be reverted: source-internal exact deduplication uses `(text,label)`, preserving contradictions for explicit conflict removal; and cross-source evaluation removes globally contradictory normalized groups before source-specific splitting.

Large model checkpoints are not redistributed. Fixed-test per-example predictions are treated as verification evidence. All eight aligned 5,654-row fixed-test prediction CSVs are now shipped in `results/predictions/`, enabling independent recomputation of the paper's seven exact McNemar comparisons.

## v1.4 evidence, environment, and manuscript synchronization

- The two character per-sample prediction CSVs are included under `results/predictions/`.
- The original saved CodeBERT seed-42, BERT-base seed-42, Random Forest, XGBoost, word LinearSVC, and word Logistic Regression prediction CSVs were subsequently exported from the restored Kaggle output and added unchanged.
- `verify_reported_results.py` is strict by default, checks row alignment, and recomputes the reported fixed-test metrics and seven exact McNemar comparisons from all eight prediction files. It passes on this release. `--aggregate-only` remains explicitly non-independent.
- `results/error_analysis/codebert_three_errors.csv` identifies the three CodeBERT mistakes directly from the released predictions, and `codebert_vs_char_linearsvc_error_rows.csv` contains the complete 11-row error union.
- Historical Kaggle package versions were not captured in the available logs. The exact pins in `requirements-lock.txt` are a frozen compatibility/reproduction environment, not a claim about the exact historical Kaggle image.
- Logged CodeBERT and character wall-clock times have different scopes and hardware paths; they are not serving-latency measurements.


## Final manuscript synchronization

Repository release v1.4.0 packages `paper/SQLShield_Paper_IJIES_final.docx`. That manuscript is a writing/positioning refinement of the already verified experimental evidence: the fixed-test metrics, seven exact McNemar comparisons, three CodeBERT error rows, fuzzy-sensitivity values, and cross-source diagnostics remain backed by the same machine-readable artifacts and aligned per-sample predictions. No experimental output was edited to force agreement with the manuscript.
