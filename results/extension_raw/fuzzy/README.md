# Near-duplicate (fuzzy) sensitivity — raw per-threshold evidence

This directory holds the per-threshold outputs of `scripts/run_fuzzy_sensitivity.py`, which backs
**Table 8** and limitation (7) in the manuscript.

## Contents

| Threshold | Predictions | History | Result |
| --- | --- | --- | --- |
| τ = 0.9 | `predictions_tau0p90.csv` (5,652 rows) | `history_tau0p90.csv` | `result_tau0p90.json` |
| τ = 0.8 | `predictions_tau0p80.csv` (5,754 rows) | `history_tau0p80.csv` | `result_tau0p80.json` |
| τ = 0.7 | `predictions_tau0p70.csv` (5,525 rows) | `history_tau0p70.csv` | `result_tau0p70.json` |

Prediction columns: `text`, `source`, `fuzzy_cluster_id`, `true_label`, `pred_label`, `prob_sqli`, `correct`.

Test-partition sizes differ across thresholds by design: each τ induces a different connected-component
grouping and therefore a different row-level `StratifiedGroupKFold` partition. The three thresholds are
**not** paired evaluations on identical instances, and the τ=1.0 baseline uses the original stratified
normalized-group split rather than `StratifiedGroupKFold`. See limitation (7).

## Recompute Table 8

```python
import json, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

for tag in ["0p90", "0p80", "0p70"]:
    d = pd.read_csv(f"results/extension_raw/fuzzy/predictions_tau{tag}.csv")
    j = json.load(open(f"results/extension_raw/fuzzy/result_tau{tag}.json"))
    tn, fp, fn, tp = confusion_matrix(d.true_label, d.pred_label).ravel()
    assert (tn, fp, fn, tp) == (j["tn"], j["fp"], j["fn"], j["tp"])
    print(j["threshold"], len(d), fp + fn, round(f1_score(d.true_label, d.pred_label), 6))
# 0.9 5652 5 0.998865
# 0.8 5754 9 0.998063
# 0.7 5525 5 0.998837
```

## Verify family isolation directly

`fuzzy_cluster_id` makes the leakage-control claim checkable rather than merely asserted: no cluster in a
test partition carries both labels.

```python
d = pd.read_csv("results/extension_raw/fuzzy/predictions_tau0p70.csv")
assert (d.groupby("fuzzy_cluster_id").true_label.nunique() > 1).sum() == 0
```

Confirmed for all three thresholds (0 mixed-label clusters at τ = 0.9, 0.8, and 0.7).

## Wall-clock provenance for Table 7

`elapsed_seconds` in the three result files are 4030.73, 4032.60, and 4058.07, whose mean is 4040.5 s
(67.3 min) — the CodeBERT figure reported in Table 7. These are end-to-end retraining, validation, and
test-evaluation timings on an NVIDIA T4, not inference latency; see limitation (9).

## Not redistributed

Model checkpoints (`checkpoint.pt`, ~513 MB per threshold) and the per-threshold `train.csv`,
`val.csv`, and `test.csv` splits are omitted. The splits are reproducible from the corpus with
`scripts/run_fuzzy_sensitivity.py` at seed 42.
