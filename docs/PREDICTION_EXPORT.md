> **Release status (v1.4.0):** all eight aligned fixed-test prediction CSVs, plus two cross-source and two multi-seed prediction CSVs, are already exported and included in `results/predictions/`. Per-sample predictions for six multi-seed runs were not retained upstream; see `results/predictions/README.md`. The procedure below is retained for rebuilding the evidence bundle from Kaggle.

# Exporting fixed-test prediction evidence from Kaggle

No GPU and no retraining are needed. The exporter only copies the saved CSV files.

If the corrected output directory is already present:

```python
from pathlib import Path
import shutil

base = Path('/kaggle/working/sqlshield_corrected_validation')
names = [
    'corrected_group_split_pilot__CodeBERT__seed42.csv',
    'corrected_group_split_pilot__BERT-base__seed42.csv',
    'classical__Random_Forest.csv',
    'classical__XGBoost.csv',
    'classical__SVM.csv',
    'classical__Logistic_Regression.csv',
    'classical__Char_LinearSVC.csv',
    'classical__Char_Logistic_Regression.csv',
]
out = Path('/kaggle/working/sqlshield_prediction_evidence')
out.mkdir(parents=True, exist_ok=True)
for name in names:
    src = base / 'predictions' / name
    if not src.exists():
        print('MISSING:', src)
    else:
        shutil.copy2(src, out / name)
archive = shutil.make_archive('/kaggle/working/sqlshield_prediction_evidence', 'zip', out)
print(archive)
```

Then place all eight CSVs under `results/predictions/` and run:

```bash
python scripts/verify_reported_results.py
python scripts/analyze_fixed_test_errors.py
```

The verifier checks alignment of all 5,654 test texts/labels, recomputes confusion metrics, recomputes all seven exact McNemar p-values, and recomputes Holm and Bonferroni corrections.
