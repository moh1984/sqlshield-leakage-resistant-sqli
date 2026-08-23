#!/usr/bin/env python3
"""Export the eight fixed-test prediction CSVs saved by the SQLShield Kaggle runs.

This script does not retrain any model. It only copies already-saved prediction evidence.
"""
from pathlib import Path
import argparse, shutil, sys

EXPECTED = {
    "CodeBERT": "corrected_group_split_pilot__CodeBERT__seed42.csv",
    "BERT-base": "corrected_group_split_pilot__BERT-base__seed42.csv",
    "Random Forest": "classical__Random_Forest.csv",
    "XGBoost": "classical__XGBoost.csv",
    "Word LinearSVC": "classical__SVM.csv",
    "Word Logistic Regression": "classical__Logistic_Regression.csv",
    "Char LinearSVC": "classical__Char_LinearSVC.csv",
    "Char Logistic Regression": "classical__Char_Logistic_Regression.csv",
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-dir', required=True, help='Path containing the predictions/ directory')
    ap.add_argument('--out-dir', default=None, help='Destination; default: <repo>/results/predictions')
    ap.add_argument('--zip', action='store_true', help='Also create a ZIP next to the destination directory')
    args=ap.parse_args()
    base=Path(args.base_dir)
    src=base/'predictions'
    root=Path(__file__).resolve().parents[1]
    out=Path(args.out_dir) if args.out_dir else root/'results'/'predictions'
    out.mkdir(parents=True, exist_ok=True)
    missing=[]
    for model,name in EXPECTED.items():
        p=src/name
        if not p.exists():
            missing.append((model,name))
            continue
        shutil.copy2(p,out/name)
        print(f'copied: {model}: {name}')
    if missing:
        print('\nMissing saved predictions:', file=sys.stderr)
        for model,name in missing:
            print(f'  - {model}: {name}', file=sys.stderr)
        print('\nNo model was retrained. Restore the complete Kaggle output and rerun this exporter.', file=sys.stderr)
        raise SystemExit(2)
    if args.zip:
        archive=shutil.make_archive(str(out), 'zip', out)
        print(f'ZIP: {archive}')
    print(f'Complete prediction evidence: {out}')

if __name__=='__main__':
    main()
