#!/usr/bin/env python3
"""Analyze CodeBERT vs character-LinearSVC per-sample error complementarity."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results'/'predictions'
OUT=ROOT/'results'/'error_analysis'
CB=P/'corrected_group_split_pilot__CodeBERT__seed42.csv'
CH=P/'classical__Char_LinearSVC.csv'

def load(path):
    if not path.exists():
        raise SystemExit(f'Missing required prediction evidence: {path}\nRun scripts/export_saved_predictions.py first.')
    d=pd.read_csv(path)
    need={'text','true_label','pred_label'}
    if not need.issubset(d.columns):
        raise SystemExit(f'{path.name} missing columns: {sorted(need-set(d.columns))}')
    return d

def main():
    cb, ch = load(CB), load(CH)
    if len(cb)!=5654 or len(ch)!=5654:
        raise SystemExit('Expected 5,654 rows in both prediction files.')
    if not (cb.text.astype(str).to_numpy()==ch.text.astype(str).to_numpy()).all():
        raise SystemExit('Prediction files are not aligned on identical test text/order.')
    if not (cb.true_label.to_numpy()==ch.true_label.to_numpy()).all():
        raise SystemExit('Prediction files disagree on true labels.')
    cb_bad=cb.pred_label.to_numpy()!=cb.true_label.to_numpy()
    ch_bad=ch.pred_label.to_numpy()!=ch.true_label.to_numpy()
    shared=cb_bad & ch_bad
    cb_only=cb_bad & ~ch_bad
    ch_only=~cb_bad & ch_bad
    union=cb_bad | ch_bad
    summary=pd.DataFrame([{
        'codebert_errors':int(cb_bad.sum()), 'char_linearsvc_errors':int(ch_bad.sum()),
        'codebert_only_errors':int(cb_only.sum()), 'char_linearsvc_only_errors':int(ch_only.sum()),
        'shared_errors':int(shared.sum()), 'error_union':int(union.sum()),
        'error_set_jaccard':float(shared.sum()/union.sum())
    }])
    OUT.mkdir(parents=True,exist_ok=True)
    summary.to_csv(OUT/'error_overlap_summary_from_predictions.csv',index=False)
    base=cb[['text','source','true_label']].copy() if 'source' in cb.columns else cb[['text','true_label']].copy()
    base['codebert_pred']=cb.pred_label
    base['char_linearsvc_pred']=ch.pred_label
    base['category']='correct_both'
    base.loc[shared,'category']='shared_error'
    base.loc[cb_only,'category']='codebert_only_error'
    base.loc[ch_only,'category']='char_linearsvc_only_error'
    base.loc[union].to_csv(OUT/'codebert_vs_char_linearsvc_error_rows.csv',index=False)
    print(summary.to_string(index=False))
    print(f'\nSaved: {OUT}')

if __name__=='__main__': main()
