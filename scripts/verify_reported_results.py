#!/usr/bin/env python3
"""Verify SQLShield reported results.

Default mode is strict and recomputes fixed-test metrics and exact McNemar tests from
per-sample prediction CSVs. It intentionally fails if prediction evidence is incomplete.
Use --aggregate-only only for an internal consistency audit of packaged summary files;
that mode is NOT independent prediction-level verification.
"""
from pathlib import Path
import argparse, math, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/'paper_reported'
P=ROOT/'results'/'predictions'

EXPECTED={
    'CodeBERT':'corrected_group_split_pilot__CodeBERT__seed42.csv',
    'BERT-base':'corrected_group_split_pilot__BERT-base__seed42.csv',
    'Random Forest':'classical__Random_Forest.csv',
    'XGBoost':'classical__XGBoost.csv',
    'Word LinearSVC':'classical__SVM.csv',
    'Word Logistic Regression':'classical__Logistic_Regression.csv',
    'Char LinearSVC':'classical__Char_LinearSVC.csv',
    'Char Logistic Regression':'classical__Char_Logistic_Regression.csv',
}
FIXED_NAME={'Word LinearSVC':'SVM','Word Logistic Regression':'Logistic Regression'}

def close(a,b,tol=2e-6):
    return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)

def score_col(d):
    for c in ['prob_sqli','score_sqli','probability','score','decision_score']:
        if c in d.columns:
            return c
    return None

def holm_adjust(ps):
    ps=np.asarray(ps,float); m=len(ps); order=np.argsort(ps); out=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*ps[idx]
        running=max(running,val)
        out[idx]=min(1.0,running)
    return out

def load_predictions():
    missing=[(m,n) for m,n in EXPECTED.items() if not (P/n).exists()]
    if missing:
        print('STRICT PREDICTION VERIFICATION: INCOMPLETE EVIDENCE BUNDLE',file=sys.stderr)
        print('Missing per-sample prediction files:',file=sys.stderr)
        for m,n in missing: print(f'  - {m}: results/predictions/{n}',file=sys.stderr)
        print('\nRun scripts/export_saved_predictions.py against the restored Kaggle output.',file=sys.stderr)
        raise SystemExit(2)
    out={m:pd.read_csv(P/n) for m,n in EXPECTED.items()}
    for m,d in out.items():
        need={'text','true_label','pred_label'}
        if not need.issubset(d.columns):
            raise SystemExit(f'{m}: missing prediction columns {sorted(need-set(d.columns))}')
        if len(d)!=5654: raise SystemExit(f'{m}: expected 5654 rows, found {len(d)}')
    ref=out['CodeBERT']
    for m,d in out.items():
        if not (ref['text'].astype(str).to_numpy()==d['text'].astype(str).to_numpy()).all():
            raise SystemExit(f'{m}: test text/order differs from CodeBERT')
        if not (ref.true_label.to_numpy()==d.true_label.to_numpy()).all():
            raise SystemExit(f'{m}: true labels differ from CodeBERT')
    return out

def recompute_metrics(d):
    y=d.true_label.astype(int).to_numpy(); pred=d.pred_label.astype(int).to_numpy()
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    r={'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
       'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),
       'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'errors':int(fp+fn),'n':len(y)}
    sc=score_col(d)
    if sc:
        s=d[sc].astype(float).to_numpy()
        r['roc_auc']=roc_auc_score(y,s); r['pr_auc']=average_precision_score(y,s)
    return r

def aggregate_consistency():
    fixed=pd.read_csv(R/'fixed_test_seed42_FINAL.csv')
    for _,r in fixed.iterrows():
        n=int(r.n); tn,fp,fn,tp=map(int,[r.tn,r.fp,r.fn,r.tp])
        assert n==tn+fp+fn+tp
        assert int(r.errors)==fp+fn
        assert close(r.accuracy,(tn+tp)/n)
    mc=pd.read_csv(R/'exact_mcnemar_FINAL.csv')
    for _,r in mc.iterrows():
        b=int(r.model1_wrong_model2_correct); c=int(r.model1_correct_model2_wrong)
        p=binomtest(min(b,c),n=b+c,p=0.5,alternative='two-sided').pvalue if b+c else 1.0
        assert close(p,r.exact_p_value,tol=1e-12)
    fuzzy=pd.read_csv(R/'fuzzy_cluster_audit_FINAL.csv')
    assert (fuzzy.mixed_label_fuzzy_clusters==0).all()
    print('AGGREGATE CONSISTENCY AUDIT: PASSED')
    print('This is NOT independent prediction-level verification. Use default mode after exporting all per-sample predictions.')

def strict():
    pred=load_predictions()
    reported=pd.read_csv(R/'fixed_test_seed42_FINAL.csv').set_index('model')
    recomputed={}
    for model,d in pred.items():
        m=recompute_metrics(d); recomputed[model]=m
        key=FIXED_NAME.get(model,model)
        rr=reported.loc[key]
        for c in ['accuracy','precision','recall','f1']:
            if not close(m[c],rr[c]): raise AssertionError(f'{model} {c}: {m[c]} != {rr[c]}')
        for c in ['tn','fp','fn','tp','errors','n']:
            if int(m[c])!=int(rr[c]): raise AssertionError(f'{model} {c}: {m[c]} != {rr[c]}')
        if 'roc_auc' in m and not close(m['roc_auc'],rr['roc_auc'],tol=5e-6):
            raise AssertionError(f'{model} roc_auc mismatch')
        if 'pr_auc' in m and not close(m['pr_auc'],rr['pr_auc'],tol=5e-6):
            raise AssertionError(f'{model} pr_auc mismatch')

    cb=pred['CodeBERT']; y=cb.true_label.to_numpy(); cb_bad=cb.pred_label.to_numpy()!=y
    comps=[]
    for model,d in pred.items():
        if model=='CodeBERT': continue
        bad=d.pred_label.to_numpy()!=y
        b=int((cb_bad & ~bad).sum()); c=int((~cb_bad & bad).sum()); n=b+c
        p=binomtest(min(b,c),n=n,p=0.5,alternative='two-sided').pvalue if n else 1.0
        comps.append((f'CodeBERT vs {model}',b,c,n,p))
    raw=[x[4] for x in comps]; holm=holm_adjust(raw); bonf=np.minimum(1.0,np.asarray(raw)*len(raw))
    rep=pd.read_csv(R/'exact_mcnemar_FINAL.csv').set_index('comparison')
    for i,(name,b,c,n,p) in enumerate(comps):
        rr=rep.loc[name]
        assert b==int(rr.model1_wrong_model2_correct) and c==int(rr.model1_correct_model2_wrong)
        assert close(p,rr.exact_p_value,tol=1e-12)
        assert close(holm[i],rr.holm_adjusted_p,tol=1e-12)
        assert close(bonf[i],rr.bonferroni_adjusted_p,tol=1e-12)
    print('STRICT PREDICTION VERIFICATION: PASSED')
    print('Recomputed fixed-test confusion metrics and all seven exact McNemar comparisons from 5,654 aligned per-sample predictions.')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--aggregate-only',action='store_true',help='Internal consistency audit only; not independent verification')
    args=ap.parse_args()
    aggregate_consistency() if args.aggregate_only else strict()
