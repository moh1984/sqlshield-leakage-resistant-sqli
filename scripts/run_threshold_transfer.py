#!/usr/bin/env python3
"""Select a threshold on Validation B only and transfer it unchanged to external A."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sqlshield.pipeline import (
    OUT, PRED_DIR, CKPT_DIR, DEVICE, load_sources, remove_global_normalized_conflicts,
    source_train_val, make_loader, evaluate_transformer, compute_metrics,
)

MODEL_ID="microsoft/codebert-base"
SEED=42


def main():
    a,b=load_sources()
    a_cross,b_cross,_=remove_global_normalized_conflicts(a,b)
    _,b_val=source_train_val(b_cross)

    ckpt=CKPT_DIR / "cross_source_B_sqliv2_TO_A_sajid576__CodeBERT__seed42.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Run scripts/run_cross_source.py first; missing checkpoint: {ckpt}")

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,num_labels=2,id2label={0:"benign",1:"sqli"},label2id={"benign":0,"sqli":1}
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt,map_location=DEVICE))
    loader=make_loader(b_val,tok,False,SEED)
    vm,_,y,p=evaluate_transformer(model,loader)

    precision,recall,thr=precision_recall_curve(y,p)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    t_f1=float(thr[int(np.nanargmax(f1))])
    fpr,tpr,troc=roc_curve(y,p)
    finite=np.isfinite(troc); ids=np.where(finite)[0]
    t_bal=float(troc[ids[np.argmax((tpr-fpr)[finite])]])

    ext_path=PRED_DIR / "cross_source_B_sqliv2_TO_A_sajid576__CodeBERT__seed42.csv"
    if not ext_path.exists():
        raise FileNotFoundError(ext_path)
    ext=pd.read_csv(ext_path); ey=ext.true_label.to_numpy(); ep=ext.prob_sqli.to_numpy()
    cases={"default_0.5":.5,"validation_B_best_F1":t_f1,"validation_B_best_balanced_acc":t_bal}
    rows=[]
    for name,t in cases.items():
        m=compute_metrics(ey,(ep>=t).astype(int),ep)
        rows.append(dict(threshold_source=name,threshold=t,accuracy=m.accuracy,precision=m.precision,recall=m.recall,f1=m.f1,roc_auc=m.roc_auc,pr_auc=m.pr_auc,balanced_accuracy=m.balanced_accuracy,mcc=m.mcc,tn=m.tn,fp=m.fp,fn=m.fn,tp=m.tp,errors=m.errors))
    out=pd.DataFrame(rows)
    out.to_csv(OUT / "cross_source_B_to_A_threshold_transfer.csv",index=False)
    print("=== VALIDATION-B ORIGINAL PERFORMANCE ===")
    print(vm)
    print("\n=== B -> A THRESHOLD TRANSFER RESULTS ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
