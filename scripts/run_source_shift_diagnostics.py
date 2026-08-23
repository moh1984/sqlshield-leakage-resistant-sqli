#!/usr/bin/env python3
"""Diagnose source fingerprints and source-conditioned fixed-test behavior."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

from sqlshield.pipeline import (
    OUT, PRED_DIR, load_sources, remove_global_normalized_conflicts,
    residual_external, compute_metrics,
)


def structural_features(df, source_name):
    x = df[["text"]].copy()
    s = x["text"].astype(str)
    x["source"] = source_name
    x["char_length"] = s.str.len()
    x["word_count"] = s.str.split().str.len()
    x["digit_count"] = s.str.count(r"\d")
    x["quote_count"] = s.str.count("'") + s.str.count('"')
    x["paren_count"] = s.str.count(r"\(") + s.str.count(r"\)")
    x["semicolon"] = s.str.contains(";", regex=False).astype(int)
    x["starts_select"] = s.str.match(r"(?i)^\s*select\b").astype(int)
    x["starts_insert"] = s.str.match(r"(?i)^\s*insert\b").astype(int)
    x["starts_update"] = s.str.match(r"(?i)^\s*update\b").astype(int)
    x["starts_delete"] = s.str.match(r"(?i)^\s*delete\b").astype(int)
    patterns = {
        "has_where": r"\bwhere\b", "has_from": r"\bfrom\b", "has_union": r"\bunion\b",
        "has_join": r"\bjoin\b", "has_or": r"\bor\b", "has_and": r"\band\b",
        "has_group_by": r"\bgroup\s+by\b", "has_order_by": r"\border\s+by\b",
        "has_comment_dash": r"--", "has_comment_hash": r"#", "has_comment_block": r"/\*",
    }
    for col, pat in patterns.items():
        x[col] = s.str.contains(pat, case=False, regex=True).astype(int)
    return x


def find_transformer(model):
    for p in [
        PRED_DIR / f"corrected_group_split_multiseed__{model}__seed42.csv",
        PRED_DIR / f"corrected_group_split_pilot__{model}__seed42.csv",
    ]:
        if p.exists(): return p
    return None


def main():
    a, b = load_sources()
    a_cross, b_cross, _ = remove_global_normalized_conflicts(a, b)
    a_unique, _ = residual_external(b_cross, a_cross)
    b_unique, _ = residual_external(a_cross, b_cross)
    a_benign = a_unique[a_unique.label == 0].copy().reset_index(drop=True)
    b_benign = b_unique[b_unique.label == 0].copy().reset_index(drop=True)

    fa = structural_features(a_benign, "A")
    fb = structural_features(b_benign, "B")
    features = pd.concat([fa, fb], ignore_index=True)

    continuous = ["char_length", "word_count", "digit_count", "quote_count", "paren_count"]
    binary = ["semicolon", "starts_select", "starts_insert", "starts_update", "starts_delete", "has_where", "has_from", "has_union", "has_join", "has_or", "has_and", "has_group_by", "has_order_by", "has_comment_dash", "has_comment_hash", "has_comment_block"]

    rows=[]
    for col in continuous:
        for src in ["A","B"]:
            v=features.loc[features.source==src,col]
            rows.append(dict(feature=col,source=src,type="continuous",mean=v.mean(),median=v.median(),std=v.std(),rate_pct=np.nan))
    for col in binary:
        for src in ["A","B"]:
            v=features.loc[features.source==src,col]
            rows.append(dict(feature=col,source=src,type="binary",mean=v.mean(),median=np.nan,std=np.nan,rate_pct=100*v.mean()))
    pd.DataFrame(rows).to_csv(OUT / "cross_source_structural_diagnostic.csv", index=False)

    source_df = pd.concat([
        a_benign[["text","normalized_text"]].assign(source_label=0),
        b_benign[["text","normalized_text"]].assign(source_label=1),
    ], ignore_index=True).drop_duplicates(subset=["normalized_text"]).reset_index(drop=True)
    tr, te = train_test_split(source_df, test_size=.20, stratify=source_df.source_label, random_state=42)
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=30000,min_df=2)
    Xtr=vec.fit_transform(tr.text); Xte=vec.transform(te.text)
    clf=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42).fit(Xtr,tr.source_label)
    pred=clf.predict(Xte); prob=clf.predict_proba(Xte)[:,1]
    tn,fp,fn,tp=confusion_matrix(te.source_label,pred,labels=[0,1]).ravel()
    source_result=pd.DataFrame([dict(
        task="benign_A_vs_B_source_identification", train_n=len(tr), test_n=len(te),
        accuracy=accuracy_score(te.source_label,pred), balanced_accuracy=balanced_accuracy_score(te.source_label,pred),
        f1=f1_score(te.source_label,pred), roc_auc=roc_auc_score(te.source_label,prob),
        tn=tn,fp=fp,fn=fn,tp=tp,
    )])
    source_result.to_csv(OUT / "cross_source_source_classifier.csv", index=False)

    names=np.array(vec.get_feature_names_out()); coef=clf.coef_[0]
    top_b=np.argsort(coef)[-30:][::-1]; top_a=np.argsort(coef)[:30]
    pd.concat([
        pd.DataFrame({"ngram":names[top_a],"coefficient":coef[top_a],"associated_source":"A"}),
        pd.DataFrame({"ngram":names[top_b],"coefficient":coef[top_b],"associated_source":"B"}),
    ],ignore_index=True).to_csv(OUT / "cross_source_source_specific_ngrams.csv",index=False)

    # Source-conditioned fixed-test metrics if prediction files are available.
    model_paths={
        "CodeBERT":find_transformer("CodeBERT"),
        "BERT-base":find_transformer("BERT-base"),
        "Random Forest":PRED_DIR / "classical__Random_Forest.csv",
    }
    cond=[]
    for model,path in model_paths.items():
        if path is None or not path.exists():
            continue
        d=pd.read_csv(path)
        score_col="prob_sqli" if "prob_sqli" in d else "score_sqli"
        for source,g in d.groupby("source"):
            m=compute_metrics(g.true_label,g.pred_label,g[score_col])
            cond.append(dict(model=model,source=source,n=len(g),benign=int((g.true_label==0).sum()),sqli=int((g.true_label==1).sum()),accuracy=m.accuracy,precision=m.precision,recall=m.recall,f1=m.f1,roc_auc=m.roc_auc,pr_auc=m.pr_auc,balanced_accuracy=m.balanced_accuracy,mcc=m.mcc,tn=m.tn,fp=m.fp,fn=m.fn,tp=m.tp,errors=m.errors))
    if cond:
        pd.DataFrame(cond).to_csv(OUT / "source_conditioned_test_performance_FINAL.csv",index=False)

    print("=== BENIGN SOURCE-IDENTIFICATION TEST ===")
    print(source_result.to_string(index=False))
    print("\nSaved diagnostics under:", OUT)


if __name__ == "__main__":
    main()
