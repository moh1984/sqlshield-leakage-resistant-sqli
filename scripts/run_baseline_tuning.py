#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLShield - Validation-only tuning of the classical baselines (Reviewer 2, item 5)
==================================================================================

The published baselines use one pre-specified configuration with library
defaults, so the comparison against a fine-tuned transformer is not like for
like. This script searches the classical configuration space on the validation
partition only, then scores the winner of each family on the fixed test set
exactly once.

The test partition is never consulted during selection.

Cost note
---------
The grid contains 216 configurations but only 12 distinct vectorizers, because
n-gram range and vocabulary size are the only settings that affect the TF-IDF
matrix. Each vectorizer is therefore fitted once and its transformed matrices
are reused across every classifier that shares it. That turns roughly an hour of
repeated vectorization into about fifteen minutes, with identical results.

Usage
-----
  python scripts/run_baseline_tuning.py            # full grid, CPU, ~15 min
  python scripts/run_baseline_tuning.py --fast     # 12-point sanity run

Outputs (under $SQLSHIELD_OUT)
  baseline_tuning_validation_grid.csv    every candidate, validation scores
  baseline_tuning_selected_test.csv      selected configs, test scores
  predictions/tuned__<family>.csv        per-sample test predictions
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sqlshield.pipeline import (
    OUT,
    PRED_DIR,
    banner,
    build_corrected_merged,
    compute_metrics,
    group_aware_split,
    load_sources,
)


def spaces(fast: bool):
    """Return (vectorizer specs, classifier specs) keyed so they can be crossed."""
    if fast:
        char_ngrams, char_feats = [(3, 5), (2, 5)], [50000]
        word_ngrams, word_feats = [(1, 2)], [10000]
        svc_c, lr_c, weights = [1.0, 4.0], [4.0], [None]
    else:
        char_ngrams, char_feats = [(2, 4), (3, 5), (2, 5), (3, 6)], [20000, 50000, 100000]
        word_ngrams, word_feats = [(1, 1), (1, 2), (1, 3)], [10000, 30000]
        svc_c, lr_c, weights = [0.5, 1.0, 2.0, 4.0], [1.0, 4.0, 16.0], [None, "balanced"]

    vecs = OrderedDict()
    for ng, mf in itertools.product(char_ngrams, char_feats):
        vecs[("char", ng, mf)] = dict(analyzer="char", ngram_range=ng,
                                      max_features=mf, lowercase=True)
    for ng, mf in itertools.product(word_ngrams, word_feats):
        vecs[("word", ng, mf)] = dict(ngram_range=ng, max_features=mf)

    clfs = []
    for c, w in itertools.product(svc_c, weights):
        clfs.append(("LinearSVC", dict(C=c, class_weight=w)))
    for c, w in itertools.product(lr_c, weights):
        clfs.append(("LogReg", dict(C=c, class_weight=w)))
    return vecs, clfs


def make_clf(kind, params):
    if kind == "LinearSVC":
        return LinearSVC(max_iter=5000, random_state=42, **params)
    return LogisticRegression(max_iter=2000, random_state=42, **params)


def family_of(analyzer, kind):
    return f"{'Char' if analyzer == 'char' else 'Word'} {kind}"


def scores_of(clf, X):
    return (clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba")
            else clf.decision_function(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    a, b = load_sources()
    corrected, _ = build_corrected_merged(a, b)
    train_df, val_df, test_df = group_aware_split(corrected)
    banner(f"train={len(train_df):,}  validation={len(val_df):,}  test={len(test_df):,}")

    vecs, clfs = spaces(args.fast)
    print(f"{len(vecs)} vectorizers x {len(clfs)} classifiers = "
          f"{len(vecs) * len(clfs)} configurations; selection on validation only\n")

    rows = []
    for vi, (vkey, vkw) in enumerate(vecs.items(), 1):
        analyzer, ng, mf = vkey
        t0 = time.time()
        vec = TfidfVectorizer(**vkw)
        Xtr = vec.fit_transform(train_df["text"])
        Xva = vec.transform(val_df["text"])
        vec_s = time.time() - t0
        print(f"[vectorizer {vi}/{len(vecs)}] {analyzer} {ng} mf={mf} "
              f"vocab={len(vec.vocabulary_):,} fitted in {vec_s:.1f}s")

        for kind, params in clfs:
            t1 = time.time()
            clf = make_clf(kind, params)
            clf.fit(Xtr, train_df["label"])
            m = compute_metrics(val_df["label"], clf.predict(Xva), scores_of(clf, Xva))
            rows.append({
                "family": family_of(analyzer, kind), "analyzer": analyzer,
                "ngram_range": str(ng), "max_features": mf,
                "C": params["C"], "class_weight": str(params["class_weight"]),
                "vocabulary_size": len(vec.vocabulary_),
                "val_f1": m.f1, "val_accuracy": m.accuracy, "val_errors": m.errors,
                "clf_fit_seconds": time.time() - t1, "vectorizer_seconds": vec_s,
            })
            print(f"    {kind:<10} C={params['C']:<5} w={str(params['class_weight']):<9} "
                  f"val_f1={m.f1:.6f} err={m.errors}")

    grid = pd.DataFrame(rows).sort_values(["family", "val_f1"], ascending=[True, False])
    grid.to_csv(OUT / "baseline_tuning_validation_grid.csv", index=False)

    banner("SELECTED ON VALIDATION, SCORED ONCE ON TEST")
    selected = []
    for family, g in grid.groupby("family"):
        best = g.iloc[0]
        vkw = dict(ngram_range=eval(best["ngram_range"]),
                   max_features=int(best["max_features"]))
        if best["analyzer"] == "char":
            vkw.update(analyzer="char", lowercase=True)
        w = None if best["class_weight"] == "None" else best["class_weight"]
        kind = "LinearSVC" if "LinearSVC" in family else "LogReg"

        vec = TfidfVectorizer(**vkw)
        Xtr = vec.fit_transform(train_df["text"])
        Xte = vec.transform(test_df["text"])
        clf = make_clf(kind, dict(C=float(best["C"]), class_weight=w))
        clf.fit(Xtr, train_df["label"])
        preds, sc = clf.predict(Xte), scores_of(clf, Xte)
        m = compute_metrics(test_df["label"], preds, sc)

        pd.DataFrame({
            "text": test_df["text"].values, "source": test_df["source"].values,
            "true_label": test_df["label"].values, "pred_label": preds,
            "score_sqli": sc,
            "correct": (preds == test_df["label"].values).astype(int),
        }).to_csv(PRED_DIR / f"tuned__{family.replace(' ', '_')}.csv", index=False)

        selected.append({"family": family, "ngram_range": best["ngram_range"],
                         "max_features": int(best["max_features"]), "C": best["C"],
                         "class_weight": best["class_weight"],
                         "val_f1": best["val_f1"], **asdict(m)})
        print(f"{family:<18} {best['ngram_range']:<7} mf={int(best['max_features']):<7} "
              f"C={best['C']:<5} w={best['class_weight']:<9} "
              f"test_f1={m.f1:.6f} errors={m.errors}")

    pd.DataFrame(selected).to_csv(OUT / "baseline_tuning_selected_test.csv", index=False)

    banner("READING")
    print("Published fixed-test reference: CodeBERT 3 errors (F1 0.9993),")
    print("Char LinearSVC 9 errors (F1 0.9980), exact McNemar p=0.109375.")
    print("If a tuned baseline closes that gap, the paper's claim is unaffected:")
    print("the contribution is the evaluation protocol, not detector superiority.")
    print("Per-sample predictions are written to predictions/tuned__*.csv, so")
    print("scripts/run_exact_mcnemar.py can be re-run on the tuned configurations")
    print("before any significance statement is restated.")


if __name__ == "__main__":
    main()
