#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLShield - Character n-gram baseline extension (canonical repository script)
===============================================

Purpose
-------
Add two strong character-level TF-IDF baselines to the existing corrected
SQLShield fixed split WITHOUT retraining CodeBERT or BERT-base:

  1) Char-TFIDF (3-5) + LinearSVC
  2) Char-TFIDF (3-5) + Logistic Regression

The script also rebuilds any missing original word-TFIDF baseline prediction
files using the exact settings documented in sqlshield_corrected_validation.py,
then recomputes the primary CodeBERT-vs-comparator exact McNemar family with
7 comparisons and both Holm and Bonferroni correction.

Expected existing files under BASE_DIR
--------------------------------------
  train_corrected.csv
  test_corrected.csv
  predictions/corrected_group_split_pilot__CodeBERT__seed42.csv
  predictions/corrected_group_split_pilot__BERT-base__seed42.csv

Default BASE_DIR:
  /kaggle/working/sqlshield_corrected_validation

You can override it with:
  python scripts/run_char_ngram_baselines.py --base-dir /path/to/sqlshield_corrected_validation
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


DEFAULT_BASE = Path("/kaggle/working/sqlshield_corrected_validation")
SEED = 42
WORD_MAX_FEATURES = 10_000
CHAR_MAX_FEATURES = 50_000
ALPHA = 0.05


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def metric_dict(y_true, y_pred, y_score) -> Dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "errors": int((y_true != y_pred).sum()),
        "n": int(len(y_true)),
    }


def get_scores(model: Pipeline, x: pd.Series) -> np.ndarray:
    clf = model["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def fit_and_save(
    name: str,
    model: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, float | int]]:
    print(f"\nTraining {name} ...")
    t0 = time.time()
    model.fit(train_df["text"].astype(str), train_df["label"].astype(int))
    pred = model.predict(test_df["text"].astype(str)).astype(int)
    score = np.asarray(get_scores(model, test_df["text"].astype(str)), dtype=float)
    elapsed = time.time() - t0

    metrics = metric_dict(test_df["label"].astype(int), pred, score)
    metrics["elapsed_seconds"] = float(elapsed)
    metrics["model"] = name

    out = pd.DataFrame({
        "text": test_df["text"].astype(str).values,
        "source": test_df["source"].values if "source" in test_df.columns else "unknown",
        "true_label": test_df["label"].astype(int).values,
        "pred_label": pred,
        "score_sqli": score,
        "correct": (pred == test_df["label"].astype(int).values).astype(int),
    })
    pred_path = pred_dir / f"classical__{safe_name(name)}.csv"
    out.to_csv(pred_path, index=False)

    print(
        f"{name}: F1={metrics['f1']:.6f} | Acc={metrics['accuracy']:.6f} | "
        f"ROC-AUC={metrics['roc_auc']:.6f} | PR-AUC={metrics['pr_auc']:.6f} | "
        f"BA={metrics['balanced_accuracy']:.6f} | MCC={metrics['mcc']:.6f} | "
        f"errors={metrics['errors']} (FP={metrics['fp']}, FN={metrics['fn']}) | "
        f"{elapsed:.1f}s"
    )
    print(f"Saved predictions: {pred_path}")
    return out, metrics


def char_models() -> Dict[str, Pipeline]:
    common = dict(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=CHAR_MAX_FEATURES,
        lowercase=True,
    )
    return {
        "Char LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(**common)),
            ("clf", LinearSVC(max_iter=2000, random_state=SEED)),
        ]),
        "Char Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**common)),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
    }


def word_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=WORD_MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=WORD_MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=WORD_MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", LinearSVC(max_iter=2000, random_state=SEED)),
        ]),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=WORD_MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", XGBClassifier(
                n_estimators=200,
                random_state=SEED,
                use_label_encoder=False,
                eval_metric="logloss",
            )),
        ])
    return models


def load_prediction(path: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"true_label", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(df) != len(test_df):
        raise ValueError(f"{path}: n={len(df)} but test set n={len(test_df)}")

    y_test = test_df["label"].astype(int).to_numpy()
    if not np.array_equal(df["true_label"].astype(int).to_numpy(), y_test):
        raise ValueError(f"{path}: true_label order does not match test_corrected.csv")

    if "text" in df.columns:
        a = df["text"].astype(str).to_numpy()
        b = test_df["text"].astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"{path}: text order does not match test_corrected.csv")
    return df


def find_transformer_prediction(pred_dir: Path, model: str) -> Path:
    preferred = pred_dir / f"corrected_group_split_pilot__{model}__seed42.csv"
    if preferred.exists():
        return preferred

    # Prefer the fixed-test pilot prediction over cross-source files.
    candidates = [
        p for p in pred_dir.glob(f"*{model}*seed42.csv")
        if "cross_source" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find fixed-test {model} seed-42 prediction file in {pred_dir}"
        )
    candidates.sort(key=lambda p: ("pilot" not in p.name, len(p.name), p.name))
    return candidates[0]


def exact_mcnemar(codebert_df: pd.DataFrame, other_df: pd.DataFrame) -> Tuple[int, int, int, float]:
    cb_true = codebert_df["true_label"].astype(int).to_numpy()
    cb_pred = codebert_df["pred_label"].astype(int).to_numpy()
    ot_true = other_df["true_label"].astype(int).to_numpy()
    ot_pred = other_df["pred_label"].astype(int).to_numpy()

    if not np.array_equal(cb_true, ot_true):
        raise ValueError("McNemar inputs do not share the same true-label order")

    cb_ok = cb_pred == cb_true
    ot_ok = ot_pred == ot_true
    b = int(np.sum((~cb_ok) & ot_ok))   # CodeBERT wrong, comparator correct
    c = int(np.sum(cb_ok & (~ot_ok)))   # CodeBERT correct, comparator wrong
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(b, n=n, p=0.5, alternative="two-sided").pvalue)
    return b, c, n, p


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def ensure_word_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_dir: Path,
) -> Dict[str, Path]:
    expected = {
        "Random Forest": pred_dir / "classical__Random_Forest.csv",
        "XGBoost": pred_dir / "classical__XGBoost.csv",
        "SVM": pred_dir / "classical__SVM.csv",
        "Logistic Regression": pred_dir / "classical__Logistic_Regression.csv",
    }

    models = word_models()
    for name, path in expected.items():
        valid = False
        if path.exists():
            try:
                load_prediction(path, test_df)
                valid = True
                print(f"Using existing prediction: {path.name}")
            except Exception as e:
                print(f"Existing {path.name} is unusable ({e}); rebuilding it.")
        if not valid:
            if name not in models:
                raise RuntimeError(
                    f"Need to rebuild {name}, but its dependency is unavailable. "
                    "For XGBoost, install xgboost first."
                )
            fit_and_save(name, models[name], train_df, test_df, pred_dir)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    base = args.base_dir
    pred_dir = base / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    train_path = base / "train_corrected.csv"
    test_path = base / "test_corrected.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. "
            "Point --base-dir to the corrected SQLShield output directory."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for col in ("text", "label"):
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing required column '{col}' in corrected split files")

    print("=" * 88)
    print("SQLShield CHARACTER N-GRAM BASELINE EXTENSION")
    print("=" * 88)
    print(f"Base dir: {base}")
    print(f"Train n: {len(train_df):,}")
    print(f"Test  n: {len(test_df):,}")
    print("Character TF-IDF: analyzer='char', ngram_range=(3,5), max_features=50000")

    # 1) Train the two new character baselines.
    char_rows = []
    char_paths: Dict[str, Path] = {}
    for name, model in char_models().items():
        _, metrics = fit_and_save(name, model, train_df, test_df, pred_dir)
        char_rows.append(metrics)
        char_paths[name] = pred_dir / f"classical__{safe_name(name)}.csv"

    char_results = pd.DataFrame(char_rows)
    char_results_path = base / "char_ngram_results.csv"
    char_results.to_csv(char_results_path, index=False)

    # 2) Ensure the four original word-level baselines are present/aligned.
    word_paths = ensure_word_predictions(train_df, test_df, pred_dir)

    # 3) Load fixed-test transformer predictions; no transformer retraining.
    cb_path = find_transformer_prediction(pred_dir, "CodeBERT")
    bert_path = find_transformer_prediction(pred_dir, "BERT-base")
    codebert = load_prediction(cb_path, test_df)
    bert = load_prediction(bert_path, test_df)
    print(f"\nUsing CodeBERT fixed-test predictions: {cb_path.name}")
    print(f"Using BERT-base fixed-test predictions: {bert_path.name}")

    # 4) Primary family: 7 CodeBERT comparisons.
    comparison_paths = {
        "BERT-base": bert_path,
        "Random Forest": word_paths["Random Forest"],
        "XGBoost": word_paths["XGBoost"],
        "Word LinearSVC": word_paths["SVM"],
        "Word Logistic Regression": word_paths["Logistic Regression"],
        "Char LinearSVC": char_paths["Char LinearSVC"],
        "Char Logistic Regression": char_paths["Char Logistic Regression"],
    }

    mcnemar_rows = []
    raw_p = []
    for label, path in comparison_paths.items():
        other = load_prediction(path, test_df)
        b, c, n, p = exact_mcnemar(codebert, other)
        raw_p.append(p)
        mcnemar_rows.append({
            "comparison": f"CodeBERT vs {label}",
            "cb_wrong_other_correct": b,
            "cb_correct_other_wrong": c,
            "discordant": n,
            "exact_p": p,
        })

    holm = holm_adjust(raw_p)
    m = len(raw_p)
    bonf_alpha = ALPHA / m
    for row, hp in zip(mcnemar_rows, holm):
        row["holm_p"] = hp
        row["bonferroni_p"] = min(1.0, row["exact_p"] * m)
        row["significant_holm_0.05"] = bool(hp < ALPHA)
        row["significant_bonferroni"] = bool(row["exact_p"] < bonf_alpha)

    mc = pd.DataFrame(mcnemar_rows)
    mc_path = base / "mcnemar_7comparisons_char_augmented.csv"
    mc.to_csv(mc_path, index=False)

    summary = {
        "base_dir": str(base),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "char_vectorizer": {
            "analyzer": "char",
            "ngram_range": [3, 5],
            "max_features": CHAR_MAX_FEATURES,
            "lowercase": True,
        },
        "primary_mcnemar_family_size": m,
        "alpha": ALPHA,
        "bonferroni_raw_alpha": bonf_alpha,
        "codebert_prediction_file": str(cb_path),
        "bert_prediction_file": str(bert_path),
        "char_results": char_rows,
        "mcnemar": mcnemar_rows,
    }
    summary_path = base / "char_ngram_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("CHARACTER BASELINE RESULTS")
    print("=" * 88)
    cols = [
        "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "balanced_accuracy", "mcc", "fp", "fn", "errors",
    ]
    print(char_results[cols].to_string(index=False))

    print("\n" + "=" * 88)
    print("EXACT McNEMAR - 7 PRIMARY CODEBERT COMPARISONS")
    print(f"Bonferroni raw alpha = 0.05/7 = {bonf_alpha:.9f}")
    print("=" * 88)
    print(mc.to_string(index=False))

    print("\nSaved:")
    print(f"  {char_results_path}")
    print(f"  {mc_path}")
    print(f"  {summary_path}")
    print(f"  {char_paths['Char LinearSVC']}")
    print(f"  {char_paths['Char Logistic Regression']}")
    print("\nNo CodeBERT or BERT-base training was performed.")


if __name__ == "__main__":
    main()
