#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQLShield - Corrected Validation Pipeline
=========================================

This script replaces the flawed "concat first, then detect text column" data pipeline.

It performs:

1) Correct source-specific loading:
   A: sajid576/Modified_SQL_Dataset.csv      -> Query, Label
   B: syed.../sqliv2.csv                     -> Sentence, Label

2) Strict cleaning:
   - labels must be numeric 0 or 1
   - text must be non-null and length > 2
   - exact (text, label) duplicates removed within each source
   - one known conflicting normalized group (#NAME?) is removed automatically
     if any normalized group has both labels

3) Correct merged corpus construction:
   - concatenate normalized source tables
   - remove exact duplicate text across sources
   - create normalized_text = lowercase + whitespace collapse

4) Leakage-resistant GROUP-AWARE 80/10/10 split:
   - all rows sharing the same normalized_text stay in ONE partition only
   - unique normalized groups are stratified by binary label
   - split seed fixed at 42
   - assertions verify zero normalized-group overlap across partitions

5) Classical ML baselines on corrected fixed split:
   - Random Forest (word TF-IDF)
   - XGBoost (word TF-IDF)
   - Logistic Regression (word TF-IDF)
   - LinearSVC / SVM (word TF-IDF)
   - Character LinearSVC (character 3-5-gram TF-IDF)
   - Character Logistic Regression (character 3-5-gram TF-IDF)

6) Transformer multi-seed stability:
   - CodeBERT x 5 seeds
   - BERT-base x 5 seeds
   - fixed corrected split
   - same hyperparameters as the current paper
   - per-run metrics + mean ± SD

7) Cross-source novel-pattern generalization:
   - A -> residual B (all normalized overlaps with A removed)
   - B -> residual A (all normalized overlaps with B removed)
   - reports class imbalance explicitly
   - metrics include F1, PR-AUC, balanced accuracy, MCC
   - default: CodeBERT, seed 42
   - can be expanded to 5 seeds / both transformers via config

Outputs:
  /kaggle/working/sqlshield_corrected_validation/
    dataset_audit.csv
    conflict_groups.csv
    split_audit.csv
    classical_results.csv
    transformer_multiseed_runs.csv
    transformer_multiseed_summary.csv
    cross_source_runs.csv
    cross_source_summary.csv
    cross_source_overlap.csv
    cross_source_conflicts_removed.csv
    predictions/*.csv
    histories/*.csv
    config.json
    results.json
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed as hf_set_seed,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

A_PATH = Path(os.environ.get(
    "SQLSHIELD_A_PATH",
    "/kaggle/input/datasets/sajid576/sql-injection-dataset/Modified_SQL_Dataset.csv",
))
B_PATH = Path(os.environ.get(
    "SQLSHIELD_B_PATH",
    "/kaggle/input/datasets/syedsaqlainhussain/sql-injection-dataset/sqliv2.csv",
))

A_NAME = "sajid576/sql-injection-dataset"
B_NAME = "syedsaqlainhussain/sql-injection-dataset/sqliv2.csv"

SPLIT_SEED = 42
SEEDS = [7, 21, 42, 84, 126]

MODELS = {
    "CodeBERT": "microsoft/codebert-base",
    "BERT-base": "bert-base-uncased",
}

# Cross-source default is intentionally lighter.
# To run 5 seeds externally too, set CROSS_SEEDS = SEEDS.
# To run BERT-base externally too, set CROSS_MODELS = MODELS.
CROSS_MODELS = {
    "CodeBERT": "microsoft/codebert-base",
}
CROSS_SEEDS = [42]

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRAD_CLIP = 1.0

TFIDF_MAX_FEATURES = 10000
CHAR_TFIDF_MAX_FEATURES = 50000

NUM_WORKERS = 0
PIN_MEMORY = True

OUT = Path(os.environ.get(
    "SQLSHIELD_OUT",
    "/kaggle/working/sqlshield_corrected_validation",
))
PRED_DIR = OUT / "predictions"
HIST_DIR = OUT / "histories"
CKPT_DIR = OUT / "checkpoints"

for d in [OUT, PRED_DIR, HIST_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# GENERAL UTILITIES
# ============================================================

def banner(s: str) -> None:
    print("\n" + "=" * 84)
    print(s)
    print("=" * 84)


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def read_b_csv(path: Path) -> pd.DataFrame:
    # sqliv2.csv is UTF-16 in the currently mounted Kaggle source.
    return pd.read_csv(path, encoding="utf-16", on_bad_lines="skip")


def strict_clean(
    raw: pd.DataFrame,
    text_col: str,
    label_col: str,
    source_name: str,
) -> pd.DataFrame:
    df = raw[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    df["text"] = df["text"].astype("string").str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin([0, 1])]
    df = df[df["text"].str.len() > 2].copy()

    df["label"] = df["label"].astype(int)
    df["source"] = source_name

    # Exact source-internal duplicate removal.
    df = df.drop_duplicates(subset=["text", "label"], keep="first").reset_index(drop=True)
    df["normalized_text"] = df["text"].map(normalize_text)

    return df


def load_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not A_PATH.exists():
        raise FileNotFoundError(f"Dataset A not found: {A_PATH}")
    if not B_PATH.exists():
        raise FileNotFoundError(f"Dataset B not found: {B_PATH}")

    raw_a = pd.read_csv(A_PATH, on_bad_lines="skip")
    raw_b = read_b_csv(B_PATH)

    if "Query" not in raw_a.columns or "Label" not in raw_a.columns:
        raise ValueError(f"Unexpected A columns: {list(raw_a.columns)}")
    if "Sentence" not in raw_b.columns or "Label" not in raw_b.columns:
        raise ValueError(f"Unexpected B columns: {list(raw_b.columns)}")

    a = strict_clean(raw_a, "Query", "Label", A_NAME)
    b = strict_clean(raw_b, "Sentence", "Label", B_NAME)
    return a, b


def source_audit(df: pd.DataFrame, name: str) -> Dict:
    return {
        "dataset": name,
        "rows": int(len(df)),
        "benign": int((df["label"] == 0).sum()),
        "sqli": int((df["label"] == 1).sum()),
        "unique_exact_text": int(df["text"].nunique()),
        "unique_normalized_groups": int(df["normalized_text"].nunique()),
    }


def remove_global_normalized_conflicts(
    a: pd.DataFrame,
    b: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove every normalized-text group that carries contradictory labels.

    The check is performed on the union of both sources before merged or
    cross-source evaluation.  This preserves contradictory (text, label) rows
    long enough for them to be detected instead of silently discarding one.
    """
    combined = pd.concat([a, b], ignore_index=True)
    label_counts = combined.groupby("normalized_text")["label"].nunique()
    conflict_norms = set(label_counts[label_counts > 1].index)

    conflicts = combined[
        combined["normalized_text"].isin(conflict_norms)
    ].copy().reset_index(drop=True)

    a_clean = a[
        ~a["normalized_text"].isin(conflict_norms)
    ].copy().reset_index(drop=True)
    b_clean = b[
        ~b["normalized_text"].isin(conflict_norms)
    ].copy().reset_index(drop=True)

    if len(a_clean):
        assert a_clean.groupby("normalized_text")["label"].nunique().max() == 1
    if len(b_clean):
        assert b_clean.groupby("normalized_text")["label"].nunique().max() == 1

    return a_clean, b_clean, conflicts


def build_corrected_merged(
    a: pd.DataFrame,
    b: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge correctly normalized sources.

    Any normalized_text group with contradictory labels is removed completely.
    Then exact text duplicates across sources are removed.
    """
    a_clean, b_clean, conflicts = remove_global_normalized_conflicts(a, b)
    clean = pd.concat([a_clean, b_clean], ignore_index=True)

    # Remove exact cross-source duplicates after contradiction removal.
    clean = clean.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    # Safety: every normalized group must now have exactly one label.
    assert clean.groupby("normalized_text")["label"].nunique().max() == 1

    return clean, conflicts


# ============================================================
# GROUP-AWARE 80/10/10 SPLIT
# ============================================================

def make_group_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per normalized group. Since conflicts were removed, each group has one label.
    """
    g = (
        df.groupby("normalized_text", as_index=False)
          .agg(
              label=("label", "first"),
              n_rows=("text", "size"),
          )
    )
    return g


def group_aware_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratify UNIQUE NORMALIZED GROUPS by label, then map all group members
    into a single partition.

    This guarantees no normalized_text leakage across train/val/test.
    """
    groups = make_group_table(df)

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=0.20,
        stratify=groups["label"],
        random_state=SPLIT_SEED,
    )

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        stratify=temp_groups["label"],
        random_state=SPLIT_SEED,
    )

    train_set = set(train_groups["normalized_text"])
    val_set = set(val_groups["normalized_text"])
    test_set = set(test_groups["normalized_text"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    train_df = df[df["normalized_text"].isin(train_set)].copy().reset_index(drop=True)
    val_df = df[df["normalized_text"].isin(val_set)].copy().reset_index(drop=True)
    test_df = df[df["normalized_text"].isin(test_set)].copy().reset_index(drop=True)

    # Final zero-leakage assertions.
    assert set(train_df["normalized_text"]).isdisjoint(set(val_df["normalized_text"]))
    assert set(train_df["normalized_text"]).isdisjoint(set(test_df["normalized_text"]))
    assert set(val_df["normalized_text"]).isdisjoint(set(test_df["normalized_text"]))

    return train_df, val_df, test_df


def split_audit_rows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[Dict]:
    rows = []
    total = len(train_df) + len(val_df) + len(test_df)

    for split_name, d in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        rows.append({
            "split": split_name,
            "rows": int(len(d)),
            "row_pct": float(100 * len(d) / total),
            "benign": int((d["label"] == 0).sum()),
            "sqli": int((d["label"] == 1).sum()),
            "sqli_pct": float(100 * (d["label"] == 1).mean()),
            "normalized_groups": int(d["normalized_text"].nunique()),
        })

    return rows


# ============================================================
# METRICS
# ============================================================

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int
    errors: int
    n: int


def compute_metrics(y_true, y_pred, y_score) -> Metrics:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = float("nan")

    try:
        pr = float(average_precision_score(y_true, y_score))
    except Exception:
        pr = float("nan")

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        errors=int((y_true != y_pred).sum()),
        n=int(len(y_true)),
    )


# ============================================================
# CLASSICAL BASELINES
# ============================================================

def classical_models() -> Dict[str, Pipeline]:
    return {
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
            )),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
            )),
            ("clf", XGBClassifier(
                n_estimators=200,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
            )),
            ("clf", LinearSVC(
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "Char LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=CHAR_TFIDF_MAX_FEATURES,
                lowercase=True,
            )),
            ("clf", LinearSVC(
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "Char Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=CHAR_TFIDF_MAX_FEATURES,
                lowercase=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42,
            )),
        ]),
    }


def run_classical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    banner("CLASSICAL BASELINES ON CORRECTED GROUP-AWARE SPLIT")

    rows = []

    for name, model in classical_models().items():
        print(f"\nTraining {name} ...")
        start = time.time()

        model.fit(train_df["text"], train_df["label"])
        preds = model.predict(test_df["text"])

        clf = model["clf"]
        if hasattr(clf, "predict_proba"):
            scores = model.predict_proba(test_df["text"])[:, 1]
        else:
            scores = model.decision_function(test_df["text"])

        metrics = compute_metrics(test_df["label"], preds, scores)
        row = {
            "model": name,
            **asdict(metrics),
            "elapsed_seconds": time.time() - start,
        }
        rows.append(row)

        pred_df = pd.DataFrame({
            "text": test_df["text"].values,
            "source": test_df["source"].values,
            "true_label": test_df["label"].values,
            "pred_label": preds,
            "score_sqli": scores,
            "correct": (preds == test_df["label"].values).astype(int),
        })
        pred_df.to_csv(
            PRED_DIR / f"classical__{name.replace(' ', '_')}.csv",
            index=False,
        )

        print(
            f"{name}: acc={metrics.accuracy:.6f}, "
            f"f1={metrics.f1:.6f}, roc_auc={metrics.roc_auc:.6f}, "
            f"errors={metrics.errors}, FP={metrics.fp}, FN={metrics.fn}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "classical_results.csv", index=False)
    return out


# ============================================================
# TRANSFORMER TRAINING
# ============================================================

class SQLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loader(
    df: pd.DataFrame,
    tokenizer,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)

    return DataLoader(
        SQLDataset(df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        generator=gen if shuffle else None,
    )


def evaluate_transformer(model, loader):
    model.eval()

    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            out = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            preds = out.logits.argmax(dim=-1)

            preds_all.extend(preds.cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    metrics = compute_metrics(labels_all, preds_all, probs_all)

    return (
        metrics,
        np.asarray(preds_all),
        np.asarray(labels_all),
        np.asarray(probs_all),
    )


def train_epoch(model, loader, optimizer, scheduler) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
        )

        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        bs = labels.size(0)
        total_loss += float(out.loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def train_transformer_once(
    experiment: str,
    model_name: str,
    model_id: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    banner(
        f"{experiment} | {model_name} | seed={seed} | "
        f"train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}"
    )

    set_all_seeds(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_loader = make_loader(train_df, tokenizer, True, seed)
    val_loader = make_loader(val_df, tokenizer, False, seed)
    test_loader = make_loader(test_df, tokenizer, False, seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "benign", 1: "sqli"},
        label2id={"benign": 0, "sqli": 1},
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    safe_exp = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    ckpt = CKPT_DIR / f"{safe_exp}__{safe_model}__seed{seed}.pt"

    best_val_f1 = -1.0
    best_epoch = None
    history = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        vm, _, _, _ = evaluate_transformer(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "val_accuracy": vm.accuracy,
            "val_precision": vm.precision,
            "val_recall": vm.recall,
            "val_f1": vm.f1,
            "val_roc_auc": vm.roc_auc,
            "val_pr_auc": vm.pr_auc,
        })

        print(
            f"Epoch {epoch}/{EPOCHS}: loss={loss:.6f}, "
            f"val_f1={vm.f1:.6f}, val_acc={vm.accuracy:.6f}"
        )

        if vm.f1 > best_val_f1:
            best_val_f1 = vm.f1
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    tm, preds, labels, probs = evaluate_transformer(model, test_loader)

    elapsed = time.time() - start

    pred_df = pd.DataFrame({
        "text": test_df["text"].values,
        "source": test_df["source"].values,
        "true_label": labels,
        "pred_label": preds,
        "prob_sqli": probs,
        "correct": (preds == labels).astype(int),
    })
    pred_df.to_csv(
        PRED_DIR / f"{safe_exp}__{safe_model}__seed{seed}.csv",
        index=False,
    )

    pd.DataFrame(history).to_csv(
        HIST_DIR / f"{safe_exp}__{safe_model}__seed{seed}.csv",
        index=False,
    )

    result = {
        "experiment": experiment,
        "model": model_name,
        "model_id": model_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        **asdict(tm),
        "elapsed_seconds": elapsed,
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "test_n": int(len(test_df)),
        "test_sqli": int((test_df["label"] == 1).sum()),
        "test_benign": int((test_df["label"] == 0).sum()),
    }

    print(
        f"TEST: acc={tm.accuracy:.6f}, f1={tm.f1:.6f}, "
        f"roc_auc={tm.roc_auc:.6f}, pr_auc={tm.pr_auc:.6f}, "
        f"bal_acc={tm.balanced_accuracy:.6f}, mcc={tm.mcc:.6f}, "
        f"errors={tm.errors}, FP={tm.fp}, FN={tm.fn}"
    )

    del model, tokenizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


SUMMARY_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "mcc",
    "errors",
    "fp",
    "fn",
]


def summarize_runs(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))

        for m in SUMMARY_METRICS:
            v = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(v.mean())
            row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            row[f"{m}_min"] = float(v.min())
            row[f"{m}_max"] = float(v.max())

        rows.append(row)

    return pd.DataFrame(rows)


def run_transformer_multiseed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for model_name, model_id in MODELS.items():
        for seed in SEEDS:
            rows.append(
                train_transformer_once(
                    experiment="corrected_group_split_multiseed",
                    model_name=model_name,
                    model_id=model_id,
                    seed=seed,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                )
            )

    runs = pd.DataFrame(rows)
    summary = summarize_runs(runs, ["experiment", "model"])

    runs.to_csv(OUT / "transformer_multiseed_runs.csv", index=False)
    summary.to_csv(OUT / "transformer_multiseed_summary.csv", index=False)

    return runs, summary


# ============================================================
# CROSS-SOURCE NOVEL-PATTERN GENERALIZATION
# ============================================================

def residual_external(
    train_source: pd.DataFrame,
    external_source: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    train_norms = set(train_source["normalized_text"])
    mask = external_source["normalized_text"].isin(train_norms)

    ext = external_source.loc[~mask].copy().reset_index(drop=True)

    report = {
        "external_before": int(len(external_source)),
        "removed_normalized_overlap": int(mask.sum()),
        "external_after": int(len(ext)),
        "external_remaining_pct": float(100 * len(ext) / len(external_source)),
        "benign_after": int((ext["label"] == 0).sum()),
        "sqli_after": int((ext["label"] == 1).sum()),
        "sqli_pct_after": float(100 * (ext["label"] == 1).mean()),
    }

    if ext["label"].nunique() < 2:
        raise ValueError("Residual external set has only one class.")

    return ext, report


def source_train_val(source_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-aware train/val split inside ONE source.
    """
    groups = make_group_table(source_df)

    tr_g, va_g = train_test_split(
        groups,
        test_size=0.10,
        stratify=groups["label"],
        random_state=SPLIT_SEED,
    )

    tr_set = set(tr_g["normalized_text"])
    va_set = set(va_g["normalized_text"])

    train_df = source_df[
        source_df["normalized_text"].isin(tr_set)
    ].copy().reset_index(drop=True)

    val_df = source_df[
        source_df["normalized_text"].isin(va_set)
    ].copy().reset_index(drop=True)

    assert set(train_df["normalized_text"]).isdisjoint(
        set(val_df["normalized_text"])
    )

    return train_df, val_df


def run_cross_direction(
    train_source: pd.DataFrame,
    external_source: pd.DataFrame,
    train_name: str,
    external_name: str,
) -> Tuple[List[Dict], Dict]:
    ext, overlap = residual_external(train_source, external_source)
    train_df, val_df = source_train_val(train_source)

    experiment = f"cross_source_{train_name}_TO_{external_name}"

    overlap.update({
        "experiment": experiment,
        "train_source": train_name,
        "external_source": external_name,
        "train_source_rows": int(len(train_source)),
        "train_partition_rows": int(len(train_df)),
        "val_partition_rows": int(len(val_df)),
    })

    banner(experiment)
    print(json.dumps(overlap, indent=2))

    rows = []

    for model_name, model_id in CROSS_MODELS.items():
        for seed in CROSS_SEEDS:
            r = train_transformer_once(
                experiment=experiment,
                model_name=model_name,
                model_id=model_id,
                seed=seed,
                train_df=train_df,
                val_df=val_df,
                test_df=ext,
            )
            r.update({
                "train_source": train_name,
                "external_source": external_name,
                **overlap,
            })
            rows.append(r)

    return rows, overlap


def run_cross_source(
    a: pd.DataFrame,
    b: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Cross-source evaluation must use the same global contradiction policy as
    # the merged corpus.  Otherwise a contradictory normalized group can leak
    # into a source-specific train/validation split.
    a_cross, b_cross, cross_conflicts = remove_global_normalized_conflicts(a, b)
    cross_conflicts.to_csv(OUT / "cross_source_conflicts_removed.csv", index=False)

    rows = []
    overlap_rows = []

    ab, ab_rep = run_cross_direction(
        a_cross, b_cross,
        "A_sajid576",
        "B_sqliv2",
    )
    rows.extend(ab)
    overlap_rows.append(ab_rep)

    ba, ba_rep = run_cross_direction(
        b_cross, a_cross,
        "B_sqliv2",
        "A_sajid576",
    )
    rows.extend(ba)
    overlap_rows.append(ba_rep)

    runs = pd.DataFrame(rows)
    overlap_df = pd.DataFrame(overlap_rows)
    summary = summarize_runs(
        runs,
        ["experiment", "model", "train_source", "external_source"],
    )

    runs.to_csv(OUT / "cross_source_runs.csv", index=False)
    overlap_df.to_csv(OUT / "cross_source_overlap.csv", index=False)
    summary.to_csv(OUT / "cross_source_summary.csv", index=False)

    return runs, summary, overlap_df


# ============================================================
# MAIN
# ============================================================

def main():
    banner("SQLShield Corrected Validation Pipeline")

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # --------------------------------------------------------
    # Load sources correctly
    # --------------------------------------------------------
    a, b = load_sources()

    audit = pd.DataFrame([
        source_audit(a, "A_sajid576"),
        source_audit(b, "B_sqliv2"),
    ])

    # Source overlap audit
    exact_overlap = len(set(a["text"]) & set(b["text"]))
    norm_overlap = len(
        set(a["normalized_text"]) & set(b["normalized_text"])
    )

    banner("SOURCE AUDIT")
    print(audit.to_string(index=False))
    print(f"\nExact text overlap: {exact_overlap:,}")
    print(f"Normalized overlap groups: {norm_overlap:,}")

    # --------------------------------------------------------
    # Corrected merged corpus
    # --------------------------------------------------------
    merged, conflicts = build_corrected_merged(a, b)

    conflicts.to_csv(OUT / "conflict_groups.csv", index=False)

    print(f"\nConflicting normalized groups removed: "
          f"{conflicts['normalized_text'].nunique() if len(conflicts) else 0}")
    print(f"Rows removed due to conflicting normalized labels: {len(conflicts)}")

    merged_audit = {
        "dataset": "corrected_merged",
        "rows": int(len(merged)),
        "benign": int((merged["label"] == 0).sum()),
        "sqli": int((merged["label"] == 1).sum()),
        "unique_exact_text": int(merged["text"].nunique()),
        "unique_normalized_groups": int(merged["normalized_text"].nunique()),
        "exact_source_overlap_before_merge": int(exact_overlap),
        "normalized_source_overlap_before_merge": int(norm_overlap),
        "conflicting_normalized_groups_removed": int(
            conflicts["normalized_text"].nunique() if len(conflicts) else 0
        ),
        "conflicting_rows_removed": int(len(conflicts)),
    }

    audit = pd.concat(
        [audit, pd.DataFrame([merged_audit])],
        ignore_index=True,
    )
    audit.to_csv(OUT / "dataset_audit.csv", index=False)

    banner("CORRECTED MERGED CORPUS")
    print(json.dumps(merged_audit, indent=2))

    # --------------------------------------------------------
    # Group-aware fixed split
    # --------------------------------------------------------
    train_df, val_df, test_df = group_aware_split(merged)

    split_rows = split_audit_rows(train_df, val_df, test_df)
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUT / "split_audit.csv", index=False)

    banner("GROUP-AWARE SPLIT AUDIT")
    print(split_df.to_string(index=False))

    # Explicit zero-overlap report
    overlap_checks = {
        "train_val_norm_overlap": len(
            set(train_df["normalized_text"]) & set(val_df["normalized_text"])
        ),
        "train_test_norm_overlap": len(
            set(train_df["normalized_text"]) & set(test_df["normalized_text"])
        ),
        "val_test_norm_overlap": len(
            set(val_df["normalized_text"]) & set(test_df["normalized_text"])
        ),
    }
    print("\nNormalized-group overlap across partitions:")
    print(json.dumps(overlap_checks, indent=2))

    assert all(v == 0 for v in overlap_checks.values())

    # Save split IDs/text for exact reproducibility.
    train_df.to_csv(OUT / "train_corrected.csv", index=False)
    val_df.to_csv(OUT / "val_corrected.csv", index=False)
    test_df.to_csv(OUT / "test_corrected.csv", index=False)

    # --------------------------------------------------------
    # Config snapshot
    # --------------------------------------------------------
    config = {
        "A_PATH": str(A_PATH),
        "B_PATH": str(B_PATH),
        "A_NAME": A_NAME,
        "B_NAME": B_NAME,
        "split_seed": SPLIT_SEED,
        "training_seeds": SEEDS,
        "models": MODELS,
        "cross_models": CROSS_MODELS,
        "cross_seeds": CROSS_SEEDS,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "grad_clip": GRAD_CLIP,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "normalization": "strip + lowercase + collapse internal whitespace",
        "conflict_policy": "remove normalized groups with contradictory labels",
        "split_policy": (
            "stratified split of unique normalized groups; "
            "all rows from each normalized group remain in one partition"
        ),
        "overlap_checks": overlap_checks,
        "device": str(DEVICE),
    }

    (OUT / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Classical baselines
    # --------------------------------------------------------
    classical = run_classical(train_df, test_df)

    # --------------------------------------------------------
    # Transformer 5-seed stability
    # --------------------------------------------------------
    multiseed_runs, multiseed_summary = run_transformer_multiseed(
        train_df, val_df, test_df
    )

    # --------------------------------------------------------
    # Cross-source residual generalization
    # --------------------------------------------------------
    cross_runs, cross_summary, cross_overlap = run_cross_source(a, b)

    # --------------------------------------------------------
    # Consolidated JSON
    # --------------------------------------------------------
    results = {
        "config": config,
        "dataset_audit": audit.to_dict(orient="records"),
        "split_audit": split_rows,
        "classical_results": classical.to_dict(orient="records"),
        "transformer_multiseed_runs": multiseed_runs.to_dict(orient="records"),
        "transformer_multiseed_summary": multiseed_summary.to_dict(orient="records"),
        "cross_source_runs": cross_runs.to_dict(orient="records"),
        "cross_source_summary": cross_summary.to_dict(orient="records"),
        "cross_source_overlap": cross_overlap.to_dict(orient="records"),
    }

    (OUT / "results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    banner("ALL EXPERIMENTS COMPLETE")
    print("\nTransformer multi-seed summary:")
    print(multiseed_summary.to_string(index=False))

    print("\nCross-source summary:")
    print(cross_summary.to_string(index=False))

    print("\nImportant output files:")
    for name in [
        "dataset_audit.csv",
        "split_audit.csv",
        "classical_results.csv",
        "transformer_multiseed_runs.csv",
        "transformer_multiseed_summary.csv",
        "cross_source_runs.csv",
        "cross_source_summary.csv",
        "cross_source_overlap.csv",
        "results.json",
    ]:
        print(" ", OUT / name)

    print(
        "\nNOTE: The residual cross-source external sets are highly imbalanced "
        "(few SQLi samples after normalized-overlap removal). Interpret PR-AUC, "
        "SQLi recall/F1, balanced accuracy, MCC, FP, and FN alongside accuracy."
    )


if __name__ == "__main__":
    main()
