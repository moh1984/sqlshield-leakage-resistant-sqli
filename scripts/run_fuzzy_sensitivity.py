#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQLShield - Fuzzy / Near-Duplicate Sensitivity Experiment (canonical repository script)
=========================================================

Purpose
-------
Evaluate how CodeBERT seed-42 performance changes when the current exact
normalized-text grouping is replaced by stronger near-duplicate family grouping.

Default sensitivity thresholds:
    1.0  = existing baseline (current exact normalized_text split; NO retraining)
    0.9  = fuzzy family split + CodeBERT retraining
    0.8  = fuzzy family split + CodeBERT retraining
    0.7  = fuzzy family split + CodeBERT retraining

Near-duplicate definition
-------------------------
1) Start from the EXISTING corrected corpus reconstructed from:
       train_corrected.csv
       val_corrected.csv
       test_corrected.csv
2) Collapse exact normalized_text groups (lowercase + whitespace collapse).
3) Represent each unique normalized_text with 5-character shingles.
4) Use MinHash-LSH only as a CANDIDATE GENERATOR.
5) Verify every candidate pair with EXACT Jaccard similarity.
6) Add an edge when exact Jaccard >= target threshold.
7) Connected components of this verified graph are treated as fuzzy families.
8) All rows from one fuzzy family are placed in ONE partition only.
9) Split assignment uses row-level StratifiedGroupKFold to preserve 80/10/10 size and class balance.

Important methodological note
-----------------------------
MinHash-LSH is approximate candidate generation. The script lowers the LSH
candidate threshold below the target threshold (default margin=0.05), then
performs exact Jaccard verification. This reduces but cannot mathematically
guarantee elimination of every possible missed candidate. Report this honestly
as a sensitivity analysis rather than an absolute proof of semantic uniqueness.

Mixed-label fuzzy families are NOT deleted. They are kept intact in one split,
which avoids near-duplicate leakage while preserving difficult benign/SQLi
contrasts. Cluster-majority label is used only to stratify cluster assignment.

CodeBERT training protocol matches the existing SQLShield corrected pipeline:
    model: microsoft/codebert-base
    seed: 42
    max_length: 128
    batch_size: 32
    epochs: 4
    AdamW lr: 2e-5
    weight_decay: 0.01
    warmup: 10%
    gradient clipping: 1.0
    best checkpoint: validation F1

Outputs
-------
<base-dir>/fuzzy_sensitivity/
    fuzzy_sensitivity_results.csv
    fuzzy_cluster_audit.csv
    fuzzy_sensitivity_summary.json
    threshold_0p90/
        cluster_assignments.csv
        train.csv
        val.csv
        test.csv
        codebert_result.json
        predictions.csv
        history.csv
        checkpoint.pt
    threshold_0p80/
        ...
    threshold_0p70/
        ...

Run on Kaggle
-------------
    python scripts/run_fuzzy_sensitivity.py \
      --base-dir /kaggle/working/sqlshield_corrected_validation

Cluster-only dry run (no CodeBERT):
    python scripts/run_fuzzy_sensitivity.py \
      --base-dir /kaggle/working/sqlshield_corrected_validation \
      --cluster-only
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from datasketch import MinHash, MinHashLSH
except ImportError as e:
    raise SystemExit(
        "\nERROR: datasketch is not installed.\n"
        "Run this Kaggle cell first:\n\n"
        "    !pip install -q datasketch\n\n"
        "Then run the script again.\n"
    ) from e

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
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def banner(msg: str) -> None:
    print("\n" + "=" * 96)
    print(msg)
    print("=" * 96)


def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def threshold_tag(t: float) -> str:
    return f"{t:.2f}".replace(".", "p")


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Match a reproducibility-oriented setup.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def char_shingles(text: str, k: int) -> frozenset[str]:
    s = str(text)
    if len(s) <= k:
        return frozenset([s])
    return frozenset(s[i:i+k] for i in range(len(s) - k + 1))


def exact_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    if inter == 0:
        return 0.0
    union = len(a) + len(b) - inter
    return inter / union


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

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

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        errors=int((y_true != y_pred).sum()),
        n=int(len(y_true)),
    )


# ---------------------------------------------------------------------
# Load current corrected corpus
# ---------------------------------------------------------------------

def load_corrected_corpus(base_dir: Path) -> pd.DataFrame:
    files = [
        ("train", base_dir / "train_corrected.csv"),
        ("val", base_dir / "val_corrected.csv"),
        ("test", base_dir / "test_corrected.csv"),
    ]

    frames = []
    for part, path in files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        df = pd.read_csv(path)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        if "source" not in df.columns:
            df["source"] = "unknown"
        df = df.copy()
        df["original_partition"] = part
        frames.append(df)

    corpus = pd.concat(frames, ignore_index=True)
    corpus["text"] = corpus["text"].astype(str)
    corpus["label"] = pd.to_numeric(corpus["label"], errors="raise").astype(int)

    bad = ~corpus["label"].isin([0, 1])
    if bad.any():
        raise ValueError("Corpus contains labels outside {0,1}.")

    corpus["normalized_text"] = corpus["text"].map(normalize_text)

    # Existing corrected split should contain no normalized_text group
    # with contradictory labels.
    conflicts = (
        corpus.groupby("normalized_text")["label"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    if len(conflicts):
        raise ValueError(
            f"Found {len(conflicts)} exact normalized groups with conflicting labels. "
            "This should not occur in the corrected corpus."
        )

    return corpus


def make_exact_group_table(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for norm, g in corpus.groupby("normalized_text", sort=False):
        label = int(g["label"].iloc[0])
        rows.append({
            "normalized_text": norm,
            "label": label,
            "row_count": int(len(g)),
            "source_count": int(g["source"].nunique()),
        })
    groups = pd.DataFrame(rows).reset_index(drop=True)
    groups["group_index"] = np.arange(len(groups), dtype=int)
    return groups


# ---------------------------------------------------------------------
# MinHash preparation and fuzzy clustering
# ---------------------------------------------------------------------

def build_shingles_and_minhashes(
    groups: pd.DataFrame,
    shingle_size: int,
    num_perm: int,
    minhash_seed: int,
) -> Tuple[List[frozenset[str]], List[MinHash]]:
    banner(
        f"Preparing {len(groups):,} exact normalized groups | "
        f"{shingle_size}-char shingles | num_perm={num_perm}"
    )

    shingles: List[frozenset[str]] = []
    minhashes: List[MinHash] = []

    start = time.time()

    for i, text in enumerate(groups["normalized_text"].tolist()):
        sset = char_shingles(text, shingle_size)
        shingles.append(sset)

        mh = MinHash(num_perm=num_perm, seed=minhash_seed)
        mh.update_batch([s.encode("utf-8", errors="ignore") for s in sset])
        minhashes.append(mh)

        if (i + 1) % 5000 == 0 or (i + 1) == len(groups):
            elapsed = time.time() - start
            print(
                f"  MinHash prepared: {i+1:,}/{len(groups):,} "
                f"({elapsed/60:.1f} min)"
            )

    return shingles, minhashes


def cluster_groups_at_threshold(
    groups: pd.DataFrame,
    shingles: Sequence[frozenset[str]],
    minhashes: Sequence[MinHash],
    target_threshold: float,
    num_perm: int,
    lsh_margin: float,
) -> Tuple[pd.DataFrame, Dict]:
    tag = threshold_tag(target_threshold)
    candidate_threshold = max(0.10, target_threshold - lsh_margin)

    banner(
        f"FUZZY CLUSTERING threshold={target_threshold:.2f} | "
        f"LSH candidate threshold={candidate_threshold:.2f}"
    )

    n = len(groups)
    uf = UnionFind(n)

    print("Building MinHash-LSH index...")
    lsh = MinHashLSH(threshold=candidate_threshold, num_perm=num_perm)
    for i, mh in enumerate(minhashes):
        lsh.insert(str(i), mh)
        if (i + 1) % 10000 == 0 or (i + 1) == n:
            print(f"  Indexed {i+1:,}/{n:,}")

    candidate_pairs = 0
    verified_edges = 0
    start = time.time()

    print("Querying candidates and exact-verifying Jaccard...")
    for i, mh in enumerate(minhashes):
        for key in lsh.query(mh):
            j = int(key)
            if j <= i:
                continue
            candidate_pairs += 1
            jac = exact_jaccard(shingles[i], shingles[j])
            if jac + 1e-12 >= target_threshold:
                uf.union(i, j)
                verified_edges += 1

        if (i + 1) % 5000 == 0 or (i + 1) == n:
            elapsed = time.time() - start
            print(
                f"  Queried {i+1:,}/{n:,} | candidates={candidate_pairs:,} | "
                f"verified edges={verified_edges:,} | {elapsed/60:.1f} min"
            )

    roots = [uf.find(i) for i in range(n)]
    # Stable compact component IDs.
    unique_roots = {}
    component_ids = []
    for r in roots:
        if r not in unique_roots:
            unique_roots[r] = len(unique_roots)
        component_ids.append(unique_roots[r])

    assign = groups.copy()
    assign["fuzzy_cluster_id"] = component_ids

    # Cluster-level statistics.
    cluster_rows = []
    for cid, g in assign.groupby("fuzzy_cluster_id", sort=True):
        row0 = int(g.loc[g["label"] == 0, "row_count"].sum())
        row1 = int(g.loc[g["label"] == 1, "row_count"].sum())
        majority = 1 if row1 >= row0 else 0
        cluster_rows.append({
            "fuzzy_cluster_id": int(cid),
            "exact_group_count": int(len(g)),
            "row_count": int(g["row_count"].sum()),
            "label0_rows": row0,
            "label1_rows": row1,
            "mixed_label": bool(row0 > 0 and row1 > 0),
            "stratify_label": int(majority),
        })

    cluster_df = pd.DataFrame(cluster_rows)
    assign = assign.merge(
        cluster_df[
            ["fuzzy_cluster_id", "exact_group_count", "row_count",
             "label0_rows", "label1_rows", "mixed_label", "stratify_label"]
        ],
        on="fuzzy_cluster_id",
        how="left",
        suffixes=("_exact_group", "_cluster"),
    )

    mixed = cluster_df["mixed_label"].astype(bool)

    audit = {
        "threshold": float(target_threshold),
        "candidate_lsh_threshold": float(candidate_threshold),
        "num_exact_normalized_groups": int(n),
        "num_fuzzy_clusters": int(cluster_df.shape[0]),
        "verified_similarity_edges": int(verified_edges),
        "candidate_pairs_exact_checked": int(candidate_pairs),
        "singleton_fuzzy_clusters": int((cluster_df["exact_group_count"] == 1).sum()),
        "multi_group_fuzzy_clusters": int((cluster_df["exact_group_count"] > 1).sum()),
        "mixed_label_fuzzy_clusters": int(mixed.sum()),
        "rows_in_mixed_label_clusters": int(cluster_df.loc[mixed, "row_count"].sum()),
        "max_exact_groups_in_cluster": int(cluster_df["exact_group_count"].max()),
        "max_rows_in_cluster": int(cluster_df["row_count"].max()),
        "median_exact_groups_per_cluster": float(cluster_df["exact_group_count"].median()),
        "mean_exact_groups_per_cluster": float(cluster_df["exact_group_count"].mean()),
        "candidate_generation": "MinHash-LSH approximate candidates + exact Jaccard verification",
        "cluster_definition": "connected components of exact-Jaccard>=threshold edges",
    }

    return assign, audit


# ---------------------------------------------------------------------
# Cluster-aware split
# ---------------------------------------------------------------------

def split_corpus_by_fuzzy_clusters(
    corpus: pd.DataFrame,
    assignments: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    v3 splitter: 10-fold StratifiedGroupKFold at ROW level.

    Why:
    - Every fuzzy family must remain entirely in one partition.
    - We also want approximately 80/10/10 rows and nearly identical class
      prevalence across train/validation/test.
    - StratifiedGroupKFold directly optimizes this row-level stratification
      subject to non-overlapping groups.

    Procedure:
    1) Map every row to its fuzzy_cluster_id.
    2) Build 10 stratified, group-disjoint folds with shuffle=True, seed=42.
    3) Choose one fold for validation and one for test using ONLY row/class
       balance (never model outcomes). The other eight folds form training.
    4) Exhaustively score all ordered val/test fold pairs and choose the pair
       closest to target 80/10/10 total rows and benign/SQLi row counts.

    This avoids the v2 failure mode where a custom greedy allocator could
    accidentally leave a validation partition with only one class.
    """
    group_to_cluster = dict(
        zip(assignments["normalized_text"], assignments["fuzzy_cluster_id"])
    )

    df = corpus.copy()
    df["fuzzy_cluster_id"] = df["normalized_text"].map(group_to_cluster)

    if df["fuzzy_cluster_id"].isna().any():
        raise RuntimeError("Some corpus rows were not mapped to fuzzy clusters.")

    y = df["label"].astype(int).to_numpy()
    groups = df["fuzzy_cluster_id"].astype(int).to_numpy()
    X_dummy = np.zeros((len(df), 1), dtype=np.uint8)

    sgkf = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=seed,
    )

    fold_id = np.full(len(df), -1, dtype=int)
    for fold, (_, hold_idx) in enumerate(sgkf.split(X_dummy, y, groups)):
        fold_id[hold_idx] = fold

    if (fold_id < 0).any():
        raise RuntimeError("Some rows were not assigned to a CV fold.")

    df["_fold"] = fold_id

    # Group integrity: every fuzzy cluster must occur in exactly one fold.
    group_fold_counts = df.groupby("fuzzy_cluster_id")["_fold"].nunique()
    if int(group_fold_counts.max()) != 1:
        raise RuntimeError("A fuzzy cluster was split across CV folds.")

    total_n = int(len(df))
    total_benign = int((df["label"] == 0).sum())
    total_sqli = int((df["label"] == 1).sum())

    target = {
        "train": {
            "n": 0.80 * total_n,
            "benign": 0.80 * total_benign,
            "sqli": 0.80 * total_sqli,
        },
        "val": {
            "n": 0.10 * total_n,
            "benign": 0.10 * total_benign,
            "sqli": 0.10 * total_sqli,
        },
        "test": {
            "n": 0.10 * total_n,
            "benign": 0.10 * total_benign,
            "sqli": 0.10 * total_sqli,
        },
    }

    fold_stats = {}
    for f in range(10):
        g = df[df["_fold"] == f]
        fold_stats[f] = {
            "n": int(len(g)),
            "benign": int((g["label"] == 0).sum()),
            "sqli": int((g["label"] == 1).sum()),
        }

    def partition_score(actual: Dict[str, int], wanted: Dict[str, float]) -> float:
        # Equal emphasis on total rows and each class count.
        score = 0.0
        for k in ("n", "benign", "sqli"):
            denom = max(wanted[k], 1.0)
            rel = (actual[k] - wanted[k]) / denom
            score += rel * rel
        return score

    best = None

    # Ordered val/test pairs. Selection uses ONLY label/size balance.
    for val_fold in range(10):
        for test_fold in range(10):
            if test_fold == val_fold:
                continue

            val_actual = fold_stats[val_fold]
            test_actual = fold_stats[test_fold]

            train_actual = {
                k: (
                    (total_n if k == "n" else
                     total_benign if k == "benign" else
                     total_sqli)
                    - val_actual[k]
                    - test_actual[k]
                )
                for k in ("n", "benign", "sqli")
            }

            score = (
                partition_score(val_actual, target["val"])
                + partition_score(test_actual, target["test"])
                + partition_score(train_actual, target["train"])
            )

            candidate = (score, val_fold, test_fold)
            if best is None or candidate < best:
                best = candidate

    _, val_fold, test_fold = best
    train_folds = set(range(10)) - {val_fold, test_fold}

    train_df = df[df["_fold"].isin(train_folds)].copy().reset_index(drop=True)
    val_df = df[df["_fold"] == val_fold].copy().reset_index(drop=True)
    test_df = df[df["_fold"] == test_fold].copy().reset_index(drop=True)

    # Hard group-overlap assertions.
    a = set(train_df["fuzzy_cluster_id"])
    b = set(val_df["fuzzy_cluster_id"])
    c = set(test_df["fuzzy_cluster_id"])
    assert not (a & b)
    assert not (a & c)
    assert not (b & c)

    def pstats(name: str, x: pd.DataFrame) -> Dict:
        n = int(len(x))
        benign = int((x["label"] == 0).sum())
        sqli = int((x["label"] == 1).sum())

        if x["label"].nunique() < 2:
            raise RuntimeError(
                f"{name} contains only one class after StratifiedGroupKFold."
            )

        return {
            f"{name}_n": n,
            f"{name}_benign": benign,
            f"{name}_sqli": sqli,
            f"{name}_sqli_pct": float(100.0 * sqli / max(n, 1)),
            f"{name}_clusters": int(x["fuzzy_cluster_id"].nunique()),
            f"{name}_row_target": float(target[name]["n"]),
            f"{name}_benign_target": float(target[name]["benign"]),
            f"{name}_sqli_target": float(target[name]["sqli"]),
            f"{name}_row_deviation": float(n - target[name]["n"]),
            f"{name}_benign_deviation": float(benign - target[name]["benign"]),
            f"{name}_sqli_deviation": float(sqli - target[name]["sqli"]),
        }

    audit = {
        "splitter": "StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)",
        "selected_val_fold": int(val_fold),
        "selected_test_fold": int(test_fold),
        "fold_pair_selection": "minimum row/class-balance deviation only; no model outcomes",
    }
    audit.update(pstats("train", train_df))
    audit.update(pstats("val", val_df))
    audit.update(pstats("test", test_df))

    # Remove internal fold marker before saving/model training.
    for part in (train_df, val_df, test_df):
        part.drop(columns=["_fold"], inplace=True)

    return train_df, val_df, test_df, audit


# ---------------------------------------------------------------------
# CodeBERT training - mirrors corrected SQLShield pipeline
# ---------------------------------------------------------------------

class SQLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
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
    max_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        SQLDataset(df, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=gen if shuffle else None,
    )


def evaluate_transformer(model, loader, device):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

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


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        bs = labels.size(0)
        total_loss += float(out.loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def train_codebert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    args,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU not detected. On Kaggle enable a GPU accelerator "
            "(T4 is sufficient), then rerun. Use --allow-cpu only if you "
            "accept a very slow run."
        )

    banner(
        f"CodeBERT fuzzy sensitivity | threshold={args.current_threshold:.2f} | "
        f"seed={args.seed} | train={len(train_df):,} | "
        f"val={len(val_df):,} | test={len(test_df):,} | device={device}"
    )

    set_all_seeds(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    train_loader = make_loader(
        train_df, tokenizer, args.max_len, args.batch_size,
        args.num_workers, True, args.seed
    )
    val_loader = make_loader(
        val_df, tokenizer, args.max_len, args.batch_size,
        args.num_workers, False, args.seed
    )
    test_loader = make_loader(
        test_df, tokenizer, args.max_len, args.batch_size,
        args.num_workers, False, args.seed
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        id2label={0: "benign", 1: "sqli"},
        label2id={"benign": 0, "sqli": 1},
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    ckpt = out_dir / "checkpoint.pt"
    best_val_f1 = -1.0
    best_epoch = None
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_clip
        )
        vm, _, _, _ = evaluate_transformer(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "val_accuracy": vm.accuracy,
            "val_precision": vm.precision,
            "val_recall": vm.recall,
            "val_f1": vm.f1,
            "val_roc_auc": vm.roc_auc,
            "val_pr_auc": vm.pr_auc,
            "val_balanced_accuracy": vm.balanced_accuracy,
            "val_mcc": vm.mcc,
        })

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"loss={loss:.6f} | val_f1={vm.f1:.6f} | "
            f"val_acc={vm.accuracy:.6f}"
        )

        if vm.f1 > best_val_f1:
            best_val_f1 = vm.f1
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    tm, preds, labels, probs = evaluate_transformer(model, test_loader, device)
    elapsed = time.time() - start

    pred_df = pd.DataFrame({
        "text": test_df["text"].values,
        "source": test_df["source"].values,
        "fuzzy_cluster_id": test_df["fuzzy_cluster_id"].values,
        "true_label": labels,
        "pred_label": preds,
        "prob_sqli": probs,
        "correct": (preds == labels).astype(int),
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    result = {
        "threshold": float(args.current_threshold),
        "grouping": "fuzzy_5char_jaccard_connected_components",
        "model": "CodeBERT",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        **asdict(tm),
        "elapsed_seconds": float(elapsed),
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "test_n": int(len(test_df)),
        "test_sqli": int((test_df["label"] == 1).sum()),
        "test_benign": int((test_df["label"] == 0).sum()),
    }

    with open(out_dir / "codebert_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"TEST threshold={args.current_threshold:.2f}: "
        f"acc={tm.accuracy:.6f} | f1={tm.f1:.6f} | "
        f"roc_auc={tm.roc_auc:.6f} | pr_auc={tm.pr_auc:.6f} | "
        f"bal_acc={tm.balanced_accuracy:.6f} | mcc={tm.mcc:.6f} | "
        f"errors={tm.errors} | FP={tm.fp} | FN={tm.fn}"
    )

    del model, tokenizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------
# Existing threshold-1.0 baseline
# ---------------------------------------------------------------------

def existing_baseline_row(base_dir: Path) -> Dict:
    pred = (
        base_dir / "predictions" /
        "corrected_group_split_pilot__CodeBERT__seed42.csv"
    )
    if not pred.exists():
        raise FileNotFoundError(
            "Existing CodeBERT seed-42 prediction file not found:\n"
            f"  {pred}"
        )

    df = pd.read_csv(pred)
    true_col = "true_label"
    pred_col = "pred_label"
    if "prob_sqli" in df.columns:
        score_col = "prob_sqli"
    elif "score_sqli" in df.columns:
        score_col = "score_sqli"
    else:
        raise ValueError(
            f"{pred.name} needs prob_sqli or score_sqli for AUC metrics."
        )

    m = compute_metrics(df[true_col], df[pred_col], df[score_col])

    return {
        "threshold": 1.0,
        "grouping": "existing_exact_normalized_text_baseline",
        "model": "CodeBERT",
        "model_id": "microsoft/codebert-base",
        "seed": 42,
        "best_epoch": None,
        "best_val_f1": None,
        **asdict(m),
        "elapsed_seconds": None,
        "train_n": 45288,
        "val_n": 5679,
        "test_n": int(len(df)),
        "test_sqli": int((df[true_col] == 1).sum()),
        "test_benign": int((df[true_col] == 0).sum()),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/kaggle/working/sqlshield_corrected_validation"),
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default="0.9,0.8,0.7",
        help="Comma-separated fuzzy Jaccard thresholds. Baseline 1.0 is read from existing predictions.",
    )
    p.add_argument("--shingle-size", type=int, default=5)
    p.add_argument("--num-perm", type=int, default=256)
    p.add_argument(
        "--lsh-margin",
        type=float,
        default=0.05,
        help="Candidate LSH threshold = target - margin; exact Jaccard decides final edges.",
    )
    p.add_argument("--minhash-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-id", type=str, default="microsoft/codebert-base")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--cluster-only",
        action="store_true",
        help="Build fuzzy clusters/splits and audits, but do not train CodeBERT.",
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CodeBERT training without CUDA (very slow).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun threshold even if codebert_result.json already exists.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir
    out_root = base_dir / "fuzzy_sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    for t in thresholds:
        if not (0.0 < t < 1.0):
            raise ValueError(
                "Fuzzy thresholds supplied to --thresholds must be between 0 and 1. "
                "Threshold 1.0 is the existing baseline and is not retrained."
            )
    thresholds = sorted(set(thresholds), reverse=True)

    banner("SQLShield FUZZY / NEAR-DUPLICATE SENSITIVITY EXPERIMENT")
    print(f"Base dir:       {base_dir}")
    print(f"Output dir:     {out_root}")
    print(f"Thresholds:     baseline 1.0 + {thresholds}")
    print(f"Shingles:       {args.shingle_size}-char")
    print(f"MinHash perms:  {args.num_perm}")
    print(f"LSH margin:     {args.lsh_margin}")
    print(f"CodeBERT seed:  {args.seed}")
    print(f"Cluster only:   {args.cluster_only}")

    corpus = load_corrected_corpus(base_dir)
    groups = make_exact_group_table(corpus)

    print(f"\nReconstructed corrected corpus: {len(corpus):,} rows")
    print(f"Exact normalized groups:        {len(groups):,}")
    print(
        f"Class counts: benign={(corpus['label']==0).sum():,} | "
        f"SQLi={(corpus['label']==1).sum():,}"
    )

    # Existing exact-group baseline.
    results_rows = [existing_baseline_row(base_dir)]
    audit_rows = [{
        "threshold": 1.0,
        "candidate_lsh_threshold": None,
        "num_exact_normalized_groups": int(len(groups)),
        "num_fuzzy_clusters": int(len(groups)),
        "verified_similarity_edges": None,
        "candidate_pairs_exact_checked": None,
        "singleton_fuzzy_clusters": None,
        "multi_group_fuzzy_clusters": None,
        "mixed_label_fuzzy_clusters": 0,
        "rows_in_mixed_label_clusters": 0,
        "max_exact_groups_in_cluster": 1,
        "max_rows_in_cluster": int(groups["row_count"].max()),
        "median_exact_groups_per_cluster": 1.0,
        "mean_exact_groups_per_cluster": 1.0,
        "candidate_generation": "not applicable - existing exact normalized_text baseline",
        "cluster_definition": "exact normalized_text equality",
        "train_n": 45288,
        "val_n": 5679,
        "test_n": 5654,
    }]

    # Compute shingles/minhashes once and reuse across thresholds.
    shingles, minhashes = build_shingles_and_minhashes(
        groups,
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        minhash_seed=args.minhash_seed,
    )

    for t in thresholds:
        args.current_threshold = t
        tag = threshold_tag(t)
        tdir = out_root / f"threshold_{tag}"
        tdir.mkdir(parents=True, exist_ok=True)

        result_json = tdir / "codebert_result.json"

        # If already fully run and not forcing, still ensure audit/splits exist
        # by rebuilding clustering deterministically, but skip model retraining.
        assign, cluster_audit = cluster_groups_at_threshold(
            groups,
            shingles,
            minhashes,
            target_threshold=t,
            num_perm=args.num_perm,
            lsh_margin=args.lsh_margin,
        )

        train_df, val_df, test_df, split_audit = split_corpus_by_fuzzy_clusters(
            corpus, assign, seed=args.seed
        )

        assign.to_csv(tdir / "cluster_assignments.csv", index=False)
        train_df.to_csv(tdir / "train.csv", index=False)
        val_df.to_csv(tdir / "val.csv", index=False)
        test_df.to_csv(tdir / "test.csv", index=False)

        cluster_audit.update(split_audit)
        audit_rows.append(cluster_audit)

        print("\nSplit audit:")
        for k, v in split_audit.items():
            print(f"  {k}: {v}")

        if args.cluster_only:
            continue

        if result_json.exists() and not args.force:
            print(f"\nExisting result found; skipping retraining: {result_json}")
            with open(result_json, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = train_codebert(
                train_df, val_df, test_df, tdir, args
            )
        results_rows.append(result)

        # Save progress after every threshold.
        pd.DataFrame(results_rows).sort_values(
            "threshold", ascending=False
        ).to_csv(out_root / "fuzzy_sensitivity_results.csv", index=False)

        pd.DataFrame(audit_rows).sort_values(
            "threshold", ascending=False
        ).to_csv(out_root / "fuzzy_cluster_audit.csv", index=False)

    # Final summaries.
    results_df = pd.DataFrame(results_rows).sort_values(
        "threshold", ascending=False
    )
    audit_df = pd.DataFrame(audit_rows).sort_values(
        "threshold", ascending=False
    )

    results_df.to_csv(
        out_root / "fuzzy_sensitivity_results.csv", index=False
    )
    audit_df.to_csv(
        out_root / "fuzzy_cluster_audit.csv", index=False
    )

    summary = {
        "experiment": "SQLShield fuzzy near-duplicate sensitivity",
        "baseline_threshold_1p0_definition": (
            "existing exact normalized_text group-aware split; "
            "lowercase + whitespace collapse"
        ),
        "fuzzy_definition": (
            f"{args.shingle_size}-character shingles; MinHash-LSH candidate generation; "
            "exact Jaccard verification; connected-component family clustering"
        ),
        "thresholds": [1.0] + thresholds,
        "num_perm": args.num_perm,
        "lsh_margin": args.lsh_margin,
        "minhash_seed": args.minhash_seed,
        "codebert_seed": args.seed,
        "codebert_model_id": args.model_id,
        "training": {
            "max_length": args.max_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "grad_clip": args.grad_clip,
            "best_checkpoint_metric": "validation F1",
        },
        "important_interpretation_note": (
            "MinHash-LSH is approximate candidate generation. Exact Jaccard is used "
            "to accept edges, but LSH may still miss some truly similar pairs. "
            "This experiment should be reported as a sensitivity analysis, not as "
            "a mathematical guarantee that every semantic near-duplicate was removed."
        ),
        "results": results_df.to_dict(orient="records"),
        "cluster_audit": audit_df.to_dict(orient="records"),
    }

    with open(
        out_root / "fuzzy_sensitivity_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, default=str)

    banner("FUZZY SENSITIVITY COMPLETE")
    print("\nCodeBERT sensitivity results:")
    if len(results_df):
        cols = [
            c for c in [
                "threshold", "accuracy", "precision", "recall", "f1",
                "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
                "fp", "fn", "errors", "train_n", "val_n", "test_n"
            ] if c in results_df.columns
        ]
        print(results_df[cols].to_string(index=False))

    print("\nCluster audit:")
    audit_cols = [
        c for c in [
            "threshold", "num_fuzzy_clusters", "verified_similarity_edges",
            "mixed_label_fuzzy_clusters", "rows_in_mixed_label_clusters",
            "max_exact_groups_in_cluster", "max_rows_in_cluster",
            "train_n", "val_n", "test_n"
        ] if c in audit_df.columns
    ]
    print(audit_df[audit_cols].to_string(index=False))

    print("\nSaved:")
    print(" ", out_root / "fuzzy_sensitivity_results.csv")
    print(" ", out_root / "fuzzy_cluster_audit.csv")
    print(" ", out_root / "fuzzy_sensitivity_summary.json")

    if args.cluster_only:
        print(
            "\nNOTE: --cluster-only was used, so only threshold 1.0 has model metrics. "
            "Inspect the cluster audit/splits, then rerun without --cluster-only."
        )


if __name__ == "__main__":
    main()
