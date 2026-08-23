"""Lightweight preprocessing primitives used to regression-test the corrected protocol."""
from __future__ import annotations

import re
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"\s+", " ", s)


def strict_clean(raw: pd.DataFrame, text_col: str, label_col: str, source_name: str) -> pd.DataFrame:
    df = raw[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype("string").str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin([0, 1])]
    df = df[df["text"].str.len() > 2].copy()
    df["label"] = df["label"].astype(int)
    df["source"] = source_name
    df = df.drop_duplicates(subset=["text", "label"], keep="first").reset_index(drop=True)
    df["normalized_text"] = df["text"].map(normalize_text)
    return df


def remove_global_normalized_conflicts(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([a, b], ignore_index=True)
    label_counts = combined.groupby("normalized_text")["label"].nunique()
    conflict_norms = set(label_counts[label_counts > 1].index)
    conflicts = combined[combined["normalized_text"].isin(conflict_norms)].copy().reset_index(drop=True)
    a_clean = a[~a["normalized_text"].isin(conflict_norms)].copy().reset_index(drop=True)
    b_clean = b[~b["normalized_text"].isin(conflict_norms)].copy().reset_index(drop=True)
    if len(a_clean):
        assert a_clean.groupby("normalized_text")["label"].nunique().max() == 1
    if len(b_clean):
        assert b_clean.groupby("normalized_text")["label"].nunique().max() == 1
    return a_clean, b_clean, conflicts


def make_group_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("normalized_text", as_index=False).agg(label=("label", "first"), n_rows=("text", "size"))


def group_aware_split(df: pd.DataFrame):
    groups = make_group_table(df)
    train_groups, temp_groups = train_test_split(groups, test_size=0.20, stratify=groups["label"], random_state=SPLIT_SEED)
    val_groups, test_groups = train_test_split(temp_groups, test_size=0.50, stratify=temp_groups["label"], random_state=SPLIT_SEED)
    tr=set(train_groups.normalized_text); va=set(val_groups.normalized_text); te=set(test_groups.normalized_text)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    return (
        df[df.normalized_text.isin(tr)].copy().reset_index(drop=True),
        df[df.normalized_text.isin(va)].copy().reset_index(drop=True),
        df[df.normalized_text.isin(te)].copy().reset_index(drop=True),
    )
