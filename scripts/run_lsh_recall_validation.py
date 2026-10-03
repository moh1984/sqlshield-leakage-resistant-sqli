#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLShield - MinHash-LSH candidate recall validation (Reviewer 2, item 3)
========================================================================

The near-duplicate sensitivity analysis uses MinHash-LSH as a candidate
generator and verifies each candidate with exact Jaccard. That controls
precision but says nothing about recall: a pair above the target threshold that
LSH never proposes is silently missed, so the fuzzy families are a lower bound
on the near-duplicate structure by an unmeasured amount.

This script measures that amount on a tractable subset. It samples unique
normalized texts, computes every pairwise exact Jaccard over 5-character
shingles, and compares the set of true pairs at each threshold against the pairs
the LSH stage would have proposed under the same parameters used in
scripts/run_fuzzy_sensitivity.py.

Complexity is quadratic, so keep the sample at a few thousand texts: 5,000 texts
is 12.5 million pairs and runs in a few minutes.

Usage
-----
  python scripts/run_lsh_recall_validation.py                  # 5,000 texts
  python scripts/run_lsh_recall_validation.py --sample 8000 --repeats 3

Outputs (under $SQLSHIELD_OUT)
  lsh_candidate_recall.csv
"""
from __future__ import annotations

import argparse
import itertools
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from datasketch import MinHash, MinHashLSH

from sqlshield.pipeline import (
    OUT,
    banner,
    build_corrected_merged,
    load_sources,
)

# These must match the defaults of scripts/run_fuzzy_sensitivity.py exactly,
# because the point of this script is to measure the recall of the candidate
# generator that actually produced Table 8, not of a similar one.
SHINGLE = 5
NUM_PERM = 256
MINHASH_SEED = 12345
THRESHOLDS = [0.9, 0.8, 0.7]
LSH_MARGIN = 0.05
CANDIDATE_FLOOR = 0.10     # candidate threshold = max(0.10, target - margin)


def shingles(s: str, k: int = SHINGLE) -> frozenset:
    """Identical to char_shingles() in run_fuzzy_sensitivity.py."""
    s = str(s)
    if len(s) <= k:
        return frozenset([s])
    return frozenset(s[i:i + k] for i in range(len(s) - k + 1))


def jaccard(x: set, y: set) -> float:
    if not x or not y:
        return 0.0
    inter = len(x & y)
    return inter / (len(x) + len(y) - inter)


def lsh_candidates(texts, sigs, threshold):
    lsh = MinHashLSH(threshold=max(CANDIDATE_FLOOR, threshold - LSH_MARGIN),
                     num_perm=NUM_PERM)
    for i, m in enumerate(sigs):
        lsh.insert(str(i), m)
    pairs = set()
    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            j = int(j)
            if i < j:
                pairs.add((i, j))
            elif j < i:
                pairs.add((j, i))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=1, help="independent samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a, b = load_sources()
    corrected, _ = build_corrected_merged(a, b)
    uniq = corrected["normalized_text"].drop_duplicates().tolist()
    banner(f"{len(uniq):,} unique normalized texts in the corrected corpus")

    rows = []
    for rep in range(args.repeats):
        rnd = random.Random(args.seed + rep)
        sample = rnd.sample(uniq, min(args.sample, len(uniq)))
        sets = [shingles(t) for t in sample]

        t0 = time.time()
        sigs = []
        for sh in sets:
            m = MinHash(num_perm=NUM_PERM, seed=MINHASH_SEED)
            m.update_batch([g.encode("utf-8", errors="ignore") for g in sh])
            sigs.append(m)
        sig_s = time.time() - t0

        t0 = time.time()
        truth = {th: set() for th in THRESHOLDS}
        for i, j in itertools.combinations(range(len(sample)), 2):
            s = jaccard(sets[i], sets[j])
            for th in THRESHOLDS:
                if s >= th:
                    truth[th].add((i, j))
        exact_s = time.time() - t0

        for th in THRESHOLDS:
            cand = lsh_candidates(sample, sigs, th)
            true_pairs = truth[th]
            found = len(true_pairs & cand)
            recall = found / len(true_pairs) if true_pairs else float("nan")
            rows.append({
                "repeat": rep, "sample_size": len(sample), "threshold": th,
                "true_pairs": len(true_pairs), "lsh_candidates": len(cand),
                "true_pairs_found": found, "candidate_recall": recall,
                "missed_pairs": len(true_pairs) - found,
                "candidate_precision": (found / len(cand)) if cand else float("nan"),
                "minhash_seconds": sig_s, "exhaustive_seconds": exact_s,
                "num_perm": NUM_PERM, "minhash_seed": MINHASH_SEED,
                "shingle_k": SHINGLE, "lsh_margin": LSH_MARGIN,
            })
            print(f"rep {rep} tau={th}: true={len(true_pairs):,} "
                  f"candidates={len(cand):,} recall={recall:.4f} "
                  f"missed={len(true_pairs) - found:,}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lsh_candidate_recall.csv", index=False)
    banner("CANDIDATE RECALL")
    print(df.to_string(index=False))
    print("\nRecall below 1.0 means the fuzzy families in Table 8 under-merge by that")
    print("proportion, so the reported near-duplicate effect is a lower bound. Quote")
    print("the measured recall in Section 3.5 instead of the present unquantified")
    print("caveat, and note that this subset result need not hold corpus-wide.")


if __name__ == "__main__":
    main()
