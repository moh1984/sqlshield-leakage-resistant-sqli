#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLShield - Repeated independent group-aware splits (Reviewer 2, item 2)
========================================================================

The published comparison rests on one group-aware partition. The five training
seeds vary optimizer and initialization behaviour on that fixed partition, so
they measure optimization variability, not partition uncertainty. With three to
nine errors separating the top models on a 5,654-row test set, a different
valid partition could reorder them.

This script regenerates the whole partition K times with different split seeds,
holding the corrected corpus, the grouping rule and the 80/10/10 proportions
fixed, and reports the distribution of every metric and of the paired
CodeBERT-minus-baseline error difference across partitions.

Classical baselines cost seconds per split, so they are run on many more
partitions than the transformers: --classical-splits controls how many
partitions the classical family sees (default 20) and --splits controls how many
the transformers see (default 5). The transformer partitions are the first
--splits entries of the same seed list, so the two families are always compared
on a common subset.

Usage
-----
  python scripts/run_multi_split.py                                  # 20 classical partitions, minutes
  python scripts/run_multi_split.py --transformers                   # + both transformers on 5 of them
  python scripts/run_multi_split.py --transformers --models CodeBERT # one model, half the GPU time

Outputs (under $SQLSHIELD_OUT)
  multi_split_runs.csv        one row per split per model
  multi_split_summary.csv     mean, SD, min, max and range per model
  multi_split_rankings.csv    model ranking within each split
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from sqlshield.pipeline import (
    MODELS,
    OUT,
    banner,
    build_corrected_merged,
    classical_models,
    compute_metrics,
    load_sources,
    make_group_table,
    train_transformer_once,
)


def group_aware_split_seeded(df: pd.DataFrame, seed: int):
    """group_aware_split with the partition seed exposed, everything else identical."""
    groups = make_group_table(df)
    train_g, temp_g = train_test_split(
        groups, test_size=0.20, stratify=groups["label"], random_state=seed)
    val_g, test_g = train_test_split(
        temp_g, test_size=0.50, stratify=temp_g["label"], random_state=seed)

    sets = [set(g["normalized_text"]) for g in (train_g, val_g, test_g)]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) \
        and sets[1].isdisjoint(sets[2])
    return tuple(df[df["normalized_text"].isin(s)].copy().reset_index(drop=True)
                 for s in sets)


def score_classical(name, model, train_df, test_df):
    t0 = time.time()
    model.fit(train_df["text"], train_df["label"])
    preds = model.predict(test_df["text"])
    clf = model["clf"]
    scores = (model.predict_proba(test_df["text"])[:, 1]
              if hasattr(clf, "predict_proba") else model.decision_function(test_df["text"]))
    m = compute_metrics(test_df["label"], preds, scores)
    return {"model": name, "family": "classical", **asdict(m),
            "elapsed_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5,
                    help="partitions the transformers are run on")
    ap.add_argument("--classical-splits", type=int, default=20,
                    help="partitions the classical baselines are run on (cheap)")
    ap.add_argument("--split-seeds", type=int, nargs="+", default=None,
                    help="explicit partition seeds; defaults to 42, 101, 202, ...")
    ap.add_argument("--transformers", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--train-seed", type=int, default=42,
                    help="transformer training seed, held fixed so that only the partition varies")
    args = ap.parse_args()

    n_all = max(args.splits, args.classical_splits)
    seeds = args.split_seeds or ([42] + [101 * i for i in range(1, n_all)])
    classical_seeds = seeds[:args.classical_splits]
    transformer_seeds = seeds[:args.splits]

    a, b = load_sources()
    corrected, _ = build_corrected_merged(a, b)
    banner(f"corrected corpus {len(corrected):,} rows\n"
           f"classical partitions ({len(classical_seeds)}): {classical_seeds}\n"
           f"transformer partitions ({len(transformer_seeds) if args.transformers else 0}): "
           f"{transformer_seeds if args.transformers else '-'}")

    rows = []
    for split_seed in classical_seeds:
        train_df, val_df, test_df = group_aware_split_seeded(corrected, split_seed)
        banner(f"split seed {split_seed} | train={len(train_df):,} "
               f"val={len(val_df):,} test={len(test_df):,}")

        for name, model in classical_models().items():
            r = score_classical(name, model, train_df, test_df)
            r.update({"split_seed": split_seed, "train_seed": None})
            rows.append(r)
            print(f"  {name:<26} f1={r['f1']:.6f} errors={r['errors']}")

        if args.transformers and split_seed in transformer_seeds:
            for model_name in args.models:
                r = train_transformer_once(
                    experiment=f"multi_split_seed{split_seed}",
                    model_name=model_name, model_id=MODELS[model_name],
                    seed=args.train_seed,
                    train_df=train_df, val_df=val_df, test_df=test_df)
                r.update({"family": "transformer", "split_seed": split_seed,
                          "train_seed": args.train_seed})
                rows.append(r)

        pd.DataFrame(rows).to_csv(OUT / "multi_split_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "multi_split_runs.csv", index=False)

    metrics = ["accuracy", "f1", "roc_auc", "balanced_accuracy", "mcc", "errors"]
    summary = (runs.groupby("model")[metrics]
                   .agg(["mean", "std", "min", "max"])
                   .round(6))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_partitions"] = runs.groupby("model")["split_seed"].nunique()
    summary["error_range"] = summary["errors_max"] - summary["errors_min"]
    summary = summary.sort_values("f1_mean", ascending=False)
    summary.to_csv(OUT / "multi_split_summary.csv")

    # rank only on the partitions every model saw, so the comparison is paired
    common = set(runs.groupby("split_seed")["model"].nunique().pipe(
        lambda s: s[s == runs["model"].nunique()]).index)
    paired = runs[runs["split_seed"].isin(common)] if common else runs
    rankings = (paired.assign(rank=paired.groupby("split_seed")["errors"]
                              .rank(method="min", ascending=True))
                      .pivot_table(index="model", columns="split_seed", values="rank"))
    rankings["rank_min"] = rankings.min(axis=1)
    rankings["rank_max"] = rankings.max(axis=1)
    rankings.to_csv(OUT / "multi_split_rankings.csv")

    banner("DISTRIBUTION ACROSS PARTITIONS")
    print(summary.to_string())
    banner("RANK PER PARTITION (1 = fewest errors)")
    print(rankings.to_string())

    banner("READING")
    print("Report mean and SD across partitions in place of the single-partition")
    print("values in Table 4, and state the error range explicitly. If a model's")
    print("rank changes across partitions, the single-split ranking in the current")
    print("manuscript cannot support an ordering claim and the text must say so.")
    print()
    print("The n_partitions column records how many partitions each model was run")
    print("on. Classical and transformer families are deliberately run on different")
    print("numbers of partitions, so say so in the caption; the ranking table above")
    print("uses only the partitions every model saw.")


if __name__ == "__main__":
    main()
