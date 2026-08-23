#!/usr/bin/env python3
"""Run CodeBERT/BERT-base on the corrected fixed split for selected training seeds."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlshield.pipeline import (
    OUT,
    MODELS,
    load_sources,
    build_corrected_merged,
    group_aware_split,
    train_transformer_once,
    summarize_runs,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+", default=["all"],
        choices=["all", "CodeBERT", "BERT-base"],
        help="Transformer models to run.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[7, 21, 42, 84, 126],
        help="Training seeds. The paper uses 7 21 42 84 126.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    selected = list(MODELS) if "all" in args.models else args.models

    a, b = load_sources()
    merged, _ = build_corrected_merged(a, b)
    train_df, val_df, test_df = group_aware_split(merged)
    assert (len(train_df), len(val_df), len(test_df)) == (45288, 5679, 5654)

    rows = []
    for model_name in selected:
        for seed in args.seeds:
            rows.append(
                train_transformer_once(
                    experiment="corrected_group_split_multiseed",
                    model_name=model_name,
                    model_id=MODELS[model_name],
                    seed=seed,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                )
            )

    runs = pd.DataFrame(rows)
    summary = summarize_runs(runs, ["experiment", "model"])
    runs.to_csv(OUT / "transformer_multiseed_runs_FINAL.csv", index=False)
    summary.to_csv(OUT / "transformer_multiseed_summary_FINAL.csv", index=False)

    print("\n=== TRANSFORMER RUNS ===")
    print(runs[["model", "seed", "best_epoch", "accuracy", "f1", "errors", "fp", "fn"]].to_string(index=False))
    print("\n=== TRANSFORMER SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
