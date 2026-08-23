#!/usr/bin/env python3
"""Reproduce the corrected fixed split and six classical baselines (four word-level + two character-level)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlshield.pipeline import (
    OUT,
    load_sources,
    build_corrected_merged,
    group_aware_split,
    run_classical,
)


def main():
    a, b = load_sources()
    merged, conflicts = build_corrected_merged(a, b)
    train_df, val_df, test_df = group_aware_split(merged)

    assert len(merged) == 56621
    assert (len(train_df), len(val_df), len(test_df)) == (45288, 5679, 5654)
    assert len(conflicts) == 2

    OUT.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT / "train_corrected.csv", index=False)
    val_df.to_csv(OUT / "val_corrected.csv", index=False)
    test_df.to_csv(OUT / "test_corrected.csv", index=False)
    conflicts.to_csv(OUT / "conflict_groups.csv", index=False)

    results = run_classical(train_df, test_df)
    results.to_csv(OUT / "classical_results_FINAL.csv", index=False)

    cols = [
        "model", "accuracy", "precision", "recall", "f1", "roc_auc",
        "pr_auc", "balanced_accuracy", "mcc", "tn", "fp", "fn", "tp",
        "errors",
    ]
    print("\n=== FINAL CLASSICAL RESULTS ===")
    print(results[cols].sort_values("f1", ascending=False).to_string(index=False))
    print("\nSaved:", OUT / "classical_results_FINAL.csv")


if __name__ == "__main__":
    main()
