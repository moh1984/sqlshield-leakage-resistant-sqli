#!/usr/bin/env python3
"""Run the corrected bidirectional normalized-overlap-free cross-source experiment."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlshield.pipeline import OUT, load_sources, run_cross_source


def main():
    a, b = load_sources()
    runs, summary, overlap = run_cross_source(a, b)

    # Preserve explicit FINAL filenames used by the paper artifact.
    runs.to_csv(OUT / "cross_source_runs_FINAL.csv", index=False)
    overlap.to_csv(OUT / "cross_source_overlap_FINAL.csv", index=False)

    print("\n=== CROSS-SOURCE RUNS ===")
    cols = [
        "experiment", "model", "seed", "best_epoch", "accuracy", "precision",
        "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
        "tn", "fp", "fn", "tp", "errors", "test_n",
    ]
    print(runs[cols].to_string(index=False))
    print("\n=== CROSS-SOURCE OVERLAP AUDIT ===")
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
