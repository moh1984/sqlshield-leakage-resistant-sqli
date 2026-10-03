#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLShield - Export the reproducibility artefacts the manuscript promises
=========================================================================

The data availability statement says the repository releases the exact
train/validation/test row assignments, the group identifiers for every
fuzzy-family split, and the preprocessing outputs immediately before and after
each deduplication stage. This script regenerates the preprocessing outputs
and the primary split deterministically from the raw sources; the fuzzy-family
assignments come from run_fuzzy_sensitivity.py --cluster-only, which owns the
exact-group table that clustering needs. Together the two let a reader reproduce
the reported numbers from the datasets rather than verifying metrics against
predictions that were already computed.

Nothing here retrains anything; it reruns the preprocessing and splitting path
only, which takes a few minutes on a CPU.

Usage
-----
  python scripts/export_reproducibility_artifacts.py

Outputs (under $SQLSHIELD_OUT/artifacts)
  preprocessing/source_A_cleaned.csv          per-source rows after strict cleaning
  preprocessing/source_B_cleaned.csv
  preprocessing/conflicted_groups.csv         the label-conflicted normalized groups
  preprocessing/corrected_corpus.csv          the 56,621-row merged corpus
  preprocessing/stage_counts.csv              row counts at every stage
  splits/primary_group_split.csv              row -> train/validation/test
  MANIFEST.csv                                file, rows, sha256

The fuzzy-family cluster assignments are produced by
scripts/run_fuzzy_sensitivity.py --cluster-only rather than here, because the
clustering functions operate on the exact-group table that script builds, with
its row_count and source_count columns, not on a bare list of normalized texts.
Duplicating that table here would be a second implementation of the same thing
and could drift from it.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sqlshield.pipeline import (
    OUT,
    banner,
    build_corrected_merged,
    group_aware_split,
    load_sources,
    remove_global_normalized_conflicts,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    argparse.ArgumentParser().parse_args()

    root = Path(OUT) / "artifacts"
    (root / "preprocessing").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)

    a, b = load_sources()
    banner(f"cleaned rows: Source A {len(a):,}  Source B {len(b):,}")
    a.to_csv(root / "preprocessing" / "source_A_cleaned.csv", index=False)
    b.to_csv(root / "preprocessing" / "source_B_cleaned.csv", index=False)

    a_nc, b_nc, conflicts = remove_global_normalized_conflicts(a, b)
    if isinstance(conflicts, pd.DataFrame):
        conflicts.to_csv(root / "preprocessing" / "conflicted_groups.csv", index=False)
    else:
        pd.DataFrame({"note": [str(conflicts)]}).to_csv(
            root / "preprocessing" / "conflicted_groups.csv", index=False)

    corrected, audit = build_corrected_merged(a, b)
    corrected.to_csv(root / "preprocessing" / "corrected_corpus.csv", index=False)

    stages = pd.DataFrame([
        {"stage": "Source A after strict cleaning", "rows": len(a)},
        {"stage": "Source B after strict cleaning", "rows": len(b)},
        {"stage": "Source A after conflict removal", "rows": len(a_nc)},
        {"stage": "Source B after conflict removal", "rows": len(b_nc)},
        {"stage": "sum before cross-source dedup", "rows": len(a_nc) + len(b_nc)},
        {"stage": "corrected corpus", "rows": len(corrected)},
        {"stage": "removed by exact cross-source dedup",
         "rows": len(a_nc) + len(b_nc) - len(corrected)},
        {"stage": "distinct normalized groups",
         "rows": int(corrected["normalized_text"].nunique())},
    ])
    stages.to_csv(root / "preprocessing" / "stage_counts.csv", index=False)
    print(stages.to_string(index=False))

    train_df, val_df, test_df = group_aware_split(corrected)
    assign = pd.concat([
        train_df.assign(partition="train"),
        val_df.assign(partition="validation"),
        test_df.assign(partition="test"),
    ], ignore_index=True)[["text", "normalized_text", "label", "source", "partition"]]
    assign.to_csv(root / "splits" / "primary_group_split.csv", index=False)
    banner(f"primary split: train {len(train_df):,}  validation {len(val_df):,}  "
           f"test {len(test_df):,}")

    banner("FUZZY FAMILIES")
    print("Cluster assignments are produced by scripts/run_fuzzy_sensitivity.py, which")
    print("owns the exact-group table that clustering depends on. Run:")
    print()
    print("  python scripts/run_fuzzy_sensitivity.py --cluster-only")
    print()
    print("then copy each threshold_0pXX/cluster_assignments.csv into")
    print("results/artifacts/splits/fuzzy_cluster_assignments_tauXX.csv.")

    rows = []
    for p in sorted(root.rglob("*.csv")):
        if p.name == "MANIFEST.csv":
            continue
        n = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
        rows.append({"file": str(p.relative_to(root)), "rows": n, "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(root / "MANIFEST.csv", index=False)

    banner("ARTIFACTS WRITTEN")
    print(pd.DataFrame(rows)[["file", "rows"]].to_string(index=False))
    print(f"\nCopy {root} into the repository as results/artifacts/ and reference it")
    print("from the data availability statement.")


if __name__ == "__main__":
    main()
