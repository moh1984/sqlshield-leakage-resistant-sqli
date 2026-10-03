#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLShield - Cross-source evaluation extended to every model (Reviewer 2, item 4)
================================================================================

The published cross-source experiment trains CodeBERT alone, so a failure in the
B->A direction cannot be attributed either to the detector or to the datasets.
This script runs the identical bidirectional overlap-free protocol for the
classical baselines and, optionally, for both transformers, so the two
explanations can be separated.

Protocol (unchanged from sqlshield.pipeline):
  1. apply the global normalized-conflict policy to both sources
  2. split the training source group-aware into 90/10 train/validation
  3. remove from the external source every row whose normalized_text occurs in
     the training source, leaving the overlap-free residual
  4. score on that residual

Usage
-----
  python scripts/run_cross_source_all_models.py                 # classical only, minutes on CPU
  python scripts/run_cross_source_all_models.py --transformers  # adds CodeBERT and BERT-base
  python scripts/run_cross_source_all_models.py --transformers --seeds 7 21 42 84 126

Outputs (under $SQLSHIELD_OUT)
  cross_source_all_models_runs.csv
  cross_source_all_models_overlap.csv
  predictions/cross_source_<dir>__<model>.csv
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sqlshield.pipeline import (
    MODELS,
    OUT,
    PRED_DIR,
    banner,
    classical_models,
    compute_metrics,
    load_sources,
    remove_global_normalized_conflicts,
    residual_external,
    source_train_val,
    summarize_runs,
    train_transformer_once,
)

DIRECTIONS = [("A_sajid576", "B_sqliv2"), ("B_sqliv2", "A_sajid576")]


def score_classical(name, model, train_df, val_df, ext, experiment):
    """Fit on the training source and score the overlap-free residual."""
    start = time.time()
    # fitted on the training partition alone, so that the classical models see
    # exactly the rows the transformer sees and the comparison stays like for like
    model.fit(train_df["text"], train_df["label"])
    preds = model.predict(ext["text"])
    clf = model["clf"]
    if hasattr(clf, "predict_proba"):
        scores = model.predict_proba(ext["text"])[:, 1]
    else:
        scores = model.decision_function(ext["text"])

    m = compute_metrics(ext["label"], preds, scores)
    pd.DataFrame({
        "text": ext["text"].values,
        "source": ext["source"].values,
        "true_label": ext["label"].values,
        "pred_label": preds,
        "score_sqli": scores,
        "correct": (preds == ext["label"].values).astype(int),
    }).to_csv(PRED_DIR / f"{experiment}__{name.replace(' ', '_')}.csv", index=False)

    return {
        "experiment": experiment,
        "model": name,
        "family": "classical",
        "seed": 42,
        **asdict(m),
        "elapsed_seconds": time.time() - start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformers", action="store_true",
                    help="also run CodeBERT and BERT-base (GPU, ~67 min per run)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--skip-classical", action="store_true")
    args = ap.parse_args()

    a, b = load_sources()
    a_cross, b_cross, _ = remove_global_normalized_conflicts(a, b)
    by_name = {"A_sajid576": a_cross, "B_sqliv2": b_cross}

    rows, overlaps = [], []

    for train_name, ext_name in DIRECTIONS:
        train_source, external_source = by_name[train_name], by_name[ext_name]
        ext, overlap = residual_external(train_source, external_source)
        train_df, val_df = source_train_val(train_source)
        experiment = f"cross_source_{train_name}_TO_{ext_name}"

        overlap.update({
            "experiment": experiment,
            "train_source": train_name,
            "external_source": ext_name,
            "train_partition_rows": int(len(train_df)),
            "val_partition_rows": int(len(val_df)),
        })
        overlaps.append(overlap)
        banner(f"{experiment}  residual n={len(ext):,}  SQLi={overlap['sqli_pct_after']:.2f}%")

        if not args.skip_classical:
            for name, model in classical_models().items():
                r = score_classical(name, model, train_df, val_df, ext, experiment)
                r.update({"train_source": train_name, "external_source": ext_name})
                rows.append(r)
                print(f"  {name:<26} f1={r['f1']:.6f}  bal_acc={r['balanced_accuracy']:.6f} "
                      f"roc_auc={r['roc_auc']:.6f}  errors={r['errors']}")

        if args.transformers:
            for model_name, model_id in MODELS.items():
                for seed in args.seeds:
                    r = train_transformer_once(
                        experiment=experiment, model_name=model_name,
                        model_id=model_id, seed=seed,
                        train_df=train_df, val_df=val_df, test_df=ext,
                    )
                    r.update({"family": "transformer", "train_source": train_name,
                              "external_source": ext_name})
                    rows.append(r)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "cross_source_all_models_runs.csv", index=False)
    pd.DataFrame(overlaps).to_csv(OUT / "cross_source_all_models_overlap.csv", index=False)

    banner("CROSS-SOURCE, ALL MODELS")
    cols = ["experiment", "model", "seed", "f1", "roc_auc", "pr_auc",
            "balanced_accuracy", "mcc", "fp", "fn", "errors", "n"]
    print(runs[cols].to_string(index=False))

    if runs["seed"].nunique() > 1:
        summarize_runs(runs, ["experiment", "model"]).to_csv(
            OUT / "cross_source_all_models_summary.csv", index=False)

    banner("READING")
    print("If every model collapses in the B->A direction, the failure belongs to the")
    print("datasets. If only the transformers collapse, it belongs to the detector.")
    print("Report balanced accuracy and ROC-AUC together: the published CodeBERT run")
    print("fails at the transferred operating point while still ranking above chance.")


if __name__ == "__main__":
    main()
