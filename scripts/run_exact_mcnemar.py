#!/usr/bin/env python3
"""Exact paired McNemar tests for the final seven CodeBERT comparisons."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sqlshield.pipeline import OUT, PRED_DIR


def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of these prediction files exists:\n" + "\n".join(map(str, paths)))


def transformer_path(model: str) -> Path:
    return first_existing(
        PRED_DIR / f"corrected_group_split_multiseed__{model}__seed42.csv",
        PRED_DIR / f"corrected_group_split_pilot__{model}__seed42.csv",
    )


def exact_mcnemar(df1, df2, name1, name2):
    y = df1["true_label"].to_numpy()
    c1 = df1["pred_label"].to_numpy() == y
    c2 = df2["pred_label"].to_numpy() == y
    b = int(((~c1) & c2).sum())
    c = int((c1 & (~c2)).sum())
    n = b + c
    p = 1.0 if n == 0 else binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return {
        "comparison": f"{name1} vs {name2}",
        "model1_errors": int((~c1).sum()),
        "model2_errors": int((~c2).sum()),
        "model1_wrong_model2_correct": b,
        "model1_correct_model2_wrong": c,
        "discordant_pairs": n,
        "exact_p_value": float(p),
    }


def main():
    paths = {
        "CodeBERT": transformer_path("CodeBERT"),
        "BERT-base": transformer_path("BERT-base"),
        "Random Forest": PRED_DIR / "classical__Random_Forest.csv",
        "XGBoost": PRED_DIR / "classical__XGBoost.csv",
        "Word LinearSVC": PRED_DIR / "classical__SVM.csv",
        "Word Logistic Regression": PRED_DIR / "classical__Logistic_Regression.csv",
        "Char LinearSVC": PRED_DIR / "classical__Char_LinearSVC.csv",
        "Char Logistic Regression": PRED_DIR / "classical__Char_Logistic_Regression.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} predictions: {path}")

    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    ref = dfs["CodeBERT"]
    assert len(ref) == 5654
    for name, d in dfs.items():
        assert len(d) == 5654, f"N mismatch: {name}"
        assert (d["text"].astype(str).to_numpy() == ref["text"].astype(str).to_numpy()).all(), f"Text mismatch: {name}"
        assert (d["true_label"].to_numpy() == ref["true_label"].to_numpy()).all(), f"Label mismatch: {name}"

    comparisons = [
        ("CodeBERT", "BERT-base"),
        ("CodeBERT", "Random Forest"),
        ("CodeBERT", "XGBoost"),
        ("CodeBERT", "Word LinearSVC"),
        ("CodeBERT", "Word Logistic Regression"),
        ("CodeBERT", "Char LinearSVC"),
        ("CodeBERT", "Char Logistic Regression"),
    ]
    out = pd.DataFrame([exact_mcnemar(dfs[a], dfs[b], a, b) for a, b in comparisons])

    m = len(out)
    out["bonferroni_adjusted_p"] = np.minimum(out["exact_p_value"] * m, 1.0)
    out["significant_raw_0.05"] = out["exact_p_value"] < 0.05
    out["significant_bonferroni"] = out["bonferroni_adjusted_p"] < 0.05

    pvals = out["exact_p_value"].to_numpy()
    order = np.argsort(pvals)
    holm = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        holm[idx] = min(running, 1.0)
    out["holm_adjusted_p"] = holm
    out["significant_holm"] = out["holm_adjusted_p"] < 0.05

    out.to_csv(OUT / "exact_mcnemar_FINAL.csv", index=False)
    print("Same fixed test set for all eight models: VERIFIED")
    print("\n=== FINAL EXACT McNEMAR RESULTS (7-comparison family) ===")
    print(out.to_string(index=False))
    print("\nBonferroni family-wise raw alpha:", 0.05 / m)


if __name__ == "__main__":
    main()
