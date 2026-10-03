#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLShield - Controlled inference benchmark (Reviewer 2, item 6)
===============================================================

Table 7 in the manuscript reports training wall-clock on mixed scopes and mixed
hardware, which does not support a deployment-cost claim. This script measures
what a deployment comparison actually needs: per-query latency at batch size 1,
throughput at a larger batch size, resident and peak memory, parameter count and
on-disk model size, all on one documented device.

Latency depends on architecture and input length, not on the value of the fitted
weights, so the transformers are measured with their pretrained encoder and a
classification head; a fine-tuned checkpoint is used instead when one is found
under $SQLSHIELD_OUT/checkpoints. The classical pipelines are fitted first,
because their vectorizer vocabulary does affect transform cost.

Usage
-----
  python scripts/run_inference_benchmark.py                     # CPU and GPU if present
  python scripts/run_inference_benchmark.py --n 2000 --repeats 5
  python scripts/run_inference_benchmark.py --device cpu

Outputs (under $SQLSHIELD_OUT)
  inference_benchmark.csv
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sqlshield.pipeline import (
    MAX_LEN,
    MODELS,
    OUT,
    banner,
    build_corrected_merged,
    classical_models,
    group_aware_split,
    load_sources,
)

CKPT_DIR = OUT / "checkpoints"


def bytes_of(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) + \
           sum(b.numel() * b.element_size() for b in model.buffers())


def time_transformer(model_name, model_id, texts, device, batch, repeats):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    ckpts = sorted(CKPT_DIR.glob(f"*{model_name}*seed42*.pt")) if CKPT_DIR.exists() else []
    loaded = False
    if ckpts:
        try:
            model.load_state_dict(torch.load(ckpts[0], map_location="cpu"))
            loaded = True
        except Exception:
            pass
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = bytes_of(model) / 1024 ** 2

    def run(bs, sample):
        lat = []
        with torch.no_grad():
            for i in range(0, len(sample), bs):
                chunk = sample[i:i + bs]
                enc = tok(list(chunk), truncation=True, padding="max_length",
                          max_length=MAX_LEN, return_tensors="pt").to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(**enc)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    run(batch, texts[:batch * 2])  # warm-up

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    single = []
    for _ in range(repeats):
        single += run(1, texts[:200])
    batched = []
    for _ in range(repeats):
        batched += run(batch, texts)

    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2
            if device.type == "cuda" else float("nan"))

    row = {
        "model": model_name, "family": "transformer",
        "weights": "fine-tuned checkpoint" if loaded else "pretrained + fresh head",
        "device": str(device), "max_len": MAX_LEN,
        "params_millions": n_params / 1e6, "model_size_mb": size_mb,
        "latency_batch1_ms_median": statistics.median(single),
        "latency_batch1_ms_p95": float(np.percentile(single, 95)),
        "batch_size": batch,
        "throughput_qps": batch * len(batched) / (sum(batched) / 1000.0),
        "peak_gpu_mem_mb": peak,
    }
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def time_classical(name, model, train_df, texts, batch, repeats):
    """Timed the same way as the transformers: per-query at batch 1, and in
    chunks of `batch` for throughput. Scoring all 1,000 queries in one call
    would measure a different quantity and could not be compared with a
    batch-32 transformer figure."""
    model.fit(train_df["text"], train_df["label"])
    vocab = len(model["tfidf"].vocabulary_)
    model.predict(texts[:batch])  # warm-up

    single = []
    for _ in range(repeats):
        for t in texts[:200]:
            t0 = time.perf_counter()
            model.predict([t])
            single.append((time.perf_counter() - t0) * 1000.0)

    batched = []
    for _ in range(repeats):
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            t0 = time.perf_counter()
            model.predict(chunk)
            batched.append(((time.perf_counter() - t0) * 1000.0, len(chunk)))
    queries = sum(n for _, n in batched)
    seconds = sum(ms for ms, _ in batched) / 1000.0

    return {
        "model": name, "family": "classical", "weights": "fitted on train split",
        "device": "cpu", "max_len": None,
        "params_millions": float("nan"), "model_size_mb": float("nan"),
        "vocabulary_size": vocab,
        "latency_batch1_ms_median": statistics.median(single),
        "latency_batch1_ms_p95": float(np.percentile(single, 95)),
        "batch_size": batch,
        "throughput_qps": queries / seconds,
        "peak_gpu_mem_mb": float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="queries drawn from the fixed test set")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu"))

    a, b = load_sources()
    corrected, _ = build_corrected_merged(a, b)
    train_df, _, test_df = group_aware_split(corrected)
    texts = test_df["text"].astype(str).tolist()[:args.n]

    banner(f"inference benchmark | device={device} | n={len(texts)} | batch={args.batch}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    rows = []
    for name, model in classical_models().items():
        rows.append(time_classical(name, model, train_df, texts, args.batch, args.repeats))
        print(f"  {name:<26} batch1 median={rows[-1]['latency_batch1_ms_median']:.3f} ms  "
              f"{rows[-1]['throughput_qps']:,.0f} q/s")

    for model_name, model_id in MODELS.items():
        rows.append(time_transformer(model_name, model_id, texts, device,
                                     args.batch, args.repeats))
        print(f"  {model_name:<26} batch1 median={rows[-1]['latency_batch1_ms_median']:.3f} ms  "
              f"{rows[-1]['throughput_qps']:,.0f} q/s  "
              f"{rows[-1]['params_millions']:.1f}M params")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inference_benchmark.csv", index=False)
    banner("INFERENCE BENCHMARK")
    print(df.to_string(index=False))
    print("\nReport the device, batch size and max sequence length alongside every")
    print("number; latency at max_length padding is an upper bound on the dynamic-")
    print("padding cost a real deployment would see.")


if __name__ == "__main__":
    main()
