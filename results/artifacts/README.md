# Reproducibility artefacts

These are the intermediate preprocessing and partitioning artefacts the
manuscript's data availability statement describes: which row went to which
partition, which normalized group joined which fuzzy family, and how many rows
survived each preprocessing stage.

**No query text is included.** Each row is keyed by `normalized_sha256`, the
first 16 hex characters of the SHA-256 of its normalized text. The repository
does not redistribute the Kaggle corpora, and these derived tables keep that
policy while still pinning every assignment exactly. The key is collision-free
on this corpus: 45,915 distinct groups give 45,915 distinct keys.

To attach the text, rebuild the corpus from Kaggle and join on the key:

```python
import hashlib, pandas as pd
from sqlshield.pipeline import load_sources, build_corrected_merged

a, b = load_sources()
corrected, _ = build_corrected_merged(a, b)
corrected["normalized_sha256"] = corrected["normalized_text"].map(
    lambda s: hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16])

split = pd.read_csv("results/artifacts/splits/primary_group_split.csv.gz")
df = corrected.merge(split[["normalized_sha256", "partition"]].drop_duplicates(),
                     on="normalized_sha256", how="left")
assert df["partition"].notna().all()
print(df["partition"].value_counts())   # 45,288 / 5,679 / 5,654
```

| File | Contents |
|---|---|
| `preprocessing/source_A_cleaned.csv.gz` | Source A after strict cleaning |
| `preprocessing/source_B_cleaned.csv.gz` | Source B after strict cleaning |
| `preprocessing/conflicted_groups.csv.gz` | the one label-conflicted normalized group |
| `preprocessing/corrected_corpus.csv.gz` | the 56,621-row corrected corpus |
| `preprocessing/stage_counts.csv.gz` | row count at every preprocessing stage |
| `splits/primary_group_split.csv.gz` | row to train/validation/test assignment |
| `splits/fuzzy_cluster_assignments_tau09\|08\|07.csv.gz` | fuzzy-family cluster id per group, with the mixed-label flag |
| `MANIFEST.csv` | rows and SHA-256 of each file as generated |

## Regenerating them

They are derived, not primary, so they can be rebuilt from the two raw Kaggle
sources at any time. Two commands are needed, because the fuzzy clustering
operates on the exact-group table that `run_fuzzy_sensitivity.py` builds:

```bash
export SQLSHIELD_A_PATH=/path/to/Modified_SQL_Dataset.csv
export SQLSHIELD_B_PATH=/path/to/sqliv2.csv
export SQLSHIELD_OUT=$PWD/results

python scripts/export_reproducibility_artifacts.py     # preprocessing + primary split
python scripts/run_fuzzy_sensitivity.py --cluster-only # per-threshold cluster assignments
```

The second writes `threshold_0p90|0p80|0p70/cluster_assignments.csv`; copy each
into `splits/fuzzy_cluster_assignments_tau09|08|07.csv`.

## Expected values

A correct run reproduces every published intermediate. If any of these differs,
stop: the environment is not reproducing the published preprocessing and no
downstream number should be trusted.

| Stage | Rows |
|---|---:|
| Source A after strict cleaning | 30,766 |
| Source B after strict cleaning | 33,537 |
| Source A after conflict removal | 30,764 |
| Sum entering the merge | 64,301 |
| Removed by exact cross-source deduplication | 7,680 |
| Corrected corpus | 56,621 |
| Distinct normalized groups | 45,915 |
| Primary split (train / validation / test) | 45,288 / 5,679 / 5,654 |
| Fuzzy families at tau = 0.9 / 0.8 / 0.7 | 43,675 / 40,705 / 38,526 |

The `mixed_label` column is zero for every family at all three thresholds, which
is the direct check of the claim in Section 4.7 that consolidation never merges
groups carrying conflicting labels.
