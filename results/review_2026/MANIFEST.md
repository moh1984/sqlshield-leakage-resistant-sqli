# What is here, and what still has to be added

The CSV files in this directory are complete and were produced by the scripts in
`scripts/`. Three groups of artefacts are still missing.

## Per-sample predictions and training curves

Every run wrote a `predictions/` and a `histories/` directory beside its CSV.
These are not included here and should be copied from the sessions that produced
them, because they are what allows a reader to recompute the paired significance
tests rather than trusting the summary numbers:

- `predictions/tuned__*.csv` — tuned classical configurations, so that
  `scripts/run_exact_mcnemar.py` can be rerun against them.
- `predictions/cross_source_*__*.csv` — every cross-source run, both families.
- `predictions/` and `histories/` for each partition of the multi-split runs.

## Promised by the manuscript, and shipped

`results/artifacts/` now contains the files themselves, gzip-compressed, together
with the commands that regenerate them:

- `splits/primary_group_split.csv` — train/validation/test assignment per row.
- `splits/fuzzy_cluster_assignments_tau09|08|07.csv` — fuzzy-family cluster
  assignments with the mixed-label flag, from `run_fuzzy_sensitivity.py
  --cluster-only`.
- `preprocessing/` — per-source cleaned rows, the label-conflicted group, the
  corrected corpus, and `stage_counts.csv` giving the row count at every stage.
- `MANIFEST.csv` — rows and SHA-256 for each file.

Reruns during this revision reproduced every published intermediate: 30,766 and
33,537 cleaned rows, one label-conflicted group, 64,301 rows entering the merge,
7,680 removed by exact cross-source deduplication, 56,621 rows across 45,915
normalized groups, the 45,288/5,679/5,654 partition, and 43,675, 40,705 and
38,526 fuzzy families at tau = 0.9, 0.8 and 0.7.

Four of the ten multi-seed transformer prediction files survive; the other six
were never written in the original sessions, which `provenance.json` records and
the manuscript now states.

## Reviewer items still open

- **R2-1.** No post-2024 detector has been reimplemented. Section 7.2 of the
  manuscript gives the comparative discussion Reviewer 1 offered as an
  alternative, arguing from these measurements why a cross-paper accuracy table
  would mislead.
- **R2-2.** The transformers were rerun on three partitions, enough to show that
  the leading ordering is unstable but too few for a reliable distribution. Two
  further partitions, about 2 h 35 m on two T4s, would close this.
- **R2-3.** Candidate recall is measured; repeated fuzzy-family splits, which
  would separate grouping strength from split construction and make the rows of
  Table 8 paired, have not been run and would need a new script.

## Environment

`provenance.json` records that the package versions of the original sessions
were never captured. The manuscript describes the released `requirements-lock.txt`
as a frozen compatibility environment that reproduces the reported results, not
as an exact historical export. Do not replace that wording with a stronger claim.
The 2026 verification session ran Python 3.12.13, PyTorch 2.10.0 with CUDA 12.8
and scikit-learn 1.6.1 on one NVIDIA Tesla T4 with an Intel Xeon host at
2.00 GHz, 2 physical cores, 4 logical threads and 31 GB of RAM.

## Verified during the revision

Running the pipeline on the partition seed of the published split reproduces
Table 4 exactly: CodeBERT at three errors (FP=1, FN=2, F1=0.999319), BERT-base
at six, character LinearSVC at nine, Random Forest at sixteen, character
Logistic Regression at twenty, XGBoost at twenty-one, word LinearSVC at
twenty-five and word Logistic Regression at fifty-two. The per-epoch curves also
reproduce, including the epoch-4 CodeBERT training loss of 0.000423.
