# Superseded results — kept for transparency, do not cite

Three files here were produced by earlier versions of the scripts and have been
replaced. They are kept so that a reader who sees two different numbers for the
same quantity can find out why.

**`cross_source_all_models_runs_TRAINVAL_superseded.csv`** — the first
cross-source run, in which the classical models were fitted on train plus
validation and therefore saw about ten per cent more data than the transformers.
Refitting on the training partition alone changed character LinearSVC from 22
errors to 21, so the conclusion is unaffected, but the comparison is only
like-for-like in the current file.

**`lsh_candidate_recall_128perm_superseded.csv`** — measured with 128 MinHash
permutations and no seed, while `run_fuzzy_sensitivity.py`, which produced
Table 8, uses 256 permutations and seed 12345. The recall figures therefore
described a different candidate generator from the one under test. The current
script matches the published parameters.

**`inference_benchmark_unbatched_classical_superseded.csv`** — classical
throughput was measured by scoring all 1,000 queries in a single `predict` call
while the transformers were measured in batches of 32, so the two families were
not timed the same way. The current script chunks the classical path at the same
batch size.
