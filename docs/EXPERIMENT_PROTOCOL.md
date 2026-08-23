# SQLShield experiment protocol — v1.4

1. Parse Source A (`Query`) and Source B (`Sentence`) separately.
2. Clean labels/text within source; preserve contradictions until normalized conflict detection.
3. Remove contradictory normalized groups and exact cross-source duplicate text.
4. Primary mixed-source benchmark: stratified group split over exact `normalized_text` groups (split seed 42), yielding 45,288 / 5,679 / 5,654 rows.
5. Fixed-test models: CodeBERT, BERT-base, four word-TF-IDF models, and two character 3-5-gram TF-IDF models.
6. Transformer stability: seeds 7, 21, 42, 84, 126 on the same split.
7. Paired inference: exact two-sided McNemar for CodeBERT vs seven comparators, with Holm and Bonferroni correction. Verification should be recomputed from per-sample predictions.
8. Accuracy uncertainty: report 95% Wilson score intervals for fixed-test accuracy.
9. Fuzzy sensitivity: 5-character shingles, 256-permutation MinHash, MinHash-LSH candidate threshold = target tau - 0.05, exact Jaccard verification, connected components, row-level StratifiedGroupKFold for tau 0.9/0.8/0.7, CodeBERT seed 42.
10. Fuzzy interpretation caveat: tau=1.0 and tau<1.0 use different split-construction procedures, so the sensitivity table does not isolate a pure threshold effect.
11. Cross-source residual transfer: remove normalized overlap before fitting/evaluation, then A->B and B->A with source-domain validation only.
12. Source fingerprint diagnostic: benign-only character n-gram source classifier.
13. Computational reporting: retain wall-clock scope/hardware and do not label fit+train timings as inference latency.
