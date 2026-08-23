# GitHub upload checklist — v1.4

This package is arranged as a repository root.

## Before making the repository public

The packaged release already contains all eight aligned fixed-test prediction CSVs; no Kaggle restoration or retraining is required for verification. From the repository root run:

```bash
pytest -q
python scripts/verify_reported_results.py
python scripts/analyze_fixed_test_errors.py
sha256sum -c SHA256SUMS.txt
```

The strict command should print `STRICT PREDICTION VERIFICATION: PASSED`. The GitHub Actions workflow repeats the lightweight regression/verification checks on pushes and pull requests.

## Upload

1. Extract the ZIP locally.
2. Open the extracted `sqlshield-leakage-resistant-sqli` folder.
3. Upload the **contents** of that folder so `README.md` is at the repository root.
4. Commit the upload.

```bash
cd sqlshield-leakage-resistant-sqli
git init
git add .
git commit -m "Release SQLShield v1.4.0 evidence-completion artifact"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Additional checks

- verify the included MIT / CC BY 4.0 licensing information in `LICENSE` and the README License section;
- verify author/citation metadata in `CITATION.cff`, and fill in `repository-code` after creating the public GitHub repository;
- the raw Kaggle CSVs are not redistributed, but the released prediction files carry a `text` column
  reproducing roughly 64% of the corpus by normalized group; confirm this is compatible with the source
  datasets' terms before making the repository public, or ship hashed text instead;
- do not commit credentials or tokens;
- avoid large `.pt` checkpoints unless Git LFS/release assets are intentionally used;
- keep the manuscript, README, machine-readable result tables, and prediction evidence synchronized;
- `requirements-lock.txt` is a frozen compatibility environment; it is not claimed to be an exact historical export of the Kaggle image;
- regenerate `SHA256SUMS.txt` as the last step before pushing and before creating the GitHub Release.
