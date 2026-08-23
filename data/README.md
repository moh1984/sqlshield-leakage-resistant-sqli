# Dataset acquisition

SQLShield uses two public Kaggle SQL-injection datasets. Raw dataset files are not redistributed in this repository.

## Source A

- Kaggle slug: `sajid576/sql-injection-dataset`
- File: `Modified_SQL_Dataset.csv`
- Required text column: `Query`
- Required label column: `Label`

## Source B

- Kaggle slug: `syedsaqlainhussain/sql-injection-dataset`
- File: `sqliv2.csv`
- Encoding: UTF-16
- Required text column: `Sentence`
- Required label column: `Label`

## Canonical Kaggle paths

```text
/kaggle/input/datasets/sajid576/sql-injection-dataset/Modified_SQL_Dataset.csv
/kaggle/input/datasets/syedsaqlainhussain/sql-injection-dataset/sqliv2.csv
```

For local execution, set `SQLSHIELD_A_PATH` and `SQLSHIELD_B_PATH` (and optionally `SQLSHIELD_OUT`) as described in the repository README.

## Important preprocessing rule

Do **not** concatenate the two raw CSV files before resolving their source-specific text columns. Source A uses `Query`; Source B uses `Sentence`. The canonical pipeline cleans each source separately and only then merges them.
