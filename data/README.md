# Data

## `raw/`

The raw dataset **is not committed to the repository** (see `.gitignore`). Raw
financial data is sensitive, and we intentionally keep it out of version
control.

### Dataset: Default of Credit Card Clients

* **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients*.
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
  (Lichman, 2013; original authors Yeh & Lien, 2009).
* **Size:** 30,000 observations × 24 predictor columns + 1 target.
* **Target:** `default_payment_next_month` — whether the client defaulted on
  their credit card payment in the next month (`1` = default, `0` = no default).

### How to acquire the data

Run the protected download routine (ships with the repo, no external data is
required):

```bash
python -m src.data.load_data
```

Or manually download the file and place it at `data/raw/cc_default.xls`:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
```

## `processed/`

Intermediate, cleaned and feature-engineered data frames that are produced by
the pipeline. Like `raw/`, processed data is git-ignored and regenerated from
`raw/` by the training pipeline.