# Raw data directory

Raw financial data is git-ignored and must be downloaded separately. See
[`data/README.md`](../data/README.md). Run `python -m src.data.load_data` to
download the UCI *Default of Credit Card Clients* dataset, or place
`cc_default.xls` here manually.

The `.gitkeep` file only exists so the directory is preserved in git.

`data/raw/README.md` is committed for documentation purposes; all other
contents are excluded.