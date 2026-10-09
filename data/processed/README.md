# Processed data directory

Processed/feature-engineered data frames are written here by the training
pipeline and are git-ignored (financial data should not live in version
control). Regenerate them with `python -m src.models.train`.