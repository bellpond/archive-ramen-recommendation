# Ramen recommendation models

## NMFBaseCF
---
NMFBaseCF is a collaborative filtering (CF) model based on non-negative matrix factorization (NMF).

The input csv file must contain 'user_id', 'store_id' and 'score' columns.
```bash
python main.py --model 'nmf-cf' \
--csv_path 'data/review_data.csv' --latent_dim 16 --normalize \
--user_id 319543 --top_n 10 --include_known
```

