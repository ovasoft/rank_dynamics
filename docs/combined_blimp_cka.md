## BLiMP accuracy, final CKA-vs-FR, and perplexity (triangulation)

| Run | BLiMP accuracy | n pairs | Final CKA vs FR | % of FR-FR baseline | Final metric | Best val PPL |
|---|---|---|---|---|---|---|
| GradTrunc8 | 0.5775 | 1600 | 0.0645 | 68.6% | 72.81 (val_ppl) | 72.81 |
| ReLoRA_r8 | 0.4950 | 1600 | 0.0527 | 56.1% | 604.58 (val_ppl) | 604.58 |
| LR_r8_dsn_dynamic | 0.5144 | 1600 | 0.0400 | 42.6% | 114.73 (val_ppl) | 114.73 |
| LR_r8_tuned | 0.5200 | 1600 | 0.0398 | 42.3% | 138.24 (val_ppl) | 138.24 |
| LR_r8 | 0.5563 | 1600 | 0.0349 | 37.2% | 456.60 (val_ppl) | 456.60 |
| LR_r8_dsn_flat | 0.4875 | 1600 | 0.0328 | 34.8% | 109.67 (val_ppl) | 106.99 |
| LR_r8_dsn_static | 0.5150 | 1600 | 0.0313 | 33.3% | 117.18 (val_ppl) | 117.18 |
| LR_specmatch8 | 0.5325 | 1600 | 0.0287 | 30.5% | 438.75 (val_ppl) | 438.75 |

Pearson correlation (BLiMP accuracy, CKA vs FR): 0.398 (n=8)
Pearson correlation (BLiMP accuracy, perplexity): -0.079 (n=8)  
Pearson correlation (CKA vs FR, perplexity): -0.048 (n=8) 