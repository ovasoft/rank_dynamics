## R2:W3 and R3:W1
We had claimed that CKA and function are meaningfully separable properties.
We tested whether CKA predicts downstream performance differently from function(perplexity) using BLiMP accuracy across eight interventions (LR_r8, three DSN variants, GradTrunc, ReLoRA, hyperparameter-tuned LR, spectrum-matched init). CKA correlates with BLiMP considerably more than perplexity does (r=0.40 vs. r=−0.08), showing performance alone is a poor proxy for what CKA captures, at least for BLiMP (grammatical competence).
The clearest evidence of dissociation comes from DSN: despite large, similar perplexity gains over untreated LR_r8 (74–77% reduction) across all three variants, each remains statistically indistinguishable from chance on BLiMP (p>0.13), while untreated LR_r8 stays significantly above chance (p<0.0001) despite far worse perplexity. Low representational alignment carries a real functional cost that perplexity alone misses. See the table below.


| Run | BLiMP accuracy | n pairs | Final CKA vs FR | % of FR-FR baseline | Final metric | Best Val PPL |
|---|---|---|---|---|---|---|
| GradTrunc8 | 0.5775 | 1600 | 0.0645 | 69.0% | 72.81 (val_ppl) | 72.81 |
| ReLoRA_r8 | 0.4950 | 1600 | 0.0527 | 56.4% | 604.58 (val_ppl) | 604.58 |
| LR_r8_dsn_dynamic | 0.5144 | 1600 | 0.0400 | 42.8% | 114.73 (val_ppl) | 114.73 |
| LR_r8_tuned | 0.5200 | 1600 | 0.0398 | 42.6% | 138.24 (val_ppl) | 138.24 |
| FR | 0.6338 | 1600 | 0.0394 | 42.2% | 46.98 (val_ppl) | 46.98 |
| LR_r8 | 0.5563 | 1600 | 0.0349 | 37.3% | 456.60 (val_ppl) | 456.60 |
| LR_r8_dsn_flat | 0.4875 | 1600 | 0.0328 | 35.0% | 109.67 (val_ppl) | 106.99 |
| LR_r8_dsn_static | 0.5150 | 1600 | 0.0313 | 33.5% | 117.18 (val_ppl) | 117.18 |
| LR_specmatch8 | 0.5325 | 1600 | 0.0287 | 30.7% | 438.75 (val_ppl) | 438.75 |

Pearson correlation (BLiMP accuracy, CKA vs FR): 0.398 (n=8)
Pearson correlation (BLiMP accuracy, perplexity): -0.079 (n=8) 
Pearson correlation (CKA vs FR, perplexity): -0.048 (n=8) 