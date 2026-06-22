# All-method evaluation report

The selected rows below minimize rank-Wasserstein within each method/NFE.

| Method | NFE | Temperature | Entropy | Gen-PPL | Rank-W |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mirror | 0 | — | 5.153 | 1476 | 3.508 |
| Periodic | 0 | — | 5.972 | 21.8 | 1.092 |
| Phrase bank | 0 | — | 4.97 | 122.9 | 1.435 |
| Top-k IID | 0 | — | 3.561 | 146 | 1.424 |
| CANDI | 8 | 0.624 | 3.821 | 20.97 | 0.241 |
| DUO | 8 | 0.625 | 3.384 | 17.68 | 0.09826 |
| MDLM | 8 | 0.624 | 2.999 | 14.58 | 0.2381 |
| CANDI | 16 | 0.624 | 4.052 | 15.38 | 0.1646 |
| DUO | 16 | 0.625 | 3.83 | 16.24 | 0.05462 |
| MDLM | 16 | 0.690 | 3.786 | 16.74 | 0.104 |
| CANDI | 32 | 0.690 | 4.557 | 18.47 | 0.1522 |
| DUO | 32 | 0.625 | 4.104 | 14.89 | 0.02386 |
| MDLM | 32 | 0.690 | 3.898 | 12.57 | 0.1302 |
| CANDI | 64 | 0.690 | 4.562 | 15.81 | 0.1239 |
| DUO | 64 | 0.625 | 4.164 | 13.56 | 0.05001 |
| MDLM | 64 | 0.755 | 4.601 | 17.56 | 0.1159 |
| AR | 128 | 0.900 | 5.271 | 13.96 | 0.02606 |
| CANDI | 128 | 0.690 | 4.489 | 14.09 | 0.1394 |
| DUO | 128 | 0.650 | 4.295 | 14.14 | 0.02461 |
| MDLM | 128 | 0.755 | 4.536 | 15.48 | 0.08811 |

## Figures

- `all_metrics_efficiency_summary.png`
- `all_metric_tradeoffs.png`
- `rank_wasserstein_efficiency_pareto.png`
- `parameter_free_metrics_summary.png`
- `../parameter_free_analysis/owt-128/plots/`
- `summaries/`
