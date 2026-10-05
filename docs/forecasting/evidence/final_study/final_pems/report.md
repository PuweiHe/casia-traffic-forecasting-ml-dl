# Locked PEMS-BAY evaluation

Task: 12 past five-minute speed readings to predict the next 12, in mph. Chronological 70/10/20 partitions with partition-contained windows and training-only standardization. The exact validation selection, six checkpoint hashes, evaluation source and provenance ledger were frozen before scoring. No configuration was retuned using test results.

| Configuration | Test MAE mean ± sample SD (mph) | Parameters | Timed training, three seeds (h) |
|---|---:|---:|---:|
| stgcn_reference | 1.685255 ± 0.005201 | 35,940 | 3.544 |
| stgcn_search_005 | 1.611418 ± 0.008581 | 637,516 | 11.634 |

Joint tuning reduced mean test MAE by 4.381%; all three matched seeds improved. The tuned model has 17.74 times as many parameters. This is an accuracy/cost trade-off, not a new STGCN architecture.

| Configuration | 15-min MAE | 30-min MAE | 60-min MAE | Overall RMSE |
|---|---:|---:|---:|---:|
| stgcn_reference | 1.388794 | 1.738003 | 2.096078 | 3.832526 |
| stgcn_search_005 | 1.328291 | 1.671875 | 1.989762 | 3.761244 |

| Configuration | Slow (<30 mph) MAE | Moderate (30–55 mph) MAE | Free flow (≥55 mph) MAE | Missing history (>25%) MAE |
|---|---:|---:|---:|---:|
| stgcn_reference | 6.991305 | 6.435984 | 1.147857 | N/A |
| stgcn_search_005 | 6.527192 | 6.259507 | 1.094857 | N/A |

Baseline test MAE: persistence 2.169196 mph, moving_mean 2.751034 mph.

## Temporal uncertainty

For the fixed representative seed42, paired24-hour block resampling used37 blocks
and5000 resamples. Candidate-minus-reference MAE was−0.080873mph, with an approximate
95% interval[−0.093002,−0.068785]. The relative reduction interval was[4.267%,5.309%].
This is supplemental conditional uncertainty for that seed/period, distinct from
three-seed training variation. Daily blocks approximate dependence; the final partial
block is retained. The diagnostic was specified after aggregate scoring and is not
a preregistered significance test. Full values and raw block sums remain in JSON.

## Attribution and limitations

- Seed17 participated in validation screening; final test was reserved until all six models and selection were frozen.
- Joint hyperparameter tuning, larger capacity and different batch size; not an isolated architectural innovation.
- Three seeds quantify training variation, not uncertainty across cities or future years.
- Existing-file provenance audit plus user attestation; deleted/external history cannot be technically ruled out.
- Reconstructed measurements obtained in 2026, not recovered 2024 internship results.
- No demonstrated deployment or congestion-reduction benefit.
- Training seconds sum timed training/validation loops; exclude setup, checkpoint overhead and queue wait.

Exact source files and hashes are in `trial_metrics.csv`; unrounded subgroup metrics and observation counts are in `summary.json`.
