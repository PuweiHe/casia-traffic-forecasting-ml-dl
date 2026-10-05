# Matched-core validation analysis

Complete matched cohort: 18/18 verified artifacts, six configurations with seeds 17/42/73. METR-LA is exploratory. No new test evaluation was performed.

The joint tuned STGCN configuration reduced mean validation MAE from 2.869269 to 2.813788 mph (1.93%). All three paired seeds improved.
The missing-aware last-speed residual worsened validation MAE at both compact and tuned settings in all three paired seeds. This negative result is retained; no residual improvement is claimed.

| Configuration | Seeds available | Mean validation MAE (mph) | Sample SD | Parameters |
|---|---|---:|---:|---:|
| dcrnn_selected_control | [17, 42, 73] | 2.868116 | 0.008061 | 888705 |
| stgcn_compact_res0 | [17, 42, 73] | 2.869269 | 0.009284 | 24612 |
| stgcn_compact_res1 | [17, 42, 73] | 2.908631 | 0.010744 | 24612 |
| stgcn_tuned_res0 | [17, 42, 73] | 2.813788 | 0.005083 | 150636 |
| stgcn_tuned_res1 | [17, 42, 73] | 2.849414 | 0.006447 | 150636 |
| sttn_selected_control | [17, 42, 73] | 2.977292 | 0.014552 | 152831 |

Positive paired differences below mean higher error; negative differences mean lower error.

| Matched effect | Mean variant minus reference (mph) | Sample SD of paired differences |
|---|---:|---:|
| tuning_with_residual_off | -0.055481 | 0.013124 |
| residual_on_compact | 0.039362 | 0.019041 |
| residual_on_tuned | 0.035627 | 0.010705 |

Source paths and exact result hashes are in `trial_metrics.csv`; frozen cohort fingerprints and unrounded paired effects are in `summary.json`. All six configurations have three verified seeds. Cross-architecture scores are descriptive validation comparisons; historical selection budgets differ.

The tuned configuration uses more parameters. Its result supports validation-based configuration selection, not a novel architecture claim or isolated learning-rate contribution. Three seeds provide limited uncertainty evidence; sample SD is not a confidence interval. Independent test, subgroup errors, serving parity and final cost/latency assessment remain pending.
