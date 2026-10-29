# Traffic forecasting: modeling decisions and engineering evidence

A reproducibility study of the forecasting workflow associated with the 2024 CASIA internship. Reported benchmark measurements come from the documented reproduction; original internship benchmark files are unavailable.

## The problem

Forecast traffic speed from historical sensor measurements while preventing temporal leakage and making model selection, interruption recovery, and inference inspectable. The primary public task uses 12 five-minute history steps to predict the next 12 on PEMS-BAY.

## Decisions, evidence, and trade-offs

| Decision | Evidence | What it demonstrates |
| --- | --- | --- |
| Split chronologically and fit scaling on training data | [Protocol and validity](FINAL_STUDY.md#evaluation-validity) | Temporal ML evaluation and leakage control |
| Screen 13 STGCN configurations, then compare three matched seeds | [Screen results](evidence/final_study/pems_screen_results.json) | Bounded tuning, replication, and experimental bookkeeping |
| Lock selected weights before test scoring | [Frozen selection](evidence/final_study/pems_test/locked_selection.json) | Separation of model selection from final evaluation |
| Report accuracy together with model cost | [Final report](FINAL_STUDY.md) | MAE 1.6853 to 1.6114 mph, 4.38% reduction, with 17.74x parameters |
| Test graph residual and Transformer embedding hypotheses | [Matched ablations](FINAL_STUDY.md#matched-ablations-and-negative-results) | Mechanism-driven experiments and retained negative findings |
| Persist optimizer, scheduler, RNG, and model state with atomic checkpoints | [Reproduction implementation](../../reproduction/model_training/) | Recoverable training and reproducibility |
| Package weights, scaler, graph, and sensor order together | [Inference implementation](../../traffic_forecasting/stgcn_inference.py) | Training-serving consistency and input validation |

## Transformer contribution

The speed-only STAEformer adaptation completed 11 trials spanning screening, replication, and embedding variants. Static spatial embeddings slightly outperformed adaptive embeddings on exploratory METR-LA validation; removing embeddings substantially worsened error. Calendar covariates were omitted and parameter counts differ, so this is neither an exact published-setting reproduction nor a controlled parameter-matched proof. The useful outcome is an explanation of what was tested and what the evidence supports, rather than a claim that attention always wins.

## Interview narrative

**Situation / task:** traffic forecasting needed a comparable temporal benchmark and a reliable path from experiments to inference.

**Action:** audited chronological partitions and missing values; established baselines; bounded the STGCN search; replicated selected configurations; evaluated residual and embedding hypotheses; implemented checkpoint recovery and validated inference artifacts.

**Result:** three matched PEMS seeds all improved, with mean held-out MAE reduced by 4.38%. The larger model imposes a measured cost. Exploratory METR-LA comparisons and failed hypotheses remain documented separately.

For MLE interviews, walk through data identity, model selection, artifact packaging, and serving parity. For SWE/SDE interviews, use writer locks, atomic persistence, bounded retries, schema validation, and testable failure handling as concrete design examples. The serving component is a local prototype; it is not evidence of production traffic, uptime, or online business impact.

## Verify without retraining

1. Read the [final report](FINAL_STUDY.md) and inspect per-seed metrics and limitations.
2. Use the environment in [reproduction instructions](REPRODUCE_FINAL_STUDY.md). Run the released-evidence and inference tests below. They check existing artifacts and synthetic inputs, not a new search or final test scoring.
3. Start the released inference artifact locally using the documented command. Keep source fingerprints and published artifacts unchanged.

```bash
python -m unittest discover -s tests/forecasting -p 'test_release_evidence.py' -v
python -m unittest discover -s tests/forecasting -p 'test_stgcn_inference.py' -v
```

Further training is not required to support the present claims. Cross-city generalization, parameter-matched architecture rankings, or production robustness would require separately designed evidence and budgets.

## Operational interpretation

The construction-monitoring context motivates the work, but public-benchmark MAE is not a measured expressway operating benefit. A deployment would first define the operator decision, forecast lead time, false-alarm tolerance, and data-freshness requirement. Model choice should then account for memory, end-to-end latency, and error costs.

Implemented evidence includes input validation, immutable artifact identity, checkpoint recovery checks, and HTTP regression tests. The server is serial and local. Authentication, capacity planning, bounded queues, production monitoring, shadow rollout, and rollback operations are proposed next steps, not delivered production features. Finite predictions on all-missing inputs do not establish reliable forecasts in that condition.

Keep offline error reduction distinct from unmeasured congestion, revenue, or staffing effects. For a system-design discussion, explain how delayed labels would support quality monitoring and why drift alone should not automatically promote a retrained model.
