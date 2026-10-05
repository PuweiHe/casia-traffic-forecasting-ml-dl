> **Completed study:** training and locked PEMS evaluation are complete. See
> [final report](FINAL_STUDY.md) and [reproduction instructions](REPRODUCE_FINAL_STUDY.md).
> The dated planning and scheduling text below is historical, not live status.

# Focused traffic forecasting workflow v2

Updated: 2026-10-03. This document describes the current bounded study and
supersedes the broad CUDA search as the active plan. Historical protocols,
completed results and interrupted checkpoints are preserved.

## Task and evaluation contract

Predict the next 12 five-minute speed readings from the preceding 12 readings.
The primary selection metric is the equal-weight mean of validation MAE over
all 12 horizons, in mph. Report RMSE, 15/30/60-minute errors, speed strata and
missing-history strata separately. Earlier single-step MAEs are a different
task and must not be compared directly with these results.

Use chronological 70/10/20 partitions, partition-contained target windows and
train-only normalization. Audit masks, graph provenance, sensor order and
inference independence from future targets. METR-LA is exploratory because its
test period was previously inspected. PEMS-BAY test-use provenance must be
reviewed before calling it independent confirmation. Do not inspect tests to
choose configurations, hardware or optimization strategies.

## Bounded experiment matrix

| Branch | Planned runs | Purpose | Evidence status |
|---|---:|---|---|
| METR-LA STGCN compact/tuned × missing-aware residual off/on | 4 cells × 3 seeds = 12 | Matched tuning, residual and interaction effects | Formal training in progress |
| METR-LA DCRNN and STTN controls | 2 models × 3 seeds = 6 | Controlled architecture references | Formal tasks submitted |
| METR-LA speed-only STAEformer | 3 screening + 2 winner replication + 6 ablation = 11 | Focused Transformer comparison and embedding mechanism | All 11 trials completed; independent final test pending |
| PEMS-BAY STGCN development screen | 13 seed-17 candidates | Development evidence for confirmation | Local MPS training in progress |
| PEMS-BAY confirmation | 2 approaches × 3 seeds = 6; optional contrasting model adds 3 | Independent dataset confirmation after provenance audit | Pending |

Seeds for formal comparisons are 17, 42 and 73. The main new formal scope is
29 METR-LA runs plus 6–9 PEMS-BAY confirmation runs; development screening,
historical searches and diagnostics are additional work, not hidden exclusions.
Historical candidates are reused only with exact data/config/seed/code/protocol
matches and complete provenance. Disclose unequal historical search budgets;
do not claim equal-budget architecture superiority.

STGCN retains direct multi-step output and the full graph. The residual uses
per-sensor last-valid speed with an explicit all-missing fallback. Omitted
recursive-output and graph-top-k sweeps cannot support measured claims about
those changes. STAEformer is a controlled speed-only adaptation of the CIKM
2023 model; time-of-day/day-of-week features are disabled for common-input
comparisons. Spatial-only and zero-adaptive-embedding ablations retain the
relevant width but are not parameter-matched.

## Selection and current findings

The new matched-core budget is at most 100 epochs, minimum 30, patience 20.
Freeze learning-rate schedules and precision before training; use convergence
curves rather than more epochs as evidence of improvement. STAEformer selection
used the prespecified rule: within 1% relative validation MAE, prefer lower
measured inference cost. This is a practical threshold, not a significance test.

Completed STAEformer validation showed the spatial-only variant slightly better
than the adaptive variant across three seeds. Preserve this negative mechanism
finding and the parameter-count confound. Exact aggregate metrics, uncertainty
and raw-file provenance will be published after the report audit; no independent
test improvement or original-paper reproduction is claimed here.

A pooled-MAE versus equal-horizon-MAE discrepancy in local screening requires
validation-only rescoring and a versioned ranking audit before confirmation.
Audit effective DCRNN curriculum settings before claiming paper equivalence.

## Scheduling and recovery

One GPU per independent trial; no DDP is needed for this matrix. Current formal
matched-core comparisons use one frozen V100 32 GB hardware/software class,
with at most two concurrent workers. Preserve historical A100 checkpoints;
do not migrate them to another GPU class or silently mix numerical protocols.

Six configured CUDA preflights passed full-batch forward/backward, finite-gradient,
optimizer and inference checks plus three diagnostic epochs. Measured diagnostic
epoch times were approximately 18–21 seconds for STGCN, 70 seconds for STTN and
226 seconds for DCRNN. These are short diagnostic timings, not full-study timings.

Three original formal STGCN tasks were preserved. Fifteen pending trials were
reassigned exactly once into eight continuous-worker bundles. Four bundles request
4 hours, one requests 1 hour, and three DCRNN bundles request 9 hours each. The new
array starts after the original array exits, with a two-worker concurrency cap.
The cost estimate is 100 × measured epoch seconds × 1.35 + 120 setup seconds;
add 600 seconds and round the requested limit up to an hour.

This scheduling-only change retains the numerical manifest, seeds, precision,
output directories and per-trial checkpoints. Fewer allocation events do not
establish a measured speedup or guarantee a shorter queue. Actual wall times and
queue delays remain to be measured.

Each epoch writes atomic progress and selected weights; resume restores model,
optimizer, scheduler, RNG and early-stopping state. Per-trial locks prevent duplicate
writers; source/data/hardware identities reject incompatible resumes. Time-limit
requeue is bounded to three retries. The bundle supervisor terminates and reaps the
actual writer before exiting. A reproduced signal-handler wait deadlock was fixed;
real-process termination regression tests passed locally and on the server.

A durable submission intent must be reconciled with scheduler state before retrying.
Diagnose deterministic failures instead of blind resubmission. Train only inside
allocated compute jobs. Slurm jobs and queued dependencies persist independently
of a laptop/browser connection; monitoring and local MPS training require an awake
host. Keep one two-hour monitor, private directories/files (700/600), umask 077,
and checksum-verified backups outside temporary scratch storage.

## Remaining delivery gates

1. Finish bounded training and audit configuration reuse, metric consistency and
   test-use provenance. Preserve incomplete cells and failed configurations.
2. Run matched multi-seed ablations, report mean/variation and paired effects.
3. Freeze final choices, then perform locked evaluation; retain negative results.
4. Publish validation curves, error analysis, parameter counts, training cost and
   batch-1/batched inference latency with hardware labels.
5. Package weights, scaler, sensor order, graph/config/source hashes and reload parity;
   implement a validated 12-to-12 inference interface with training-serving consistency.
6. Publish only reviewed code, selected artifacts and aggregate evidence. Exclude raw
   archives, private records, credentials, personalized server details and resume copies.

These experiments are a later reconstruction/extension. They must not be presented
as recovered original internship logs, measured 2024 results, online deployment
benefits or demonstrated congestion reduction. Interview/resume claims require
finished evidence and explicit personal-contribution attribution.

## References

- [STAEformer (CIKM 2023)](https://arxiv.org/abs/2308.10425)
- [Random search for hyperparameter optimization](https://jmlr.org/papers/v13/bergstra12a.html)
- [Model-selection bias](https://jmlr.org/papers/v11/cawley10a.html)
- [DCRNN training recipe](https://github.com/liyaguang/DCRNN/blob/master/data/model/dcrnn_la.yaml)
- [CARC running jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs)
