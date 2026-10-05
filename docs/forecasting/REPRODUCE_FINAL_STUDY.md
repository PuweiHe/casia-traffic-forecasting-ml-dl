# Reproduce the final study

## Inspect or serve the released models

Use Python3.10 with `pip install -r requirements-final-study.txt`. This file matches the measured local numerical stack; older requirements files describe other studies. Exact numerical replay additionally depends on hardware/OS. The measured local platform was Apple M4 Pro, macOS26.5.2, Python3.10.19, PyTorch2.10.0. CUDA trials were separate frozen V10032GB protocols; do not migrate their checkpoints to MPS.

Two selected seed42 artifacts are included under `artifacts/forecasting/`: `pems_stgcn_reference_seed42.pt` and `pems_stgcn_search_005_seed42.pt`. Seed42 was fixed as the representative before test evaluation; it was not selected as the best test seed. Both artifacts retain their original immutable `validation_candidate_not_final_test` format field. Their completed test association and release decision are documented in the final report; the old field was not rewritten to bypass source-hash checks.

```bash
python -m traffic_forecasting.stgcn_inference serve \
  --artifact artifacts/forecasting/pems_stgcn_search_005_seed42.pt \
  --device cpu --port 8765
```

The default listener is loopback-only. Use `/health` and JSON `/predict`. Inputs contain exactly `sensor_ids`, `timestamps` and `speed_mph`; the artifact defines sensor order,12 five-minute UTC history timestamps per sample, and speed shape[B,12,325,1]. Use JSON null for missing readings. No actual traffic time-series records are included. See [interface and limitations](STGCN_SERVING_CANDIDATE.md). This is a local prototype, not an authenticated public production service.

## Regenerate figures and audit published numbers

```bash
python scripts/plot_final_study.py
python -m unittest discover -s tests/forecasting -p 'test_release_evidence.py'
python -m unittest discover -s tests/forecasting -p 'test_stgcn_inference.py'
```

`docs/forecasting/evidence/final_study/PUBLIC_MANIFEST.json` hashes the included aggregate evidence. Individual test JSON files bind to the original immutable selection hash. Training curves, screen candidates, three-seed tests, METR matched comparisons and latency samples are included. Raw archives, recordings and personalized server control files are excluded.

## Retrain the published PEMS configuration sequence

Obtain the original `pems-bay.h5` and `distances_bay_2017.csv` from the DCRNN authors' dataset distribution and place them in `../model_training/data/PEMS_BAY`. Install `h5py` if preparing raw HDF5 files; it is optional for model inference. Run `python -m traffic_forecasting.prepare_pems_bay --root ../model_training/data/PEMS_BAY`. Compare generated hashes to the published `pems_data_manifest.json`; do not assume a different archive/timezone policy is the same data.

The unchanged archival evaluation helpers expect this sibling layout:

```text
study/
  Bigscity-LibCity-master/   # clone this repository using this directory name
  model_training/           # private local outputs; not committed
```

Copy `reproduction/model_training/*.py` into that sibling `model_training/`, then create its `protocols/` directory and copy the three `configs/forecasting/local_pems_stgcn_mps_*_v1.json` protocols there. The copied files are source snapshots, not a claim that output weights have already been regenerated. On an MPS-capable Mac, run from the repository root:

```bash
python -m traffic_forecasting.local_pems_stgcn_mps \
  --protocol configs/forecasting/local_pems_stgcn_mps_screen_v1.json \
  --data-dir ../model_training/data \
  --output-dir ../model_training/runs/local_pems_stgcn_mps_screen_v1
python -m traffic_forecasting.local_pems_stgcn_mps_confirm \
  --protocol configs/forecasting/local_pems_stgcn_mps_confirm_v1.json \
  --data-dir ../model_training/data \
  --output-dir ../model_training/runs/local_pems_stgcn_mps_confirm_v1
python -m traffic_forecasting.local_pems_stgcn_mps_seed73 \
  --protocol configs/forecasting/local_pems_stgcn_mps_seed73_v1.json \
  --data-dir ../model_training/data \
  --output-dir ../model_training/runs/local_pems_stgcn_mps_seed73_v1
```

These are validation-only entries with source/identity checks and atomic checkpoints. The confirmation entries impose epoch-boundary wall budgets; their archived external supervisor supplied the original hard cap. Do not claim the CLI alone supplies that hard cap. Do not launch two processes into one output directory, change frozen code or extend a stopped budget silently. The study used full-batch smoke checks before long runs.

After six complete artifacts, `analyze_pems_confirmation.py` and `validate_training_artifacts.py` check results against selected/progress weights, histories, data and declarations. `locked_pems_evaluation.py freeze` then `evaluate` shows the original gated evaluation implementation. It additionally requires a locally reviewed `test_provenance_ledger.json` with status `reviewed_independent_with_disclosed_scope`. Do not fabricate a review or reuse our attestation as your own. A rerun after viewing published test scores is numerical replication, not a new independent model-selection exercise. Original locked test reports are available without re-scoring.

The copied helpers preserve original paths/layout and source hashes for audit. New hardware/software/configuration studies require separate protocols and output roots; exact cross-device bitwise reproduction is not promised. Only the two representative serving models are distributed; reproducing all six seeds requires retraining.

## CUDA matched cohort and Transformer controls

`configs/forecasting/matched_core_v2.json` contains the18 immutable tasks; `focused_staeformer_v1.json` contains the three Transformer screen candidates. `traffic_forecasting.focused_worker` runs an indexed validation-only task on an allocated CUDA compute node:

```bash
python -m traffic_forecasting.focused_worker \
  --plan configs/forecasting/matched_core_v2.json \
  --index 0 --data-dir /path/to/prepared/data --output /path/to/new/private/run
```

Use a separate output root for each new numerical study. The worker's setup, manifest and task locks protect shared roots. Match the intended CUDA/software/hardware record and run `--preflight` in a separate diagnostic root before formal work. Scheduling wrappers and account paths are site-specific; historical personal Slurm control files are not published. No unverified queue-time or heterogeneous-GPU speedup is claimed.
