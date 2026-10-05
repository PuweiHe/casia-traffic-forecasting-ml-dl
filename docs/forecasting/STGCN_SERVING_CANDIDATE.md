# PEMS-BAY STGCN candidate serving

This separate inference module packages the completed validation candidate. It does not change frozen training code or promote a final tested model. Candidate seed 73 and independent-test provenance review remain pending.

The artifact includes selected weights, the supplied graph, ordered sensor IDs, train-fitted normalization, the trial configuration, source/data/checkpoint fingerprints and a `validation_candidate_not_final_test` stage. The model rebuild uses the frozen STGCN graph policy `max(A, A.T)`; its Chebyshev graph buffers are reconstructed from the packaged graph rather than assumed to exist in the weight dictionary.

Package a trusted, completed local checkpoint:

```bash
python -m traffic_forecasting.stgcn_inference export \
  --selected /path/to/completed/selected.pt \
  --data-dir /path/to/PEMS_BAY \
  --output /path/to/private/artifact.pt
```

Export refuses to replace an existing artifact. Source and graph/sensor hashes must match the frozen declaration. Use trusted local artifacts; an embedded fingerprint is an integrity check, not a signature establishing trust in an unknown file.

The request has exactly three fields:

- `sensor_ids`: strings matching the artifact's sensor order exactly.
- `timestamps`: `[batch, 12]` timezone-aware ISO timestamps, consecutive and on five-minute boundaries.
- `speed_mph`: `[batch, 12, nodes, 1]`, finite JSON numbers or `null` for missing history. Nonpositive speeds follow the original training missingness policy. Observed values equal to the fitted mean remain observed even though their normalized value is zero.

The response returns 12 future steps in mph, sensor order, future timestamps, artifact identity/stage and missing-history fraction. Predictions are not clipped; clipping would change parity with offline inference. Future labels are not accepted as inputs. Batch size is bounded at 32.

Run one JSON request:

```bash
python -m traffic_forecasting.stgcn_inference predict \
  --artifact /path/to/private/artifact.pt \
  --request /path/to/request.json \
  --output /path/to/prediction.json --device mps
```

Devices are explicit. Unsupported MPS/CUDA raises an error without fallback. CPU is an intentional inference option; it does not migrate a training checkpoint or alter any experiment protocol.

For a local HTTP demonstration:

```bash
python -m traffic_forecasting.stgcn_inference serve \
  --artifact /path/to/private/artifact.pt --device mps --port 8080
```

This loads the model once and binds only `127.0.0.1`. `GET /health` checks readiness; `POST /predict` accepts JSON, with an 8 MiB body limit and a read timeout. Request histories are not logged. This single-process endpoint is a local demonstration; production authentication, TLS, concurrency and promotion policies are outside this implementation.

## Verification

Five focused tests cover training-preprocessing equivalence, prediction/label-independent schema, mask semantics, sensor/time validation, graph fingerprint rejection, unavailable-device rejection and HTTP happy/error paths. Run:

```bash
python -m unittest discover -s tests/forecasting -p test_stgcn_inference.py -v
```

The recorded real-checkpoint check used only the first PEMS-BAY training history. Saved candidate seed42 weights reloaded with maximum same-device difference of **0 mph** on both CPU and MPS. This is one checked history, not proof for every input or equality across different hardware. No test scoring was performed.

The local companion `model_training/verify_stgcn_serving.py` records raw timing samples, environment, artifact/checkpoint hashes and parity. On Apple M4 Pro, the illustrative in-process MPS p50 was 16.10 ms for batch1 and 83.18 ms for batch16 (3 warmups, 10 synchronized repetitions). Timing includes request validation, normalization, forward, device transfer and list response; it excludes HTTP transport and JSON encoding. These small repeated-history benchmarks are not production latency guarantees or training-speed improvements.
