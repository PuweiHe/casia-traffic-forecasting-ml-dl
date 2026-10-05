# PEMS-BAY bounded confirmation validation

The bounded three-task MPS confirmation completed successfully. All completed artifacts passed identity, best-weight and epoch-history verification. No new training or test evaluation was performed by this analysis.

| Configuration | Seed | Validation MAE (mph) | Selected epoch | Completed epochs | Parameters |
|---|---:|---:|---:|---:|---:|
| stgcn_reference | 17 | 1.712775 | 56 | 81 | 35940 |
| stgcn_reference | 42 | 1.704072 | 57 | 82 | 35940 |
| stgcn_reference | 73 | 1.691798 | 64 | 89 | 35940 |
| stgcn_search_005 | 17 | 1.601869 | 58 | 83 | 637516 |
| stgcn_search_005 | 42 | 1.597715 | 41 | 66 | 637516 |
| stgcn_search_005 | 73 | 1.603047 | 45 | 70 | 637516 |

The candidate reduces validation MAE by 6.475% for seed 17, 6.241% for seed 42, 5.246% for seed 73. Seed17 was used for selection; seeds42/73 are additional replications. This is complete three-seed validation evidence, not independent-test improvement.

The candidate uses 637,516 parameters versus 35,940 for the reference. Configuration changes jointly affect graph order, temporal kernel, width, dropout, learning rate and batch size; these measurements do not isolate an individual change.

Candidate seed73 is independently bounded and cannot reset its deadline. Locked evaluation and final inference-cost assessment remain pending.

Exact source paths, result hashes, unrounded metrics and limits are in `summary.json`.
