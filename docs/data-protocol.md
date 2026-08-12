# XJTU-SY data protocol

## Dataset grain

The independent experimental unit is one bearing run. A CSV is one 1.28-second, two-channel vibration snapshot containing 32,768 samples at 25.6 kHz. Snapshots are recorded at one-minute intervals.

The manifest contains one row per snapshot, but rows from the same bearing are not independent machines.

## Development split

The initial benchmark follows CRULE within each operating condition:

| Role | Bearing indices | Bearings across three conditions |
|---|---:|---:|
| Train | 1, 2 | 6 |
| Validation | 3 | 3 |
| Test | 4, 5 | 6 |

The machine-readable assignment is in `configs/splits/xjtu_sy_crule.json`.

This fixed split is for reproducible development. Final evidence must rotate held-out bearings because validation behavior can vary materially among the 15 runs.

## Main target

For snapshot `t` in a bearing with final snapshot `T`:

```text
RUL minutes = (T - t) × 1 minute
```

The last snapshot has a target of zero. This is an absolute dataset-time target and can support a user-selected maintenance horizon. Per-bearing lifetime-normalized RUL is not the main target because it cannot be converted back to remaining minutes for an unseen bearing without knowing that bearing's total lifetime.

The endpoint must be described as XJTU-SY's experimental end-of-life criterion, not a universal physical failure threshold.

## Leakage controls

1. Split complete bearings before preprocessing or sequence construction.
2. Fit normalization, feature selection, thresholds, and learned transforms only on train bearings.
3. Use validation bearings for model selection; do not inspect test metrics during development.
4. Keep uncertainty calibration separate from final test bearings.
5. Report metrics per bearing in addition to aggregate metrics.
6. Treat domain adaptation and zero-shot cross-condition generalization as different experiments.

## Verified local audit

The full audit validates:

- the 15 expected condition/bearing directories;
- continuous numeric snapshot filenames for every run;
- the expected count of snapshots per bearing;
- the two-channel CSV header;
- exactly 32,768 signal rows in every CSV;
- deterministic relative paths, split assignments, and RUL labels.

It does not yet validate every numeric value's range or derive signal features. Those checks belong to the next baseline stage.
