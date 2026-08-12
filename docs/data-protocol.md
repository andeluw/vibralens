# XJTU-SY data protocol

## Dataset grain

The independent experimental unit is one bearing run. A CSV is one 1.28-second, two-channel vibration snapshot containing 32,768 samples at 25.6 kHz. Snapshots are recorded at one-minute intervals.

The manifest contains one row per snapshot, but rows from the same bearing are not independent machines.

## Split protocol

The tracked split follows the CRULE assignment within each operating condition:

| Role | Bearing indices | Bearings across three conditions |
|---|---:|---:|
| Train | 1, 2 | 6 |
| Validation | 3 | 3 |
| Test | 4, 5 | 6 |

The machine-readable assignment is in `configs/splits/xjtu_sy_crule.json`.

This fixed assignment makes experiments reproducible while preserving the complete bearing as the independent unit. Because the dataset contains only 15 bearing runs, aggregate metrics must be accompanied by per-bearing results and their spread.

## Main target

For snapshot `t` in a bearing with final snapshot `T`:

```text
RUL minutes = (T - t) × 1 minute
```

The last snapshot has a target of zero. This is an absolute dataset-time target. Per-bearing lifetime-normalized RUL is not used because it cannot be converted back to remaining minutes for an unseen bearing without knowing that bearing's total lifetime.

The endpoint must be described as XJTU-SY's experimental end-of-life criterion, not a universal physical failure threshold.

## Leakage controls

1. Split complete bearings before preprocessing or sequence construction.
2. Fit normalization, feature selection, thresholds, and learned transforms only on train bearings.
3. Use validation bearings for model selection.
4. Keep test bearings outside tuning, calibration, and threshold selection.
5. Report metrics per bearing in addition to aggregate metrics.

## Verified local audit

The full audit validates:

- the 15 expected condition/bearing directories;
- continuous numeric snapshot filenames for every run;
- the expected count of snapshots per bearing;
- the two-channel CSV header;
- exactly 32,768 signal rows in every CSV;
- deterministic relative paths, split assignments, and RUL labels.

The manifest audit validates dataset structure rather than signal values. Numeric finiteness and derived-feature checks are performed by the feature extractor and recorded in `artifacts/features/xjtu_sy_feature_audit.json`.
