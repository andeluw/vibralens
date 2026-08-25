# XJTU-SY data protocol

## Source and permitted use

The XJTU-SY accelerated-life bearing dataset is provided by Xi'an Jiaotong
University and Changxing Sumyoung Technology. The
[official repository](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)
describes it as publicly available for validating bearing prognostics
algorithms and requests citation of the originating publication:

> B. Wang, Y. Lei, N. Li, and N. Li, "A Hybrid Prognostics Approach for
> Estimating Remaining Useful Life of Rolling Element Bearings," IEEE
> Transactions on Reliability, 69(1), 401-412, 2020.
> <https://doi.org/10.1109/TR.2018.2882682>

The official repository does not publish a standard license file. VibraLens
therefore does not redistribute the raw dataset. Reproduction requires users to
obtain it from the provider and follow the provider's terms and citation
request.

## Dataset grain

The independent experimental unit is one bearing run. A CSV is one 1.28-second, two-channel vibration snapshot containing 32,768 samples at 25.6 kHz. Snapshots are recorded at one-minute intervals.

The manifest contains one row per snapshot, but rows from the same bearing are not independent machines.

The 15 bearings are distributed across three operating conditions:

| Condition | Directory | Shaft speed | Radial load | Bearings |
|---:|---|---:|---:|---:|
| 1 | `35Hz12kN` | 2,100 rpm | 12 kN | 5 |
| 2 | `37.5Hz11kN` | 2,250 rpm | 11 kN | 5 |
| 3 | `40Hz10kN` | 2,400 rpm | 10 kN | 5 |

## Split protocol

The tracked split follows the CRULE assignment within each operating condition:

| Role | Bearing indices | Bearings across three conditions |
|---|---:|---:|
| Train | 1, 2 | 6 |
| Validation | 3 | 3 |
| Test | 4, 5 | 6 |

The machine-readable assignment is in `configs/splits/xjtu_sy_crule.json`.

This fixed assignment makes experiments reproducible while preserving the complete bearing as the independent unit. Because the dataset contains only 15 bearing runs, aggregate metrics must be accompanied by per-bearing results and their spread.

This protocol is one fixed complete-bearing holdout. It is not rotating
cross-validation or leave-one-bearing-out evaluation.

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
