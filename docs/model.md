# VibraLens RUL model

## Intended use

Model version `vibralens-rul-0.1.0` estimates remaining useful life in dataset minutes from one XJTU-SY-format vibration snapshot, the bearing's observed age, and a known operating-condition ID. The result is an empirical interval used by a simple planned-break rule.

This is experimental XJTU-SY evidence, not factory certification. The interval is not a formal safety guarantee, and the service abstains instead of extrapolating to unknown operating conditions.

## Data boundary

All 15 run-to-failure bearings have one fixed role, with no snapshots from a bearing crossing roles:

| Bearing index in each condition | Role | Bearings |
|---|---|---:|
| 1-2 | Training | 6 |
| 3 | Validation and model selection | 3 |
| 4-5 | One-time held-out evaluation | 6 |

Training uses bearing-balanced sample weights so every training run has equal total influence despite different lifetimes. Validation and test headline errors are macro-averaged across complete bearings.

## Model ladder and selection

The frozen experiment compares:

1. an age-only median-lifetime baseline by condition;
2. regularized linear Ridge candidates over horizontal, vertical, and dual-channel features, with and without age;
3. quantile histogram-gradient-boosting candidates as a nonlinear interval comparison.

The selected candidate is the simpler `ridge-vertical-age1-alpha100-empirical80`: condition, age, and 14 deterministic vertical-channel features, with Ridge `alpha=100`. Its interval is the point prediction plus or minus a bearing-balanced 80th-percentile validation residual radius of `454.504` minutes, clipped at zero.

The validation gate required at least 5% macro-bearing MAE improvement over age-only and improvement on at least two of three validation bearings. The selected candidate improved macro MAE by 59.0% and improved two bearings, so the gate passed before held-out metrics were read.

## Recorded metrics

| Split / metric | Value |
|---|---:|
| Validation age-only macro-bearing MAE | 775.916 min |
| Validation selected macro-bearing MAE | 317.883 min |
| Validation across-bearing MAE standard deviation | 114.457 min |
| Validation macro-bearing interval coverage | 80.21% |
| Validation mean interval width | 868.705 min |
| Held-out macro-bearing MAE | 347.010 min |
| Held-out across-bearing MAE standard deviation | 120.874 min |
| Held-out snapshot MAE | 363.443 min |
| Held-out macro-bearing interval coverage | 62.42% |
| Held-out snapshot interval coverage | 69.00% |
| Held-out mean interval width | 873.644 min |

The lower held-out coverage shows that the empirical interval does not transfer uniformly across bearing runs. It should be presented as uncertainty context, not a calibrated deployment guarantee.

The near-end-of-life diagnostics make the weakest case explicit: at the final `Bearing1_4` snapshot, actual RUL is zero while the model predicts `778.100` minutes and its lower bound remains `323.596` minutes. The final `Bearing3_4` point estimate is also late at `276.248` minutes, although zero remains inside that interval. Four other final held-out snapshots clip to a zero-minute point estimate. These examples are recorded for all six test bearings rather than hidden behind the aggregate score.

Machine-readable evidence is stored in:

- `artifacts/evaluation/xjtu_sy_rul_selection.json`;
- `artifacts/evaluation/xjtu_sy_rul_test.json`;
- `artifacts/models/vibralens_rul_v0_1.json`.

## Input and bundle contract

Each raw snapshot is UTF-8 CSV with this exact header:

```text
Horizontal_vibration_signals,Vertical_vibration_signals
```

It must contain exactly 32,768 finite numeric rows, two columns, sampled at 25,600 Hz. Inference also requires:

- `bearing_age_minutes`: finite and non-negative;
- `condition_id`: one of `1`, `2`, or `3`;
- `planned_break_minutes`: finite and non-negative.

The joblib bundle records its format and model versions, fitted estimator, ordered input schema, supported conditions, empirical interval radius, package versions, validation metrics, limitations, and SHA-256 fingerprints for the data/configuration/selection chain. Load joblib artifacts only from trusted sources.

## Decision rule

For interval `(pessimistic, median, optimistic)` and planned-break horizon `h`:

| Condition | Risk | Advisory |
|---|---|---|
| `h <= pessimistic` | `low` | `safe_to_wait` |
| `pessimistic < h <= optimistic` | `uncertain` | `inspect_first` |
| `h > optimistic` | `high` | `maintenance_urgent` |

An unsupported condition returns no RUL, `unknown` risk, and `inspect_first`.
