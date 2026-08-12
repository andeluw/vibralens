# Vibration feature protocol

## Purpose

This stage converts every two-channel XJTU-SY vibration snapshot into a compact, interpretable feature row for the first leakage-safe baseline. The transformation is deterministic and does not learn parameters from any split.

Feature definitions were fixed before evaluating a model. Extracting features from test snapshots is required for eventual inference and does not fit or tune anything; test labels and test performance must remain unused during model selection.

## Input contract

Each snapshot must contain:

- the exact horizontal/vertical CSV header;
- exactly two numeric columns;
- exactly the row count declared by the manifest;
- only finite numeric values;
- the sampling rate declared by the manifest.

Any violation stops extraction rather than silently imputing or dropping values.

## Features

The same 14 features are calculated independently for horizontal and vertical vibration, producing 28 columns per snapshot.

### Time domain

- mean;
- population standard deviation;
- root mean square;
- absolute peak;
- peak-to-peak amplitude;
- population skewness;
- Pearson kurtosis, where a Gaussian signal is approximately 3;
- crest factor: absolute peak divided by RMS.

Zero-variance signals receive zero skewness, kurtosis, and crest factor rather than undefined values.

### Frequency domain

The spectrum is calculated with a real FFT after subtracting the snapshot mean. Power is the squared FFT magnitude.

- dominant frequency;
- power-weighted spectral centroid;
- spectral entropy normalized by the number of FFT bins;
- relative energy below 1 kHz;
- relative energy from 1 kHz to below 5 kHz;
- relative energy from 5 kHz through the 12.8 kHz Nyquist frequency.

The three band-energy fractions sum to one for every non-constant signal.

## Reproduction

```bash
uv run vibralens-extract-features \
  --dataset-root ../XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets \
  --manifest artifacts/data/xjtu_sy_manifest.csv \
  --output-directory artifacts/features \
  --workers 8
```

The generated feature CSV is intentionally ignored by Git because it is derived from the external raw dataset. The small audit JSON is retained as reproducibility evidence.

## Verified artifact

The full local run produced:

- 9,216 snapshot rows;
- 15 bearings;
- 28 feature columns;
- zero non-finite raw or feature values;
- train/validation/test counts of 5,970 / 1,062 / 2,184;
- a SHA-256 digest stored in `artifacts/features/xjtu_sy_feature_audit.json`.

Independent checks enforce non-negative magnitude features, frequencies in `[0, 12,800]` Hz, entropy in `[0, 1]`, and per-channel energy fractions summing to one.

## Modeling boundary

No normalization, feature selection, threshold fitting, or dimensionality reduction occurs here. Any such operation must be fitted using training bearings only inside the model pipeline.

The first baseline should use this feature table without adding overlapping temporal windows. Temporal models can be compared later after the snapshot baseline establishes an honest reference point.
