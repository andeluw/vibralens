"""Deterministic model selection and artifact training for XJTU-SY RUL."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vibralens.modeling.metrics import (
    evaluate_interval_predictions,
    evaluate_point_predictions,
)
from vibralens.modeling.baselines import AgeOnlyRegressor
from vibralens.modeling.table import (
    FeatureTable,
    ModelMatrix,
    build_model_matrix,
    load_feature_table,
)


class TrainingError(ValueError):
    """Raised when experiment configuration or training state is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    model_version: str
    random_state: int
    quantiles: Tuple[float, float, float]
    feature_sets: Tuple[str, ...]
    include_age_options: Tuple[bool, ...]
    ridge_alphas: Tuple[float, ...]
    minimum_macro_mae_improvement_fraction: float
    minimum_validation_bearings_improved: int
    empirical_interval_coverage: float
    gradient_boosting_candidates: Tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    estimator_family: str
    feature_set: str
    include_age: bool
    parameters: Mapping[str, object]
    validation_point_metrics: Mapping[str, object]
    validation_interval_metrics: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "estimator_family": self.estimator_family,
            "feature_set": self.feature_set,
            "include_age": self.include_age,
            "parameters": dict(self.parameters),
            "validation_point_metrics": dict(self.validation_point_metrics),
            "validation_interval_metrics": dict(self.validation_interval_metrics),
        }


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate the frozen experiment matrix."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingError("cannot read experiment config: {}".format(path)) from error

    try:
        quantiles = tuple(float(value) for value in payload["quantiles"])
        feature_sets = tuple(str(value) for value in payload["feature_sets"])
        include_age_options = tuple(
            bool(value) for value in payload["include_age_options"]
        )
        ridge_alphas = tuple(float(value) for value in payload["ridge_alphas"])
        candidates = tuple(dict(value) for value in payload["gradient_boosting_candidates"])
        config = ExperimentConfig(
            schema_version=int(payload["schema_version"]),
            model_version=str(payload["model_version"]),
            random_state=int(payload["random_state"]),
            quantiles=quantiles,  # type: ignore[arg-type]
            feature_sets=feature_sets,
            include_age_options=include_age_options,
            ridge_alphas=ridge_alphas,
            minimum_macro_mae_improvement_fraction=float(
                payload["minimum_macro_mae_improvement_fraction"]
            ),
            minimum_validation_bearings_improved=int(
                payload["minimum_validation_bearings_improved"]
            ),
            empirical_interval_coverage=float(payload["empirical_interval_coverage"]),
            gradient_boosting_candidates=candidates,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError("experiment config has invalid fields") from error

    if config.schema_version != 1:
        raise TrainingError("unsupported experiment schema version")
    if not config.model_version:
        raise TrainingError("model_version cannot be empty")
    if len(config.quantiles) != 3 or tuple(sorted(config.quantiles)) != config.quantiles:
        raise TrainingError("quantiles must contain three ordered values")
    if any(value <= 0.0 or value >= 1.0 for value in config.quantiles):
        raise TrainingError("quantiles must lie strictly between zero and one")
    if set(config.feature_sets) - {"horizontal", "vertical", "dual"}:
        raise TrainingError("feature_sets contains an unsupported value")
    if not config.feature_sets or not config.include_age_options:
        raise TrainingError("feature and age experiment options cannot be empty")
    if any(alpha <= 0.0 for alpha in config.ridge_alphas):
        raise TrainingError("ridge alphas must be positive")
    if not 0.0 <= config.minimum_macro_mae_improvement_fraction < 1.0:
        raise TrainingError("minimum improvement fraction is invalid")
    if config.minimum_validation_bearings_improved < 1:
        raise TrainingError("minimum improved bearing count must be positive")
    if not 0.0 < config.empirical_interval_coverage < 1.0:
        raise TrainingError("empirical interval coverage must lie between zero and one")
    if not config.gradient_boosting_candidates:
        raise TrainingError("gradient boosting candidates cannot be empty")
    return config


def build_preprocessor(
    number_of_columns: int,
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """One-hot encode condition and optionally scale remaining numeric columns."""
    if number_of_columns < 2:
        raise TrainingError("model matrices require condition plus numeric inputs")
    numeric_columns = list(range(1, number_of_columns))
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "condition",
                OneHotEncoder(
                    categories=[[1.0, 2.0, 3.0]],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                [0],
            ),
            ("numeric", numeric_transformer, numeric_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_ridge_pipeline(alpha: float, number_of_columns: int) -> Pipeline:
    """Construct the fixed Ridge baseline pipeline."""
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise TrainingError("Ridge alpha must be positive and finite")
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(number_of_columns, scale_numeric=True),
            ),
            ("regressor", Ridge(alpha=float(alpha))),
        ]
    )


def bearing_balanced_sample_weights(bearing_ids: np.ndarray) -> np.ndarray:
    """Give every physical bearing equal total weight during fitting."""
    bearings = np.asarray(bearing_ids, dtype=str)
    if bearings.ndim != 1 or bearings.size == 0:
        raise TrainingError("bearing_ids must be a non-empty vector")
    if any(not bearing_id for bearing_id in bearings):
        raise TrainingError("bearing_ids cannot contain empty values")
    counts = {
        bearing_id: int(np.sum(bearings == bearing_id))
        for bearing_id in set(bearings.tolist())
    }
    return np.asarray(
        [1.0 / counts[bearing_id] for bearing_id in bearings],
        dtype=np.float64,
    )


def bearing_balanced_quantile(
    values: np.ndarray,
    bearing_ids: np.ndarray,
    quantile: float,
) -> float:
    """Return a weighted quantile where every bearing has equal total weight."""
    numeric_values = np.asarray(values, dtype=np.float64)
    bearings = np.asarray(bearing_ids, dtype=str)
    if numeric_values.ndim != 1 or numeric_values.size == 0:
        raise TrainingError("values must be a non-empty vector")
    if bearings.ndim != 1 or bearings.size != numeric_values.size:
        raise TrainingError("bearing_ids must match values")
    if not np.isfinite(numeric_values).all():
        raise TrainingError("values contain non-finite entries")
    if not 0.0 < quantile < 1.0:
        raise TrainingError("quantile must lie strictly between zero and one")
    weights = bearing_balanced_sample_weights(bearings)
    order = np.argsort(numeric_values, kind="stable")
    sorted_values = numeric_values[order]
    cumulative_weights = np.cumsum(weights[order])
    target = quantile * float(np.sum(weights))
    index = min(
        int(np.searchsorted(cumulative_weights, target, side="left")),
        sorted_values.size - 1,
    )
    return float(sorted_values[index])


def _point_metrics_by_condition(
    matrix: ModelMatrix,
    predictions: np.ndarray,
) -> Dict[str, object]:
    report = evaluate_point_predictions(
        matrix.targets,
        predictions,
        matrix.bearing_ids,
        matrix.ages_minutes,
    )
    report["per_condition"] = {
        str(condition_id): evaluate_point_predictions(
            matrix.targets[matrix.condition_ids == condition_id],
            predictions[matrix.condition_ids == condition_id],
            matrix.bearing_ids[matrix.condition_ids == condition_id],
            matrix.ages_minutes[matrix.condition_ids == condition_id],
        )
        for condition_id in sorted(set(matrix.condition_ids.tolist()))
    }
    return report


def _interval_metrics_by_condition(
    matrix: ModelMatrix,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, object]:
    report = evaluate_interval_predictions(
        matrix.targets,
        lower,
        upper,
        matrix.bearing_ids,
    )
    report["per_condition"] = {
        str(condition_id): evaluate_interval_predictions(
            matrix.targets[matrix.condition_ids == condition_id],
            lower[matrix.condition_ids == condition_id],
            upper[matrix.condition_ids == condition_id],
            matrix.bearing_ids[matrix.condition_ids == condition_id],
        )
        for condition_id in sorted(set(matrix.condition_ids.tolist()))
    }
    return report


def select_ridge_candidate(
    table: FeatureTable,
    config: ExperimentConfig,
) -> Tuple[CandidateResult, List[CandidateResult]]:
    """Evaluate the fixed Ridge grid on validation bearings."""
    train_mask = table.splits == "train"
    validation_mask = table.splits == "validation"
    results: List[CandidateResult] = []
    for feature_set in config.feature_sets:
        for include_age in config.include_age_options:
            train = build_model_matrix(
                table,
                train_mask,
                feature_set=feature_set,
                include_age=include_age,
            )
            validation = build_model_matrix(
                table,
                validation_mask,
                feature_set=feature_set,
                include_age=include_age,
            )
            if train.feature_names != validation.feature_names:
                raise TrainingError("train and validation feature schemas differ")
            for alpha in config.ridge_alphas:
                pipeline = build_ridge_pipeline(alpha, train.values.shape[1])
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    pipeline.fit(
                        train.values,
                        train.targets,
                        regressor__sample_weight=bearing_balanced_sample_weights(
                            train.bearing_ids
                        ),
                    )
                    predictions = np.maximum(
                        np.asarray(
                            pipeline.predict(validation.values),
                            dtype=np.float64,
                        ),
                        0.0,
                    )
                if not np.isfinite(predictions).all():
                    raise TrainingError(
                        "Ridge candidate {} produced non-finite predictions".format(
                            alpha
                        )
                    )
                result = CandidateResult(
                    candidate_id="ridge-{}-age{}-alpha{}".format(
                        feature_set,
                        int(include_age),
                        format(alpha, "g"),
                    ),
                    estimator_family="ridge",
                    feature_set=feature_set,
                    include_age=include_age,
                    parameters={"alpha": float(alpha)},
                    validation_point_metrics=_point_metrics_by_condition(
                        validation,
                        predictions,
                    ),
                    validation_interval_metrics={},
                )
                results.append(result)
    if not results:
        raise TrainingError("Ridge experiment matrix produced no candidates")
    ordered = sorted(
        results,
        key=lambda result: (
            float(result.validation_point_metrics["macro_bearing_mae"]),
            result.candidate_id,
        ),
    )
    return ordered[0], ordered


def calibrate_ridge_candidate(
    table: FeatureTable,
    ridge_result: CandidateResult,
    *,
    empirical_interval_coverage: float,
) -> CandidateResult:
    """Attach a bearing-balanced empirical residual interval to a Ridge result."""
    try:
        alpha = float(ridge_result.parameters["alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError("Ridge result is missing alpha") from error
    train = build_model_matrix(
        table,
        table.splits == "train",
        feature_set=ridge_result.feature_set,
        include_age=ridge_result.include_age,
    )
    validation = build_model_matrix(
        table,
        table.splits == "validation",
        feature_set=ridge_result.feature_set,
        include_age=ridge_result.include_age,
    )
    pipeline = build_ridge_pipeline(alpha, train.values.shape[1])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pipeline.fit(
            train.values,
            train.targets,
            regressor__sample_weight=bearing_balanced_sample_weights(
                train.bearing_ids
            ),
        )
        predictions = np.maximum(
            np.asarray(pipeline.predict(validation.values), dtype=np.float64),
            0.0,
        )
    if not np.isfinite(predictions).all():
        raise TrainingError("Ridge candidate produced non-finite validation predictions")
    residuals = np.abs(validation.targets - predictions)
    radius = bearing_balanced_quantile(
        residuals,
        validation.bearing_ids,
        empirical_interval_coverage,
    )
    lower = np.maximum(predictions - radius, 0.0)
    upper = predictions + radius
    return CandidateResult(
        candidate_id="{}-empirical{}".format(
            ridge_result.candidate_id,
            int(round(empirical_interval_coverage * 100)),
        ),
        estimator_family="ridge_empirical_interval",
        feature_set=ridge_result.feature_set,
        include_age=ridge_result.include_age,
        parameters={
            **dict(ridge_result.parameters),
            "empirical_interval_coverage": empirical_interval_coverage,
            "interval_radius_minutes": radius,
            "sample_weighting": "equal_total_weight_per_bearing",
        },
        validation_point_metrics=_point_metrics_by_condition(
            validation,
            predictions,
        ),
        validation_interval_metrics=_interval_metrics_by_condition(
            validation,
            lower,
            upper,
        ),
    )


def build_quantile_pipeline(
    *,
    quantile: float,
    candidate: Mapping[str, object],
    random_state: int,
    number_of_columns: int,
) -> Pipeline:
    """Construct one fitted-preprocessing quantile boosting pipeline."""
    try:
        regressor = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(quantile),
            learning_rate=float(candidate["learning_rate"]),
            max_iter=int(candidate["max_iter"]),
            max_leaf_nodes=int(candidate["max_leaf_nodes"]),
            min_samples_leaf=int(candidate["min_samples_leaf"]),
            l2_regularization=float(candidate["l2_regularization"]),
            random_state=int(random_state),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError("invalid gradient boosting candidate") from error
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(number_of_columns, scale_numeric=False),
            ),
            ("regressor", regressor),
        ]
    )


def _quantile_sort_key(result: CandidateResult) -> Tuple[float, float, float, str]:
    return (
        float(result.validation_point_metrics["macro_bearing_mae"]),
        abs(
            float(result.validation_interval_metrics["macro_bearing_coverage"])
            - 0.80
        ),
        float(result.validation_interval_metrics["mean_width_minutes"]),
        result.candidate_id,
    )


def select_quantile_candidate(
    table: FeatureTable,
    config: ExperimentConfig,
) -> Tuple[CandidateResult, List[CandidateResult]]:
    """Evaluate every configured quantile boosting candidate on validation bearings."""
    train_mask = table.splits == "train"
    validation_mask = table.splits == "validation"
    results: List[CandidateResult] = []
    for feature_set in config.feature_sets:
        for include_age in config.include_age_options:
            train = build_model_matrix(
                table,
                train_mask,
                feature_set=feature_set,
                include_age=include_age,
            )
            validation = build_model_matrix(
                table,
                validation_mask,
                feature_set=feature_set,
                include_age=include_age,
            )
            for candidate_index, candidate in enumerate(
                config.gradient_boosting_candidates,
                start=1,
            ):
                raw_predictions = []
                for quantile in config.quantiles:
                    pipeline = build_quantile_pipeline(
                        quantile=quantile,
                        candidate=candidate,
                        random_state=config.random_state,
                        number_of_columns=train.values.shape[1],
                    )
                    pipeline.fit(
                        train.values,
                        train.targets,
                        regressor__sample_weight=bearing_balanced_sample_weights(
                            train.bearing_ids
                        ),
                    )
                    raw_predictions.append(pipeline.predict(validation.values))
                lower, median, upper = postprocess_quantiles(*raw_predictions)
                result = CandidateResult(
                    candidate_id="hgbq-{}-age{}-candidate{}".format(
                        feature_set,
                        int(include_age),
                        candidate_index,
                    ),
                    estimator_family="hist_gradient_boosting_quantile",
                    feature_set=feature_set,
                    include_age=include_age,
                    parameters={
                        **dict(candidate),
                        "quantiles": list(config.quantiles),
                    },
                    validation_point_metrics=_point_metrics_by_condition(
                        validation,
                        median,
                    ),
                    validation_interval_metrics=_interval_metrics_by_condition(
                        validation,
                        lower,
                        upper,
                    ),
                )
                results.append(result)
    if not results:
        raise TrainingError("quantile experiment matrix produced no candidates")
    ordered = sorted(results, key=_quantile_sort_key)
    return ordered[0], ordered


def postprocess_quantiles(
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make three predicted quantile arrays finite, non-negative, and ordered."""
    arrays = [np.asarray(values, dtype=np.float64) for values in (lower, median, upper)]
    if any(array.ndim != 1 for array in arrays):
        raise TrainingError("quantile predictions must be one-dimensional")
    if len({array.size for array in arrays}) != 1 or arrays[0].size == 0:
        raise TrainingError("quantile predictions must have matching non-empty shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise TrainingError("quantile predictions contain non-finite values")
    ordered = np.maximum(np.sort(np.vstack(arrays), axis=0), 0.0)
    return ordered[0], ordered[1], ordered[2]


def passes_vibration_gate(
    age_only_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    *,
    minimum_improvement_fraction: float,
    minimum_bearings_improved: int,
) -> Dict[str, object]:
    """Evaluate the predeclared macro and per-bearing advancement gate."""
    try:
        age_macro = float(age_only_metrics["macro_bearing_mae"])
        candidate_macro = float(candidate_metrics["macro_bearing_mae"])
        age_per_bearing = age_only_metrics["per_bearing"]
        candidate_per_bearing = candidate_metrics["per_bearing"]
        if not isinstance(age_per_bearing, Mapping) or not isinstance(
            candidate_per_bearing, Mapping
        ):
            raise TypeError
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError("gate metrics are incomplete") from error
    if not np.isfinite(age_macro) or not np.isfinite(candidate_macro) or age_macro < 0.0:
        raise TrainingError("gate metrics must be finite and non-negative")
    if not 0.0 <= minimum_improvement_fraction < 1.0:
        raise TrainingError("minimum_improvement_fraction is invalid")
    if minimum_bearings_improved < 1:
        raise TrainingError("minimum_bearings_improved must be positive")
    if set(age_per_bearing) != set(candidate_per_bearing):
        raise TrainingError("gate reports must cover identical bearings")

    bearings_improved = 0
    for bearing_id in age_per_bearing:
        try:
            age_mae = float(age_per_bearing[bearing_id]["mae_minutes"])  # type: ignore[index]
            candidate_mae = float(
                candidate_per_bearing[bearing_id]["mae_minutes"]  # type: ignore[index]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingError("per-bearing gate metrics are incomplete") from error
        if candidate_mae < age_mae:
            bearings_improved += 1

    improvement_fraction = (
        0.0 if age_macro == 0.0 else (age_macro - candidate_macro) / age_macro
    )
    macro_passed = improvement_fraction >= minimum_improvement_fraction
    bearing_passed = bearings_improved >= minimum_bearings_improved
    return {
        "passed": bool(macro_passed and bearing_passed),
        "macro_passed": bool(macro_passed),
        "bearing_count_passed": bool(bearing_passed),
        "age_only_macro_bearing_mae": age_macro,
        "candidate_macro_bearing_mae": candidate_macro,
        "macro_improvement_fraction": float(improvement_fraction),
        "minimum_macro_improvement_fraction": float(minimum_improvement_fraction),
        "bearings_improved": bearings_improved,
        "minimum_bearings_improved": minimum_bearings_improved,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise TrainingError("cannot fingerprint {}".format(path)) from error
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingError("cannot read {}: {}".format(description, path)) from error
    if not isinstance(payload, Mapping):
        raise TrainingError("{} must contain a JSON object".format(description))
    return payload


def _age_only_validation_metrics(table: FeatureTable) -> Dict[str, object]:
    train_mask = table.splits == "train"
    validation_mask = table.splits == "validation"
    model = AgeOnlyRegressor().fit(
        table.age_minutes[train_mask],
        table.condition_ids[train_mask],
        table.rul_minutes[train_mask],
        table.bearing_ids[train_mask],
    )
    predictions = model.predict(
        table.age_minutes[validation_mask],
        table.condition_ids[validation_mask],
    )
    validation = ModelMatrix(
        values=np.empty((int(np.sum(validation_mask)), 0), dtype=np.float64),
        targets=table.rul_minutes[validation_mask],
        bearing_ids=table.bearing_ids[validation_mask],
        condition_ids=table.condition_ids[validation_mask],
        ages_minutes=table.age_minutes[validation_mask],
        feature_names=(),
    )
    return _point_metrics_by_condition(validation, predictions)


def run_selection(
    feature_path: Path,
    feature_audit_path: Path,
    manifest_path: Path,
    config_path: Path,
) -> Dict[str, object]:
    """Run validation-only model selection and return a reproducible report."""
    feature_path = Path(feature_path)
    feature_audit_path = Path(feature_audit_path)
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    feature_digest = _sha256(feature_path)
    audit = _read_json(feature_audit_path, "feature audit")
    recorded_digest = audit.get("feature_artifact_sha256")
    if recorded_digest is not None and recorded_digest != feature_digest:
        raise TrainingError("feature artifact digest disagrees with feature audit")
    if audit.get("status") not in (None, "passed"):
        raise TrainingError("feature audit does not have passed status")

    config = load_experiment_config(config_path)
    table = load_feature_table(feature_path)
    split_names = set(table.splits.tolist())
    if "train" not in split_names or "validation" not in split_names:
        raise TrainingError("feature table requires train and validation rows")

    age_metrics = _age_only_validation_metrics(table)
    ridge_best, ridge_results = select_ridge_candidate(table, config)
    quantile_best, quantile_results = select_quantile_candidate(table, config)
    selected = calibrate_ridge_candidate(
        table,
        ridge_best,
        empirical_interval_coverage=config.empirical_interval_coverage,
    )
    gate = passes_vibration_gate(
        age_metrics,
        selected.validation_point_metrics,
        minimum_improvement_fraction=config.minimum_macro_mae_improvement_fraction,
        minimum_bearings_improved=config.minimum_validation_bearings_improved,
    )

    return {
        "schema_version": 1,
        "model_version": config.model_version,
        "selection_metric": "macro_bearing_mae",
        "test_metrics_status": "not_evaluated",
        "fingerprints": {
            "features_sha256": feature_digest,
            "feature_audit_sha256": _sha256(feature_audit_path),
            "manifest_sha256": _sha256(manifest_path),
            "config_sha256": _sha256(config_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "age_only_validation": age_metrics,
        "ridge_baseline_selected": ridge_best.to_dict(),
        "ridge_candidates": [result.to_dict() for result in ridge_results],
        "selected_candidate": selected.to_dict(),
        "quantile_comparison_selected": quantile_best.to_dict(),
        "quantile_candidates": [result.to_dict() for result in quantile_results],
        "vibration_gate": gate,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and verify VibraLens RUL models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser(
        "select",
        help="Run validation-only candidate selection.",
    )
    select_parser.add_argument("--features", required=True, type=Path)
    select_parser.add_argument("--feature-audit", required=True, type=Path)
    select_parser.add_argument("--manifest", required=True, type=Path)
    select_parser.add_argument("--config", required=True, type=Path)
    select_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    if args.command == "select":
        report = run_selection(
            args.features,
            args.feature_audit,
            args.manifest,
            args.config,
        )
        _write_json(args.output, report)
        print(
            "Validation selection written to {}; passed={}".format(
                args.output,
                report["vibration_gate"]["passed"],  # type: ignore[index]
            )
        )
        return 0
    raise TrainingError("unsupported training command")


if __name__ == "__main__":
    raise SystemExit(main())
