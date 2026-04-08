from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MaxAbsScaler

from train_next_hour_regression import (
    BINARY_FIELDS,
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    TARGETS,
    inverse_log1p_target,
    load_rows,
    mean_absolute_error,
    median_absolute_error,
    parse_flag,
    parse_float,
    r2_score,
    root_mean_squared_error,
    safe_log1p_target,
    split_rows,
    target_to_naive_prediction,
    write_csv,
    write_json,
)

MIN_SEGMENT_TRAIN_ROWS = 500
MIN_SEGMENT_TEST_ROWS = 100
MAX_LOG_TARGET_PREDICTION = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train subreddit-specific gradient-descent regression baselines from the "
            "model-ready next-hour prediction table."
        )
    )
    parser.add_argument(
        "--input",
        default="data/models/reddit/prediction_next_hour.csv",
        help="Model-ready next-hour prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/reddit/sgd_by_subreddit",
        help="Directory where metrics and coefficient summaries will be written.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGETS),
        default=["upvotes", "comments"],
        help="Which next-hour targets to model.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological training fraction. The remainder is used as test data.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Regularization strength for SGDRegressor.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.15,
        help="Elastic-net blend for SGDRegressor.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum SGD iterations.",
    )
    parser.add_argument(
        "--eta0",
        type=float,
        default=0.0005,
        help="Initial learning rate for SGD.",
    )
    return parser.parse_args()


def row_to_feature_dict(row: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        payload[f"num:{field}"] = parse_float(row.get(field), 0.0)
    for field in BINARY_FIELDS:
        payload[f"bin:{field}"] = parse_flag(row.get(field))
    for field in CATEGORICAL_FIELDS:
        value = str(row.get(field, "") or "").strip().lower()
        payload[f"cat:{field}"] = value or "__none__"
    return payload


def rows_by_subreddit(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        subreddit = str(row.get("subreddit", "") or "").strip().lower()
        if subreddit:
            grouped[subreddit].append(row)
    return dict(grouped)


def safe_inverse_prediction(value: float) -> float:
    clipped = max(0.0, min(float(value), MAX_LOG_TARGET_PREDICTION))
    return inverse_log1p_target(clipped)


def fit_segment_model(
    *,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    target_name: str,
    target_column: str,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    eta0: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_feature_dicts = [row_to_feature_dict(row) for row in train_rows]
    test_feature_dicts = [row_to_feature_dict(row) for row in test_rows]

    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform(train_feature_dicts)
    test_matrix = vectorizer.transform(test_feature_dicts)

    scaler = MaxAbsScaler()
    train_matrix = scaler.fit_transform(train_matrix)
    test_matrix = scaler.transform(test_matrix)

    train_targets_raw = [max(0.0, parse_float(row.get(target_column), 0.0)) for row in train_rows]
    test_targets_raw = [max(0.0, parse_float(row.get(target_column), 0.0)) for row in test_rows]
    train_targets = [safe_log1p_target(value) for value in train_targets_raw]

    model = SGDRegressor(
        loss="huber",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        learning_rate="invscaling",
        eta0=eta0,
        power_t=0.2,
        random_state=42,
        average=True,
    )
    model.fit(train_matrix, train_targets)

    predicted_targets = [
        safe_inverse_prediction(value)
        for value in model.predict(test_matrix)
    ]
    naive_predictions = [target_to_naive_prediction(row, target_name) for row in test_rows]

    metrics = {
        "target_column": target_column,
        "model": "sklearn_sgd_log1p_subreddit_specific",
        "test_mae": mean_absolute_error(test_targets_raw, predicted_targets),
        "test_rmse": root_mean_squared_error(test_targets_raw, predicted_targets),
        "test_median_ae": median_absolute_error(test_targets_raw, predicted_targets),
        "test_r2": r2_score(test_targets_raw, predicted_targets),
        "naive_mae": mean_absolute_error(test_targets_raw, naive_predictions),
        "naive_rmse": root_mean_squared_error(test_targets_raw, naive_predictions),
        "naive_median_ae": median_absolute_error(test_targets_raw, naive_predictions),
        "naive_r2": r2_score(test_targets_raw, naive_predictions),
        "feature_count": len(vectorizer.get_feature_names_out()),
        "iterations_run": int(model.n_iter_),
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "eta0": eta0,
    }

    coefficient_rows = [
        {
            "feature": feature,
            "coefficient": float(coefficient),
            "abs_coefficient": abs(float(coefficient)),
            "target": target_name,
        }
        for feature, coefficient in zip(vectorizer.get_feature_names_out(), model.coef_)
    ]
    coefficient_rows.sort(key=lambda row: (-row["abs_coefficient"], row["feature"]))

    return metrics, coefficient_rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    train_rows, test_rows = split_rows(rows, args.train_fraction)
    train_rows_by_subreddit = rows_by_subreddit(train_rows)
    test_rows_by_subreddit = rows_by_subreddit(test_rows)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "targets": args.targets,
        "subreddit_segments": {},
    }
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    all_subreddits = sorted(set(train_rows_by_subreddit) | set(test_rows_by_subreddit))
    for subreddit in all_subreddits:
        subreddit_train_rows = train_rows_by_subreddit.get(subreddit, [])
        subreddit_test_rows = test_rows_by_subreddit.get(subreddit, [])
        if (
            len(subreddit_train_rows) < MIN_SEGMENT_TRAIN_ROWS
            or len(subreddit_test_rows) < MIN_SEGMENT_TEST_ROWS
        ):
            continue

        subreddit_payload: dict[str, Any] = {
            "train_rows": len(subreddit_train_rows),
            "test_rows": len(subreddit_test_rows),
            "targets": {},
        }
        for target_name in args.targets:
            target_column = TARGETS[target_name]
            metrics, top_coefficients = fit_segment_model(
                train_rows=subreddit_train_rows,
                test_rows=subreddit_test_rows,
                target_name=target_name,
                target_column=target_column,
                alpha=args.alpha,
                l1_ratio=args.l1_ratio,
                max_iter=args.max_iter,
                eta0=args.eta0,
            )
            metrics["train_rows"] = len(subreddit_train_rows)
            metrics["test_rows"] = len(subreddit_test_rows)
            subreddit_payload["targets"][target_name] = metrics
            metric_rows.append(
                {
                    "subreddit": subreddit,
                    "target": target_name,
                    "train_rows": len(subreddit_train_rows),
                    "test_rows": len(subreddit_test_rows),
                    "test_r2": metrics["test_r2"],
                    "test_mae": metrics["test_mae"],
                    "test_rmse": metrics["test_rmse"],
                    "naive_r2": metrics["naive_r2"],
                    "naive_mae": metrics["naive_mae"],
                    "naive_rmse": metrics["naive_rmse"],
                    "iterations_run": metrics["iterations_run"],
                    "feature_count": metrics["feature_count"],
                }
            )
            for rank, row in enumerate(top_coefficients[:25], start=1):
                coefficient_rows.append(
                    {
                        "subreddit": subreddit,
                        "target": target_name,
                        "feature_rank": rank,
                        "feature": row["feature"],
                        "coefficient": row["coefficient"],
                        "abs_coefficient": row["abs_coefficient"],
                    }
                )

        payload["subreddit_segments"][subreddit] = subreddit_payload

    write_json(output_dir / "sgd_metrics.json", payload)
    write_csv(
        output_dir / "sgd_subreddit_metrics.csv",
        [
            "subreddit",
            "target",
            "train_rows",
            "test_rows",
            "test_r2",
            "test_mae",
            "test_rmse",
            "naive_r2",
            "naive_mae",
            "naive_rmse",
            "iterations_run",
            "feature_count",
        ],
        metric_rows,
    )
    write_csv(
        output_dir / "sgd_subreddit_coefficients.csv",
        ["subreddit", "target", "feature_rank", "feature", "coefficient", "abs_coefficient"],
        coefficient_rows,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
