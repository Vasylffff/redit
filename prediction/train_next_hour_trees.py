from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train tree-based next-hour prediction baselines from the model-ready "
            "prediction table using scikit-learn."
        )
    )
    parser.add_argument(
        "--input",
        default="data/models/reddit/prediction_next_hour.csv",
        help="Model-ready next-hour prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/reddit/trees",
        help="Directory where metrics, importances, and test predictions will be written.",
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
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the ensemble.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=18,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Tree max_features setting.",
    )
    parser.add_argument(
        "--segment-by-subreddit",
        action="store_true",
        help="Also train separate tree benchmarks for each subreddit.",
    )
    parser.add_argument(
        "--segments-only",
        action="store_true",
        help="Skip the global benchmark and run only the segmented subreddit models.",
    )
    return parser.parse_args()


@dataclass
class TreeFeatureMatrix:
    vectorizer: DictVectorizer
    feature_names: list[str]


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


def build_train_matrix(rows: list[dict[str, str]]) -> tuple[Any, TreeFeatureMatrix]:
    feature_dicts = [row_to_feature_dict(row) for row in rows]
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_dicts)
    return matrix, TreeFeatureMatrix(
        vectorizer=vectorizer,
        feature_names=list(vectorizer.get_feature_names_out()),
    )


def transform_rows(rows: list[dict[str, str]], matrix_spec: TreeFeatureMatrix) -> Any:
    feature_dicts = [row_to_feature_dict(row) for row in rows]
    return matrix_spec.vectorizer.transform(feature_dicts)


def train_single_target(
    *,
    target_name: str,
    target_column: str,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_matrix, matrix_spec = build_train_matrix(train_rows)
    test_matrix = transform_rows(test_rows, matrix_spec)

    train_targets_raw = [parse_float(row.get(target_column), 0.0) for row in train_rows]
    test_targets_raw = [parse_float(row.get(target_column), 0.0) for row in test_rows]
    train_targets = [safe_log1p_target(value) for value in train_targets_raw]

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=1,
    )
    model.fit(train_matrix, train_targets)

    predicted_log_targets = model.predict(test_matrix)
    predicted_targets = [inverse_log1p_target(value) for value in predicted_log_targets]
    naive_predictions = [target_to_naive_prediction(row, target_name) for row in test_rows]

    metrics = {
        "target_column": target_column,
        "model": "sklearn_extra_trees_log1p_global",
        "test_mae": mean_absolute_error(test_targets_raw, predicted_targets),
        "test_rmse": root_mean_squared_error(test_targets_raw, predicted_targets),
        "test_median_ae": median_absolute_error(test_targets_raw, predicted_targets),
        "test_r2": r2_score(test_targets_raw, predicted_targets),
        "naive_mae": mean_absolute_error(test_targets_raw, naive_predictions),
        "naive_rmse": root_mean_squared_error(test_targets_raw, naive_predictions),
        "naive_median_ae": median_absolute_error(test_targets_raw, naive_predictions),
        "naive_r2": r2_score(test_targets_raw, naive_predictions),
        "feature_count": len(matrix_spec.feature_names),
        "tree_params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        },
    }

    importances = sorted(
        zip(matrix_spec.feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    importance_rows = [
        {
            "feature": feature,
            "importance": importance,
            "target": target_name,
        }
        for feature, importance in importances
    ]

    prediction_rows: list[dict[str, Any]] = []
    for row, actual, predicted, naive in zip(test_rows, test_targets_raw, predicted_targets, naive_predictions):
        prediction_rows.append(
            {
                "snapshot_time_utc": row.get("snapshot_time_utc"),
                "subreddit": row.get("subreddit"),
                "post_id": row.get("post_id"),
                "title": row.get("title"),
                "age_bucket": row.get("age_bucket"),
                "activity_state": row.get("activity_state"),
                "listing_type": row.get("listing_type"),
                "target": target_name,
                "actual_next_delta": actual,
                "predicted_next_delta": predicted,
                "naive_next_delta": naive,
            }
        )

    return metrics, importance_rows, prediction_rows


def rows_by_subreddit(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        subreddit = str(row.get("subreddit", "") or "").strip().lower()
        if subreddit:
            grouped[subreddit].append(row)
    return dict(grouped)


def main() -> None:
    args = parse_args()
    if args.segments_only and not args.segment_by_subreddit:
        raise SystemExit("--segments-only requires --segment-by-subreddit.")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    train_rows, test_rows = split_rows(rows, args.train_fraction)

    metrics_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "targets": {},
    }

    if not args.segments_only:
        for target_name in args.targets:
            target_column = TARGETS[target_name]
            metrics, importance_rows, prediction_rows = train_single_target(
                target_name=target_name,
                target_column=target_column,
                train_rows=train_rows,
                test_rows=test_rows,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
            )
            metrics_payload["targets"][target_name] = metrics

            importance_path = output_dir / f"trees_{target_name}_feature_importance.csv"
            predictions_path = output_dir / f"trees_{target_name}_test_predictions.csv"
            write_csv(importance_path, list(importance_rows[0].keys()) if importance_rows else ["feature", "importance", "target"], importance_rows)
            write_csv(predictions_path, list(prediction_rows[0].keys()) if prediction_rows else ["snapshot_time_utc"], prediction_rows)

    if args.segment_by_subreddit:
        segmented_payload: dict[str, Any] = {}
        segmented_metric_rows: list[dict[str, Any]] = []
        segmented_importance_rows: list[dict[str, Any]] = []
        train_rows_by_subreddit = rows_by_subreddit(train_rows)
        test_rows_by_subreddit = rows_by_subreddit(test_rows)
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
                metrics, importance_rows, _prediction_rows = train_single_target(
                    target_name=target_name,
                    target_column=target_column,
                    train_rows=subreddit_train_rows,
                    test_rows=subreddit_test_rows,
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features=args.max_features,
                )
                metrics["model"] = "sklearn_extra_trees_log1p_subreddit_specific"
                metrics["train_rows"] = len(subreddit_train_rows)
                metrics["test_rows"] = len(subreddit_test_rows)
                subreddit_payload["targets"][target_name] = metrics
                segmented_metric_rows.append(
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
                        "feature_count": metrics["feature_count"],
                    }
                )
                for rank, importance_row in enumerate(importance_rows[:25], start=1):
                    segmented_importance_rows.append(
                        {
                            "subreddit": subreddit,
                            "target": target_name,
                            "feature_rank": rank,
                            "feature": importance_row["feature"],
                            "importance": importance_row["importance"],
                        }
                    )

            segmented_payload[subreddit] = subreddit_payload

        metrics_payload["subreddit_segments"] = segmented_payload
        write_csv(
            output_dir / "trees_subreddit_metrics.csv",
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
                "feature_count",
            ],
            segmented_metric_rows,
        )
        write_csv(
            output_dir / "trees_subreddit_feature_importance.csv",
            ["subreddit", "target", "feature_rank", "feature", "importance"],
            segmented_importance_rows,
        )

    metrics_path = output_dir / "trees_metrics.json"
    write_json(metrics_path, metrics_payload)
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
