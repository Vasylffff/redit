from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NUMERIC_FIELDS = [
    "title_length_chars",
    "body_length_chars",
    "title_word_count",
    "body_word_count",
    "title_uppercase_ratio",
    "age_hours_at_snapshot",
    "rank_within_snapshot",
    "previous_rank_within_snapshot",
    "rank_change_from_previous_snapshot",
    "abs_rank_change_from_previous_snapshot",
    "upvotes_at_snapshot",
    "comment_count_at_snapshot",
    "upvote_ratio_at_snapshot",
    "upvote_velocity_per_hour",
    "comment_velocity_per_hour",
    "hours_since_previous_snapshot",
    "previous_gap_ratio_to_expected",
    "next_gap_ratio_to_expected",
    "previous_to_next_gap_ratio",
    "expected_snapshot_gap_hours",
    "max_gap_hours_seen",
    "irregular_gap_count",
    "large_gap_count",
    "regular_gap_share",
    "previous_upvote_velocity_per_hour",
    "previous_comment_velocity_per_hour",
    "previous_upvote_delta_from_previous_snapshot",
    "previous_comment_delta_from_previous_snapshot",
    "upvote_velocity_change_from_previous",
    "comment_velocity_change_from_previous",
    "upvote_velocity_acceleration_per_hour2",
    "comment_velocity_acceleration_per_hour2",
    "upvote_velocity_ratio_to_previous",
    "comment_velocity_ratio_to_previous",
    "upvote_velocity_ratio_to_prior_peak",
    "comment_velocity_ratio_to_prior_peak",
    "recent_upvote_velocity_mean_2",
    "recent_comment_velocity_mean_2",
    "upvote_velocity_ratio_to_recent_mean_2",
    "comment_velocity_ratio_to_recent_mean_2",
    "alive_upvote_velocity_threshold",
    "alive_comment_velocity_threshold",
    "surging_upvote_velocity_threshold",
    "surging_comment_velocity_threshold",
    "dead_upvote_velocity_threshold",
    "dead_comment_velocity_threshold",
    "hours_since_first_seen_snapshot",
    "snapshot_hour_utc",
    "sequence_position",
    "night_snapshot_count",
    "day_snapshot_count",
    "night_snapshot_share",
    "prior_same_title_posts_24h_subreddit",
    "prior_same_story_posts_24h_subreddit",
    "listing_run_length_snapshots",
    "hours_in_current_listing",
    "comment_sample_count",
    "comment_sample_coverage_ratio",
    "top_level_comment_sample_count",
    "reply_comment_sample_count",
    "top_level_comment_sample_share",
    "unique_commenter_count_sample",
    "deleted_comment_share_sample",
    "question_comment_share_sample",
    "avg_comment_upvotes_sample",
    "max_comment_upvotes_sample",
    "avg_comment_body_word_count_sample",
    "max_comment_body_word_count_sample",
    "newest_comment_age_minutes_sample",
    "oldest_comment_age_minutes_sample",
    "comment_sample_span_minutes",
]

CATEGORICAL_FIELDS = [
    "subreddit",
    "listing_type",
    "previous_listing_type",
    "listing_transition",
    "activity_state",
    "analysis_priority",
    "age_bucket",
    "link_domain_category",
    "content_topic_primary",
    "previous_gap_bucket",
    "next_gap_bucket",
]

BINARY_FIELDS = [
    "is_video",
    "has_images",
    "is_fresh_1h",
    "is_fresh_6h",
    "is_old_24h",
    "is_night_snapshot_utc",
    "listing_changed_from_previous",
    "rank_improved_from_previous",
    "rank_worsened_from_previous",
    "rank_unchanged_from_previous",
    "title_has_question_mark",
    "title_has_exclamation_mark",
    "title_has_colon",
    "title_has_quotes",
    "title_has_number",
    "title_starts_with_question_word",
    "content_has_breaking_word",
    "content_has_update_word",
    "content_has_live_word",
    "content_has_trailer_word",
    "content_has_review_word",
    "content_has_guide_word",
    "content_has_leak_word",
    "content_has_ama_word",
    "content_has_analysis_word",
    "content_has_report_word",
    "same_title_seen_before_24h_subreddit",
    "same_story_seen_before_24h_subreddit",
    "has_comment_sample",
    "op_replied_in_comment_sample",
    "previous_gap_is_regular",
    "next_gap_is_regular",
    "previous_gap_is_large",
    "next_gap_is_large",
]

TARGETS = {
    "upvotes": "upvote_delta_next_snapshot",
    "comments": "comment_delta_next_snapshot",
}
MIN_BUCKET_TRAIN_ROWS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train simple next-hour ridge regression baselines from the model-ready "
            "prediction table without external ML libraries."
        )
    )
    parser.add_argument(
        "--input",
        default="data/models/reddit/prediction_next_hour.csv",
        help="Model-ready next-hour prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/reddit/regression",
        help="Directory where metrics, coefficients, and test predictions will be written.",
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
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization strength applied to non-bias coefficients.",
    )
    parser.add_argument(
        "--max-category-values",
        type=int,
        default=12,
        help="Cap the number of one-hot values kept per categorical field.",
    )
    return parser.parse_args()


def parse_float(value: Any, default: float = 0.0) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_flag(value: Any) -> float:
    return 1.0 if str(value or "").strip() == "1" else 0.0


def parse_dt(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def safe_log1p_target(value: float) -> float:
    return math.log1p(max(0.0, value))


def inverse_log1p_target(value: float) -> float:
    return max(0.0, math.expm1(value))


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def median_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    errors = sorted(abs(a - p) for a, p in zip(actual, predicted))
    mid = len(errors) // 2
    if len(errors) % 2:
        return errors[mid]
    return (errors[mid - 1] + errors[mid]) / 2.0


def r2_score(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    mean_actual = sum(actual) / len(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


@dataclass
class FeatureSpec:
    numeric_fields: list[str]
    binary_fields: list[str]
    categorical_fields: list[str]
    categories_by_field: dict[str, list[str]]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]

    def feature_names(self) -> list[str]:
        names = ["bias"]
        names.extend(f"num:{field}" for field in self.numeric_fields)
        names.extend(f"bin:{field}" for field in self.binary_fields)
        for field in self.categorical_fields:
            for category in self.categories_by_field[field]:
                names.append(f"cat:{field}={category}")
            names.append(f"cat:{field}=__other__")
        return names


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligibility_field = "eligible_for_regular_cadence_label"
    if rows and eligibility_field not in rows[0]:
        eligibility_field = "eligible_for_next_hour_label"
    eligible = [row for row in rows if str(row.get(eligibility_field, "")).strip() == "1"]
    if not eligible:
        raise SystemExit("No eligible next-hour rows were found.")
    eligible.sort(key=lambda row: parse_dt(row["snapshot_time_utc"]))
    return eligible


def split_rows(rows: list[dict[str, str]], train_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    split_index = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    return rows[:split_index], rows[split_index:]


def build_feature_spec(train_rows: list[dict[str, str]], max_category_values: int) -> FeatureSpec:
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}

    for field in NUMERIC_FIELDS:
        values = [parse_float(row.get(field), 0.0) for row in train_rows]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = math.sqrt(variance) or 1.0
        numeric_means[field] = mean_value
        numeric_stds[field] = std_value

    categories_by_field: dict[str, list[str]] = {}
    for field in CATEGORICAL_FIELDS:
        counts: dict[str, int] = defaultdict(int)
        for row in train_rows:
            value = str(row.get(field, "") or "").strip().lower()
            if value:
                counts[value] += 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        categories_by_field[field] = [value for value, _ in ordered[:max_category_values]]

    return FeatureSpec(
        numeric_fields=NUMERIC_FIELDS,
        binary_fields=BINARY_FIELDS,
        categorical_fields=CATEGORICAL_FIELDS,
        categories_by_field=categories_by_field,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
    )


def vectorize_row(row: dict[str, str], spec: FeatureSpec) -> list[float]:
    vector = [1.0]

    for field in spec.numeric_fields:
        raw_value = parse_float(row.get(field), 0.0)
        mean_value = spec.numeric_means[field]
        std_value = spec.numeric_stds[field]
        vector.append((raw_value - mean_value) / std_value)

    for field in spec.binary_fields:
        vector.append(parse_flag(row.get(field)))

    for field in spec.categorical_fields:
        value = str(row.get(field, "") or "").strip().lower()
        categories = spec.categories_by_field[field]
        matched = False
        for category in categories:
            hit = 1.0 if value == category else 0.0
            vector.append(hit)
            matched = matched or bool(hit)
        vector.append(0.0 if matched or not value else 1.0)

    return vector


def solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(matrix)
    aug = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            continue
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def fit_ridge_regression(
    vectors: list[list[float]],
    targets: list[float],
    ridge_alpha: float,
) -> list[float]:
    dimension = len(vectors[0])
    xtx = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    xty = [0.0 for _ in range(dimension)]

    for vector, target in zip(vectors, targets):
        for i in range(dimension):
            xty[i] += vector[i] * target
            row_i = xtx[i]
            vi = vector[i]
            for j in range(dimension):
                row_i[j] += vi * vector[j]

    for i in range(1, dimension):
        xtx[i][i] += ridge_alpha

    return solve_linear_system(xtx, xty)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def target_to_naive_prediction(row: dict[str, str], target_name: str) -> float:
    hours = parse_float(row.get("hours_to_next_snapshot"), 1.0)
    if target_name == "upvotes":
        return max(0.0, parse_float(row.get("upvote_velocity_per_hour"), 0.0) * hours)
    return max(0.0, parse_float(row.get("comment_velocity_per_hour"), 0.0) * hours)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_bucket_models(
    rows: list[dict[str, str]],
    max_category_values: int,
    ridge_alpha: float,
    target_column: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        bucket = str(row.get("age_bucket", "") or "").strip().lower() or "__unknown__"
        rows_by_bucket[bucket].append(row)

    models_by_bucket: dict[str, dict[str, Any]] = {}
    counts_by_bucket = {bucket: len(bucket_rows) for bucket, bucket_rows in rows_by_bucket.items()}

    for bucket, bucket_rows in rows_by_bucket.items():
        if len(bucket_rows) < MIN_BUCKET_TRAIN_ROWS:
            continue
        spec = build_feature_spec(bucket_rows, max_category_values)
        vectors = [vectorize_row(row, spec) for row in bucket_rows]
        targets = [safe_log1p_target(max(0.0, parse_float(row.get(target_column), 0.0))) for row in bucket_rows]
        coefficients = fit_ridge_regression(vectors, targets, ridge_alpha)
        models_by_bucket[bucket] = {
            "spec": spec,
            "coefficients": coefficients,
            "feature_names": spec.feature_names(),
            "train_rows": len(bucket_rows),
        }

    return models_by_bucket, counts_by_bucket


def predict_with_bucketed_model(
    row: dict[str, str],
    *,
    bucket_models: dict[str, dict[str, Any]],
    global_spec: FeatureSpec,
    global_coefficients: list[float],
) -> tuple[float, str]:
    bucket = str(row.get("age_bucket", "") or "").strip().lower() or "__unknown__"
    bucket_model = bucket_models.get(bucket)
    if bucket_model is not None:
        vector = vectorize_row(row, bucket_model["spec"])
        return inverse_log1p_target(dot(vector, bucket_model["coefficients"])), bucket
    vector = vectorize_row(row, global_spec)
    return inverse_log1p_target(dot(vector, global_coefficients)), "__global_fallback__"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    train_rows, test_rows = split_rows(rows, args.train_fraction)
    global_spec = build_feature_spec(train_rows, args.max_category_values)
    global_feature_names = global_spec.feature_names()

    train_vectors = [vectorize_row(row, global_spec) for row in train_rows]

    combined_metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_csv": str(input_path),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "ridge_alpha": args.ridge_alpha,
        "targets": {},
    }

    for target_name in args.targets:
        target_column = TARGETS[target_name]
        train_target_raw = [max(0.0, parse_float(row.get(target_column), 0.0)) for row in train_rows]
        test_target_raw = [max(0.0, parse_float(row.get(target_column), 0.0)) for row in test_rows]
        train_target = [safe_log1p_target(value) for value in train_target_raw]

        global_coefficients = fit_ridge_regression(train_vectors, train_target, args.ridge_alpha)
        bucket_models, train_rows_by_bucket = train_bucket_models(
            train_rows,
            args.max_category_values,
            args.ridge_alpha,
            target_column,
        )

        regression_predictions_raw: list[float] = []
        global_predictions_raw: list[float] = []
        naive_predictions_raw: list[float] = []
        prediction_rows: list[dict[str, Any]] = []
        test_rows_by_bucket: dict[str, int] = defaultdict(int)
        bucket_usage_counts: dict[str, int] = defaultdict(int)

        for row, actual in zip(test_rows, test_target_raw):
            bucket_name = str(row.get("age_bucket", "") or "").strip().lower() or "__unknown__"
            test_rows_by_bucket[bucket_name] += 1
            bucketed_predicted_value, model_scope = predict_with_bucketed_model(
                row,
                bucket_models=bucket_models,
                global_spec=global_spec,
                global_coefficients=global_coefficients,
            )
            global_predicted_value = inverse_log1p_target(
                dot(vectorize_row(row, global_spec), global_coefficients)
            )
            naive_value = target_to_naive_prediction(row, target_name)
            regression_predictions_raw.append(bucketed_predicted_value)
            global_predictions_raw.append(global_predicted_value)
            naive_predictions_raw.append(naive_value)
            bucket_usage_counts[model_scope] += 1
            prediction_rows.append(
                {
                    "target": target_name,
                    "snapshot_time_utc": row["snapshot_time_utc"],
                    "subreddit": row["subreddit"],
                    "listing_type": row["listing_type"],
                    "age_bucket": row["age_bucket"],
                    "post_id": row["post_id"],
                    "title": row["title"],
                    "actual_next_delta": round(actual, 6),
                    "predicted_next_delta": round(bucketed_predicted_value, 6),
                    "global_predicted_next_delta": round(global_predicted_value, 6),
                    "naive_velocity_baseline": round(naive_value, 6),
                    "model_scope": model_scope,
                }
            )

        coefficient_rows: list[dict[str, Any]] = []
        global_coefficient_rows = [
            {"scope": "__global__", "feature": name, "coefficient": value, "train_rows": len(train_rows)}
            for name, value in zip(global_feature_names, global_coefficients)
        ]
        coefficient_rows.extend(global_coefficient_rows)
        for bucket, bucket_model in sorted(bucket_models.items()):
            coefficient_rows.extend(
                {
                    "scope": bucket,
                    "feature": name,
                    "coefficient": value,
                    "train_rows": bucket_model["train_rows"],
                }
                for name, value in zip(bucket_model["feature_names"], bucket_model["coefficients"])
            )
        coefficient_rows_sorted = sorted(coefficient_rows, key=lambda row: (row["scope"], -abs(float(row["coefficient"])), row["feature"]))
        global_top_coefficients = sorted(global_coefficient_rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)[:15]
        bucket_top_coefficients = {
            bucket: sorted(
                [
                    row for row in coefficient_rows
                    if row["scope"] == bucket
                ],
                key=lambda row: abs(float(row["coefficient"])),
                reverse=True,
            )[:10]
            for bucket in bucket_models
        }

        metrics = {
            "target_column": target_column,
            "model": "pure_python_ridge_log1p_age_bucketed",
            "test_mae": mean_absolute_error(test_target_raw, regression_predictions_raw),
            "test_rmse": root_mean_squared_error(test_target_raw, regression_predictions_raw),
            "test_median_ae": median_absolute_error(test_target_raw, regression_predictions_raw),
            "test_r2": r2_score(test_target_raw, regression_predictions_raw),
            "global_test_r2": r2_score(test_target_raw, global_predictions_raw),
            "global_test_mae": mean_absolute_error(test_target_raw, global_predictions_raw),
            "naive_mae": mean_absolute_error(test_target_raw, naive_predictions_raw),
            "naive_rmse": root_mean_squared_error(test_target_raw, naive_predictions_raw),
            "naive_median_ae": median_absolute_error(test_target_raw, naive_predictions_raw),
            "naive_r2": r2_score(test_target_raw, naive_predictions_raw),
            "global_top_coefficients": global_top_coefficients,
            "bucket_top_coefficients": bucket_top_coefficients,
            "bucket_model_train_rows": train_rows_by_bucket,
            "bucket_model_test_rows": dict(test_rows_by_bucket),
            "bucket_model_usage_counts": dict(bucket_usage_counts),
            "bucket_models_built": sorted(bucket_models.keys()),
        }
        combined_metrics["targets"][target_name] = metrics

        write_csv(
            output_dir / f"regression_{target_name}_test_predictions.csv",
            list(prediction_rows[0].keys()) if prediction_rows else [
                "target",
                "snapshot_time_utc",
                "subreddit",
                "listing_type",
                "age_bucket",
                "post_id",
                "title",
                "actual_next_delta",
                "predicted_next_delta",
                "global_predicted_next_delta",
                "naive_velocity_baseline",
                "model_scope",
            ],
            prediction_rows,
        )
        write_csv(
            output_dir / f"regression_{target_name}_coefficients.csv",
            ["scope", "feature", "coefficient", "train_rows"],
            coefficient_rows_sorted,
        )

    write_json(output_dir / "regression_metrics.json", combined_metrics)
    print(json.dumps(combined_metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
