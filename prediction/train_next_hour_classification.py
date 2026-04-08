from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_next_hour_regression import (
    BINARY_FIELDS,
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    load_rows,
    parse_float,
    split_rows,
    write_csv,
    write_json,
)


TargetFunc = Callable[[dict[str, str]], int | None]
THRESHOLD_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train practical yes/no classifiers for the Reddit next-snapshot tasks "
            "using the model-ready prediction table."
        )
    )
    parser.add_argument(
        "--input",
        default="data/models/reddit/prediction_next_hour.csv",
        help="Model-ready next-hour prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/reddit/classification",
        help="Directory where classification metrics and predictions will be written.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological training fraction. The remainder is used as test data.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGETS),
        default=["alive_next", "surging_next", "cooling_or_dead_next"],
        help="Which binary next-snapshot targets to train.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=120,
        help="Number of trees in the ExtraTreesClassifier ensemble.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=14,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=10,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Tree max_features setting.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def parse_flag(value: Any) -> float:
    return 1.0 if clean_text(value) == "1" else 0.0


def row_to_feature_dict(row: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        payload[f"num:{field}"] = parse_float(row.get(field), 0.0)
    for field in BINARY_FIELDS:
        payload[f"bin:{field}"] = parse_flag(row.get(field))
    for field in CATEGORICAL_FIELDS:
        value = clean_text(row.get(field)).lower()
        payload[f"cat:{field}"] = value or "__none__"
    return payload


def alive_target(row: dict[str, str]) -> int | None:
    value = clean_text(row.get("alive_next_snapshot"))
    return int(value) if value in {"0", "1"} else None


def surging_target(row: dict[str, str]) -> int | None:
    value = clean_text(row.get("surging_next_snapshot"))
    return int(value) if value in {"0", "1"} else None


def high_upvote_growth_target(row: dict[str, str]) -> int | None:
    value = clean_text(row.get("high_upvote_growth_next_snapshot"))
    return int(value) if value in {"0", "1"} else None


def high_comment_growth_target(row: dict[str, str]) -> int | None:
    value = clean_text(row.get("high_comment_growth_next_snapshot"))
    return int(value) if value in {"0", "1"} else None


def cooling_or_dead_target(row: dict[str, str]) -> int | None:
    next_state = clean_text(row.get("next_activity_state")).lower()
    if not next_state:
        return None
    return int(next_state in {"cooling", "dead"})


TARGETS: dict[str, TargetFunc] = {
    "alive_next": alive_target,
    "surging_next": surging_target,
    "cooling_or_dead_next": cooling_or_dead_target,
    "high_upvote_growth_next": high_upvote_growth_target,
    "high_comment_growth_next": high_comment_growth_target,
}


def naive_prediction(row: dict[str, str], target_name: str) -> int:
    current_state = clean_text(row.get("activity_state")).lower()
    if target_name == "alive_next":
        return int(current_state in {"alive", "surging", "emerging", "cooling"})
    if target_name == "surging_next":
        return int(current_state == "surging")
    if target_name == "cooling_or_dead_next":
        return int(current_state in {"cooling", "dead"})
    if target_name == "high_upvote_growth_next":
        velocity = parse_float(row.get("upvote_velocity_per_hour"), 0.0)
        threshold = parse_float(row.get("alive_upvote_velocity_threshold"), 0.0)
        return int(velocity >= threshold)
    if target_name == "high_comment_growth_next":
        velocity = parse_float(row.get("comment_velocity_per_hour"), 0.0)
        threshold = parse_float(row.get("alive_comment_velocity_threshold"), 0.0)
        return int(velocity >= threshold)
    return 0


def collect_labeled_rows(rows: list[dict[str, str]], target_func: TargetFunc) -> list[dict[str, str]]:
    labeled: list[dict[str, str]] = []
    for row in rows:
        if target_func(row) is not None:
            labeled.append(row)
    return labeled


def build_matrix(rows: list[dict[str, str]]) -> tuple[Any, DictVectorizer]:
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform([row_to_feature_dict(row) for row in rows])
    return matrix, vectorizer


def transform_matrix(rows: list[dict[str, str]], vectorizer: DictVectorizer) -> Any:
    return vectorizer.transform([row_to_feature_dict(row) for row in rows])


def safe_auc(actual: list[int], probabilities: list[float]) -> float | None:
    if len(set(actual)) < 2:
        return None
    try:
        return float(roc_auc_score(actual, probabilities))
    except ValueError:
        return None


def metric_row(
    *,
    segment: str,
    actual: list[int],
    predicted: list[int],
    probabilities: list[float],
    naive_predicted: list[int],
) -> dict[str, Any]:
    return {
        "segment": segment,
        "row_count": len(actual),
        "positive_rate": (sum(actual) / len(actual)) if actual else 0.0,
        "accuracy": accuracy_score(actual, predicted) if actual else 0.0,
        "balanced_accuracy": balanced_accuracy_score(actual, predicted) if len(set(actual)) > 1 else 0.0,
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": safe_auc(actual, probabilities),
        "naive_accuracy": accuracy_score(actual, naive_predicted) if actual else 0.0,
        "naive_balanced_accuracy": balanced_accuracy_score(actual, naive_predicted) if len(set(actual)) > 1 else 0.0,
        "naive_precision": precision_score(actual, naive_predicted, zero_division=0),
        "naive_recall": recall_score(actual, naive_predicted, zero_division=0),
        "naive_f1": f1_score(actual, naive_predicted, zero_division=0),
    }


def threshold_labels(probabilities: list[float], threshold: float) -> list[int]:
    return [int(prob >= threshold) for prob in probabilities]


def threshold_metric_rows(
    *,
    actual: list[int],
    probabilities: list[float],
    naive_predicted: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        predicted = threshold_labels(probabilities, threshold)
        row = metric_row(
            segment=f"threshold_{threshold:.2f}",
            actual=actual,
            predicted=predicted,
            probabilities=probabilities,
            naive_predicted=naive_predicted,
        )
        row["threshold"] = threshold
        row["predicted_positive_count"] = int(sum(predicted))
        rows.append(row)
    return rows


def choose_balanced_threshold(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.5
    best_row = max(
        rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["f1"]),
            -abs(float(row["threshold"]) - 0.6),
        ),
    )
    return float(best_row["threshold"])


def choose_conservative_threshold(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.7
    eligible = [row for row in rows if float(row["recall"]) >= 0.5]
    candidate_rows = eligible or rows
    best_row = max(
        candidate_rows,
        key=lambda row: (
            float(row["precision"]),
            float(row["accuracy"]),
            float(row["threshold"]),
        ),
    )
    return float(best_row["threshold"])


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_rows(input_path)
    metrics_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "targets": {},
    }

    for target_name in args.targets:
        target_func = TARGETS[target_name]
        labeled_rows = collect_labeled_rows(base_rows, target_func)
        if len(labeled_rows) < 1000:
            continue

        train_rows, test_rows = split_rows(labeled_rows, args.train_fraction)
        train_target = [int(target_func(row) or 0) for row in train_rows]
        test_target = [int(target_func(row) or 0) for row in test_rows]

        train_matrix, vectorizer = build_matrix(train_rows)
        test_matrix = transform_matrix(test_rows, vectorizer)

        model = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        model.fit(train_matrix, train_target)

        predicted = list(model.predict(test_matrix))
        probabilities_matrix = model.predict_proba(test_matrix)
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
        probabilities = [float(row[positive_index]) for row in probabilities_matrix]
        naive_predicted = [naive_prediction(row, target_name) for row in test_rows]
        threshold_rows = threshold_metric_rows(
            actual=test_target,
            probabilities=probabilities,
            naive_predicted=naive_predicted,
        )
        balanced_threshold = choose_balanced_threshold(threshold_rows)
        conservative_threshold = choose_conservative_threshold(threshold_rows)
        balanced_predicted = threshold_labels(probabilities, balanced_threshold)
        conservative_predicted = threshold_labels(probabilities, conservative_threshold)

        overall_metrics = metric_row(
            segment="overall",
            actual=test_target,
            predicted=predicted,
            probabilities=probabilities,
            naive_predicted=naive_predicted,
        )

        by_subreddit_rows: dict[str, list[tuple[int, int, float, int]]] = defaultdict(list)
        prediction_rows: list[dict[str, Any]] = []
        for row, actual, pred, prob, naive in zip(test_rows, test_target, predicted, probabilities, naive_predicted):
            subreddit = clean_text(row.get("subreddit")).lower()
            by_subreddit_rows[subreddit].append((actual, pred, prob, naive))
            prediction_rows.append(
                {
                    "target": target_name,
                    "snapshot_time_utc": row.get("snapshot_time_utc"),
                    "subreddit": row.get("subreddit"),
                    "post_id": row.get("post_id"),
                    "title": row.get("title"),
                    "activity_state": row.get("activity_state"),
                    "listing_type": row.get("listing_type"),
                    "actual_label": actual,
                    "predicted_label": pred,
                    "predicted_probability": round(prob, 6),
                    "balanced_threshold": balanced_threshold,
                    "predicted_label_balanced": int(prob >= balanced_threshold),
                    "conservative_threshold": conservative_threshold,
                    "predicted_label_conservative": int(prob >= conservative_threshold),
                    "naive_label": naive,
                }
            )

        subreddit_metric_rows: list[dict[str, Any]] = []
        for subreddit, tuples in sorted(by_subreddit_rows.items()):
            actual = [item[0] for item in tuples]
            pred = [item[1] for item in tuples]
            probs = [item[2] for item in tuples]
            naive = [item[3] for item in tuples]
            subreddit_metric_rows.append(
                metric_row(
                    segment=subreddit,
                    actual=actual,
                    predicted=pred,
                    probabilities=probs,
                    naive_predicted=naive,
                )
            )

        class_counts = Counter(test_target)
        importance_rows = sorted(
            (
                {
                    "feature": feature,
                    "importance": float(importance),
                    "target": target_name,
                }
                for feature, importance in zip(vectorizer.get_feature_names_out(), model.feature_importances_)
            ),
            key=lambda row: row["importance"],
            reverse=True,
        )

        metrics_payload["targets"][target_name] = {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "test_class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
            "model": "sklearn_extra_trees_classifier",
            "tree_params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "max_features": args.max_features,
            },
            "overall_metrics": overall_metrics,
            "balanced_threshold": balanced_threshold,
            "balanced_threshold_metrics": metric_row(
                segment="balanced_threshold",
                actual=test_target,
                predicted=balanced_predicted,
                probabilities=probabilities,
                naive_predicted=naive_predicted,
            ),
            "conservative_threshold": conservative_threshold,
            "conservative_threshold_metrics": metric_row(
                segment="conservative_threshold",
                actual=test_target,
                predicted=conservative_predicted,
                probabilities=probabilities,
                naive_predicted=naive_predicted,
            ),
            "top_features": importance_rows[:15],
        }

        write_csv(
            output_dir / f"classification_{target_name}_metrics_by_subreddit.csv",
            list(subreddit_metric_rows[0].keys()) if subreddit_metric_rows else ["segment"],
            subreddit_metric_rows,
        )
        write_csv(
            output_dir / f"classification_{target_name}_test_predictions.csv",
            list(prediction_rows[0].keys()) if prediction_rows else ["target"],
            prediction_rows,
        )
        write_csv(
            output_dir / f"classification_{target_name}_feature_importance.csv",
            ["feature", "importance", "target"],
            importance_rows,
        )
        write_csv(
            output_dir / f"classification_{target_name}_threshold_metrics.csv",
            list(threshold_rows[0].keys()) if threshold_rows else ["threshold"],
            threshold_rows,
        )

    write_json(output_dir / "classification_metrics.json", metrics_payload)
    print(json.dumps(metrics_payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
