from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_naive_forecast import recommendation_label, state_multiplier
from train_next_hour_regression import (
    mean_absolute_error,
    median_absolute_error,
    parse_float,
    r2_score,
    root_mean_squared_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the practical naive forecast rule against the current "
            "next-hour prediction dataset."
        )
    )
    parser.add_argument(
        "--input",
        default="data/models/reddit/prediction_next_hour.csv",
        help="Input prediction_next_hour.csv path.",
    )
    parser.add_argument(
        "--overall-output",
        default="data/history/reddit/naive_forecast_evaluation_overall.csv",
        help="Where to write the overall naive forecast evaluation CSV.",
    )
    parser.add_argument(
        "--subreddit-output",
        default="data/history/reddit/naive_forecast_evaluation_by_subreddit.csv",
        help="Where to write the per-subreddit evaluation CSV.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def actual_recommendation_label(row: dict[str, str]) -> str:
    actual_upvotes = max(0.0, parse_float(row.get("upvote_delta_next_snapshot"), 0.0))
    actual_comments = max(0.0, parse_float(row.get("comment_delta_next_snapshot"), 0.0))
    next_state = clean_text(row.get("next_activity_state"))
    return recommendation_label(
        state=next_state,
        predicted_upvotes=actual_upvotes,
        predicted_comments=actual_comments,
        alive_upvote_threshold=parse_float(row.get("alive_upvote_velocity_threshold"), 0.0),
        alive_comment_threshold=parse_float(row.get("alive_comment_velocity_threshold"), 0.0),
        surging_upvote_threshold=parse_float(row.get("surging_upvote_velocity_threshold"), 0.0),
        surging_comment_threshold=parse_float(row.get("surging_comment_velocity_threshold"), 0.0),
    )


def predicted_recommendation_label(row: dict[str, str]) -> tuple[float, float, str]:
    state = clean_text(row.get("activity_state"))
    multiplier = state_multiplier(state)
    predicted_upvotes = max(0.0, parse_float(row.get("upvote_velocity_per_hour"), 0.0)) * multiplier
    predicted_comments = max(0.0, parse_float(row.get("comment_velocity_per_hour"), 0.0)) * multiplier
    label = recommendation_label(
        state=state,
        predicted_upvotes=predicted_upvotes,
        predicted_comments=predicted_comments,
        alive_upvote_threshold=parse_float(row.get("alive_upvote_velocity_threshold"), 0.0),
        alive_comment_threshold=parse_float(row.get("alive_comment_velocity_threshold"), 0.0),
        surging_upvote_threshold=parse_float(row.get("surging_upvote_velocity_threshold"), 0.0),
        surging_comment_threshold=parse_float(row.get("surging_comment_velocity_threshold"), 0.0),
    )
    return predicted_upvotes, predicted_comments, label


def build_metric_row(label: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    actual_upvotes: list[float] = []
    predicted_upvotes: list[float] = []
    actual_comments: list[float] = []
    predicted_comments: list[float] = []
    exact_matches = 0
    predicted_surge = 0
    true_surge = 0
    predicted_active = 0
    true_active = 0
    predicted_die = 0
    true_die = 0

    for row in rows:
        pred_up, pred_comments, pred_label = predicted_recommendation_label(row)
        actual_label = actual_recommendation_label(row)
        act_up = max(0.0, parse_float(row.get("upvote_delta_next_snapshot"), 0.0))
        act_comments = max(0.0, parse_float(row.get("comment_delta_next_snapshot"), 0.0))

        actual_upvotes.append(act_up)
        predicted_upvotes.append(pred_up)
        actual_comments.append(act_comments)
        predicted_comments.append(pred_comments)

        if pred_label == actual_label:
            exact_matches += 1
        if pred_label == "surge_watch":
            predicted_surge += 1
            if actual_label == "surge_watch":
                true_surge += 1
        if pred_label in {"surge_watch", "active_watch"}:
            predicted_active += 1
            if actual_label in {"surge_watch", "active_watch"}:
                true_active += 1
        if pred_label in {"cooling_watch", "dying_watch", "stop_watch"}:
            predicted_die += 1
            if actual_label in {"cooling_watch", "dying_watch", "stop_watch"}:
                true_die += 1

    row_count = len(rows)
    return {
        "segment": label,
        "row_count": row_count,
        "upvote_mae": mean_absolute_error(actual_upvotes, predicted_upvotes),
        "upvote_rmse": root_mean_squared_error(actual_upvotes, predicted_upvotes),
        "upvote_median_ae": median_absolute_error(actual_upvotes, predicted_upvotes),
        "upvote_r2": r2_score(actual_upvotes, predicted_upvotes),
        "comment_mae": mean_absolute_error(actual_comments, predicted_comments),
        "comment_rmse": root_mean_squared_error(actual_comments, predicted_comments),
        "comment_median_ae": median_absolute_error(actual_comments, predicted_comments),
        "comment_r2": r2_score(actual_comments, predicted_comments),
        "recommendation_exact_match_rate": (exact_matches / row_count) if row_count else 0.0,
        "predicted_surge_count": predicted_surge,
        "predicted_surge_precision": (true_surge / predicted_surge) if predicted_surge else None,
        "predicted_active_count": predicted_active,
        "predicted_active_precision": (true_active / predicted_active) if predicted_active else None,
        "predicted_die_count": predicted_die,
        "predicted_die_precision": (true_die / predicted_die) if predicted_die else None,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    overall_output_path = Path(args.overall_output)
    subreddit_output_path = Path(args.subreddit_output)

    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows = load_rows(input_path)
    overall_rows = [build_metric_row("overall", rows)]

    rows_by_subreddit: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_subreddit[clean_text(row.get("subreddit")).lower()].append(row)
    subreddit_rows = [
        build_metric_row(subreddit, subreddit_rows)
        for subreddit, subreddit_rows in sorted(rows_by_subreddit.items())
    ]

    write_csv(overall_output_path, overall_rows)
    write_csv(subreddit_output_path, subreddit_rows)

    print(f"Saved {len(overall_rows)} row(s) to {overall_output_path}")
    print(f"Saved {len(subreddit_rows)} row(s) to {subreddit_output_path}")


if __name__ == "__main__":
    main()
