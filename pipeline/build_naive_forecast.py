from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


STATE_MULTIPLIERS = {
    "surging": 1.0,
    "alive": 0.9,
    "emerging": 0.8,
    "cooling": 0.45,
    "dying": 0.15,
    "dead": 0.0,
}

STATE_DECAY = {
    "surging": 0.9,
    "alive": 0.8,
    "emerging": 0.72,
    "cooling": 0.45,
    "dying": 0.2,
    "dead": 0.0,
}

STATE_DIE_SOON_BASE = {
    "surging": 0.08,
    "alive": 0.22,
    "emerging": 0.34,
    "cooling": 0.72,
    "dying": 0.9,
    "dead": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build practical forecast tables from the current post status table "
            "using the simple next-hour naive velocity rule."
        )
    )
    parser.add_argument(
        "--input",
        default="data/history/reddit/latest_post_status.csv",
        help="Input latest post status CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/history/reddit/naive_next_hour_forecast_latest.csv",
        help="Output CSV path for the latest naive forecast view.",
    )
    parser.add_argument(
        "--top-output",
        default="data/history/reddit/naive_forecast_leaderboard.csv",
        help="Output CSV path for the top-ranked forecast leaderboard.",
    )
    parser.add_argument(
        "--watchlist-output",
        default="data/history/reddit/naive_forecast_watchlist_by_subreddit.csv",
        help="Output CSV path for the compact subreddit watchlists.",
    )
    parser.add_argument(
        "--top-limit",
        type=int,
        default=250,
        help="How many ranked rows to keep in the compact overall leaderboard view.",
    )
    parser.add_argument(
        "--per-subreddit-limit",
        type=int,
        default=20,
        help="How many rows to keep per subreddit in the watchlist output.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_log1p(value: float | None) -> float:
    if value is None:
        return 0.0
    return math.log1p(max(0.0, value))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def state_multiplier(state: str) -> float:
    return STATE_MULTIPLIERS.get(clean_text(state).lower(), 0.7)


def state_decay(state: str) -> float:
    return STATE_DECAY.get(clean_text(state).lower(), 0.7)


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


def recommendation_label(
    *,
    state: str,
    predicted_upvotes: float,
    predicted_comments: float,
    alive_upvote_threshold: float,
    alive_comment_threshold: float,
    surging_upvote_threshold: float,
    surging_comment_threshold: float,
) -> str:
    state_key = clean_text(state).lower()
    if (
        predicted_upvotes >= surging_upvote_threshold
        or predicted_comments >= surging_comment_threshold
    ):
        return "surge_watch"
    if (
        predicted_upvotes >= alive_upvote_threshold
        or predicted_comments >= alive_comment_threshold
    ):
        return "active_watch"
    if state_key == "dying":
        return "dying_watch"
    if state_key == "cooling":
        return "cooling_watch"
    if state_key == "dead":
        return "stop_watch"
    return "low_watch"


def predicted_flow_state(
    *,
    state: str,
    predicted_upvotes: float,
    predicted_comments: float,
    die_soon_score: float,
    alive_upvote_threshold: float,
    alive_comment_threshold: float,
    surging_upvote_threshold: float,
    surging_comment_threshold: float,
) -> str:
    state_key = clean_text(state).lower()
    if (
        predicted_upvotes >= surging_upvote_threshold
        or predicted_comments >= surging_comment_threshold
    ):
        return "surging"
    if (
        predicted_upvotes >= alive_upvote_threshold
        or predicted_comments >= alive_comment_threshold
    ):
        return "alive"
    if die_soon_score >= 0.9 or state_key == "dead":
        return "dead"
    if die_soon_score >= 0.65 or state_key == "dying":
        return "dying"
    if state_key in {"cooling", "dying", "dead"}:
        return "cooling"
    return "cooling"


def decayed_horizon_total(base_delta: float, state: str, horizon_hours: int) -> float:
    if horizon_hours <= 0:
        return 0.0
    decay = state_decay(state)
    if base_delta <= 0 or decay <= 0:
        return 0.0
    if abs(decay - 1.0) < 1e-9:
        return base_delta * horizon_hours
    return base_delta * sum(decay ** step for step in range(horizon_hours))


def compute_die_soon_score(
    *,
    state: str,
    upvote_velocity: float,
    comment_velocity: float,
    dead_upvote_threshold: float,
    dead_comment_threshold: float,
    alive_upvote_threshold: float,
    alive_comment_threshold: float,
    current_attention_score: float,
    general_popularity_score: float,
    observed_hours: float,
) -> float:
    state_key = clean_text(state).lower()
    if state_key == "dead":
        return 1.0

    alive_ratio = (
        (
            min((safe_ratio(upvote_velocity, alive_upvote_threshold) or 0.0), 4.0)
            + min((safe_ratio(comment_velocity, alive_comment_threshold) or 0.0), 4.0)
        )
        / 2.0
    )
    dead_ratio = (
        (
            min((safe_ratio(upvote_velocity, dead_upvote_threshold) or 0.0), 8.0)
            + min((safe_ratio(comment_velocity, dead_comment_threshold) or 0.0), 8.0)
        )
        / 2.0
    )
    low_motion_component = 1.0 - clamp01(alive_ratio / 1.5)
    near_dead_component = 1.0 - clamp01(dead_ratio / 3.0)
    attention_penalty = clamp01(current_attention_score / 50.0)
    popularity_penalty = clamp01(general_popularity_score / 75.0)
    time_bonus = clamp01(observed_hours / 6.0)
    base = STATE_DIE_SOON_BASE.get(state_key, 0.35)
    score = (
        (0.42 * base)
        + (0.28 * low_motion_component)
        + (0.22 * near_dead_component)
        + (0.08 * time_bonus)
        - (0.12 * attention_penalty)
        - (0.08 * popularity_penalty)
    )
    return clamp01(score)


def build_forecast_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    forecast_rows: list[dict[str, Any]] = []
    for row in rows:
        latest_state = clean_text(row.get("latest_activity_state"))
        current_attention_score = parse_float(row.get("current_attention_score")) or 0.0
        general_popularity_score = parse_float(row.get("general_popularity_score")) or 0.0
        upvote_velocity = max(0.0, parse_float(row.get("last_upvote_velocity_per_hour")) or 0.0)
        comment_velocity = max(0.0, parse_float(row.get("last_comment_velocity_per_hour")) or 0.0)
        observed_hours = parse_float(row.get("observed_hours")) or 0.0
        multiplier = state_multiplier(latest_state)
        adjusted_upvote_delta = upvote_velocity * multiplier
        adjusted_comment_delta = comment_velocity * multiplier

        alive_upvote_threshold = parse_float(row.get("alive_upvote_velocity_threshold")) or 0.0
        alive_comment_threshold = parse_float(row.get("alive_comment_velocity_threshold")) or 0.0
        surging_upvote_threshold = parse_float(row.get("surging_upvote_velocity_threshold")) or 0.0
        surging_comment_threshold = parse_float(row.get("surging_comment_velocity_threshold")) or 0.0
        dead_upvote_threshold = parse_float(row.get("dead_upvote_velocity_threshold")) or 0.0
        dead_comment_threshold = parse_float(row.get("dead_comment_velocity_threshold")) or 0.0

        predicted_upvotes_3h = decayed_horizon_total(adjusted_upvote_delta, latest_state, 3)
        predicted_comments_3h = decayed_horizon_total(adjusted_comment_delta, latest_state, 3)
        predicted_upvotes_6h = decayed_horizon_total(adjusted_upvote_delta, latest_state, 6)
        predicted_comments_6h = decayed_horizon_total(adjusted_comment_delta, latest_state, 6)
        die_soon_score = compute_die_soon_score(
            state=latest_state,
            upvote_velocity=upvote_velocity,
            comment_velocity=comment_velocity,
            dead_upvote_threshold=dead_upvote_threshold,
            dead_comment_threshold=dead_comment_threshold,
            alive_upvote_threshold=alive_upvote_threshold,
            alive_comment_threshold=alive_comment_threshold,
            current_attention_score=current_attention_score,
            general_popularity_score=general_popularity_score,
            observed_hours=observed_hours,
        )
        recommendation = recommendation_label(
            state=latest_state,
            predicted_upvotes=adjusted_upvote_delta,
            predicted_comments=adjusted_comment_delta,
            alive_upvote_threshold=alive_upvote_threshold,
            alive_comment_threshold=alive_comment_threshold,
            surging_upvote_threshold=surging_upvote_threshold,
            surging_comment_threshold=surging_comment_threshold,
        )
        predicted_state_next_hour = predicted_flow_state(
            state=latest_state,
            predicted_upvotes=adjusted_upvote_delta,
            predicted_comments=adjusted_comment_delta,
            die_soon_score=die_soon_score,
            alive_upvote_threshold=alive_upvote_threshold,
            alive_comment_threshold=alive_comment_threshold,
            surging_upvote_threshold=surging_upvote_threshold,
            surging_comment_threshold=surging_comment_threshold,
        )
        die_soon_label = "likely_to_die_soon" if die_soon_score >= 0.65 else "likely_to_continue"
        forecast_priority_score = (
            safe_log1p(adjusted_upvote_delta)
            + safe_log1p(adjusted_comment_delta * 4.0)
            + (current_attention_score * 0.15)
            + (general_popularity_score * 0.03)
            - (die_soon_score * 2.0)
        )

        forecast_rows.append(
            {
                "subreddit": clean_text(row.get("subreddit")),
                "post_id": clean_text(row.get("post_id")),
                "title": clean_text(row.get("title")),
                "url": clean_text(row.get("url")),
                "author": clean_text(row.get("author")),
                "last_seen_time_utc": clean_text(row.get("last_seen_time_utc")),
                "latest_activity_state": latest_state,
                "analysis_priority": clean_text(row.get("analysis_priority")),
                "latest_rank_seen": clean_text(row.get("latest_rank_seen")),
                "listing_types_seen": clean_text(row.get("listing_types_seen")),
                "snapshot_count": clean_text(row.get("snapshot_count")),
                "observed_hours": observed_hours,
                "current_attention_score": current_attention_score,
                "general_popularity_score": general_popularity_score,
                "last_upvote_velocity_per_hour": upvote_velocity,
                "last_comment_velocity_per_hour": comment_velocity,
                "naive_predicted_upvote_delta_next_hour": adjusted_upvote_delta,
                "naive_predicted_comment_delta_next_hour": adjusted_comment_delta,
                "naive_predicted_upvote_delta_next_3h": predicted_upvotes_3h,
                "naive_predicted_comment_delta_next_3h": predicted_comments_3h,
                "naive_predicted_upvote_delta_next_6h": predicted_upvotes_6h,
                "naive_predicted_comment_delta_next_6h": predicted_comments_6h,
                "state_multiplier": multiplier,
                "state_decay": state_decay(latest_state),
                "die_soon_score": die_soon_score,
                "die_soon_label": die_soon_label,
                "predicted_flow_state_next_hour": predicted_state_next_hour,
                "forecast_priority_score": forecast_priority_score,
                "forecast_recommendation": recommendation,
            }
        )

    forecast_rows.sort(
        key=lambda row: (
            -float(row["forecast_priority_score"]),
            -float(row["naive_predicted_upvote_delta_next_hour"]),
            -float(row["naive_predicted_comment_delta_next_hour"]),
            row["subreddit"],
            row["post_id"],
        )
    )
    for index, row in enumerate(forecast_rows, start=1):
        row["forecast_rank_overall"] = index
    subreddit_counts: dict[str, int] = {}
    for row in forecast_rows:
        subreddit = row["subreddit"].lower()
        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
        row["forecast_rank_in_subreddit"] = subreddit_counts[subreddit]
    return forecast_rows


def build_watchlist_rows(
    forecast_rows: list[dict[str, Any]],
    *,
    per_subreddit_limit: int,
) -> list[dict[str, Any]]:
    watchlist_rows: list[dict[str, Any]] = []
    per_subreddit_counts: dict[str, int] = {}
    for row in forecast_rows:
        subreddit = clean_text(row.get("subreddit")).lower()
        recommendation = clean_text(row.get("forecast_recommendation"))
        if recommendation not in {"surge_watch", "active_watch", "cooling_watch", "dying_watch"}:
            continue
        if per_subreddit_counts.get(subreddit, 0) >= per_subreddit_limit:
            continue
        per_subreddit_counts[subreddit] = per_subreddit_counts.get(subreddit, 0) + 1
        watchlist_rows.append(
            {
                "subreddit": row.get("subreddit"),
                "watchlist_rank_in_subreddit": per_subreddit_counts[subreddit],
                "forecast_rank_overall": row.get("forecast_rank_overall"),
                "post_id": row.get("post_id"),
                "title": row.get("title"),
                "url": row.get("url"),
                "latest_activity_state": row.get("latest_activity_state"),
                "predicted_flow_state_next_hour": row.get("predicted_flow_state_next_hour"),
                "forecast_recommendation": recommendation,
                "die_soon_label": row.get("die_soon_label"),
                "die_soon_score": row.get("die_soon_score"),
                "naive_predicted_upvote_delta_next_hour": row.get("naive_predicted_upvote_delta_next_hour"),
                "naive_predicted_comment_delta_next_hour": row.get("naive_predicted_comment_delta_next_hour"),
                "naive_predicted_upvote_delta_next_3h": row.get("naive_predicted_upvote_delta_next_3h"),
                "naive_predicted_comment_delta_next_3h": row.get("naive_predicted_comment_delta_next_3h"),
                "naive_predicted_upvote_delta_next_6h": row.get("naive_predicted_upvote_delta_next_6h"),
                "naive_predicted_comment_delta_next_6h": row.get("naive_predicted_comment_delta_next_6h"),
                "forecast_priority_score": row.get("forecast_priority_score"),
                "current_attention_score": row.get("current_attention_score"),
                "general_popularity_score": row.get("general_popularity_score"),
            }
        )
    return watchlist_rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    top_output_path = Path(args.top_output)
    watchlist_output_path = Path(args.watchlist_output)

    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows = load_rows(input_path)
    forecast_rows = build_forecast_rows(rows)
    leaderboard_rows = forecast_rows[: max(0, args.top_limit)]
    watchlist_rows = build_watchlist_rows(
        forecast_rows,
        per_subreddit_limit=max(0, args.per_subreddit_limit),
    )

    write_csv(output_path, forecast_rows)
    write_csv(top_output_path, leaderboard_rows)
    write_csv(watchlist_output_path, watchlist_rows)

    print(f"Saved {len(forecast_rows)} row(s) to {output_path}")
    print(f"Saved {len(leaderboard_rows)} row(s) to {top_output_path}")
    print(f"Saved {len(watchlist_rows)} row(s) to {watchlist_output_path}")


if __name__ == "__main__":
    main()
