from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REDDIT_COMMENTS_URL_RE = re.compile(
    r"(?:https?://(?:www\.)?reddit\.com)?/r/(?P<subreddit>[^/]+)/comments/(?P<post_id>[A-Za-z0-9_]+)/",
    re.IGNORECASE,
)

STATE_PRIORITY = {
    "surging": 0,
    "alive": 1,
    "emerging": 2,
    "cooling": 3,
    "unknown": 4,
    "dead": 9,
}

ANALYSIS_PRIORITY = {
    "highest": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "very_low": 4,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive number.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined exact-post observation pool with two lanes: "
            "a fixed prediction cohort and a rolling live-watch pool."
        )
    )
    parser.add_argument(
        "--source",
        default="data/history/reddit/latest_post_status.csv",
        help="Latest post status CSV produced by build_reddit_history.py.",
    )
    parser.add_argument(
        "--output",
        default="data/tracking/free_observation_pool_latest.csv",
        help="Where to write the combined observation pool CSV used by the collector.",
    )
    parser.add_argument(
        "--prediction-output",
        default="data/tracking/prediction_observation_pool_latest.csv",
        help="Where to write the fixed prediction cohort CSV.",
    )
    parser.add_argument(
        "--live-output",
        default="data/tracking/live_watch_pool_latest.csv",
        help="Where to write the rolling live-watch pool CSV.",
    )
    parser.add_argument(
        "--max-posts",
        type=positive_int,
        default=1000,
        help="Maximum number of posts to keep in the combined observation pool overall.",
    )
    parser.add_argument(
        "--per-subreddit-limit",
        type=positive_int,
        default=250,
        help="Maximum number of posts to keep per subreddit in the combined observation pool.",
    )
    parser.add_argument(
        "--prediction-max-posts",
        type=positive_int,
        default=600,
        help="Maximum number of posts to keep in the fixed prediction cohort.",
    )
    parser.add_argument(
        "--prediction-per-subreddit-limit",
        type=positive_int,
        default=150,
        help="Maximum number of prediction-cohort posts to keep per subreddit.",
    )
    parser.add_argument(
        "--live-max-posts",
        type=positive_int,
        default=400,
        help="Maximum number of posts to keep in the rolling live-watch pool.",
    )
    parser.add_argument(
        "--live-per-subreddit-limit",
        type=positive_int,
        default=100,
        help="Maximum number of live-watch posts to keep per subreddit.",
    )
    parser.add_argument(
        "--prediction-target-hours",
        type=positive_float,
        default=24.0,
        help="How long each fixed prediction cohort post should stay in the cohort.",
    )
    parser.add_argument(
        "--prediction-max-admission-age-hours",
        type=positive_float,
        default=3.0,
        help="Maximum current post age for admitting a new post into the prediction cohort.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_datetime(value: Any) -> datetime | None:
    text = clean_text(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def isoformat_or_empty(value: datetime | None) -> str:
    return value.astimezone(timezone.utc).isoformat() if value else ""


def normalize_post_id(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return text if text.lower().startswith("t3_") else f"t3_{text}"


def parse_reddit_post_target(url: str) -> tuple[str, str]:
    text = clean_text(url)
    if not text:
        return "", ""
    match = REDDIT_COMMENTS_URL_RE.search(text)
    if not match:
        return "", ""
    return clean_text(match.group("subreddit")).lower(), clean_text(match.group("post_id"))


def canonical_key(row: dict[str, Any]) -> tuple[str, str] | None:
    url_subreddit, url_post_id = parse_reddit_post_target(row.get("url", ""))
    subreddit = url_subreddit or clean_text(row.get("subreddit")).lower()
    post_id = normalize_post_id(row.get("post_id")) or normalize_post_id(url_post_id)
    if not subreddit or not post_id:
        return None
    return subreddit, post_id


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Latest status CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Latest status CSV is empty.")
    required = {"subreddit", "post_id", "url", "latest_activity_state", "last_seen_time_utc"}
    if not required.issubset(rows[0].keys()):
        joined = ", ".join(sorted(required))
        raise SystemExit(f"Latest status CSV must contain: {joined}")
    return rows


def load_optional_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_status_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = canonical_key(row)
        if key is None:
            continue
        latest[key] = dict(row)
    return latest


def compute_reference_now(rows: list[dict[str, Any]]) -> datetime:
    times = [
        parse_datetime(row.get("last_seen_time_utc"))
        for row in rows
        if parse_datetime(row.get("last_seen_time_utc")) is not None
    ]
    if times:
        return max(times)
    return datetime.now(tz=timezone.utc)


def admission_age_hours(row: dict[str, Any]) -> float | None:
    age_minutes = parse_float(row.get("age_at_last_seen_minutes"))
    if age_minutes is not None:
        return age_minutes / 60.0
    created_at = parse_datetime(row.get("created_at"))
    last_seen = parse_datetime(row.get("last_seen_time_utc"))
    if created_at is None or last_seen is None:
        return None
    return max(0.0, (last_seen - created_at).total_seconds() / 3600.0)


def prediction_end_time(row: dict[str, Any], admitted_at: datetime, target_hours: float) -> datetime:
    created_at = parse_datetime(row.get("created_at"))
    if created_at is not None:
        return created_at + timedelta(hours=target_hours)
    return admitted_at + timedelta(hours=target_hours)


def is_prediction_admission_eligible(row: dict[str, Any], max_age_hours: float) -> bool:
    if not clean_text(row.get("url")):
        return False
    if canonical_key(row) is None:
        return False
    age_hours = admission_age_hours(row)
    if age_hours is None or age_hours > max_age_hours:
        return False
    return True


def prediction_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    state = clean_text(row.get("latest_activity_state")).lower()
    analysis = clean_text(row.get("analysis_priority")).lower()
    age_hours = admission_age_hours(row)
    snapshot_count = parse_int(row.get("snapshot_count")) or 0
    return (
        0 if clean_text(row.get("fresh_at_first_seen_1h")) == "1" else 1,
        0 if clean_text(row.get("fresh_at_first_seen_6h")) == "1" else 1,
        ANALYSIS_PRIORITY.get(analysis, 9),
        STATE_PRIORITY.get(state, 9),
        0 if clean_text(row.get("seen_in_new")) == "1" else 1,
        age_hours if age_hours is not None else 10**9,
        snapshot_count,
        -(parse_float(row.get("current_attention_score")) or 0.0),
        -(parse_float(row.get("last_comment_velocity_per_hour")) or 0.0),
        -(parse_float(row.get("last_upvote_velocity_per_hour")) or 0.0),
    )


def live_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    state = clean_text(row.get("latest_activity_state")).lower()
    analysis = clean_text(row.get("analysis_priority")).lower()
    last_seen = parse_datetime(row.get("last_seen_time_utc"))
    age_minutes = parse_float(row.get("age_at_last_seen_minutes"))
    snapshot_count = parse_int(row.get("snapshot_count")) or 0
    return (
        ANALYSIS_PRIORITY.get(analysis, 9),
        STATE_PRIORITY.get(state, 9),
        0 if clean_text(row.get("seen_in_new")) == "1" else 1,
        0 if clean_text(row.get("seen_in_rising")) == "1" else 1,
        0 if clean_text(row.get("seen_in_hot")) == "1" else 1,
        age_minutes if age_minutes is not None else 10**12,
        -(last_seen.timestamp() if last_seen else 0),
        snapshot_count,
        -(parse_float(row.get("last_comment_velocity_per_hour")) or 0.0),
        -(parse_float(row.get("last_upvote_velocity_per_hour")) or 0.0),
    )


def merge_row(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update({key: value for key, value in extra.items() if clean_text(value) != ""})
    return merged


def annotate_prediction_row(
    row: dict[str, Any],
    *,
    admitted_at: datetime,
    expires_at: datetime,
    target_hours: float,
    retained: bool,
) -> dict[str, Any]:
    return {
        **dict(row),
        "tracking_window": "prediction_cohort",
        "pool_role": "prediction_cohort",
        "cohort_admitted_time_utc": isoformat_or_empty(admitted_at),
        "cohort_expires_at_utc": isoformat_or_empty(expires_at),
        "cohort_target_hours": f"{target_hours:.2f}",
        "cohort_retained_from_previous_run": "1" if retained else "0",
        "cohort_is_active": "1",
    }


def annotate_live_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        "tracking_window": "live_watch",
        "pool_role": "live_watch",
        "cohort_admitted_time_utc": "",
        "cohort_expires_at_utc": "",
        "cohort_target_hours": "",
        "cohort_retained_from_previous_run": "",
        "cohort_is_active": "",
    }


def select_prediction_cohort(
    rows: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
    *,
    target_hours: float,
    max_age_hours: float,
    max_posts: int,
    per_subreddit_limit: int,
    reference_now: datetime,
) -> list[dict[str, Any]]:
    latest_by_key = latest_status_by_key(rows)
    previous_by_key = {
        key: row
        for row in previous_rows
        if (key := canonical_key(row)) is not None
    }

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    selected_keys: set[tuple[str, str]] = set()

    carried_rows: list[dict[str, Any]] = []
    for key, previous in previous_by_key.items():
        expires_at = parse_datetime(previous.get("cohort_expires_at_utc"))
        if expires_at is None or reference_now > expires_at:
            continue
        current = latest_by_key.get(key)
        merged = merge_row(previous, current or {})
        admitted_at = parse_datetime(merged.get("cohort_admitted_time_utc")) or reference_now
        subreddit, _post_id = key
        if counts[subreddit] >= per_subreddit_limit:
            continue
        carried_rows.append(
            annotate_prediction_row(
                merged,
                admitted_at=admitted_at,
                expires_at=expires_at,
                target_hours=target_hours,
                retained=True,
            )
        )
        counts[subreddit] += 1
        selected_keys.add(key)
        if len(carried_rows) >= max_posts:
            break

    carried_rows.sort(
        key=lambda row: (
            parse_datetime(row.get("cohort_admitted_time_utc")) or reference_now,
            clean_text(row.get("subreddit")).lower(),
            clean_text(row.get("post_id")).lower(),
        )
    )
    selected.extend(carried_rows)

    admission_candidates = []
    for row in rows:
        key = canonical_key(row)
        if key is None or key in selected_keys:
            continue
        subreddit, _post_id = key
        if counts[subreddit] >= per_subreddit_limit:
            continue
        if not is_prediction_admission_eligible(row, max_age_hours):
            continue
        admission_candidates.append(dict(row))
    admission_candidates.sort(key=prediction_sort_key)

    for row in admission_candidates:
        key = canonical_key(row)
        if key is None or key in selected_keys:
            continue
        subreddit, _post_id = key
        if counts[subreddit] >= per_subreddit_limit or len(selected) >= max_posts:
            continue
        admitted_at = reference_now
        expires_at = prediction_end_time(row, admitted_at, target_hours)
        if reference_now > expires_at:
            continue
        selected.append(
            annotate_prediction_row(
                row,
                admitted_at=admitted_at,
                expires_at=expires_at,
                target_hours=target_hours,
                retained=False,
            )
        )
        counts[subreddit] += 1
        selected_keys.add(key)
    return selected


def select_live_watch_pool(
    rows: list[dict[str, Any]],
    *,
    excluded_keys: set[tuple[str, str]],
    max_posts: int,
    per_subreddit_limit: int,
) -> list[dict[str, Any]]:
    eligible_rows = []
    for row in rows:
        key = canonical_key(row)
        if key is None or key in excluded_keys:
            continue
        if not clean_text(row.get("url")):
            continue
        if clean_text(row.get("latest_activity_state")).lower() == "dead":
            continue
        eligible_rows.append(dict(row))

    ordered = sorted(eligible_rows, key=live_sort_key)
    counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    selected: list[dict[str, Any]] = []

    for row in ordered:
        key = canonical_key(row)
        if key is None or key in seen:
            continue
        subreddit, _post_id = key
        if counts[subreddit] >= per_subreddit_limit:
            continue
        selected.append(annotate_live_row(row))
        counts[subreddit] += 1
        seen.add(key)
        if len(selected) >= max_posts:
            break
    return selected


def rank_rows(rows: list[dict[str, Any]], *, prefix: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        subreddit = clean_text(row.get("subreddit")).lower()
        counts[subreddit] += 1
        ranked.append(
            {
                **dict(row),
                f"{prefix}_rank": index,
                f"{prefix}_rank_in_subreddit": counts[subreddit],
            }
        )
    return ranked


def combine_pools(
    prediction_rows: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
    *,
    max_posts: int,
    per_subreddit_limit: int,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    combined: list[dict[str, Any]] = []

    for row in [*prediction_rows, *live_rows]:
        key = canonical_key(row)
        if key is None or key in seen:
            continue
        subreddit, _post_id = key
        if counts[subreddit] >= per_subreddit_limit:
            continue
        seen.add(key)
        counts[subreddit] += 1
        combined.append(dict(row))
        if len(combined) >= max_posts:
            break

    return rank_rows(combined, prefix="observation_pool")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    combined_output_path = Path(args.output)
    prediction_output_path = Path(args.prediction_output)
    live_output_path = Path(args.live_output)

    rows = load_rows(source_path)
    previous_prediction_rows = load_optional_rows(prediction_output_path)
    reference_now = compute_reference_now(rows)

    prediction_rows = select_prediction_cohort(
        rows,
        previous_prediction_rows,
        target_hours=args.prediction_target_hours,
        max_age_hours=args.prediction_max_admission_age_hours,
        max_posts=args.prediction_max_posts,
        per_subreddit_limit=args.prediction_per_subreddit_limit,
        reference_now=reference_now,
    )
    prediction_rows = rank_rows(prediction_rows, prefix="prediction_cohort")

    prediction_keys = {
        key
        for row in prediction_rows
        if (key := canonical_key(row)) is not None
    }
    live_rows = select_live_watch_pool(
        rows,
        excluded_keys=prediction_keys,
        max_posts=args.live_max_posts,
        per_subreddit_limit=args.live_per_subreddit_limit,
    )
    live_rows = rank_rows(live_rows, prefix="live_watch")

    combined_rows = combine_pools(
        prediction_rows,
        live_rows,
        max_posts=args.max_posts,
        per_subreddit_limit=args.per_subreddit_limit,
    )

    if not combined_rows:
        raise SystemExit("No posts were selected for the combined observation pool.")

    write_rows(prediction_output_path, prediction_rows)
    write_rows(live_output_path, live_rows)
    write_rows(combined_output_path, combined_rows)

    prediction_new = sum(clean_text(row.get("cohort_retained_from_previous_run")) != "1" for row in prediction_rows)
    prediction_retained = len(prediction_rows) - prediction_new
    print(f"Selected {len(prediction_rows)} post(s) into the prediction cohort.")
    print(f"- retained from prior run: {prediction_retained}")
    print(f"- newly admitted this run: {prediction_new}")
    print(f"Prediction cohort written to {prediction_output_path}")
    print(f"Selected {len(live_rows)} post(s) into the live-watch pool.")
    print(f"Live-watch pool written to {live_output_path}")
    print(f"Combined observation pool written to {combined_output_path}")


if __name__ == "__main__":
    main()
