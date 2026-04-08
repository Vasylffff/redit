from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import ceil, floor, log1p
from pathlib import Path
from statistics import mean, median
from typing import Any

from normalize_reddit_json import (
    author_name,
    clean_text,
    extract_domain,
    isoformat_or_empty,
    load_items,
    parse_datetime,
)


CONFIG_NAME_RE = re.compile(
    r"reddit_r_(?P<subreddit>.+)_(?P<suffix>new|hot|rising|top_day|top_week)\.json$",
    re.IGNORECASE,
)

RAW_TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}_\d{6})_")
REDDIT_COMMENTS_URL_RE = re.compile(
    r"(?:https?://(?:www\.)?reddit\.com)?/r/(?P<subreddit>[^/]+)/comments/(?P<post_id>[A-Za-z0-9_]+)/",
    re.IGNORECASE,
)
DEFAULT_EXCLUDED_SUBREDDITS = ("pasta",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build historical Reddit flow tables from all saved raw JSON files, "
            "combining previous snapshots with newly collected ones."
        )
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/reddit_json",
        help="Directory containing raw Reddit JSON files and sibling metadata JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/history/reddit",
        help="Directory where merged history CSV files will be written.",
    )
    parser.add_argument(
        "--exclude-subreddits",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_SUBREDDITS),
        help=(
            "Optional subreddit names to exclude from the merged history. "
            "Defaults to known test/noise subreddits."
        ),
    )
    return parser.parse_args()


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def safe_sum(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return float(ordered[lower])
    blend = position - lower
    return float((ordered[lower] * (1 - blend)) + (ordered[upper] * blend))


def parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = clean_text(value)
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def normalize_subreddit(value: str) -> str:
    text = clean_text(value)
    if text.lower().startswith("r/"):
        return text[2:]
    return text


def parse_reddit_post_target(url: str) -> tuple[str, str]:
    text = clean_text(url)
    if not text:
        return "", ""
    match = REDDIT_COMMENTS_URL_RE.search(text)
    if not match:
        return "", ""
    return normalize_subreddit(match.group("subreddit")), clean_text(match.group("post_id"))


def normalize_post_id(
    value: Any,
    *,
    parsed_value: Any = None,
    url: str = "",
) -> str:
    direct = clean_text(value)
    parsed = clean_text(parsed_value)
    _url_subreddit, url_post_id = parse_reddit_post_target(url)
    candidate = direct or parsed or url_post_id
    if not candidate:
        return ""
    return candidate if candidate.lower().startswith("t3_") else f"t3_{candidate}"


def normalize_comment_id(value: Any, *, parsed_value: Any = None) -> str:
    direct = clean_text(value)
    parsed = clean_text(parsed_value)
    candidate = direct or parsed
    if not candidate:
        return ""
    return candidate if candidate.lower().startswith("t1_") else f"t1_{candidate}"


def infer_post_subreddit(
    *,
    url: str,
    parsed_community_name: Any,
    community_name: Any,
    category: Any,
    fallback_subreddit: str,
) -> str:
    url_subreddit, _url_post_id = parse_reddit_post_target(url)
    return (
        url_subreddit
        or normalize_subreddit(clean_text(parsed_community_name))
        or normalize_subreddit(clean_text(community_name))
        or normalize_subreddit(clean_text(category))
        or normalize_subreddit(fallback_subreddit)
    )


def canonical_post_group_key(row: dict[str, Any]) -> tuple[str, str] | None:
    subreddit = infer_post_subreddit(
        url=clean_text(row.get("url")),
        parsed_community_name=row.get("subreddit"),
        community_name=row.get("subreddit"),
        category=row.get("subreddit"),
        fallback_subreddit=clean_text(row.get("subreddit")),
    )
    post_id = normalize_post_id(
        row.get("post_id"),
        parsed_value=row.get("parsed_post_id"),
        url=clean_text(row.get("url")),
    )
    if not subreddit or not post_id:
        return None
    return subreddit, post_id


def listing_type(listing: str, top_time: str) -> str:
    base = clean_text(listing)
    period = clean_text(top_time)
    if base == "top" and period:
        return f"top_{period}"
    return base


def parse_raw_timestamp_from_name(path: Path) -> datetime | None:
    match = RAW_TIMESTAMP_RE.match(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize_subreddit_names(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    return {normalize_subreddit(value).lower() for value in values if clean_text(value)}


def infer_from_input_file(metadata: dict[str, Any]) -> tuple[str, str, str]:
    input_file = clean_text(metadata.get("input_file"))
    if not input_file:
        return "", "", ""
    match = CONFIG_NAME_RE.search(Path(input_file).name)
    if not match:
        return "", "", ""
    subreddit = normalize_subreddit(match.group("subreddit"))
    suffix = match.group("suffix").lower()
    if suffix == "top_day":
        return subreddit, "top", "day"
    if suffix == "top_week":
        return subreddit, "top", "week"
    return subreddit, suffix, ""


def infer_snapshot_context(
    *,
    raw_file: Path,
    metadata: dict[str, Any],
    posts: list[dict[str, Any]],
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    inferred_subreddit, inferred_listing, inferred_top_time = infer_from_input_file(metadata)

    first_post = posts[0] if posts else {}
    subreddit = (
        normalize_subreddit(clean_text(metadata.get("subreddit")))
        or inferred_subreddit
        or normalize_subreddit(clean_text(first_post.get("parsedCommunityName")))
        or normalize_subreddit(clean_text(first_post.get("communityName")))
        or normalize_subreddit(clean_text(first_post.get("category")))
    )
    listing = clean_text(metadata.get("listing")) or inferred_listing
    top_time = clean_text(metadata.get("top_time")) or inferred_top_time
    listing_label = listing_type(listing, top_time)

    saved_at = parse_datetime(clean_text(metadata.get("saved_at")))
    if saved_at is None:
        saved_at = parse_raw_timestamp_from_name(raw_file)
    if saved_at is None:
        first_scraped = parse_datetime(clean_text(first_post.get("scrapedAt")))
        saved_at = first_scraped

    return {
        "snapshot_id": raw_file.stem,
        "snapshot_time_dt": saved_at,
        "snapshot_time_utc": isoformat_or_empty(saved_at),
        "subreddit": subreddit,
        "listing": listing,
        "top_time": top_time,
        "listing_type": listing_label,
        "source_type": clean_text(metadata.get("source_type")),
        "schedule_name": clean_text(metadata.get("schedule_name")),
        "cadence_label": clean_text(metadata.get("cadence_label")),
        "scheduled_hour": clean_text(metadata.get("scheduled_hour")),
        "input_file": clean_text(metadata.get("input_file")),
        "batch_manifest": clean_text(metadata.get("batch_manifest")),
        "schedule_plan": clean_text(metadata.get("schedule_plan")),
        "output_file": clean_text(metadata.get("output_file")) or str(raw_file),
        "metadata_file": str(raw_file.with_name(f"{raw_file.stem}_metadata.json")),
        "item_count": metadata.get("item_count"),
        "post_count": len(posts),
        "comment_count": len(comments),
    }


def age_bucket(age_minutes: float | None) -> str:
    if age_minutes is None:
        return "unknown"
    if age_minutes < 30:
        return "under_30m"
    if age_minutes < 60:
        return "30m_to_1h"
    if age_minutes < 180:
        return "1h_to_3h"
    if age_minutes < 360:
        return "3h_to_6h"
    if age_minutes < 720:
        return "6h_to_12h"
    if age_minutes < 1440:
        return "12h_to_24h"
    return "over_24h"


def classify_activity_state(
    *,
    age_minutes: float | None,
    hours_since_previous_snapshot: float | None,
    upvote_delta_from_previous_snapshot: float | None,
    comment_delta_from_previous_snapshot: float | None,
    still_visible_next_snapshot: int | None,
    alive_upvote_velocity_threshold: float,
    alive_comment_velocity_threshold: float,
    surging_upvote_velocity_threshold: float,
    surging_comment_velocity_threshold: float,
    dead_upvote_velocity_threshold: float,
    dead_comment_velocity_threshold: float,
) -> str:
    dying_min_age_minutes = 720.0
    dead_min_age_minutes = 2160.0
    near_zero_dead_multiplier = 0.35

    if hours_since_previous_snapshot is None:
        if age_minutes is not None and age_minutes <= 60:
            return "emerging"
        return "unknown"

    upvote_velocity = (
        upvote_delta_from_previous_snapshot / hours_since_previous_snapshot
        if upvote_delta_from_previous_snapshot is not None
        and hours_since_previous_snapshot > 0
        else None
    )
    comment_velocity = (
        comment_delta_from_previous_snapshot / hours_since_previous_snapshot
        if comment_delta_from_previous_snapshot is not None
        and hours_since_previous_snapshot > 0
        else None
    )
    positive_growth = (
        (upvote_delta_from_previous_snapshot or 0) > 0
        or (comment_delta_from_previous_snapshot or 0) > 0
    )
    above_dead_floor = (
        (upvote_velocity or 0) > dead_upvote_velocity_threshold
        or (comment_velocity or 0) > dead_comment_velocity_threshold
    )
    low_motion = not above_dead_floor
    near_flat_motion = (
        (upvote_velocity or 0) <= (dead_upvote_velocity_threshold * near_zero_dead_multiplier)
        and (comment_velocity or 0) <= (dead_comment_velocity_threshold * near_zero_dead_multiplier)
    )

    if (comment_velocity or 0) >= surging_comment_velocity_threshold or (
        upvote_velocity or 0
    ) >= surging_upvote_velocity_threshold:
        return "surging"
    if (comment_velocity or 0) >= alive_comment_velocity_threshold or (
        upvote_velocity or 0
    ) >= alive_upvote_velocity_threshold:
        return "alive"
    if positive_growth and above_dead_floor:
        return "alive"
    if (
        age_minutes is not None
        and age_minutes >= dead_min_age_minutes
        and low_motion
        and (near_flat_motion or not positive_growth)
    ):
        return "dead"
    if (
        age_minutes is not None
        and age_minutes >= dying_min_age_minutes
        and low_motion
        and (near_flat_motion or not positive_growth)
    ):
        return "dying"
    if still_visible_next_snapshot == 1:
        return "cooling"
    if age_minutes is not None and age_minutes <= 60:
        return "emerging"
    return "cooling"


def analysis_priority_for_state(activity_state: str, age_minutes: float | None) -> str:
    state = clean_text(activity_state)
    if state == "surging":
        return "highest"
    if state in {"alive", "emerging"}:
        return "high"
    if state == "dying":
        return "low"
    if state == "cooling":
        if age_minutes is not None and age_minutes <= 360:
            return "medium"
        return "low"
    if state == "dead":
        return "very_low"
    return "low"


def stabilize_lifecycle_state(
    *,
    latest_state: str,
    snapshot_count: int,
    age_minutes: float | None,
    last_upvote_velocity_per_hour: float | None,
    last_comment_velocity_per_hour: float | None,
    dead_upvote_velocity_threshold: float,
    dead_comment_velocity_threshold: float,
) -> str:
    dead_min_age_minutes = 720.0
    state = clean_text(latest_state)
    if state in {"surging", "alive", "emerging"}:
        return state
    if snapshot_count < 2:
        return "unknown"
    if state == "cooling" and snapshot_count < 3:
        return "unknown"
    if state == "dying":
        if snapshot_count < 3:
            return "unknown"
        upvote_velocity = last_upvote_velocity_per_hour or 0.0
        comment_velocity = last_comment_velocity_per_hour or 0.0
        if (
            upvote_velocity > dead_upvote_velocity_threshold
            or comment_velocity > dead_comment_velocity_threshold
        ):
            return "cooling"
        return "dying"
    if state == "dead":
        if snapshot_count < 3:
            return "unknown"
        upvote_velocity = last_upvote_velocity_per_hour or 0.0
        comment_velocity = last_comment_velocity_per_hour or 0.0
        if age_minutes is not None and age_minutes < dead_min_age_minutes:
            return "dying"
        if (
            upvote_velocity > dead_upvote_velocity_threshold
            or comment_velocity > dead_comment_velocity_threshold
        ):
            return "dying"
    return state


def default_activity_thresholds() -> dict[str, Any]:
    return {
        "threshold_source": "default",
        "sample_count": 0,
        "positive_upvote_samples": 0,
        "positive_comment_samples": 0,
        "alive_upvote_velocity_threshold": 20.0,
        "alive_comment_velocity_threshold": 2.0,
        "surging_upvote_velocity_threshold": 100.0,
        "surging_comment_velocity_threshold": 10.0,
        "dead_upvote_velocity_threshold": 5.0,
        "dead_comment_velocity_threshold": 0.5,
    }


def derive_activity_thresholds(
    *,
    all_upvote_velocities: list[float],
    all_comment_velocities: list[float],
    fallback: dict[str, Any] | None = None,
    threshold_source: str,
) -> dict[str, Any]:
    base = dict(fallback or default_activity_thresholds())
    positive_upvote_velocities = [value for value in all_upvote_velocities if value > 0]
    positive_comment_velocities = [value for value in all_comment_velocities if value > 0]

    alive_upvote = percentile(positive_upvote_velocities, 0.35)
    alive_comment = percentile(positive_comment_velocities, 0.35)
    surging_upvote = percentile(positive_upvote_velocities, 0.80)
    surging_comment = percentile(positive_comment_velocities, 0.80)
    dead_upvote = percentile(positive_upvote_velocities, 0.10)
    dead_comment = percentile(positive_comment_velocities, 0.10)

    alive_upvote = max(1.0, alive_upvote if alive_upvote is not None else base["alive_upvote_velocity_threshold"])
    alive_comment = max(0.25, alive_comment if alive_comment is not None else base["alive_comment_velocity_threshold"])

    surging_upvote = max(
        alive_upvote * 3.0,
        surging_upvote if surging_upvote is not None else base["surging_upvote_velocity_threshold"],
    )
    surging_comment = max(
        alive_comment * 3.0,
        surging_comment if surging_comment is not None else base["surging_comment_velocity_threshold"],
    )

    dead_upvote = min(
        alive_upvote * 0.5,
        dead_upvote if dead_upvote is not None else base["dead_upvote_velocity_threshold"],
    )
    dead_comment = min(
        alive_comment * 0.5,
        dead_comment if dead_comment is not None else base["dead_comment_velocity_threshold"],
    )

    return {
        "threshold_source": threshold_source,
        "sample_count": len(all_upvote_velocities) or len(all_comment_velocities),
        "positive_upvote_samples": len(positive_upvote_velocities),
        "positive_comment_samples": len(positive_comment_velocities),
        "alive_upvote_velocity_threshold": float(alive_upvote),
        "alive_comment_velocity_threshold": float(alive_comment),
        "surging_upvote_velocity_threshold": float(surging_upvote),
        "surging_comment_velocity_threshold": float(surging_comment),
        "dead_upvote_velocity_threshold": float(max(0.1, dead_upvote)),
        "dead_comment_velocity_threshold": float(max(0.05, dead_comment)),
    }


def build_activity_thresholds(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    all_subreddits = sorted(
        {clean_text(row.get("subreddit")) for row in rows if clean_text(row.get("subreddit"))}
    )
    relevant_rows = [
        row
        for row in rows
        if row.get("seen_in_previous_snapshot") == 1
        and clean_text(row.get("listing_type"))
    ]
    by_subreddit: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"upvotes": [], "comments": []}
    )
    global_upvotes: list[float] = []
    global_comments: list[float] = []

    for row in relevant_rows:
        subreddit = clean_text(row.get("subreddit"))
        upvote_velocity = parse_float(row.get("upvote_velocity_per_hour"))
        comment_velocity = parse_float(row.get("comment_velocity_per_hour"))
        if upvote_velocity is not None:
            by_subreddit[subreddit]["upvotes"].append(upvote_velocity)
            global_upvotes.append(upvote_velocity)
        if comment_velocity is not None:
            by_subreddit[subreddit]["comments"].append(comment_velocity)
            global_comments.append(comment_velocity)

    default_thresholds = default_activity_thresholds()
    global_thresholds = derive_activity_thresholds(
        all_upvote_velocities=global_upvotes,
        all_comment_velocities=global_comments,
        fallback=default_thresholds,
        threshold_source="global_empirical" if (global_upvotes or global_comments) else "default",
    )

    threshold_map: dict[str, dict[str, Any]] = {"__global__": global_thresholds}
    threshold_rows: list[dict[str, Any]] = [
        {"subreddit": "__global__", **global_thresholds}
    ]

    for subreddit in all_subreddits:
        values = by_subreddit.get(subreddit, {"upvotes": [], "comments": []})
        positive_upvotes = [value for value in values["upvotes"] if value > 0]
        positive_comments = [value for value in values["comments"] if value > 0]
        enough_data = (
            len(values["upvotes"]) >= 30
            and len(positive_upvotes) >= 15
            and len(positive_comments) >= 10
        )
        if enough_data:
            thresholds = derive_activity_thresholds(
                all_upvote_velocities=values["upvotes"],
                all_comment_velocities=values["comments"],
                fallback=global_thresholds,
                threshold_source="subreddit_empirical",
            )
        else:
            thresholds = dict(global_thresholds)
            thresholds["threshold_source"] = "global_fallback"
            thresholds["sample_count"] = len(values["upvotes"]) or len(values["comments"])
            thresholds["positive_upvote_samples"] = len(positive_upvotes)
            thresholds["positive_comment_samples"] = len(positive_comments)
        threshold_map[subreddit] = thresholds
        threshold_rows.append({"subreddit": subreddit, **thresholds})

    return threshold_map, threshold_rows


def build_post_snapshot_rows(
    *,
    posts: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshot_time = context["snapshot_time_dt"]
    for rank, post in enumerate(posts, start=1):
        created_at = parse_datetime(clean_text(post.get("createdAt")))
        scraped_at = parse_datetime(clean_text(post.get("scrapedAt")))
        if snapshot_time and created_at:
            age_minutes = (snapshot_time - created_at).total_seconds() / 60
        elif scraped_at and created_at:
            age_minutes = (scraped_at - created_at).total_seconds() / 60
        else:
            age_minutes = None

        link = clean_text(post.get("link"))
        url = clean_text(post.get("url"))
        image_urls = post.get("imageUrls", []) or []
        post_subreddit = infer_post_subreddit(
            url=url,
            parsed_community_name=post.get("parsedCommunityName"),
            community_name=post.get("communityName"),
            category=post.get("category"),
            fallback_subreddit=context["subreddit"],
        )
        post_id = normalize_post_id(
            post.get("id"),
            parsed_value=post.get("parsedId"),
            url=url,
        )

        rows.append(
            {
                "snapshot_id": context["snapshot_id"],
                "snapshot_time_utc": context["snapshot_time_utc"],
                "subreddit": post_subreddit,
                "listing": context["listing"],
                "top_time": context["top_time"],
                "listing_type": context["listing_type"],
                "schedule_name": context["schedule_name"],
                "cadence_label": context["cadence_label"],
                "scheduled_hour": context["scheduled_hour"],
                "post_id": post_id,
                "parsed_post_id": clean_text(post.get("parsedId")),
                "url": url,
                "external_link": link,
                "link_domain": extract_domain(link or url),
                "title": clean_text(post.get("title")),
                "body": clean_text(post.get("body")),
                "author": author_name(post.get("username")),
                "author_id": clean_text(post.get("userId")),
                "created_at": isoformat_or_empty(created_at),
                "scraped_at": isoformat_or_empty(scraped_at),
                "age_minutes_at_snapshot": age_minutes,
                "age_bucket": age_bucket(age_minutes),
                "is_fresh_30m": int(age_minutes is not None and age_minutes <= 30),
                "is_fresh_1h": int(age_minutes is not None and age_minutes <= 60),
                "is_fresh_3h": int(age_minutes is not None and age_minutes <= 180),
                "is_fresh_6h": int(age_minutes is not None and age_minutes <= 360),
                "is_old_12h": int(age_minutes is not None and age_minutes >= 720),
                "is_old_24h": int(age_minutes is not None and age_minutes >= 1440),
                "upvotes_at_snapshot": post.get("upVotes"),
                "comment_count_at_snapshot": post.get("numberOfComments"),
                "upvote_ratio_at_snapshot": post.get("upVoteRatio"),
                "flair": clean_text(post.get("flair")),
                "is_video": bool(post.get("isVideo", False)),
                "has_images": bool(image_urls),
                "rank_within_snapshot": rank,
                "seen_in_snapshot": 1,
                "source_type": context["source_type"],
                "input_file": context["input_file"],
            }
        )
    return rows


def build_comment_snapshot_rows(
    *,
    comments: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshot_time = context["snapshot_time_dt"]
    for comment in comments:
        created_at = parse_datetime(clean_text(comment.get("createdAt")))
        scraped_at = parse_datetime(clean_text(comment.get("scrapedAt")))
        if snapshot_time and created_at:
            age_minutes = (snapshot_time - created_at).total_seconds() / 60
        elif scraped_at and created_at:
            age_minutes = (scraped_at - created_at).total_seconds() / 60
        else:
            age_minutes = None

        url = clean_text(comment.get("url"))
        post_id = normalize_post_id(comment.get("postId"))
        comment_id = normalize_comment_id(comment.get("id"), parsed_value=comment.get("parsedId"))
        parent_id = clean_text(comment.get("parentId"))
        subreddit = (
            normalize_subreddit(clean_text(comment.get("category")))
            or normalize_subreddit(clean_text(comment.get("communityName")))
            or context["subreddit"]
        )
        body = clean_text(comment.get("body"))
        rows.append(
            {
                "snapshot_id": context["snapshot_id"],
                "snapshot_time_utc": context["snapshot_time_utc"],
                "subreddit": subreddit,
                "listing": context["listing"],
                "top_time": context["top_time"],
                "listing_type": context["listing_type"],
                "schedule_name": context["schedule_name"],
                "cadence_label": context["cadence_label"],
                "scheduled_hour": context["scheduled_hour"],
                "post_id": post_id,
                "comment_id": comment_id,
                "parsed_comment_id": clean_text(comment.get("parsedId")),
                "parent_id": parent_id,
                "is_top_level_comment": int(parent_id.lower().startswith("t3_")) if parent_id else 0,
                "url": url,
                "author": author_name(comment.get("username")),
                "author_id": clean_text(comment.get("userId")),
                "body": body,
                "body_length_chars": len(body),
                "body_word_count": len(body.split()) if body else 0,
                "created_at": isoformat_or_empty(created_at),
                "scraped_at": isoformat_or_empty(scraped_at),
                "age_minutes_at_snapshot": age_minutes,
                "age_bucket": age_bucket(age_minutes),
                "upvotes_at_snapshot": comment.get("upVotes"),
                "reply_count_at_snapshot": comment.get("numberOfreplies"),
                "source_type": context["source_type"],
                "input_file": context["input_file"],
            }
        )
    return rows


def build_comment_post_aggregates(
    comment_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comment_rows:
        subreddit = normalize_subreddit(clean_text(row.get("subreddit")))
        post_id = normalize_post_id(row.get("post_id"))
        if not subreddit or not post_id:
            continue
        grouped[(subreddit, post_id)].append(row)

    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        per_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            per_snapshot[clean_text(row.get("snapshot_id"))].append(row)

        ordered_snapshots = sorted(
            per_snapshot.items(),
            key=lambda item: (
                parse_datetime(item[1][0].get("snapshot_time_utc")) or datetime.min.replace(tzinfo=timezone.utc),
                item[0],
            ),
        )
        latest_snapshot_id, latest_rows = ordered_snapshots[-1]

        all_scores = [parse_float(row.get("upvotes_at_snapshot")) for row in rows]
        all_replies = [parse_float(row.get("reply_count_at_snapshot")) for row in rows]
        latest_scores = [parse_float(row.get("upvotes_at_snapshot")) for row in latest_rows]
        latest_replies = [parse_float(row.get("reply_count_at_snapshot")) for row in latest_rows]
        latest_unique_commenters = {
            clean_text(row.get("author")).lower()
            for row in latest_rows
            if clean_text(row.get("author"))
        }
        latest_top_level_count = sum(
            1 for row in latest_rows if str(row.get("is_top_level_comment", "")).strip() == "1"
        )
        latest_reply_count = len(latest_rows) - latest_top_level_count

        max_sample_size_seen = max(len(snapshot_rows) for _snapshot_id, snapshot_rows in ordered_snapshots)
        max_total_comment_upvotes_seen = max(
            (
                sum(parse_float(row.get("upvotes_at_snapshot")) or 0.0 for row in snapshot_rows)
                for _snapshot_id, snapshot_rows in ordered_snapshots
            ),
            default=0.0,
        )
        max_total_comment_replies_seen = max(
            (
                sum(parse_float(row.get("reply_count_at_snapshot")) or 0.0 for row in snapshot_rows)
                for _snapshot_id, snapshot_rows in ordered_snapshots
            ),
            default=0.0,
        )

        aggregates[key] = {
            "comment_snapshot_count": len(ordered_snapshots),
            "latest_comment_snapshot_id": latest_snapshot_id,
            "latest_comment_snapshot_time_utc": clean_text(latest_rows[0].get("snapshot_time_utc")),
            "latest_comment_sample_count": len(latest_rows),
            "latest_top_level_comment_sample_count": latest_top_level_count,
            "latest_reply_comment_sample_count": latest_reply_count,
            "latest_unique_commenter_count": len(latest_unique_commenters),
            "latest_total_comment_upvotes": safe_sum(
                [value for value in latest_scores if value is not None]
            ),
            "latest_total_comment_replies": safe_sum(
                [value for value in latest_replies if value is not None]
            ),
            "latest_avg_comment_upvotes": safe_mean(
                [value for value in latest_scores if value is not None]
            ),
            "latest_max_comment_upvotes": max(
                (value for value in latest_scores if value is not None),
                default=None,
            ),
            "latest_max_comment_replies": max(
                (value for value in latest_replies if value is not None),
                default=None,
            ),
            "total_sampled_comment_upvotes": safe_sum(
                [value for value in all_scores if value is not None]
            ),
            "total_sampled_comment_replies": safe_sum(
                [value for value in all_replies if value is not None]
            ),
            "max_comment_upvotes_seen": max(
                (value for value in all_scores if value is not None),
                default=None,
            ),
            "max_comment_replies_seen": max(
                (value for value in all_replies if value is not None),
                default=None,
            ),
            "max_comment_sample_count_seen": max_sample_size_seen,
            "max_total_comment_upvotes_seen": float(max_total_comment_upvotes_seen),
            "max_total_comment_replies_seen": float(max_total_comment_replies_seen),
        }

    return aggregates


def compute_comment_engagement_score(
    *,
    latest_comment_sample_count: float | None,
    latest_total_comment_upvotes: float | None,
    latest_total_comment_replies: float | None,
    latest_unique_commenter_count: float | None,
    max_comment_upvotes_seen: float | None,
    max_comment_replies_seen: float | None,
) -> float:
    return float(
        (1.2 * log1p(max(0.0, latest_comment_sample_count or 0.0)))
        + (1.3 * log1p(max(0.0, latest_total_comment_upvotes or 0.0)))
        + (1.4 * log1p(max(0.0, latest_total_comment_replies or 0.0)))
        + (0.9 * log1p(max(0.0, latest_unique_commenter_count or 0.0)))
        + (0.7 * log1p(max(0.0, max_comment_upvotes_seen or 0.0)))
        + (0.8 * log1p(max(0.0, max_comment_replies_seen or 0.0)))
    )


def compute_general_popularity_score(
    *,
    max_upvotes: float | None,
    max_comments: float | None,
    total_upvote_growth: float | None,
    total_comment_growth: float | None,
    last_upvote_velocity_per_hour: float | None,
    last_comment_velocity_per_hour: float | None,
    comment_engagement_score: float | None,
) -> float:
    return float(
        (1.5 * log1p(max(0.0, max_upvotes or 0.0)))
        + (1.6 * log1p(max(0.0, max_comments or 0.0)))
        + (1.0 * log1p(max(0.0, total_upvote_growth or 0.0)))
        + (1.1 * log1p(max(0.0, total_comment_growth or 0.0)))
        + (0.8 * log1p(max(0.0, last_upvote_velocity_per_hour or 0.0)))
        + (1.0 * log1p(max(0.0, last_comment_velocity_per_hour or 0.0)))
        + (comment_engagement_score or 0.0)
    )


def compute_current_attention_score(
    *,
    latest_activity_state: str,
    latest_listing_type: str,
    age_at_last_seen_minutes: float | None,
    latest_rank_seen: float | None,
    last_upvote_velocity_per_hour: float | None,
    last_comment_velocity_per_hour: float | None,
    latest_total_comment_upvotes: float | None,
    latest_total_comment_replies: float | None,
    latest_unique_commenter_count: float | None,
) -> float:
    state = clean_text(latest_activity_state)
    listing = clean_text(latest_listing_type)

    raw_flow = (
        (1.8 * log1p(max(0.0, last_upvote_velocity_per_hour or 0.0)))
        + (2.0 * log1p(max(0.0, last_comment_velocity_per_hour or 0.0)))
        + (0.9 * log1p(max(0.0, latest_total_comment_upvotes or 0.0)))
        + (1.0 * log1p(max(0.0, latest_total_comment_replies or 0.0)))
        + (0.7 * log1p(max(0.0, latest_unique_commenter_count or 0.0)))
    )

    state_multiplier = {
        "surging": 1.25,
        "alive": 1.0,
        "emerging": 1.05,
        "cooling": 0.65,
        "dying": 0.35,
        "dead": 0.15,
        "unknown": 0.35,
    }.get(state, 0.5)
    state_bonus = {
        "surging": 1.6,
        "alive": 0.9,
        "emerging": 1.1,
        "cooling": 0.2,
        "dying": -0.7,
        "dead": -1.5,
        "unknown": -0.2,
    }.get(state, 0.0)
    listing_bonus = {
        "hot": 1.7,
        "rising": 1.5,
        "new": 1.0,
        "top_day": 0.9,
        "top_week": 0.3,
    }.get(listing, 0.0)

    if age_at_last_seen_minutes is None:
        freshness_bonus = 0.0
    elif age_at_last_seen_minutes <= 60:
        freshness_bonus = 1.3
    elif age_at_last_seen_minutes <= 360:
        freshness_bonus = 0.8
    elif age_at_last_seen_minutes <= 1440:
        freshness_bonus = 0.2
    else:
        freshness_bonus = -0.6

    rank_bonus = (
        (2.2 / log1p(max(2.0, latest_rank_seen + 1.5)))
        if latest_rank_seen is not None and latest_rank_seen > 0
        else 0.0
    )

    return float((raw_flow * state_multiplier) + state_bonus + listing_bonus + freshness_bonus + rank_bonus)


def build_snapshot_catalog_rows(
    *,
    context: dict[str, Any],
    post_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in post_rows:
        grouped_rows[clean_text(row.get("subreddit"))].append(row)

    multi_subreddit_snapshot = len([key for key in grouped_rows if key]) > 1
    snapshot_rows: list[dict[str, Any]] = []

    for subreddit, subreddit_rows in grouped_rows.items():
        upvotes = [
            float(row["upvotes_at_snapshot"])
            for row in subreddit_rows
            if isinstance(row.get("upvotes_at_snapshot"), (int, float))
        ]
        comments = [
            float(row["comment_count_at_snapshot"])
            for row in subreddit_rows
            if isinstance(row.get("comment_count_at_snapshot"), (int, float))
        ]
        domains = [row["link_domain"] for row in subreddit_rows if row.get("link_domain")]
        domain_counts = Counter(domains)
        top_domain = domain_counts.most_common(1)[0][0] if domain_counts else ""
        top_domain_share = (
            domain_counts[top_domain] / len(subreddit_rows)
            if subreddit_rows and top_domain
            else None
        )

        snapshot_rows.append(
            {
                "snapshot_id": context["snapshot_id"],
                "snapshot_time_utc": context["snapshot_time_utc"],
                "subreddit": subreddit or context["subreddit"],
                "listing": context["listing"],
                "top_time": context["top_time"],
                "listing_type": context["listing_type"],
                "schedule_name": context["schedule_name"],
                "cadence_label": context["cadence_label"],
                "scheduled_hour": context["scheduled_hour"],
                "source_type": context["source_type"],
                "input_file": context["input_file"],
                "batch_manifest": context["batch_manifest"],
                "schedule_plan": context["schedule_plan"],
                "output_file": context["output_file"],
                "metadata_file": context["metadata_file"],
                "item_count": None if multi_subreddit_snapshot else context["item_count"],
                "post_count_in_snapshot": len(subreddit_rows),
                "comment_count_in_snapshot": None
                if multi_subreddit_snapshot
                else context["comment_count"],
                "unique_link_domains": len(set(domains)),
                "average_upvotes": safe_mean(upvotes),
                "median_upvotes": safe_median(upvotes),
                "average_comment_count": safe_mean(comments),
                "median_comment_count": safe_median(comments),
                "share_of_posts_with_comments": (
                    sum(
                        1
                        for row in subreddit_rows
                        if isinstance(row.get("comment_count_at_snapshot"), (int, float))
                        and row["comment_count_at_snapshot"] > 0
                    )
                    / len(subreddit_rows)
                    if subreddit_rows
                    else None
                ),
                "share_of_posts_with_external_links": (
                    sum(1 for row in subreddit_rows if row.get("external_link")) / len(subreddit_rows)
                    if subreddit_rows
                    else None
                ),
                "top_domain_by_frequency": top_domain,
                "top_domain_share": top_domain_share,
                "_snapshot_time_dt": context["snapshot_time_dt"],
                "_post_ids": {row["post_id"] for row in subreddit_rows if row.get("post_id")},
            }
        )

    return snapshot_rows


def enrich_subreddit_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subreddit"], row["listing_type"])].append(row)

    for group_rows in grouped.values():
        group_rows.sort(
            key=lambda row: (
                row.get("_snapshot_time_dt") or datetime.min.replace(tzinfo=timezone.utc),
                row["snapshot_id"],
            )
        )
        previous: dict[str, Any] | None = None
        for row in group_rows:
            if previous is None:
                row["hours_since_previous_snapshot"] = None
                row["new_post_count_since_previous_snapshot"] = None
                row["persisting_post_count_from_previous_snapshot"] = None
            else:
                current_dt = row.get("_snapshot_time_dt")
                previous_dt = previous.get("_snapshot_time_dt")
                if current_dt and previous_dt:
                    row["hours_since_previous_snapshot"] = (
                        current_dt - previous_dt
                    ).total_seconds() / 3600
                else:
                    row["hours_since_previous_snapshot"] = None
                previous_ids = previous.get("_post_ids", set())
                current_ids = row.get("_post_ids", set())
                row["new_post_count_since_previous_snapshot"] = len(
                    current_ids - previous_ids
                )
                row["persisting_post_count_from_previous_snapshot"] = len(
                    current_ids & previous_ids
                )
            previous = row

    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        cleaned = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        cleaned_rows.append(cleaned)
    cleaned_rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            row.get("listing_type", ""),
            row.get("snapshot_time_utc", ""),
            row.get("snapshot_id", ""),
        )
    )
    return cleaned_rows


def enrich_post_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_snapshot: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    snapshot_rows_by_group: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    snapshot_time_by_id: dict[str, datetime | None] = {}

    for row in rows:
        snapshot_id = row["snapshot_id"]
        post_id = row["post_id"]
        if post_id:
            rows_by_snapshot[snapshot_id][post_id] = row
        snapshot_rows_by_group[(row["subreddit"], row["listing_type"])][snapshot_id].append(row)
        snapshot_time_by_id[snapshot_id] = parse_datetime(row.get("snapshot_time_utc"))

    for (subreddit, listing_type_value), snapshots in snapshot_rows_by_group.items():
        ordered_snapshot_ids = sorted(
            snapshots.keys(),
            key=lambda snapshot_id: (
                snapshot_time_by_id.get(snapshot_id) or datetime.min.replace(tzinfo=timezone.utc),
                snapshot_id,
            ),
        )
        for index, snapshot_id in enumerate(ordered_snapshot_ids):
            previous_snapshot_id = (
                ordered_snapshot_ids[index - 1] if index > 0 else ""
            )
            next_snapshot_id = (
                ordered_snapshot_ids[index + 1] if index + 1 < len(ordered_snapshot_ids) else ""
            )
            previous_snapshot_time = snapshot_time_by_id.get(previous_snapshot_id) if previous_snapshot_id else None
            next_snapshot_time = parse_datetime(rows_by_snapshot[next_snapshot_id][next(iter(rows_by_snapshot[next_snapshot_id]))]["snapshot_time_utc"]) if next_snapshot_id and rows_by_snapshot.get(next_snapshot_id) else None
            current_snapshot_time = snapshot_time_by_id.get(snapshot_id)

            for row in snapshots[snapshot_id]:
                current_upvotes = parse_float(row.get("upvotes_at_snapshot"))
                current_comments = parse_float(row.get("comment_count_at_snapshot"))

                if previous_snapshot_id:
                    row["previous_snapshot_id"] = previous_snapshot_id
                    row["previous_snapshot_time_utc"] = isoformat_or_empty(previous_snapshot_time)
                    if current_snapshot_time and previous_snapshot_time:
                        row["hours_since_previous_snapshot"] = (
                            current_snapshot_time - previous_snapshot_time
                        ).total_seconds() / 3600
                    else:
                        row["hours_since_previous_snapshot"] = None

                    previous_row = rows_by_snapshot.get(previous_snapshot_id, {}).get(row["post_id"])
                    if previous_row:
                        previous_upvotes = parse_float(previous_row.get("upvotes_at_snapshot"))
                        previous_comments = parse_float(previous_row.get("comment_count_at_snapshot"))
                        row["seen_in_previous_snapshot"] = 1
                        row["previous_rank_within_snapshot"] = previous_row.get("rank_within_snapshot")
                        row["previous_upvotes_at_snapshot"] = previous_row.get("upvotes_at_snapshot")
                        row["previous_comment_count_at_snapshot"] = previous_row.get(
                            "comment_count_at_snapshot"
                        )
                        row["upvote_delta_from_previous_snapshot"] = (
                            current_upvotes - previous_upvotes
                            if current_upvotes is not None and previous_upvotes is not None
                            else None
                        )
                        row["comment_delta_from_previous_snapshot"] = (
                            current_comments - previous_comments
                            if current_comments is not None and previous_comments is not None
                            else None
                        )
                    else:
                        row["seen_in_previous_snapshot"] = 0
                        row["previous_rank_within_snapshot"] = None
                        row["previous_upvotes_at_snapshot"] = None
                        row["previous_comment_count_at_snapshot"] = None
                        row["upvote_delta_from_previous_snapshot"] = None
                        row["comment_delta_from_previous_snapshot"] = None
                else:
                    row["previous_snapshot_id"] = ""
                    row["previous_snapshot_time_utc"] = ""
                    row["hours_since_previous_snapshot"] = None
                    row["seen_in_previous_snapshot"] = None
                    row["previous_rank_within_snapshot"] = None
                    row["previous_upvotes_at_snapshot"] = None
                    row["previous_comment_count_at_snapshot"] = None
                    row["upvote_delta_from_previous_snapshot"] = None
                    row["comment_delta_from_previous_snapshot"] = None

                if next_snapshot_id:
                    row["next_snapshot_id"] = next_snapshot_id
                    row["next_snapshot_time_utc"] = isoformat_or_empty(next_snapshot_time)
                    if current_snapshot_time and next_snapshot_time:
                        row["hours_to_next_snapshot"] = (
                            next_snapshot_time - current_snapshot_time
                        ).total_seconds() / 3600
                    else:
                        row["hours_to_next_snapshot"] = None

                    next_row = rows_by_snapshot.get(next_snapshot_id, {}).get(row["post_id"])
                    if next_row:
                        row["still_visible_next_snapshot"] = 1
                        row["next_rank_within_snapshot"] = next_row.get("rank_within_snapshot")
                        row["next_upvotes_at_snapshot"] = next_row.get("upvotes_at_snapshot")
                        row["next_comment_count_at_snapshot"] = next_row.get(
                            "comment_count_at_snapshot"
                        )
                        next_upvotes = parse_float(next_row.get("upvotes_at_snapshot"))
                        row["upvote_delta_to_next_snapshot"] = (
                            next_upvotes - current_upvotes
                            if current_upvotes is not None
                            and next_upvotes is not None
                            else None
                        )
                        next_comments = parse_float(next_row.get("comment_count_at_snapshot"))
                        row["comment_delta_to_next_snapshot"] = (
                            next_comments - current_comments
                            if current_comments is not None
                            and next_comments is not None
                            else None
                        )
                    else:
                        row["still_visible_next_snapshot"] = 0
                        row["next_rank_within_snapshot"] = None
                        row["next_upvotes_at_snapshot"] = None
                        row["next_comment_count_at_snapshot"] = None
                        row["upvote_delta_to_next_snapshot"] = None
                        row["comment_delta_to_next_snapshot"] = None
                else:
                    row["next_snapshot_id"] = ""
                    row["next_snapshot_time_utc"] = ""
                    row["hours_to_next_snapshot"] = None
                    row["still_visible_next_snapshot"] = None
                    row["next_rank_within_snapshot"] = None
                    row["next_upvotes_at_snapshot"] = None
                    row["next_comment_count_at_snapshot"] = None
                    row["upvote_delta_to_next_snapshot"] = None
                    row["comment_delta_to_next_snapshot"] = None

                hours_since_previous = parse_float(row.get("hours_since_previous_snapshot"))
                upvote_delta_previous = parse_float(row.get("upvote_delta_from_previous_snapshot"))
                comment_delta_previous = parse_float(row.get("comment_delta_from_previous_snapshot"))
                row["upvote_velocity_per_hour"] = (
                    upvote_delta_previous / hours_since_previous
                    if upvote_delta_previous is not None
                    and hours_since_previous is not None
                    and hours_since_previous > 0
                    else None
                )
                row["comment_velocity_per_hour"] = (
                    comment_delta_previous / hours_since_previous
                    if comment_delta_previous is not None
                    and hours_since_previous is not None
                    and hours_since_previous > 0
                    else None
                )
    threshold_map, _threshold_rows = build_activity_thresholds(rows)

    for row in rows:
        subreddit = clean_text(row.get("subreddit"))
        thresholds = threshold_map.get(subreddit, threshold_map["__global__"])
        row["activity_threshold_source"] = thresholds["threshold_source"]
        row["alive_upvote_velocity_threshold"] = thresholds["alive_upvote_velocity_threshold"]
        row["alive_comment_velocity_threshold"] = thresholds["alive_comment_velocity_threshold"]
        row["surging_upvote_velocity_threshold"] = thresholds["surging_upvote_velocity_threshold"]
        row["surging_comment_velocity_threshold"] = thresholds["surging_comment_velocity_threshold"]
        row["dead_upvote_velocity_threshold"] = thresholds["dead_upvote_velocity_threshold"]
        row["dead_comment_velocity_threshold"] = thresholds["dead_comment_velocity_threshold"]

        hours_since_previous = parse_float(row.get("hours_since_previous_snapshot"))
        upvote_delta_previous = parse_float(row.get("upvote_delta_from_previous_snapshot"))
        comment_delta_previous = parse_float(row.get("comment_delta_from_previous_snapshot"))
        row["activity_state"] = classify_activity_state(
            age_minutes=parse_float(row.get("age_minutes_at_snapshot")),
            hours_since_previous_snapshot=hours_since_previous,
            upvote_delta_from_previous_snapshot=upvote_delta_previous,
            comment_delta_from_previous_snapshot=comment_delta_previous,
            still_visible_next_snapshot=row.get("still_visible_next_snapshot"),
            alive_upvote_velocity_threshold=thresholds["alive_upvote_velocity_threshold"],
            alive_comment_velocity_threshold=thresholds["alive_comment_velocity_threshold"],
            surging_upvote_velocity_threshold=thresholds["surging_upvote_velocity_threshold"],
            surging_comment_velocity_threshold=thresholds["surging_comment_velocity_threshold"],
            dead_upvote_velocity_threshold=thresholds["dead_upvote_velocity_threshold"],
            dead_comment_velocity_threshold=thresholds["dead_comment_velocity_threshold"],
        )
        row["is_alive_at_snapshot"] = int(
            row["activity_state"] in {"emerging", "surging", "alive", "cooling"}
        )
        row["is_dead_at_snapshot"] = int(row["activity_state"] == "dead")

    rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            row.get("listing_type", ""),
            row.get("snapshot_time_utc", ""),
            row.get("rank_within_snapshot", 0),
            row.get("post_id", ""),
        )
    )
    return rows


def build_post_lifecycle_rows(
    rows: list[dict[str, Any]],
    *,
    comment_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        canonical_key = canonical_post_group_key(row)
        if canonical_key is not None:
            grouped[canonical_key].append(row)

    comment_aggregates = build_comment_post_aggregates(comment_rows or [])
    lifecycle_rows: list[dict[str, Any]] = []
    for (subreddit, post_id), group_rows in grouped.items():
        group_rows.sort(
            key=lambda row: (
                parse_datetime(row.get("snapshot_time_utc")) or datetime.min.replace(tzinfo=timezone.utc),
                row.get("snapshot_id", ""),
                row.get("rank_within_snapshot", 0),
            )
        )

        first_row = group_rows[0]
        last_row = group_rows[-1]

        first_seen = parse_datetime(first_row.get("snapshot_time_utc"))
        last_seen = parse_datetime(last_row.get("snapshot_time_utc"))
        created_at = parse_datetime(first_row.get("created_at"))
        observed_hours = (
            (last_seen - first_seen).total_seconds() / 3600
            if first_seen and last_seen
            else None
        )
        age_at_last_seen = (
            (last_seen - created_at).total_seconds() / 60
            if last_seen and created_at
            else parse_float(last_row.get("age_minutes_at_snapshot"))
        )

        upvotes = [parse_float(row.get("upvotes_at_snapshot")) for row in group_rows]
        comments = [parse_float(row.get("comment_count_at_snapshot")) for row in group_rows]
        valid_upvotes = [value for value in upvotes if value is not None]
        valid_comments = [value for value in comments if value is not None]
        comment_aggregate = comment_aggregates.get((subreddit, post_id), {})
        activity_states_seen = []
        for row in group_rows:
            activity_value = clean_text(row.get("activity_state"))
            if activity_value and activity_value not in activity_states_seen:
                activity_states_seen.append(activity_value)

        listing_types_seen = []
        for row in group_rows:
            listing_value = clean_text(row.get("listing_type"))
            if listing_value and listing_value not in listing_types_seen:
                listing_types_seen.append(listing_value)

        listing_type_set = set(listing_types_seen)

        best_rank = min(
            (
                int(row["rank_within_snapshot"])
                for row in group_rows
                if isinstance(row.get("rank_within_snapshot"), int)
            ),
            default=None,
        )
        visible_next_count = sum(
            1 for row in group_rows if row.get("still_visible_next_snapshot") == 1
        )
        snapshot_count = len(group_rows)
        raw_latest_state = clean_text(last_row.get("activity_state"))
        last_upvote_velocity = parse_float(last_row.get("upvote_velocity_per_hour"))
        last_comment_velocity = parse_float(last_row.get("comment_velocity_per_hour"))
        latest_rank_seen = parse_float(last_row.get("rank_within_snapshot"))
        dead_upvote_velocity_threshold = parse_float(last_row.get("dead_upvote_velocity_threshold")) or 5.0
        dead_comment_velocity_threshold = parse_float(last_row.get("dead_comment_velocity_threshold")) or 0.5
        latest_state = stabilize_lifecycle_state(
            latest_state=raw_latest_state,
            snapshot_count=snapshot_count,
            age_minutes=age_at_last_seen,
            last_upvote_velocity_per_hour=last_upvote_velocity,
            last_comment_velocity_per_hour=last_comment_velocity,
            dead_upvote_velocity_threshold=dead_upvote_velocity_threshold,
            dead_comment_velocity_threshold=dead_comment_velocity_threshold,
        )
        is_currently_alive = int(latest_state in {"surging", "alive", "emerging"})
        is_currently_dead = int(snapshot_count >= 2 and latest_state == "dead")
        total_upvote_growth = (
            parse_float(last_row.get("upvotes_at_snapshot"))
            - parse_float(first_row.get("upvotes_at_snapshot"))
            if parse_float(last_row.get("upvotes_at_snapshot")) is not None
            and parse_float(first_row.get("upvotes_at_snapshot")) is not None
            else None
        )
        total_comment_growth = (
            parse_float(last_row.get("comment_count_at_snapshot"))
            - parse_float(first_row.get("comment_count_at_snapshot"))
            if parse_float(last_row.get("comment_count_at_snapshot")) is not None
            and parse_float(first_row.get("comment_count_at_snapshot")) is not None
            else None
        )
        comment_engagement_score = compute_comment_engagement_score(
            latest_comment_sample_count=parse_float(comment_aggregate.get("latest_comment_sample_count")),
            latest_total_comment_upvotes=parse_float(comment_aggregate.get("latest_total_comment_upvotes")),
            latest_total_comment_replies=parse_float(comment_aggregate.get("latest_total_comment_replies")),
            latest_unique_commenter_count=parse_float(comment_aggregate.get("latest_unique_commenter_count")),
            max_comment_upvotes_seen=parse_float(comment_aggregate.get("max_comment_upvotes_seen")),
            max_comment_replies_seen=parse_float(comment_aggregate.get("max_comment_replies_seen")),
        )
        general_popularity_score = compute_general_popularity_score(
            max_upvotes=max(valid_upvotes) if valid_upvotes else None,
            max_comments=max(valid_comments) if valid_comments else None,
            total_upvote_growth=total_upvote_growth,
            total_comment_growth=total_comment_growth,
            last_upvote_velocity_per_hour=last_upvote_velocity,
            last_comment_velocity_per_hour=last_comment_velocity,
            comment_engagement_score=comment_engagement_score,
        )
        current_attention_score = compute_current_attention_score(
            latest_activity_state=latest_state,
            latest_listing_type=clean_text(last_row.get("listing_type")),
            age_at_last_seen_minutes=age_at_last_seen,
            latest_rank_seen=latest_rank_seen,
            last_upvote_velocity_per_hour=last_upvote_velocity,
            last_comment_velocity_per_hour=last_comment_velocity,
            latest_total_comment_upvotes=parse_float(comment_aggregate.get("latest_total_comment_upvotes")),
            latest_total_comment_replies=parse_float(comment_aggregate.get("latest_total_comment_replies")),
            latest_unique_commenter_count=parse_float(comment_aggregate.get("latest_unique_commenter_count")),
        )

        lifecycle_rows.append(
            {
                "subreddit": subreddit,
                "post_id": post_id,
                "title": clean_text(first_row.get("title")),
                "url": clean_text(first_row.get("url")),
                "external_link": clean_text(first_row.get("external_link")),
                "link_domain": clean_text(first_row.get("link_domain")),
                "author": clean_text(first_row.get("author")),
                "created_at": clean_text(first_row.get("created_at")),
                "first_seen_snapshot_id": clean_text(first_row.get("snapshot_id")),
                "first_seen_time_utc": clean_text(first_row.get("snapshot_time_utc")),
                "last_seen_snapshot_id": clean_text(last_row.get("snapshot_id")),
                "last_seen_time_utc": clean_text(last_row.get("snapshot_time_utc")),
                "snapshot_count": snapshot_count,
                "observed_hours": observed_hours,
                "listing_types_seen": "|".join(listing_types_seen),
                "listing_type_count": len(listing_types_seen),
                "seen_in_new": int("new" in listing_type_set),
                "seen_in_rising": int("rising" in listing_type_set),
                "seen_in_hot": int("hot" in listing_type_set),
                "seen_in_top_day": int("top_day" in listing_type_set),
                "seen_in_top_week": int("top_week" in listing_type_set),
                "activity_states_seen": "|".join(activity_states_seen),
                "best_rank_seen": best_rank,
                "first_listing_type": clean_text(first_row.get("listing_type")),
                "last_listing_type": clean_text(last_row.get("listing_type")),
                "latest_rank_seen": latest_rank_seen,
                "latest_activity_state": latest_state,
                "is_currently_alive": is_currently_alive,
                "is_currently_dead": is_currently_dead,
                "first_age_minutes": parse_float(first_row.get("age_minutes_at_snapshot")),
                "last_age_minutes": parse_float(last_row.get("age_minutes_at_snapshot")),
                "age_at_last_seen_minutes": age_at_last_seen,
                "analysis_priority": analysis_priority_for_state(
                    latest_state,
                    age_at_last_seen,
                ),
                "activity_threshold_source": clean_text(last_row.get("activity_threshold_source")),
                "alive_upvote_velocity_threshold": parse_float(last_row.get("alive_upvote_velocity_threshold")),
                "alive_comment_velocity_threshold": parse_float(last_row.get("alive_comment_velocity_threshold")),
                "surging_upvote_velocity_threshold": parse_float(last_row.get("surging_upvote_velocity_threshold")),
                "surging_comment_velocity_threshold": parse_float(last_row.get("surging_comment_velocity_threshold")),
                "dead_upvote_velocity_threshold": dead_upvote_velocity_threshold,
                "dead_comment_velocity_threshold": dead_comment_velocity_threshold,
                "fresh_at_first_seen_1h": int(first_row.get("is_fresh_1h") == 1),
                "fresh_at_first_seen_6h": int(first_row.get("is_fresh_6h") == 1),
                "old_at_last_seen_24h": int(age_at_last_seen is not None and age_at_last_seen >= 1440),
                "last_upvote_velocity_per_hour": last_upvote_velocity,
                "last_comment_velocity_per_hour": last_comment_velocity,
                "first_upvotes": parse_float(first_row.get("upvotes_at_snapshot")),
                "last_upvotes": parse_float(last_row.get("upvotes_at_snapshot")),
                "max_upvotes": max(valid_upvotes) if valid_upvotes else None,
                "first_comments": parse_float(first_row.get("comment_count_at_snapshot")),
                "last_comments": parse_float(last_row.get("comment_count_at_snapshot")),
                "max_comments": max(valid_comments) if valid_comments else None,
                "total_upvote_growth": total_upvote_growth,
                "total_comment_growth": total_comment_growth,
                "comment_snapshot_count": parse_float(comment_aggregate.get("comment_snapshot_count")),
                "latest_comment_snapshot_id": clean_text(comment_aggregate.get("latest_comment_snapshot_id")),
                "latest_comment_snapshot_time_utc": clean_text(
                    comment_aggregate.get("latest_comment_snapshot_time_utc")
                ),
                "latest_comment_sample_count": parse_float(
                    comment_aggregate.get("latest_comment_sample_count")
                ),
                "latest_top_level_comment_sample_count": parse_float(
                    comment_aggregate.get("latest_top_level_comment_sample_count")
                ),
                "latest_reply_comment_sample_count": parse_float(
                    comment_aggregate.get("latest_reply_comment_sample_count")
                ),
                "latest_unique_commenter_count": parse_float(
                    comment_aggregate.get("latest_unique_commenter_count")
                ),
                "latest_total_comment_upvotes": parse_float(
                    comment_aggregate.get("latest_total_comment_upvotes")
                ),
                "latest_total_comment_replies": parse_float(
                    comment_aggregate.get("latest_total_comment_replies")
                ),
                "latest_avg_comment_upvotes": parse_float(
                    comment_aggregate.get("latest_avg_comment_upvotes")
                ),
                "latest_max_comment_upvotes": parse_float(
                    comment_aggregate.get("latest_max_comment_upvotes")
                ),
                "latest_max_comment_replies": parse_float(
                    comment_aggregate.get("latest_max_comment_replies")
                ),
                "total_sampled_comment_upvotes": parse_float(
                    comment_aggregate.get("total_sampled_comment_upvotes")
                ),
                "total_sampled_comment_replies": parse_float(
                    comment_aggregate.get("total_sampled_comment_replies")
                ),
                "max_comment_upvotes_seen": parse_float(
                    comment_aggregate.get("max_comment_upvotes_seen")
                ),
                "max_comment_replies_seen": parse_float(
                    comment_aggregate.get("max_comment_replies_seen")
                ),
                "max_comment_sample_count_seen": parse_float(
                    comment_aggregate.get("max_comment_sample_count_seen")
                ),
                "max_total_comment_upvotes_seen": parse_float(
                    comment_aggregate.get("max_total_comment_upvotes_seen")
                ),
                "max_total_comment_replies_seen": parse_float(
                    comment_aggregate.get("max_total_comment_replies_seen")
                ),
                "comment_engagement_score": comment_engagement_score,
                "general_popularity_score": general_popularity_score,
                "current_attention_score": current_attention_score,
                "visible_next_snapshot_count": visible_next_count,
            }
        )

    lifecycle_rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            -(row.get("general_popularity_score") or 0),
            -(row.get("max_upvotes") or 0),
            -(row.get("max_comments") or 0),
            row.get("first_seen_time_utc", ""),
        )
    )
    return lifecycle_rows


def build_top_posts_rows(
    lifecycle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lifecycle_rows:
        grouped[row["subreddit"]].append(row)

    top_rows: list[dict[str, Any]] = []
    for subreddit, group_rows in grouped.items():
        group_rows.sort(
            key=lambda row: (
                -(row.get("general_popularity_score") or 0),
                -(row.get("comment_engagement_score") or 0),
                -(row.get("max_upvotes") or 0),
                -(row.get("max_comments") or 0),
                -(row.get("total_upvote_growth") or 0),
                row.get("first_seen_time_utc", ""),
            )
        )
        for rank, row in enumerate(group_rows, start=1):
            top_rows.append(
                {
                    "subreddit": subreddit,
                    "popularity_rank": rank,
                    **row,
                }
            )
    return top_rows


def build_latest_status_rows(lifecycle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in lifecycle_rows]
    rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            {"highest": 0, "high": 1, "medium": 2, "low": 3, "very_low": 4}.get(
                row.get("analysis_priority", "low"),
                9,
            ),
            -(row.get("current_attention_score") or 0),
            -(row.get("last_comment_velocity_per_hour") or 0),
            -(row.get("last_upvote_velocity_per_hour") or 0),
            -(row.get("comment_engagement_score") or 0),
            -(row.get("general_popularity_score") or 0),
            -(row.get("max_upvotes") or 0),
        )
    )
    return rows


def filter_latest_status_rows(
    rows: list[dict[str, Any]],
    *,
    states: set[str] | None = None,
    priorities: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        state = clean_text(row.get("latest_activity_state"))
        priority = clean_text(row.get("analysis_priority"))
        if states is not None and state not in states:
            continue
        if priorities is not None and priority not in priorities:
            continue
        filtered.append(dict(row))
    return filtered


def focus_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("subreddit", ""),
        {"highest": 0, "high": 1, "medium": 2, "low": 3, "very_low": 4}.get(
            row.get("analysis_priority", "low"),
            9,
        ),
        -(row.get("current_attention_score") or 0),
        -(row.get("last_comment_velocity_per_hour") or 0),
        -(row.get("last_upvote_velocity_per_hour") or 0),
        -(row.get("comment_engagement_score") or 0),
        -(row.get("general_popularity_score") or 0),
        -(row.get("total_comment_growth") or 0),
        -(row.get("total_upvote_growth") or 0),
        -(row.get("max_comments") or 0),
        -(row.get("max_upvotes") or 0),
    )


def build_analysis_focus_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focus_rows = [dict(row) for row in rows]
    focus_rows.sort(key=focus_sort_key)

    subreddit_counts: dict[str, int] = defaultdict(int)
    ranked_rows: list[dict[str, Any]] = []
    for overall_rank, row in enumerate(focus_rows, start=1):
        subreddit = clean_text(row.get("subreddit"))
        subreddit_counts[subreddit] += 1
        subreddit_rank = subreddit_counts[subreddit]

        analysis_priority = clean_text(row.get("analysis_priority"))
        latest_state = clean_text(row.get("latest_activity_state"))
        if analysis_priority == "highest" or subreddit_rank <= 5:
            recommended_action = "track_now"
        elif latest_state in {"alive", "emerging"} and subreddit_rank <= 15:
            recommended_action = "track_soon"
        elif latest_state == "dying" and subreddit_rank <= 10:
            recommended_action = "track_briefly"
        else:
            recommended_action = "analyze_only"

        ranked_rows.append(
            {
                **row,
                "focus_rank_overall": overall_rank,
                "focus_rank_in_subreddit": subreddit_rank,
                "is_top_5_in_subreddit": int(subreddit_rank <= 5),
                "is_top_15_in_subreddit": int(subreddit_rank <= 15),
                "recommended_action": recommended_action,
            }
        )
    return ranked_rows


def build_tracking_candidates_rows(
    focus_rows: list[dict[str, Any]],
    *,
    per_subreddit_limit: int = 15,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in focus_rows:
        canonical_key = canonical_post_group_key(row)
        if canonical_key is None or canonical_key in seen_keys:
            continue
        canonical_subreddit, _canonical_post_id = canonical_key
        subreddit_rank = row.get("focus_rank_in_subreddit")
        if not isinstance(subreddit_rank, int) or subreddit_rank > per_subreddit_limit:
            continue
        seen_keys.add(canonical_key)
        candidates.append(
            {
                **row,
                "subreddit": canonical_subreddit,
                "tracking_candidate_rank": subreddit_rank,
                "tracking_window": "next_hour",
            }
        )
    return candidates


def build_current_attention_leaderboard_rows(
    latest_status_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in latest_status_rows]
    rows.sort(
        key=lambda row: (
            -(row.get("current_attention_score") or 0),
            -(row.get("last_comment_velocity_per_hour") or 0),
            -(row.get("last_upvote_velocity_per_hour") or 0),
            -(row.get("comment_engagement_score") or 0),
            row.get("subreddit", ""),
            row.get("post_id", ""),
        )
    )

    subreddit_counts: dict[str, int] = defaultdict(int)
    ranked_rows: list[dict[str, Any]] = []
    for overall_rank, row in enumerate(rows, start=1):
        subreddit = clean_text(row.get("subreddit"))
        subreddit_counts[subreddit] += 1
        ranked_rows.append(
            {
                "current_attention_rank_overall": overall_rank,
                "current_attention_rank_in_subreddit": subreddit_counts[subreddit],
                **row,
            }
        )
    return ranked_rows


def build_general_popularity_leaderboard_rows(
    top_posts_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in top_posts_rows]
    rows.sort(
        key=lambda row: (
            -(row.get("general_popularity_score") or 0),
            -(row.get("comment_engagement_score") or 0),
            -(row.get("max_upvotes") or 0),
            -(row.get("max_comments") or 0),
            row.get("subreddit", ""),
            row.get("post_id", ""),
        )
    )

    subreddit_counts: dict[str, int] = defaultdict(int)
    ranked_rows: list[dict[str, Any]] = []
    for overall_rank, row in enumerate(rows, start=1):
        subreddit = clean_text(row.get("subreddit"))
        subreddit_counts[subreddit] += 1
        ranked_rows.append(
            {
                "general_popularity_rank_overall": overall_rank,
                "general_popularity_rank_in_subreddit": subreddit_counts[subreddit],
                **row,
            }
        )
    return ranked_rows


def build_subreddit_attention_latest_rows(
    latest_status_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latest_status_rows:
        grouped[clean_text(row.get("subreddit"))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for subreddit, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                -(row.get("current_attention_score") or 0),
                -(row.get("general_popularity_score") or 0),
                row.get("title", ""),
            )
        )
        top_row = rows[0] if rows else {}
        alive_now = sum(
            1 for row in rows if clean_text(row.get("latest_activity_state")) in {"surging", "alive", "emerging"}
        )
        dying_now = sum(1 for row in rows if clean_text(row.get("latest_activity_state")) == "dying")
        cooling_now = sum(
            1 for row in rows if clean_text(row.get("latest_activity_state")) in {"cooling", "dying"}
        )
        dead_now = sum(1 for row in rows if clean_text(row.get("latest_activity_state")) == "dead")
        attention_values = [row.get("current_attention_score") or 0 for row in rows]
        popularity_values = [row.get("general_popularity_score") or 0 for row in rows]

        summary_rows.append(
            {
                "subreddit": subreddit,
                "post_count": len(rows),
                "alive_or_surging_count": alive_now,
                "cooling_count": cooling_now,
                "dying_count": dying_now,
                "dead_count": dead_now,
                "total_current_attention_score": safe_sum(attention_values),
                "avg_current_attention_score": safe_mean(attention_values),
                "max_current_attention_score": max(attention_values) if attention_values else None,
                "total_general_popularity_score": safe_sum(popularity_values),
                "avg_general_popularity_score": safe_mean(popularity_values),
                "max_general_popularity_score": max(popularity_values) if popularity_values else None,
                "top_current_attention_post_id": clean_text(top_row.get("post_id")),
                "top_current_attention_title": clean_text(top_row.get("title")),
                "top_current_attention_score": top_row.get("current_attention_score"),
                "top_general_popularity_score": top_row.get("general_popularity_score"),
                "top_latest_activity_state": clean_text(top_row.get("latest_activity_state")),
                "top_last_listing_type": clean_text(top_row.get("last_listing_type")),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -(row.get("total_current_attention_score") or 0),
            -(row.get("alive_or_surging_count") or 0),
            row.get("subreddit", ""),
        )
    )
    return summary_rows


def write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    *,
    destination: Path,
    raw_files: list[Path],
    snapshot_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
    lifecycle_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "raw_file_count": len(raw_files),
        "snapshot_count": len(snapshot_rows),
        "post_snapshot_count": len(post_rows),
        "post_lifecycle_count": len(lifecycle_rows),
        "subreddit_count": len({row["subreddit"] for row in post_rows if row.get("subreddit")}),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    excluded_subreddits = normalize_subreddit_names(args.exclude_subreddits)
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw directory not found: {raw_dir}")

    raw_files = sorted(
        path
        for path in raw_dir.glob("*.json")
        if path.is_file() and not path.name.endswith("_metadata.json")
    )
    if not raw_files:
        raise SystemExit(f"No raw Apify JSON files found in {raw_dir}")

    all_post_rows: list[dict[str, Any]] = []
    all_comment_rows: list[dict[str, Any]] = []
    snapshot_catalog_rows: list[dict[str, Any]] = []

    for raw_file in raw_files:
        metadata = load_metadata(raw_file.with_name(f"{raw_file.stem}_metadata.json"))
        items = load_items(str(raw_file))
        posts = [item for item in items if clean_text(item.get("dataType")) == "post"]
        comments = [item for item in items if clean_text(item.get("dataType")) == "comment"]

        context = infer_snapshot_context(
            raw_file=raw_file,
            metadata=metadata,
            posts=posts,
            comments=comments,
        )
        if clean_text(context.get("subreddit")).lower() in excluded_subreddits:
            continue
        post_rows = build_post_snapshot_rows(posts=posts, context=context)
        comment_rows = build_comment_snapshot_rows(comments=comments, context=context)
        snapshot_catalog_rows.extend(
            build_snapshot_catalog_rows(context=context, post_rows=post_rows)
        )
        all_post_rows.extend(post_rows)
        all_comment_rows.extend(comment_rows)

    enriched_post_rows = enrich_post_snapshots(all_post_rows)
    enriched_snapshot_rows = enrich_subreddit_snapshots(snapshot_catalog_rows)
    _, activity_threshold_rows = build_activity_thresholds(enriched_post_rows)
    lifecycle_rows = build_post_lifecycle_rows(enriched_post_rows, comment_rows=all_comment_rows)
    top_posts_rows = build_top_posts_rows(lifecycle_rows)
    latest_status_rows = build_latest_status_rows(lifecycle_rows)
    alive_focus_rows = filter_latest_status_rows(
        latest_status_rows,
        states={"surging", "alive", "emerging"},
        priorities={"highest", "high"},
    )
    analysis_focus_rows = build_analysis_focus_rows(alive_focus_rows)
    current_attention_leaderboard_rows = build_current_attention_leaderboard_rows(latest_status_rows)
    general_popularity_leaderboard_rows = build_general_popularity_leaderboard_rows(top_posts_rows)
    subreddit_attention_latest_rows = build_subreddit_attention_latest_rows(latest_status_rows)
    cooling_rows = filter_latest_status_rows(
        latest_status_rows,
        states={"cooling", "dying"},
    )
    dying_rows = filter_latest_status_rows(
        latest_status_rows,
        states={"dying"},
    )
    dead_rows = filter_latest_status_rows(
        latest_status_rows,
        states={"dead"},
    )
    top_risers_rows = sorted(
        analysis_focus_rows,
        key=lambda row: (
            -(row.get("current_attention_score") or 0),
            -(row.get("last_comment_velocity_per_hour") or 0),
            -(row.get("last_upvote_velocity_per_hour") or 0),
            -(row.get("comment_engagement_score") or 0),
            -(row.get("total_comment_growth") or 0),
            -(row.get("total_upvote_growth") or 0),
        ),
    )
    tracking_candidates_rows = build_tracking_candidates_rows(analysis_focus_rows)

    output_dir = Path(args.output_dir)
    post_snapshots_path = output_dir / "post_snapshots.csv"
    comment_snapshots_path = output_dir / "comment_snapshots.csv"
    subreddit_snapshots_path = output_dir / "subreddit_snapshots.csv"
    snapshot_catalog_path = output_dir / "snapshot_catalog.csv"
    activity_thresholds_path = output_dir / "activity_thresholds.csv"
    post_lifecycles_path = output_dir / "post_lifecycles.csv"
    top_posts_path = output_dir / "top_posts.csv"
    latest_status_path = output_dir / "latest_post_status.csv"
    alive_focus_path = output_dir / "alive_posts_latest.csv"
    analysis_focus_path = output_dir / "analysis_focus_latest.csv"
    current_attention_leaderboard_path = output_dir / "current_attention_leaderboard.csv"
    general_popularity_leaderboard_path = output_dir / "general_popularity_leaderboard.csv"
    subreddit_attention_latest_path = output_dir / "subreddit_attention_latest.csv"
    cooling_path = output_dir / "cooling_posts_latest.csv"
    dying_path = output_dir / "dying_posts_latest.csv"
    dead_path = output_dir / "dead_posts_latest.csv"
    top_risers_path = output_dir / "top_risers_latest.csv"
    tracking_candidates_path = output_dir / "tracking_candidates_latest.csv"
    metadata_path = output_dir / "metadata.json"

    write_csv(enriched_post_rows, post_snapshots_path)
    write_csv(all_comment_rows, comment_snapshots_path)
    write_csv(enriched_snapshot_rows, subreddit_snapshots_path)
    write_csv(enriched_snapshot_rows, snapshot_catalog_path)
    write_csv(activity_threshold_rows, activity_thresholds_path)
    write_csv(lifecycle_rows, post_lifecycles_path)
    write_csv(top_posts_rows, top_posts_path)
    write_csv(latest_status_rows, latest_status_path)
    write_csv(alive_focus_rows, alive_focus_path)
    write_csv(analysis_focus_rows, analysis_focus_path)
    write_csv(current_attention_leaderboard_rows, current_attention_leaderboard_path)
    write_csv(general_popularity_leaderboard_rows, general_popularity_leaderboard_path)
    write_csv(subreddit_attention_latest_rows, subreddit_attention_latest_path)
    write_csv(cooling_rows, cooling_path)
    write_csv(dying_rows, dying_path)
    write_csv(dead_rows, dead_path)
    write_csv(top_risers_rows, top_risers_path)
    write_csv(tracking_candidates_rows, tracking_candidates_path)
    write_metadata(
        destination=metadata_path,
        raw_files=raw_files,
        snapshot_rows=enriched_snapshot_rows,
        post_rows=enriched_post_rows,
        lifecycle_rows=lifecycle_rows,
    )

    print(f"Saved {len(enriched_post_rows)} post snapshot row(s) to {post_snapshots_path}")
    print(f"Saved {len(all_comment_rows)} comment snapshot row(s) to {comment_snapshots_path}")
    print(
        f"Saved {len(enriched_snapshot_rows)} subreddit snapshot row(s) "
        f"to {subreddit_snapshots_path}"
    )
    print(f"Snapshot catalog written to {snapshot_catalog_path}")
    print(f"Activity thresholds written to {activity_thresholds_path}")
    print(f"Saved {len(lifecycle_rows)} post lifecycle row(s) to {post_lifecycles_path}")
    print(f"Saved {len(top_posts_rows)} ranked post row(s) to {top_posts_path}")
    print(f"Saved {len(latest_status_rows)} latest post status row(s) to {latest_status_path}")
    print(f"Saved {len(alive_focus_rows)} alive/focus row(s) to {alive_focus_path}")
    print(f"Saved {len(analysis_focus_rows)} analysis focus row(s) to {analysis_focus_path}")
    print(
        f"Saved {len(current_attention_leaderboard_rows)} attention leaderboard row(s) "
        f"to {current_attention_leaderboard_path}"
    )
    print(
        f"Saved {len(general_popularity_leaderboard_rows)} popularity leaderboard row(s) "
        f"to {general_popularity_leaderboard_path}"
    )
    print(
        f"Saved {len(subreddit_attention_latest_rows)} subreddit attention row(s) "
        f"to {subreddit_attention_latest_path}"
    )
    print(f"Saved {len(cooling_rows)} cooling row(s) to {cooling_path}")
    print(f"Saved {len(dying_rows)} dying row(s) to {dying_path}")
    print(f"Saved {len(dead_rows)} dead row(s) to {dead_path}")
    print(f"Saved {len(top_risers_rows)} top riser row(s) to {top_risers_path}")
    print(
        f"Saved {len(tracking_candidates_rows)} tracking candidate row(s) "
        f"to {tracking_candidates_path}"
    )
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
