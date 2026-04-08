from __future__ import annotations

import argparse
import bisect
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from normalize_reddit_json import clean_text, parse_datetime

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except ImportError:
    _VADER = None


NON_WORD_RE = re.compile(r"[^a-z0-9]+")
UTC_NIGHT_END_HOUR = 6
DEFAULT_STRICT_MIN_NEXT_HOURS = 0.75
DEFAULT_STRICT_MAX_NEXT_HOURS = 1.5
DEFAULT_LARGE_GAP_HOURS = 2.5
QUESTION_START_WORDS = {"how", "why", "what", "when", "who", "can", "is", "are", "does", "do"}
DOMAIN_CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("video_platform", ("youtube.com", "youtu.be", "vimeo.com", "twitch.tv")),
    ("social_platform", ("reddit.com", "redd.it", "x.com", "twitter.com", "instagram.com", "tiktok.com", "facebook.com")),
    ("government", (".gov", ".gov.uk", ".europa.eu", "senate.gov", "house.gov", "whitehouse.gov")),
    ("academic", (".edu", "arxiv.org", "nature.com", "science.org", "nih.gov")),
    ("developer_platform", ("github.com", "gitlab.com", "npmjs.com", "pypi.org")),
]
CONTENT_FLAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "content_has_breaking_word": ("breaking", "developing", "just in"),
    "content_has_update_word": ("update", "updated", "updates"),
    "content_has_live_word": ("live", "livestream"),
    "content_has_trailer_word": ("trailer", "teaser"),
    "content_has_review_word": ("review", "reviewed", "hands-on"),
    "content_has_guide_word": ("guide", "how to", "walkthrough", "explainer"),
    "content_has_leak_word": ("leak", "leaked", "rumor", "rumour"),
    "content_has_ama_word": ("ama", "ask me anything"),
    "content_has_analysis_word": ("analysis", "opinion", "editorial", "essay"),
    "content_has_report_word": ("report", "reported", "according to"),
}
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "politics_government": (
        "trump", "biden", "election", "vote", "voter", "campaign", "white house",
        "president", "senate", "congress", "parliament", "minister", "government",
        "policy", "supreme court", "politics",
    ),
    "war_geopolitics": (
        "ukraine", "russia", "iran", "israel", "gaza", "hamas", "military", "missile",
        "drone", "troops", "war", "ceasefire", "nato", "china", "taiwan", "attack",
    ),
    "business_economy": (
        "economy", "economic", "business", "startup", "funding", "market", "stocks",
        "shares", "earnings", "revenue", "tariff", "inflation", "layoffs", "jobs", "ipo",
    ),
    "science_health": (
        "study", "scientists", "research", "trial", "disease", "health", "cancer",
        "drug", "vaccine", "bees", "climate", "space", "nasa", "medical",
    ),
    "ai_software": (
        "ai", "artificial intelligence", "openai", "chatgpt", "llm", "model", "software",
        "app", "api", "cyber", "hack", "github", "microsoft", "google",
    ),
    "hardware_devices": (
        "iphone", "android", "nvidia", "amd", "intel", "chip", "gpu", "cpu",
        "laptop", "phone", "device", "hardware", "server", "console",
    ),
    "gaming_entertainment": (
        "game", "gaming", "xbox", "playstation", "ps5", "nintendo", "steam",
        "trailer", "movie", "tv", "netflix", "spotify", "episode",
    ),
}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build model-ready prediction tables from the historical Reddit post "
            "snapshot dataset."
        )
    )
    parser.add_argument(
        "--history-dir",
        default="data/history/reddit",
        help="Directory containing post_snapshots.csv and related history tables.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/reddit",
        help="Directory where prediction CSV files will be written.",
    )
    parser.add_argument(
        "--min-next-hours",
        type=positive_float,
        default=0.5,
        help="Minimum next-snapshot horizon to count as a next-hour training example.",
    )
    parser.add_argument(
        "--max-next-hours",
        type=positive_float,
        default=2.5,
        help="Maximum next-snapshot horizon to count as a next-hour training example.",
    )
    parser.add_argument(
        "--strict-min-next-hours",
        type=positive_float,
        default=DEFAULT_STRICT_MIN_NEXT_HOURS,
        help="Stricter minimum horizon for a regular-cadence next-hour example.",
    )
    parser.add_argument(
        "--strict-max-next-hours",
        type=positive_float,
        default=DEFAULT_STRICT_MAX_NEXT_HOURS,
        help="Stricter maximum horizon for a regular-cadence next-hour example.",
    )
    parser.add_argument(
        "--large-gap-hours",
        type=positive_float,
        default=DEFAULT_LARGE_GAP_HOURS,
        help="Gap threshold above which a snapshot interval is treated as a large jump.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_title_key(title: str) -> str:
    normalized = NON_WORD_RE.sub(" ", clean_text(title).lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def count_words(text: str) -> int:
    normalized = normalize_title_key(text)
    return len(normalized.split()) if normalized else 0


def uppercase_ratio(text: str) -> float | None:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    uppercase_letters = sum(1 for char in letters if char.isupper())
    return uppercase_letters / len(letters)


def starts_with_question_word(title: str) -> int:
    normalized = normalize_title_key(title)
    first_word = normalized.split()[0] if normalized else ""
    return int(first_word in QUESTION_START_WORDS)


def classify_link_domain(external_link: str, link_domain: str) -> str:
    domain = clean_text(link_domain).lower()
    if not clean_text(external_link):
        return "self_post"
    if not domain:
        return "external_unknown"
    for category, patterns in DOMAIN_CATEGORY_PATTERNS:
        if any(domain == pattern or domain.endswith(pattern) for pattern in patterns):
            return category
    if any(marker in domain for marker in ("news", "times", "post", "journal", "herald", "guardian", "cnn", "bbc", "reuters", "apnews", "bloomberg")):
        return "news_site"
    return "publisher_or_blog"


def keyword_flags(title: str, body: str) -> dict[str, int]:
    combined = f"{clean_text(title)} {clean_text(body)}".strip().lower()
    return {
        flag: int(any(keyword in combined for keyword in keywords))
        for flag, keywords in CONTENT_FLAG_KEYWORDS.items()
    }


def detect_primary_topic(title: str, body: str) -> str:
    combined = f"{clean_text(title)} {clean_text(body)}".strip().lower()
    if not combined:
        return "unknown"
    scored_topics: list[tuple[int, str]] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in combined)
        if score:
            scored_topics.append((score, topic))
    if not scored_topics:
        return "general"
    scored_topics.sort(key=lambda item: (-item[0], item[1]))
    return scored_topics[0][1]


def build_recent_repeat_indexes(
    post_rows: list[dict[str, str]],
) -> tuple[
    dict[tuple[str, str], dict[str, list[Any]]],
    dict[tuple[str, str, str], dict[str, list[Any]]],
]:
    first_seen_by_post: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
    for row in post_rows:
        key = sequence_key(row)
        if key is None:
            continue
        snapshot_dt = parse_datetime(row.get("snapshot_time_utc"))
        if snapshot_dt is None:
            continue
        snapshot_ts = snapshot_dt.timestamp()
        previous = first_seen_by_post.get(key)
        if previous is None or snapshot_ts < previous[0]:
            first_seen_by_post[key] = (snapshot_ts, row)

    title_indexes: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: {"times": [], "post_ids": []})
    story_indexes: dict[tuple[str, str, str], dict[str, list[Any]]] = defaultdict(lambda: {"times": [], "post_ids": []})

    ordered_first_seen = sorted(
        ((subreddit, post_id, first_seen_ts, row) for (subreddit, post_id), (first_seen_ts, row) in first_seen_by_post.items()),
        key=lambda item: (item[2], item[0], item[1]),
    )
    for subreddit, post_id, first_seen_ts, row in ordered_first_seen:
        title_key = normalize_title_key(row.get("title", ""))
        if title_key:
            bucket = title_indexes[(subreddit, title_key)]
            bucket["times"].append(first_seen_ts)
            bucket["post_ids"].append(post_id)
        story_key_type, story_key = canonical_story_key(row)
        if story_key:
            bucket = story_indexes[(subreddit, story_key_type, story_key)]
            bucket["times"].append(first_seen_ts)
            bucket["post_ids"].append(post_id)

    return dict(title_indexes), dict(story_indexes)


def count_prior_distinct_posts_24h(
    index_bucket: dict[str, list[Any]] | None,
    *,
    current_ts: float | None,
    current_post_id: str,
) -> int:
    if index_bucket is None or current_ts is None:
        return 0
    times = index_bucket["times"]
    post_ids = index_bucket["post_ids"]
    window_start = current_ts - (24 * 3600)
    left = bisect.bisect_left(times, window_start)
    right = bisect.bisect_left(times, current_ts)
    if right <= left:
        return 0
    count = 0
    for index in range(left, right):
        if post_ids[index] != current_post_id:
            count += 1
    return count


def comment_group_key(row: dict[str, str]) -> tuple[str, str, str] | None:
    snapshot_id = clean_text(row.get("snapshot_id"))
    subreddit = clean_text(row.get("subreddit")).lower()
    post_id = clean_text(row.get("post_id"))
    if not snapshot_id or not subreddit or not post_id:
        return None
    return snapshot_id, subreddit, post_id


def build_comment_snapshot_aggregates(
    comment_rows: list[dict[str, str]],
    *,
    post_author_by_key: dict[tuple[str, str, str], str],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped_rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in comment_rows:
        key = comment_group_key(row)
        if key is None:
            continue
        grouped_rows[key].append(row)

    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rows in grouped_rows.items():
        snapshot_id, subreddit, post_id = key
        post_author = clean_text(post_author_by_key.get((snapshot_id, subreddit, post_id), "")).lower()
        sample_size = len(rows)
        top_level_count = 0
        reply_count = 0
        unique_commenters: set[str] = set()
        deleted_count = 0
        op_replied = 0
        question_count = 0
        comment_scores: list[float] = []
        comment_word_counts: list[float] = []
        comment_ages: list[float] = []
        sentiment_scores: list[float] = []
        sentiment_weights: list[float] = []

        for row in rows:
            author = clean_text(row.get("author"))
            author_key = author.lower()
            if author_key:
                unique_commenters.add(author_key)
            if author_key == "[deleted]":
                deleted_count += 1
            if post_author and author_key == post_author:
                op_replied = 1
            if str(row.get("is_top_level_comment", "")).strip() == "1":
                top_level_count += 1
            else:
                reply_count += 1
            body = clean_text(row.get("body"))
            if "?" in body:
                question_count += 1
            score = parse_float(row.get("upvotes_at_snapshot"))
            if score is not None:
                comment_scores.append(score)
            body_word_count = parse_float(row.get("body_word_count"))
            if body_word_count is None:
                body_word_count = float(count_words(body))
            if body_word_count is not None:
                comment_word_counts.append(body_word_count)
            age_minutes = parse_float(row.get("age_minutes_at_snapshot"))
            if age_minutes is not None:
                comment_ages.append(age_minutes)
            if _VADER is not None and body and body not in ("[deleted]", "[removed]"):
                vader_score = _VADER.polarity_scores(body)["compound"]
                sentiment_scores.append(vader_score)
                weight = max(1.0, score) if score is not None else 1.0
                sentiment_weights.append(weight)

        weighted_sentiment = None
        if sentiment_scores and sentiment_weights:
            total_weight = sum(sentiment_weights)
            if total_weight > 0:
                weighted_sentiment = sum(
                    s * w for s, w in zip(sentiment_scores, sentiment_weights)
                ) / total_weight
        positive_count = sum(1 for s in sentiment_scores if s >= 0.05)
        negative_count = sum(1 for s in sentiment_scores if s <= -0.05)

        aggregates[key] = {
            "has_comment_sample": int(sample_size > 0),
            "comment_sample_count": sample_size,
            "top_level_comment_sample_count": top_level_count,
            "reply_comment_sample_count": reply_count,
            "top_level_comment_sample_share": (top_level_count / sample_size) if sample_size else None,
            "unique_commenter_count_sample": len(unique_commenters),
            "deleted_comment_share_sample": (deleted_count / sample_size) if sample_size else None,
            "question_comment_share_sample": (question_count / sample_size) if sample_size else None,
            "op_replied_in_comment_sample": op_replied,
            "avg_comment_upvotes_sample": safe_mean(comment_scores),
            "max_comment_upvotes_sample": safe_max(comment_scores),
            "avg_comment_body_word_count_sample": safe_mean(comment_word_counts),
            "max_comment_body_word_count_sample": safe_max(comment_word_counts),
            "newest_comment_age_minutes_sample": min(comment_ages) if comment_ages else None,
            "oldest_comment_age_minutes_sample": max(comment_ages) if comment_ages else None,
            "comment_sample_span_minutes": (
                (max(comment_ages) - min(comment_ages))
                if len(comment_ages) >= 2
                else None
            ),
            "sentiment_mean_sample": safe_mean(sentiment_scores),
            "sentiment_weighted_mean_sample": weighted_sentiment,
            "sentiment_positive_share_sample": (
                (positive_count / len(sentiment_scores)) if sentiment_scores else None
            ),
            "sentiment_negative_share_sample": (
                (negative_count / len(sentiment_scores)) if sentiment_scores else None
            ),
            "sentiment_variance_sample": (
                float(sum((s - safe_mean(sentiment_scores)) ** 2 for s in sentiment_scores) / len(sentiment_scores))
                if len(sentiment_scores) >= 2 and safe_mean(sentiment_scores) is not None
                else None
            ),
        }

    return aggregates


def canonical_story_key(row: dict[str, str]) -> tuple[str, str]:
    external_link = clean_text(row.get("external_link"))
    if external_link:
        parsed = urlparse(external_link)
        path = (parsed.path or "").rstrip("/")
        key = f"{parsed.netloc.lower()}{path}"
        return "external_link", key or external_link.lower()

    title_key = normalize_title_key(row.get("title", ""))
    if title_key:
        return "title", title_key

    return "post_id", clean_text(row.get("post_id"))


def write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_timeline_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline_rows: list[dict[str, Any]] = []
    for row in all_rows:
        timeline_rows.append(
            {
                "sequence_id": row.get("sequence_id"),
                "subreddit": row.get("subreddit"),
                "post_id": row.get("post_id"),
                "title": row.get("title"),
                "snapshot_id": row.get("snapshot_id"),
                "snapshot_time_utc": row.get("snapshot_time_utc"),
                "sequence_position": row.get("sequence_position"),
                "sequence_length_total": row.get("sequence_length_total"),
                "sequence_remaining_snapshots": row.get("sequence_remaining_snapshots"),
                "sequence_progress_ratio": row.get("sequence_progress_ratio"),
                "hours_since_first_seen_snapshot": row.get("hours_since_first_seen_snapshot"),
                "hours_until_last_seen_snapshot": row.get("hours_until_last_seen_snapshot"),
                "sequence_span_hours": row.get("sequence_span_hours"),
                "listing_type": row.get("listing_type"),
                "previous_listing_type": row.get("previous_listing_type"),
                "listing_transition": row.get("listing_transition"),
                "listing_changed_from_previous": row.get("listing_changed_from_previous"),
                "listing_run_length_snapshots": row.get("listing_run_length_snapshots"),
                "hours_in_current_listing": row.get("hours_in_current_listing"),
                "schedule_name": row.get("schedule_name"),
                "rank_within_snapshot": row.get("rank_within_snapshot"),
                "previous_rank_within_snapshot": row.get("previous_rank_within_snapshot"),
                "rank_change_from_previous_snapshot": row.get("rank_change_from_previous_snapshot"),
                "abs_rank_change_from_previous_snapshot": row.get("abs_rank_change_from_previous_snapshot"),
                "rank_improved_from_previous": row.get("rank_improved_from_previous"),
                "rank_worsened_from_previous": row.get("rank_worsened_from_previous"),
                "rank_unchanged_from_previous": row.get("rank_unchanged_from_previous"),
                "upvotes_at_snapshot": row.get("upvotes_at_snapshot"),
                "comment_count_at_snapshot": row.get("comment_count_at_snapshot"),
                "upvote_velocity_per_hour": row.get("upvote_velocity_per_hour"),
                "comment_velocity_per_hour": row.get("comment_velocity_per_hour"),
                "upvote_delta_from_previous_snapshot": row.get("upvote_delta_from_previous_snapshot"),
                "comment_delta_from_previous_snapshot": row.get("comment_delta_from_previous_snapshot"),
                "hours_since_previous_snapshot": row.get("hours_since_previous_snapshot"),
                "previous_gap_bucket": row.get("previous_gap_bucket"),
                "next_gap_bucket": row.get("next_gap_bucket"),
                "previous_gap_is_regular": row.get("previous_gap_is_regular"),
                "next_gap_is_regular": row.get("next_gap_is_regular"),
                "previous_gap_is_large": row.get("previous_gap_is_large"),
                "next_gap_is_large": row.get("next_gap_is_large"),
                "previous_gap_ratio_to_expected": row.get("previous_gap_ratio_to_expected"),
                "next_gap_ratio_to_expected": row.get("next_gap_ratio_to_expected"),
                "previous_to_next_gap_ratio": row.get("previous_to_next_gap_ratio"),
                "expected_snapshot_gap_hours": row.get("expected_snapshot_gap_hours"),
                "max_gap_hours_seen": row.get("max_gap_hours_seen"),
                "irregular_gap_count": row.get("irregular_gap_count"),
                "large_gap_count": row.get("large_gap_count"),
                "regular_gap_share": row.get("regular_gap_share"),
                "previous_upvote_velocity_per_hour": row.get("previous_upvote_velocity_per_hour"),
                "previous_comment_velocity_per_hour": row.get("previous_comment_velocity_per_hour"),
                "previous_upvote_delta_from_previous_snapshot": row.get("previous_upvote_delta_from_previous_snapshot"),
                "previous_comment_delta_from_previous_snapshot": row.get("previous_comment_delta_from_previous_snapshot"),
                "upvote_velocity_change_from_previous": row.get("upvote_velocity_change_from_previous"),
                "comment_velocity_change_from_previous": row.get("comment_velocity_change_from_previous"),
                "upvote_velocity_acceleration_per_hour2": row.get("upvote_velocity_acceleration_per_hour2"),
                "comment_velocity_acceleration_per_hour2": row.get("comment_velocity_acceleration_per_hour2"),
                "upvote_velocity_ratio_to_previous": row.get("upvote_velocity_ratio_to_previous"),
                "comment_velocity_ratio_to_previous": row.get("comment_velocity_ratio_to_previous"),
                "upvote_velocity_ratio_to_prior_peak": row.get("upvote_velocity_ratio_to_prior_peak"),
                "comment_velocity_ratio_to_prior_peak": row.get("comment_velocity_ratio_to_prior_peak"),
                "recent_upvote_velocity_mean_2": row.get("recent_upvote_velocity_mean_2"),
                "recent_comment_velocity_mean_2": row.get("recent_comment_velocity_mean_2"),
                "upvote_velocity_ratio_to_recent_mean_2": row.get("upvote_velocity_ratio_to_recent_mean_2"),
                "comment_velocity_ratio_to_recent_mean_2": row.get("comment_velocity_ratio_to_recent_mean_2"),
                "title_word_count": row.get("title_word_count"),
                "body_word_count": row.get("body_word_count"),
                "title_uppercase_ratio": row.get("title_uppercase_ratio"),
                "title_has_question_mark": row.get("title_has_question_mark"),
                "title_has_exclamation_mark": row.get("title_has_exclamation_mark"),
                "title_has_colon": row.get("title_has_colon"),
                "title_has_quotes": row.get("title_has_quotes"),
                "title_has_number": row.get("title_has_number"),
                "title_starts_with_question_word": row.get("title_starts_with_question_word"),
                "link_domain_category": row.get("link_domain_category"),
                "content_topic_primary": row.get("content_topic_primary"),
                "content_has_breaking_word": row.get("content_has_breaking_word"),
                "content_has_update_word": row.get("content_has_update_word"),
                "content_has_live_word": row.get("content_has_live_word"),
                "content_has_trailer_word": row.get("content_has_trailer_word"),
                "content_has_review_word": row.get("content_has_review_word"),
                "content_has_guide_word": row.get("content_has_guide_word"),
                "content_has_leak_word": row.get("content_has_leak_word"),
                "content_has_ama_word": row.get("content_has_ama_word"),
                "content_has_analysis_word": row.get("content_has_analysis_word"),
                "content_has_report_word": row.get("content_has_report_word"),
                "prior_same_title_posts_24h_subreddit": row.get("prior_same_title_posts_24h_subreddit"),
                "prior_same_story_posts_24h_subreddit": row.get("prior_same_story_posts_24h_subreddit"),
                "same_title_seen_before_24h_subreddit": row.get("same_title_seen_before_24h_subreddit"),
                "same_story_seen_before_24h_subreddit": row.get("same_story_seen_before_24h_subreddit"),
                "has_comment_sample": row.get("has_comment_sample"),
                "comment_sample_count": row.get("comment_sample_count"),
                "comment_sample_coverage_ratio": row.get("comment_sample_coverage_ratio"),
                "top_level_comment_sample_count": row.get("top_level_comment_sample_count"),
                "reply_comment_sample_count": row.get("reply_comment_sample_count"),
                "top_level_comment_sample_share": row.get("top_level_comment_sample_share"),
                "unique_commenter_count_sample": row.get("unique_commenter_count_sample"),
                "deleted_comment_share_sample": row.get("deleted_comment_share_sample"),
                "question_comment_share_sample": row.get("question_comment_share_sample"),
                "op_replied_in_comment_sample": row.get("op_replied_in_comment_sample"),
                "avg_comment_upvotes_sample": row.get("avg_comment_upvotes_sample"),
                "max_comment_upvotes_sample": row.get("max_comment_upvotes_sample"),
                "avg_comment_body_word_count_sample": row.get("avg_comment_body_word_count_sample"),
                "max_comment_body_word_count_sample": row.get("max_comment_body_word_count_sample"),
                "newest_comment_age_minutes_sample": row.get("newest_comment_age_minutes_sample"),
                "oldest_comment_age_minutes_sample": row.get("oldest_comment_age_minutes_sample"),
                "comment_sample_span_minutes": row.get("comment_sample_span_minutes"),
                "sentiment_mean_sample": row.get("sentiment_mean_sample"),
                "sentiment_weighted_mean_sample": row.get("sentiment_weighted_mean_sample"),
                "sentiment_positive_share_sample": row.get("sentiment_positive_share_sample"),
                "sentiment_negative_share_sample": row.get("sentiment_negative_share_sample"),
                "sentiment_variance_sample": row.get("sentiment_variance_sample"),
                "activity_state": row.get("activity_state"),
                "analysis_priority": row.get("analysis_priority"),
                "still_visible_next_snapshot": row.get("still_visible_next_snapshot"),
                "next_snapshot_id": row.get("next_snapshot_id"),
                "next_snapshot_time_utc": row.get("next_snapshot_time_utc"),
                "upvote_delta_next_snapshot": row.get("upvote_delta_next_snapshot"),
                "comment_delta_next_snapshot": row.get("comment_delta_next_snapshot"),
                "upvote_delta_next_hour_equivalent": row.get("upvote_delta_next_hour_equivalent"),
                "comment_delta_next_hour_equivalent": row.get("comment_delta_next_hour_equivalent"),
                "eligible_for_broad_next_hour_label": row.get("eligible_for_broad_next_hour_label"),
                "eligible_for_strict_next_hour_label": row.get("eligible_for_strict_next_hour_label"),
                "eligible_for_regular_cadence_label": row.get("eligible_for_regular_cadence_label"),
                "alive_next_snapshot": row.get("alive_next_snapshot"),
                "surging_next_snapshot": row.get("surging_next_snapshot"),
                "snapshot_hour_utc": row.get("snapshot_hour_utc"),
                "snapshot_weekday_utc": row.get("snapshot_weekday_utc"),
                "is_night_snapshot_utc": row.get("is_night_snapshot_utc"),
                "first_seen_is_night_utc": row.get("first_seen_is_night_utc"),
                "carried_from_night_to_day": row.get("carried_from_night_to_day"),
                "carried_from_day_to_night": row.get("carried_from_day_to_night"),
            }
        )

    timeline_rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            row.get("post_id", ""),
            row.get("snapshot_time_utc", ""),
            row.get("snapshot_id", ""),
        )
    )
    return timeline_rows


def sequence_key(row: dict[str, str]) -> tuple[str, str] | None:
    subreddit = clean_text(row.get("subreddit")).lower()
    post_id = clean_text(row.get("post_id"))
    if not subreddit or not post_id:
        return None
    return subreddit, post_id


def utc_hour(dt: Any) -> int | None:
    return dt.hour if dt is not None else None


def utc_weekday(dt: Any) -> int | None:
    return dt.weekday() if dt is not None else None


def is_night_hour(hour: int | None) -> int:
    return int(hour is not None and 0 <= hour < UTC_NIGHT_END_HOUR)


def safe_max(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def safe_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return (sum(present) / len(present)) if present else None


def safe_median(values: list[float | None]) -> float | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    midpoint = len(present) // 2
    if len(present) % 2:
        return present[midpoint]
    return (present[midpoint - 1] + present[midpoint]) / 2


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def gap_bucket(
    hours: float | None,
    *,
    strict_min_hours: float,
    strict_max_hours: float,
    large_gap_hours: float,
) -> str:
    if hours is None:
        return "missing"
    if hours < strict_min_hours:
        return "short"
    if hours <= strict_max_hours:
        return "regular"
    if hours <= large_gap_hours:
        return "wide"
    return "very_wide"


def unique_join(values: list[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return "|".join(ordered)


def build_sequence_metadata(
    post_rows: list[dict[str, str]],
    *,
    strict_min_next_hours: float,
    strict_max_next_hours: float,
    large_gap_hours: float,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in post_rows:
        key = sequence_key(row)
        if key is None:
            continue
        grouped_rows[key].append(row)

    snapshot_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    sequence_rows: list[dict[str, Any]] = []

    for (subreddit, post_id), rows in grouped_rows.items():
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                clean_text(row.get("snapshot_time_utc")),
                clean_text(row.get("snapshot_id")),
            ),
        )
        first_row = ordered_rows[0]
        last_row = ordered_rows[-1]
        first_seen_dt = parse_datetime(first_row.get("snapshot_time_utc"))
        last_seen_dt = parse_datetime(last_row.get("snapshot_time_utc"))
        first_seen_hour = utc_hour(first_seen_dt)
        last_seen_hour = utc_hour(last_seen_dt)
        first_seen_is_night = is_night_hour(first_seen_hour)
        last_seen_is_night = is_night_hour(last_seen_hour)
        hours_span = None
        if first_seen_dt and last_seen_dt:
            hours_span = (last_seen_dt - first_seen_dt).total_seconds() / 3600

        story_key_type, story_key = canonical_story_key(first_row)
        sequence_id = f"{subreddit}:{post_id}"
        listing_types_seen = unique_join([clean_text(row.get("listing_type")) for row in ordered_rows])
        activity_states_seen = unique_join([clean_text(row.get("activity_state")) for row in ordered_rows])
        snapshot_hours = [utc_hour(parse_datetime(row.get("snapshot_time_utc"))) for row in ordered_rows]
        night_snapshot_count = sum(is_night_hour(hour) for hour in snapshot_hours)
        day_snapshot_count = len(ordered_rows) - night_snapshot_count
        carried_from_night_to_day = int(
            first_seen_is_night == 1
            and any(hour is not None and is_night_hour(hour) == 0 for hour in snapshot_hours)
        )
        carried_from_day_to_night = int(
            first_seen_is_night == 0
            and any(hour is not None and is_night_hour(hour) == 1 for hour in snapshot_hours[1:])
        )
        max_upvotes_seen = safe_max([parse_float(row.get("upvotes_at_snapshot")) for row in ordered_rows])
        max_comments_seen = safe_max([parse_float(row.get("comment_count_at_snapshot")) for row in ordered_rows])
        max_upvote_velocity_seen = safe_max(
            [parse_float(row.get("upvote_velocity_per_hour")) for row in ordered_rows]
        )
        max_comment_velocity_seen = safe_max(
            [parse_float(row.get("comment_velocity_per_hour")) for row in ordered_rows]
        )
        upvote_velocities = [parse_float(row.get("upvote_velocity_per_hour")) for row in ordered_rows]
        comment_velocities = [parse_float(row.get("comment_velocity_per_hour")) for row in ordered_rows]
        upvote_deltas = [parse_float(row.get("upvote_delta_from_previous_snapshot")) for row in ordered_rows]
        comment_deltas = [parse_float(row.get("comment_delta_from_previous_snapshot")) for row in ordered_rows]
        sequence_length_total = len(ordered_rows)
        night_snapshot_share = (
            night_snapshot_count / sequence_length_total if sequence_length_total else None
        )
        sequence_gap_hours = [
            parse_float(row.get("hours_since_previous_snapshot"))
            for row in ordered_rows[1:]
            if parse_float(row.get("hours_since_previous_snapshot")) not in (None, 0)
        ]
        expected_snapshot_gap_hours = safe_median(sequence_gap_hours)
        max_gap_hours_seen = safe_max(sequence_gap_hours)
        irregular_gap_count = sum(
            1
            for gap in sequence_gap_hours
            if gap < strict_min_next_hours or gap > strict_max_next_hours
        )
        large_gap_count = sum(1 for gap in sequence_gap_hours if gap > large_gap_hours)
        regular_gap_share = (
            (len(sequence_gap_hours) - irregular_gap_count) / len(sequence_gap_hours)
            if sequence_gap_hours
            else None
        )

        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "subreddit": subreddit,
                "post_id": post_id,
                "story_key_type": story_key_type,
                "story_key": story_key,
                "title": clean_text(first_row.get("title")),
                "author": clean_text(first_row.get("author")),
                "created_at": clean_text(first_row.get("created_at")),
                "first_seen_snapshot_id": clean_text(first_row.get("snapshot_id")),
                "first_seen_snapshot_time_utc": clean_text(first_row.get("snapshot_time_utc")),
                "last_seen_snapshot_id": clean_text(last_row.get("snapshot_id")),
                "last_seen_snapshot_time_utc": clean_text(last_row.get("snapshot_time_utc")),
                "first_seen_listing_type": clean_text(first_row.get("listing_type")),
                "last_seen_listing_type": clean_text(last_row.get("listing_type")),
                "sequence_length_total": sequence_length_total,
                "sequence_span_hours": hours_span,
                "first_seen_hour_utc": first_seen_hour,
                "last_seen_hour_utc": last_seen_hour,
                "first_seen_weekday_utc": utc_weekday(first_seen_dt),
                "last_seen_weekday_utc": utc_weekday(last_seen_dt),
                "first_seen_is_night_utc": first_seen_is_night,
                "last_seen_is_night_utc": last_seen_is_night,
                "night_snapshot_count": night_snapshot_count,
                "day_snapshot_count": day_snapshot_count,
                "night_snapshot_share": night_snapshot_share,
                "carried_from_night_to_day": carried_from_night_to_day,
                "carried_from_day_to_night": carried_from_day_to_night,
                "expected_snapshot_gap_hours": expected_snapshot_gap_hours,
                "max_gap_hours_seen": max_gap_hours_seen,
                "irregular_gap_count": irregular_gap_count,
                "large_gap_count": large_gap_count,
                "regular_gap_share": regular_gap_share,
                "listing_types_seen": listing_types_seen,
                "activity_states_seen": activity_states_seen,
                "max_upvotes_seen": max_upvotes_seen,
                "max_comments_seen": max_comments_seen,
                "max_upvote_velocity_seen": max_upvote_velocity_seen,
                "max_comment_velocity_seen": max_comment_velocity_seen,
                "latest_activity_state": clean_text(last_row.get("activity_state")),
                "latest_analysis_priority": clean_text(last_row.get("analysis_priority")),
                "latest_age_hours": (
                    (parse_float(last_row.get("age_minutes_at_snapshot")) or 0.0) / 60
                    if parse_float(last_row.get("age_minutes_at_snapshot")) is not None
                    else None
                ),
            }
        )

        for sequence_position, row in enumerate(ordered_rows, start=1):
            index = sequence_position - 1
            snapshot_id = clean_text(row.get("snapshot_id"))
            snapshot_time_dt = parse_datetime(row.get("snapshot_time_utc"))
            snapshot_hour = utc_hour(snapshot_time_dt)
            hours_since_first_seen_snapshot = None
            hours_until_last_seen_snapshot = None
            if first_seen_dt and snapshot_time_dt:
                hours_since_first_seen_snapshot = (
                    snapshot_time_dt - first_seen_dt
                ).total_seconds() / 3600
            if last_seen_dt and snapshot_time_dt:
                hours_until_last_seen_snapshot = (
                    last_seen_dt - snapshot_time_dt
                ).total_seconds() / 3600
            current_upvote_velocity = upvote_velocities[index]
            current_comment_velocity = comment_velocities[index]
            previous_upvote_velocity = upvote_velocities[index - 1] if index > 0 else None
            previous_comment_velocity = comment_velocities[index - 1] if index > 0 else None
            previous_upvote_delta = upvote_deltas[index - 1] if index > 0 else None
            previous_comment_delta = comment_deltas[index - 1] if index > 0 else None
            current_listing_type = clean_text(row.get("listing_type"))
            previous_listing_type = clean_text(ordered_rows[index - 1].get("listing_type")) if index > 0 else ""
            current_rank = parse_int(row.get("rank_within_snapshot"))
            previous_rank = parse_int(ordered_rows[index - 1].get("rank_within_snapshot")) if index > 0 else None
            current_hours_since_previous = parse_float(row.get("hours_since_previous_snapshot"))
            next_hours = parse_float(row.get("hours_to_next_snapshot"))
            upvote_velocity_change = (
                current_upvote_velocity - previous_upvote_velocity
                if current_upvote_velocity is not None and previous_upvote_velocity is not None
                else None
            )
            comment_velocity_change = (
                current_comment_velocity - previous_comment_velocity
                if current_comment_velocity is not None and previous_comment_velocity is not None
                else None
            )
            upvote_velocity_acceleration = (
                upvote_velocity_change / current_hours_since_previous
                if upvote_velocity_change is not None
                and current_hours_since_previous not in (None, 0)
                else None
            )
            comment_velocity_acceleration = (
                comment_velocity_change / current_hours_since_previous
                if comment_velocity_change is not None
                and current_hours_since_previous not in (None, 0)
                else None
            )
            prior_upvote_peak = safe_max(upvote_velocities[:index]) if index > 0 else None
            prior_comment_peak = safe_max(comment_velocities[:index]) if index > 0 else None
            recent_upvote_velocity_mean = safe_mean(upvote_velocities[max(0, index - 2):index])
            recent_comment_velocity_mean = safe_mean(comment_velocities[max(0, index - 2):index])
            listing_transition = (
                f"{previous_listing_type or '__none__'}->{current_listing_type or '__none__'}"
                if index > 0
                else f"__start__->{current_listing_type or '__none__'}"
            )
            listing_changed = int(index > 0 and previous_listing_type != current_listing_type)
            listing_run_length_snapshots = 1
            hours_in_current_listing = 0.0
            lookback_index = index - 1
            while lookback_index >= 0:
                lookback_row = ordered_rows[lookback_index]
                if clean_text(lookback_row.get("listing_type")) != current_listing_type:
                    break
                listing_run_length_snapshots += 1
                lookback_hours = parse_float(ordered_rows[lookback_index + 1].get("hours_since_previous_snapshot"))
                if lookback_hours is not None:
                    hours_in_current_listing += lookback_hours
                lookback_index -= 1
            rank_change = (
                previous_rank - current_rank
                if previous_rank is not None and current_rank is not None
                else None
            )
            previous_gap_bucket = gap_bucket(
                current_hours_since_previous,
                strict_min_hours=strict_min_next_hours,
                strict_max_hours=strict_max_next_hours,
                large_gap_hours=large_gap_hours,
            )
            next_gap_bucket = gap_bucket(
                next_hours,
                strict_min_hours=strict_min_next_hours,
                strict_max_hours=strict_max_next_hours,
                large_gap_hours=large_gap_hours,
            )

            snapshot_meta[(snapshot_id, subreddit, post_id)] = {
                "sequence_id": sequence_id,
                "sequence_position": sequence_position,
                "sequence_length_total": sequence_length_total,
                "sequence_remaining_snapshots": sequence_length_total - sequence_position,
                "sequence_progress_ratio": (
                    sequence_position / sequence_length_total if sequence_length_total else None
                ),
                "first_seen_snapshot_id": clean_text(first_row.get("snapshot_id")),
                "first_seen_snapshot_time_utc": clean_text(first_row.get("snapshot_time_utc")),
                "last_seen_snapshot_id": clean_text(last_row.get("snapshot_id")),
                "last_seen_snapshot_time_utc": clean_text(last_row.get("snapshot_time_utc")),
                "first_seen_listing_type": clean_text(first_row.get("listing_type")),
                "last_seen_listing_type": clean_text(last_row.get("listing_type")),
                "previous_listing_type": previous_listing_type,
                "listing_transition": listing_transition,
                "listing_changed_from_previous": listing_changed,
                "listing_run_length_snapshots": listing_run_length_snapshots,
                "hours_in_current_listing": hours_in_current_listing,
                "hours_since_first_seen_snapshot": hours_since_first_seen_snapshot,
                "hours_until_last_seen_snapshot": hours_until_last_seen_snapshot,
                "sequence_span_hours": hours_span,
                "expected_snapshot_gap_hours": expected_snapshot_gap_hours,
                "max_gap_hours_seen": max_gap_hours_seen,
                "irregular_gap_count": irregular_gap_count,
                "large_gap_count": large_gap_count,
                "regular_gap_share": regular_gap_share,
                "snapshot_hour_utc": snapshot_hour,
                "snapshot_weekday_utc": utc_weekday(snapshot_time_dt),
                "is_night_snapshot_utc": is_night_hour(snapshot_hour),
                "first_seen_hour_utc": first_seen_hour,
                "first_seen_weekday_utc": utc_weekday(first_seen_dt),
                "first_seen_is_night_utc": first_seen_is_night,
                "last_seen_hour_utc": last_seen_hour,
                "last_seen_weekday_utc": utc_weekday(last_seen_dt),
                "last_seen_is_night_utc": last_seen_is_night,
                "night_snapshot_count": night_snapshot_count,
                "day_snapshot_count": day_snapshot_count,
                "night_snapshot_share": night_snapshot_share,
                "carried_from_night_to_day": carried_from_night_to_day,
                "carried_from_day_to_night": carried_from_day_to_night,
                "previous_rank_within_snapshot": previous_rank,
                "rank_change_from_previous_snapshot": rank_change,
                "abs_rank_change_from_previous_snapshot": abs(rank_change) if rank_change is not None else None,
                "rank_improved_from_previous": int(rank_change is not None and rank_change > 0),
                "rank_worsened_from_previous": int(rank_change is not None and rank_change < 0),
                "rank_unchanged_from_previous": int(rank_change == 0) if rank_change is not None else 0,
                "previous_upvote_velocity_per_hour": previous_upvote_velocity,
                "previous_comment_velocity_per_hour": previous_comment_velocity,
                "previous_gap_bucket": previous_gap_bucket,
                "next_gap_bucket": next_gap_bucket,
                "previous_gap_is_regular": int(
                    current_hours_since_previous is not None
                    and strict_min_next_hours <= current_hours_since_previous <= strict_max_next_hours
                ),
                "next_gap_is_regular": int(
                    next_hours is not None
                    and strict_min_next_hours <= next_hours <= strict_max_next_hours
                ),
                "previous_gap_is_large": int(
                    current_hours_since_previous is not None and current_hours_since_previous > large_gap_hours
                ),
                "next_gap_is_large": int(
                    next_hours is not None and next_hours > large_gap_hours
                ),
                "previous_gap_ratio_to_expected": safe_ratio(
                    current_hours_since_previous, expected_snapshot_gap_hours
                ),
                "next_gap_ratio_to_expected": safe_ratio(
                    next_hours, expected_snapshot_gap_hours
                ),
                "previous_to_next_gap_ratio": safe_ratio(
                    current_hours_since_previous, next_hours
                ),
                "previous_upvote_delta_from_previous_snapshot": previous_upvote_delta,
                "previous_comment_delta_from_previous_snapshot": previous_comment_delta,
                "upvote_velocity_change_from_previous": upvote_velocity_change,
                "comment_velocity_change_from_previous": comment_velocity_change,
                "upvote_velocity_acceleration_per_hour2": upvote_velocity_acceleration,
                "comment_velocity_acceleration_per_hour2": comment_velocity_acceleration,
                "upvote_velocity_ratio_to_previous": safe_ratio(
                    current_upvote_velocity, previous_upvote_velocity
                ),
                "comment_velocity_ratio_to_previous": safe_ratio(
                    current_comment_velocity, previous_comment_velocity
                ),
                "upvote_velocity_ratio_to_prior_peak": safe_ratio(
                    current_upvote_velocity, prior_upvote_peak
                ),
                "comment_velocity_ratio_to_prior_peak": safe_ratio(
                    current_comment_velocity, prior_comment_peak
                ),
                "recent_upvote_velocity_mean_2": recent_upvote_velocity_mean,
                "recent_comment_velocity_mean_2": recent_comment_velocity_mean,
                "upvote_velocity_ratio_to_recent_mean_2": safe_ratio(
                    current_upvote_velocity, recent_upvote_velocity_mean
                ),
                "comment_velocity_ratio_to_recent_mean_2": safe_ratio(
                    current_comment_velocity, recent_comment_velocity_mean
                ),
            }

    sequence_rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            row.get("first_seen_snapshot_time_utc", ""),
            row.get("post_id", ""),
        )
    )
    return snapshot_meta, sequence_rows


def build_prediction_rows(
    post_rows: list[dict[str, str]],
    comment_rows: list[dict[str, str]] | None = None,
    *,
    min_next_hours: float,
    max_next_hours: float,
    strict_min_next_hours: float,
    strict_max_next_hours: float,
    large_gap_hours: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_post_rows = sorted(
        post_rows,
        key=lambda row: (
            clean_text(row.get("snapshot_time_utc")),
            clean_text(row.get("snapshot_id")),
            clean_text(row.get("subreddit")).lower(),
            clean_text(row.get("post_id")),
        ),
    )
    indexed_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in ordered_post_rows:
        indexed_rows[
            (
                clean_text(row.get("snapshot_id")),
                clean_text(row.get("subreddit")).lower(),
                clean_text(row.get("post_id")),
            )
        ] = row

    sequence_meta, sequence_rows = build_sequence_metadata(
        ordered_post_rows,
        strict_min_next_hours=strict_min_next_hours,
        strict_max_next_hours=strict_max_next_hours,
        large_gap_hours=large_gap_hours,
    )
    title_repeat_indexes, story_repeat_indexes = build_recent_repeat_indexes(ordered_post_rows)
    post_author_by_key = {
        (
            clean_text(row.get("snapshot_id")),
            clean_text(row.get("subreddit")).lower(),
            clean_text(row.get("post_id")),
        ): clean_text(row.get("author"))
        for row in ordered_post_rows
    }
    comment_snapshot_aggregates = build_comment_snapshot_aggregates(
        comment_rows or [],
        post_author_by_key=post_author_by_key,
    )

    all_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []

    for row in ordered_post_rows:
        snapshot_id = clean_text(row.get("snapshot_id"))
        subreddit = clean_text(row.get("subreddit"))
        subreddit_key = subreddit.lower()
        next_snapshot_id = clean_text(row.get("next_snapshot_id"))
        post_id = clean_text(row.get("post_id"))
        next_row = (
            indexed_rows.get((next_snapshot_id, subreddit_key, post_id))
            if next_snapshot_id
            else None
        )
        sequence_info = sequence_meta.get((snapshot_id, subreddit_key, post_id), {})

        story_key_type, story_key = canonical_story_key(row)
        hours_to_next_snapshot = parse_float(row.get("hours_to_next_snapshot"))
        next_upvote_delta = parse_float(row.get("upvote_delta_to_next_snapshot"))
        next_comment_delta = parse_float(row.get("comment_delta_to_next_snapshot"))
        next_state = clean_text(next_row.get("activity_state")) if next_row else ""
        next_alive = int(next_state in {"alive", "surging", "emerging", "cooling"}) if next_row else None
        next_surging = int(next_state == "surging") if next_row else None
        next_rising = int(next_state in {"surging", "alive", "emerging"}) if next_row else None
        next_cooling = int(next_state == "cooling") if next_row else None
        next_dying = int(next_state == "dying") if next_row else None
        next_dead = int(next_state == "dead") if next_row else None
        next_weakening = int(next_state in {"cooling", "dying", "dead"}) if next_row else None
        current_snapshot_time = parse_datetime(row.get("snapshot_time_utc"))
        created_at = parse_datetime(row.get("created_at"))
        post_age_hours = None
        if current_snapshot_time and created_at:
            post_age_hours = (current_snapshot_time - created_at).total_seconds() / 3600
        current_snapshot_ts = current_snapshot_time.timestamp() if current_snapshot_time else None
        title = clean_text(row.get("title"))
        body = clean_text(row.get("body"))
        comment_sample = comment_snapshot_aggregates.get((snapshot_id, subreddit_key, post_id), {})
        title_key = normalize_title_key(title)
        title_repeat_count = count_prior_distinct_posts_24h(
            title_repeat_indexes.get((subreddit_key, title_key)) if title_key else None,
            current_ts=current_snapshot_ts,
            current_post_id=post_id,
        )
        story_repeat_count = count_prior_distinct_posts_24h(
            story_repeat_indexes.get((subreddit_key, story_key_type, story_key)) if story_key else None,
            current_ts=current_snapshot_ts,
            current_post_id=post_id,
        )
        content_flags = keyword_flags(title, body)
        title_uppercase = uppercase_ratio(title)
        broad_next_hour_eligible = int(
            hours_to_next_snapshot is not None
            and min_next_hours <= hours_to_next_snapshot <= max_next_hours
        )
        strict_next_hour_eligible = int(
            hours_to_next_snapshot is not None
            and strict_min_next_hours <= hours_to_next_snapshot <= strict_max_next_hours
        )
        regular_cadence_eligible = int(
            strict_next_hour_eligible == 1
            and (
                parse_float(row.get("hours_since_previous_snapshot")) is None
                or (
                    strict_min_next_hours
                    <= parse_float(row.get("hours_since_previous_snapshot"))
                    <= strict_max_next_hours
                )
            )
        )

        record = {
            "snapshot_id": snapshot_id,
            "snapshot_time_utc": clean_text(row.get("snapshot_time_utc")),
            "subreddit": subreddit,
            "listing_type": clean_text(row.get("listing_type")),
            "previous_listing_type": clean_text(sequence_info.get("previous_listing_type")),
            "listing_transition": clean_text(sequence_info.get("listing_transition")),
            "listing_changed_from_previous": parse_int(sequence_info.get("listing_changed_from_previous")),
            "listing_run_length_snapshots": parse_int(sequence_info.get("listing_run_length_snapshots")),
            "hours_in_current_listing": parse_float(sequence_info.get("hours_in_current_listing")),
            "schedule_name": clean_text(row.get("schedule_name")),
            "post_id": post_id,
            "parsed_post_id": clean_text(row.get("parsed_post_id")),
            "story_key_type": story_key_type,
            "story_key": story_key,
            "url": clean_text(row.get("url")),
            "external_link": clean_text(row.get("external_link")),
            "link_domain": clean_text(row.get("link_domain")),
            "link_domain_category": classify_link_domain(
                clean_text(row.get("external_link")),
                clean_text(row.get("link_domain")),
            ),
            "title": title,
            "body": body,
            "title_length_chars": len(title),
            "body_length_chars": len(body),
            "title_word_count": count_words(title),
            "body_word_count": count_words(body),
            "title_uppercase_ratio": title_uppercase,
            "title_has_question_mark": int("?" in title),
            "title_has_exclamation_mark": int("!" in title),
            "title_has_colon": int(":" in title),
            "title_has_quotes": int(any(mark in title for mark in ('"', "'", "“", "”", "‘", "’"))),
            "title_has_number": int(any(char.isdigit() for char in title)),
            "title_starts_with_question_word": starts_with_question_word(title),
            "content_topic_primary": detect_primary_topic(title, body),
            **content_flags,
            "prior_same_title_posts_24h_subreddit": title_repeat_count,
            "prior_same_story_posts_24h_subreddit": story_repeat_count,
            "same_title_seen_before_24h_subreddit": int(title_repeat_count > 0),
            "same_story_seen_before_24h_subreddit": int(story_repeat_count > 0),
            "has_comment_sample": parse_int(comment_sample.get("has_comment_sample")),
            "comment_sample_count": parse_int(comment_sample.get("comment_sample_count")),
            "comment_sample_coverage_ratio": safe_ratio(
                parse_float(comment_sample.get("comment_sample_count")),
                parse_float(row.get("comment_count_at_snapshot")),
            ),
            "top_level_comment_sample_count": parse_int(comment_sample.get("top_level_comment_sample_count")),
            "reply_comment_sample_count": parse_int(comment_sample.get("reply_comment_sample_count")),
            "top_level_comment_sample_share": parse_float(comment_sample.get("top_level_comment_sample_share")),
            "unique_commenter_count_sample": parse_int(comment_sample.get("unique_commenter_count_sample")),
            "deleted_comment_share_sample": parse_float(comment_sample.get("deleted_comment_share_sample")),
            "question_comment_share_sample": parse_float(comment_sample.get("question_comment_share_sample")),
            "op_replied_in_comment_sample": parse_int(comment_sample.get("op_replied_in_comment_sample")),
            "avg_comment_upvotes_sample": parse_float(comment_sample.get("avg_comment_upvotes_sample")),
            "max_comment_upvotes_sample": parse_float(comment_sample.get("max_comment_upvotes_sample")),
            "avg_comment_body_word_count_sample": parse_float(comment_sample.get("avg_comment_body_word_count_sample")),
            "max_comment_body_word_count_sample": parse_float(comment_sample.get("max_comment_body_word_count_sample")),
            "newest_comment_age_minutes_sample": parse_float(comment_sample.get("newest_comment_age_minutes_sample")),
            "oldest_comment_age_minutes_sample": parse_float(comment_sample.get("oldest_comment_age_minutes_sample")),
            "comment_sample_span_minutes": parse_float(comment_sample.get("comment_sample_span_minutes")),
            "sentiment_mean_sample": parse_float(comment_sample.get("sentiment_mean_sample")),
            "sentiment_weighted_mean_sample": parse_float(comment_sample.get("sentiment_weighted_mean_sample")),
            "sentiment_positive_share_sample": parse_float(comment_sample.get("sentiment_positive_share_sample")),
            "sentiment_negative_share_sample": parse_float(comment_sample.get("sentiment_negative_share_sample")),
            "sentiment_variance_sample": parse_float(comment_sample.get("sentiment_variance_sample")),
            "author": clean_text(row.get("author")),
            "created_at": clean_text(row.get("created_at")),
            "age_minutes_at_snapshot": parse_float(row.get("age_minutes_at_snapshot")),
            "age_hours_at_snapshot": post_age_hours,
            "age_bucket": clean_text(row.get("age_bucket")),
            "rank_within_snapshot": parse_int(row.get("rank_within_snapshot")),
            "previous_rank_within_snapshot": parse_int(sequence_info.get("previous_rank_within_snapshot")),
            "rank_change_from_previous_snapshot": parse_float(sequence_info.get("rank_change_from_previous_snapshot")),
            "abs_rank_change_from_previous_snapshot": parse_float(sequence_info.get("abs_rank_change_from_previous_snapshot")),
            "rank_improved_from_previous": parse_int(sequence_info.get("rank_improved_from_previous")),
            "rank_worsened_from_previous": parse_int(sequence_info.get("rank_worsened_from_previous")),
            "rank_unchanged_from_previous": parse_int(sequence_info.get("rank_unchanged_from_previous")),
            "upvotes_at_snapshot": parse_float(row.get("upvotes_at_snapshot")),
            "comment_count_at_snapshot": parse_float(row.get("comment_count_at_snapshot")),
            "upvote_ratio_at_snapshot": parse_float(row.get("upvote_ratio_at_snapshot")),
            "upvote_velocity_per_hour": parse_float(row.get("upvote_velocity_per_hour")),
            "comment_velocity_per_hour": parse_float(row.get("comment_velocity_per_hour")),
            "upvote_delta_from_previous_snapshot": parse_float(row.get("upvote_delta_from_previous_snapshot")),
            "comment_delta_from_previous_snapshot": parse_float(row.get("comment_delta_from_previous_snapshot")),
            "hours_since_previous_snapshot": parse_float(row.get("hours_since_previous_snapshot")),
            "previous_gap_bucket": clean_text(sequence_info.get("previous_gap_bucket")),
            "next_gap_bucket": clean_text(sequence_info.get("next_gap_bucket")),
            "previous_gap_is_regular": parse_int(sequence_info.get("previous_gap_is_regular")),
            "next_gap_is_regular": parse_int(sequence_info.get("next_gap_is_regular")),
            "previous_gap_is_large": parse_int(sequence_info.get("previous_gap_is_large")),
            "next_gap_is_large": parse_int(sequence_info.get("next_gap_is_large")),
            "previous_gap_ratio_to_expected": parse_float(sequence_info.get("previous_gap_ratio_to_expected")),
            "next_gap_ratio_to_expected": parse_float(sequence_info.get("next_gap_ratio_to_expected")),
            "previous_to_next_gap_ratio": parse_float(sequence_info.get("previous_to_next_gap_ratio")),
            "expected_snapshot_gap_hours": parse_float(sequence_info.get("expected_snapshot_gap_hours")),
            "max_gap_hours_seen": parse_float(sequence_info.get("max_gap_hours_seen")),
            "irregular_gap_count": parse_int(sequence_info.get("irregular_gap_count")),
            "large_gap_count": parse_int(sequence_info.get("large_gap_count")),
            "regular_gap_share": parse_float(sequence_info.get("regular_gap_share")),
            "previous_upvote_velocity_per_hour": parse_float(row.get("previous_upvote_velocity_per_hour")),
            "previous_comment_velocity_per_hour": parse_float(row.get("previous_comment_velocity_per_hour")),
            "previous_upvote_delta_from_previous_snapshot": parse_float(row.get("previous_upvote_delta_from_previous_snapshot")),
            "previous_comment_delta_from_previous_snapshot": parse_float(row.get("previous_comment_delta_from_previous_snapshot")),
            "upvote_velocity_change_from_previous": parse_float(row.get("upvote_velocity_change_from_previous")),
            "comment_velocity_change_from_previous": parse_float(row.get("comment_velocity_change_from_previous")),
            "upvote_velocity_acceleration_per_hour2": parse_float(row.get("upvote_velocity_acceleration_per_hour2")),
            "comment_velocity_acceleration_per_hour2": parse_float(row.get("comment_velocity_acceleration_per_hour2")),
            "upvote_velocity_ratio_to_previous": parse_float(row.get("upvote_velocity_ratio_to_previous")),
            "comment_velocity_ratio_to_previous": parse_float(row.get("comment_velocity_ratio_to_previous")),
            "upvote_velocity_ratio_to_prior_peak": parse_float(row.get("upvote_velocity_ratio_to_prior_peak")),
            "comment_velocity_ratio_to_prior_peak": parse_float(row.get("comment_velocity_ratio_to_prior_peak")),
            "recent_upvote_velocity_mean_2": parse_float(row.get("recent_upvote_velocity_mean_2")),
            "recent_comment_velocity_mean_2": parse_float(row.get("recent_comment_velocity_mean_2")),
            "upvote_velocity_ratio_to_recent_mean_2": parse_float(row.get("upvote_velocity_ratio_to_recent_mean_2")),
            "comment_velocity_ratio_to_recent_mean_2": parse_float(row.get("comment_velocity_ratio_to_recent_mean_2")),
            "activity_state": clean_text(row.get("activity_state")),
            "analysis_priority": clean_text(row.get("analysis_priority")),
            "is_video": int(clean_text(row.get("is_video")).lower() == "true"),
            "has_images": int(clean_text(row.get("has_images")).lower() == "true"),
            "is_fresh_1h": int(clean_text(row.get("is_fresh_1h")) == "1"),
            "is_fresh_6h": int(clean_text(row.get("is_fresh_6h")) == "1"),
            "is_old_24h": int(clean_text(row.get("is_old_24h")) == "1"),
            "activity_threshold_source": clean_text(row.get("activity_threshold_source")),
            "alive_upvote_velocity_threshold": parse_float(row.get("alive_upvote_velocity_threshold")),
            "alive_comment_velocity_threshold": parse_float(row.get("alive_comment_velocity_threshold")),
            "surging_upvote_velocity_threshold": parse_float(row.get("surging_upvote_velocity_threshold")),
            "surging_comment_velocity_threshold": parse_float(row.get("surging_comment_velocity_threshold")),
            "dead_upvote_velocity_threshold": parse_float(row.get("dead_upvote_velocity_threshold")),
            "dead_comment_velocity_threshold": parse_float(row.get("dead_comment_velocity_threshold")),
            "next_snapshot_id": next_snapshot_id,
            "next_snapshot_time_utc": clean_text(row.get("next_snapshot_time_utc")),
            "hours_to_next_snapshot": hours_to_next_snapshot,
            "still_visible_next_snapshot": parse_int(row.get("still_visible_next_snapshot")),
            "next_upvotes_at_snapshot": parse_float(row.get("next_upvotes_at_snapshot")),
            "next_comment_count_at_snapshot": parse_float(row.get("next_comment_count_at_snapshot")),
            "upvote_delta_next_snapshot": next_upvote_delta,
            "comment_delta_next_snapshot": next_comment_delta,
            "upvote_delta_next_hour_equivalent": safe_ratio(next_upvote_delta, hours_to_next_snapshot),
            "comment_delta_next_hour_equivalent": safe_ratio(next_comment_delta, hours_to_next_snapshot),
            "next_activity_state": next_state,
            "alive_next_snapshot": next_alive,
            "surging_next_snapshot": next_surging,
            "rising_next_snapshot": next_rising,
            "cooling_next_snapshot": next_cooling,
            "dying_next_snapshot": next_dying,
            "dead_next_snapshot": next_dead,
            "weakening_next_snapshot": next_weakening,
            "high_comment_growth_next_snapshot": (
                int(
                    next_comment_delta is not None
                    and next_comment_delta >= (parse_float(row.get("alive_comment_velocity_threshold")) or 0)
                )
                if next_comment_delta is not None
                else None
            ),
            "high_upvote_growth_next_snapshot": (
                int(
                    next_upvote_delta is not None
                    and next_upvote_delta >= (parse_float(row.get("alive_upvote_velocity_threshold")) or 0)
                )
                if next_upvote_delta is not None
                else None
            ),
            "eligible_for_broad_next_hour_label": broad_next_hour_eligible,
            "eligible_for_strict_next_hour_label": strict_next_hour_eligible,
            "eligible_for_regular_cadence_label": regular_cadence_eligible,
            "eligible_for_next_hour_label": regular_cadence_eligible,
            **sequence_info,
        }
        all_rows.append(record)
        if record["eligible_for_next_hour_label"] == 1:
            hourly_rows.append(record)

    all_rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            row.get("snapshot_time_utc", ""),
            row.get("listing_type", ""),
            row.get("rank_within_snapshot") or 0,
            row.get("post_id", ""),
        )
    )
    hourly_rows.sort(
        key=lambda row: (
            row.get("subreddit", ""),
            row.get("snapshot_time_utc", ""),
            row.get("listing_type", ""),
            row.get("rank_within_snapshot") or 0,
            row.get("post_id", ""),
        )
    )
    return all_rows, hourly_rows, sequence_rows


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir)
    output_dir = Path(args.output_dir)

    post_rows = load_csv(history_dir / "post_snapshots.csv")
    comment_snapshots_path = history_dir / "comment_snapshots.csv"
    comment_rows = load_csv(comment_snapshots_path) if comment_snapshots_path.is_file() else []
    all_rows, hourly_rows, sequence_rows = build_prediction_rows(
        post_rows,
        comment_rows,
        min_next_hours=args.min_next_hours,
        max_next_hours=args.max_next_hours,
        strict_min_next_hours=args.strict_min_next_hours,
        strict_max_next_hours=args.strict_max_next_hours,
        large_gap_hours=args.large_gap_hours,
    )
    timeline_rows = build_timeline_rows(all_rows)

    all_rows_path = output_dir / "prediction_all_snapshots.csv"
    hourly_rows_path = output_dir / "prediction_next_hour.csv"
    sequence_rows_path = output_dir / "prediction_sequences.csv"
    timeline_rows_path = output_dir / "post_timeline_points.csv"
    write_csv(all_rows, all_rows_path)
    write_csv(hourly_rows, hourly_rows_path)
    write_csv(sequence_rows, sequence_rows_path)
    write_csv(timeline_rows, timeline_rows_path)

    print(f"Saved {len(all_rows)} prediction row(s) to {all_rows_path}")
    print(f"Saved {len(hourly_rows)} next-hour training row(s) to {hourly_rows_path}")
    print(f"Saved {len(sequence_rows)} sequence row(s) to {sequence_rows_path}")
    print(f"Saved {len(timeline_rows)} timeline row(s) to {timeline_rows_path}")


if __name__ == "__main__":
    main()
