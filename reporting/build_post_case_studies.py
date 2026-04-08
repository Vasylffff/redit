from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build readable case-study summaries for a small set of high-priority "
            "forecast/watch posts."
        )
    )
    parser.add_argument(
        "--forecast-input",
        default="data/history/reddit/naive_forecast_leaderboard.csv",
        help="Forecast leaderboard CSV used to choose candidate posts.",
    )
    parser.add_argument(
        "--timeline-input",
        default="data/models/reddit/post_timeline_points.csv",
        help="Timeline CSV with per-post snapshot points.",
    )
    parser.add_argument(
        "--comments-input",
        default="data/history/reddit/comment_snapshots.csv",
        help="Comment snapshot CSV for top sampled comments.",
    )
    parser.add_argument(
        "--output",
        default="data/history/reddit/post_case_studies_latest.csv",
        help="Output CSV path for compact case-study summaries.",
    )
    parser.add_argument(
        "--markdown-output",
        default="data/history/reddit/post_case_studies_latest.md",
        help="Output Markdown path for human-readable case studies.",
    )
    parser.add_argument(
        "--overall-limit",
        type=int,
        default=5,
        help="How many top overall posts to include.",
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


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Required CSV not found: {path}")
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


def truncate_text(text: str, limit: int = 160) -> str:
    normalized = " ".join(clean_text(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def choose_posts(
    forecast_rows: list[dict[str, str]],
    *,
    overall_limit: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in forecast_rows[: max(0, overall_limit)]:
        key = (clean_text(row.get("subreddit")).lower(), clean_text(row.get("post_id")))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)

    per_subreddit_best: dict[str, dict[str, str]] = {}
    for row in forecast_rows:
        subreddit = clean_text(row.get("subreddit")).lower()
        if subreddit and subreddit not in per_subreddit_best:
            per_subreddit_best[subreddit] = row
    for subreddit in sorted(per_subreddit_best):
        row = per_subreddit_best[subreddit]
        key = (clean_text(row.get("subreddit")).lower(), clean_text(row.get("post_id")))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)

    return selected


def main() -> None:
    args = parse_args()
    forecast_rows = load_rows(Path(args.forecast_input))
    timeline_rows = load_rows(Path(args.timeline_input))
    comment_rows = load_rows(Path(args.comments_input))

    selected_rows = choose_posts(forecast_rows, overall_limit=args.overall_limit)

    timeline_by_post: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in timeline_rows:
        key = (clean_text(row.get("subreddit")).lower(), clean_text(row.get("post_id")))
        if key[0] and key[1]:
            timeline_by_post[key].append(row)

    comments_by_post: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    latest_comment_snapshot_by_post: dict[tuple[str, str], str] = {}
    for row in comment_rows:
        key = (clean_text(row.get("subreddit")).lower(), clean_text(row.get("post_id")))
        snapshot_time = clean_text(row.get("snapshot_time_utc"))
        if not key[0] or not key[1]:
            continue
        if snapshot_time >= latest_comment_snapshot_by_post.get(key, ""):
            latest_comment_snapshot_by_post[key] = snapshot_time
        comments_by_post[key].append(row)

    output_rows: list[dict[str, Any]] = []
    markdown_lines = ["# Post Case Studies", ""]

    for case_rank, forecast_row in enumerate(selected_rows, start=1):
        subreddit = clean_text(forecast_row.get("subreddit"))
        post_id = clean_text(forecast_row.get("post_id"))
        key = (subreddit.lower(), post_id)
        post_timeline = sorted(
            timeline_by_post.get(key, []),
            key=lambda row: clean_text(row.get("snapshot_time_utc")),
        )
        latest_comment_snapshot_time = latest_comment_snapshot_by_post.get(key, "")
        latest_comments = [
            row
            for row in comments_by_post.get(key, [])
            if clean_text(row.get("snapshot_time_utc")) == latest_comment_snapshot_time
        ]
        latest_comments.sort(
            key=lambda row: (
                -(parse_float(row.get("upvotes_at_snapshot")) or 0.0),
                -(parse_float(row.get("reply_count_at_snapshot")) or 0.0),
                clean_text(row.get("comment_id")),
            )
        )

        listing_path_parts: list[str] = []
        for row in post_timeline:
            listing_type = clean_text(row.get("listing_type"))
            if listing_type and (not listing_path_parts or listing_path_parts[-1] != listing_type):
                listing_path_parts.append(listing_type)
        listing_path = " -> ".join(listing_path_parts)

        first_row = post_timeline[0] if post_timeline else {}
        last_row = post_timeline[-1] if post_timeline else {}
        recent_points = post_timeline[-3:] if len(post_timeline) >= 3 else post_timeline
        recent_timeline = " | ".join(
            (
                f"{clean_text(row.get('snapshot_time_utc'))[-8:]} "
                f"u={int(parse_float(row.get('upvotes_at_snapshot')) or 0)} "
                f"c={int(parse_float(row.get('comment_count_at_snapshot')) or 0)} "
                f"state={clean_text(row.get('activity_state'))}"
            )
            for row in recent_points
        )

        top_comment_1 = latest_comments[0] if len(latest_comments) >= 1 else {}
        top_comment_2 = latest_comments[1] if len(latest_comments) >= 2 else {}
        top_comment_1_excerpt = truncate_text(clean_text(top_comment_1.get("body")))
        top_comment_2_excerpt = truncate_text(clean_text(top_comment_2.get("body")))

        narrative = (
            f"{clean_text(forecast_row.get('latest_activity_state'))} post in r/{subreddit} "
            f"with forecast {clean_text(forecast_row.get('forecast_recommendation'))}; "
            f"expected next hour about +{int(parse_float(forecast_row.get('naive_predicted_upvote_delta_next_hour')) or 0)} "
            f"upvotes and +{int(parse_float(forecast_row.get('naive_predicted_comment_delta_next_hour')) or 0)} comments. "
            f"It has {len(post_timeline)} tracked snapshots over about "
            f"{round(parse_float(forecast_row.get('observed_hours')) or 0.0, 2)}h and a die-soon score of "
            f"{round(parse_float(forecast_row.get('die_soon_score')) or 0.0, 3)}."
        )

        output_rows.append(
            {
                "case_rank": case_rank,
                "subreddit": subreddit,
                "post_id": post_id,
                "title": clean_text(forecast_row.get("title")),
                "url": clean_text(forecast_row.get("url")),
                "latest_activity_state": clean_text(forecast_row.get("latest_activity_state")),
                "forecast_recommendation": clean_text(forecast_row.get("forecast_recommendation")),
                "die_soon_label": clean_text(forecast_row.get("die_soon_label")),
                "die_soon_score": parse_float(forecast_row.get("die_soon_score")),
                "current_attention_score": parse_float(forecast_row.get("current_attention_score")),
                "general_popularity_score": parse_float(forecast_row.get("general_popularity_score")),
                "naive_predicted_upvote_delta_next_hour": parse_float(forecast_row.get("naive_predicted_upvote_delta_next_hour")),
                "naive_predicted_comment_delta_next_hour": parse_float(forecast_row.get("naive_predicted_comment_delta_next_hour")),
                "naive_predicted_upvote_delta_next_3h": parse_float(forecast_row.get("naive_predicted_upvote_delta_next_3h")),
                "naive_predicted_comment_delta_next_3h": parse_float(forecast_row.get("naive_predicted_comment_delta_next_3h")),
                "naive_predicted_upvote_delta_next_6h": parse_float(forecast_row.get("naive_predicted_upvote_delta_next_6h")),
                "naive_predicted_comment_delta_next_6h": parse_float(forecast_row.get("naive_predicted_comment_delta_next_6h")),
                "timeline_snapshot_count": len(post_timeline),
                "listing_path": listing_path,
                "first_seen_time_utc": clean_text(first_row.get("snapshot_time_utc")),
                "last_seen_time_utc": clean_text(last_row.get("snapshot_time_utc")) or clean_text(forecast_row.get("last_seen_time_utc")),
                "first_upvotes": parse_float(first_row.get("upvotes_at_snapshot")),
                "last_upvotes": parse_float(last_row.get("upvotes_at_snapshot")),
                "first_comments": parse_float(first_row.get("comment_count_at_snapshot")),
                "last_comments": parse_float(last_row.get("comment_count_at_snapshot")),
                "recent_timeline_points": recent_timeline,
                "top_comment_1_excerpt": top_comment_1_excerpt,
                "top_comment_1_upvotes": parse_float(top_comment_1.get("upvotes_at_snapshot")),
                "top_comment_1_replies": parse_float(top_comment_1.get("reply_count_at_snapshot")),
                "top_comment_2_excerpt": top_comment_2_excerpt,
                "top_comment_2_upvotes": parse_float(top_comment_2.get("upvotes_at_snapshot")),
                "top_comment_2_replies": parse_float(top_comment_2.get("reply_count_at_snapshot")),
                "narrative_summary": narrative,
            }
        )

        markdown_lines.extend(
            [
                f"## {case_rank}. r/{subreddit}: {clean_text(forecast_row.get('title'))}",
                "",
                narrative,
                "",
                f"- Listing path: {listing_path or 'unknown'}",
                f"- Recent timeline: {recent_timeline or 'no timeline points'}",
                f"- Top comment 1: {top_comment_1_excerpt or 'n/a'}",
                f"- Top comment 2: {top_comment_2_excerpt or 'n/a'}",
                "",
            ]
        )

    output_path = Path(args.output)
    markdown_output_path = Path(args.markdown_output)
    write_csv(output_path, output_rows)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")

    print(f"Saved {len(output_rows)} row(s) to {output_path}")
    print(f"Saved case-study markdown to {markdown_output_path}")


if __name__ == "__main__":
    main()
