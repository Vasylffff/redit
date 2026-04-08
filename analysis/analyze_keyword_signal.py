from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


TOKEN_RE = re.compile(r"[a-z][a-z0-9']+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "has",
    "had",
    "into",
    "about",
    "their",
    "they",
    "them",
    "would",
    "there",
    "were",
    "what",
    "when",
    "where",
    "which",
    "will",
    "your",
    "just",
    "than",
    "then",
    "because",
    "while",
    "through",
    "being",
    "after",
    "before",
    "under",
    "over",
    "still",
    "only",
    "more",
    "most",
    "some",
    "such",
    "many",
    "much",
    "very",
    "like",
    "also",
    "into",
    "onto",
    "between",
    "could",
    "should",
    "would",
    "been",
    "being",
    "than",
    "then",
    "here",
    "there",
    "these",
    "those",
    "isnt",
    "dont",
    "cant",
    "it's",
    "im",
    "youre",
    "weve",
    "ive",
    "our",
    "out",
    "off",
    "its",
    "why",
    "how",
    "who",
    "all",
    "any",
    "can",
    "not",
    "are",
    "was",
    "but",
    "you",
    "his",
    "her",
    "she",
    "him",
    "did",
    "get",
    "got",
    "one",
    "two",
    "three",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which post and comment keywords are associated with stronger or weaker "
            "Reddit outcomes in the current local dataset."
        )
    )
    parser.add_argument(
        "--post-input",
        default="data/models/reddit/prediction_next_hour.csv",
        help="Model-ready next-hour prediction CSV used for post keyword analysis.",
    )
    parser.add_argument(
        "--comment-input",
        default="data/history/reddit/comment_snapshots.csv",
        help="Comment snapshot CSV used for comment keyword analysis.",
    )
    parser.add_argument(
        "--status-input",
        default="data/history/reddit/latest_post_status.csv",
        help="Latest post status CSV used to join comment keywords to parent post quality.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis/reddit",
        help="Directory where keyword analysis outputs will be written.",
    )
    parser.add_argument(
        "--min-post-keyword-count",
        type=int,
        default=25,
        help="Minimum number of post rows a keyword must appear in.",
    )
    parser.add_argument(
        "--min-comment-keyword-count",
        type=int,
        default=20,
        help="Minimum number of comment rows a keyword must appear in.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many strongest and weakest keywords to include in the summary JSON.",
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


def parse_int(value: Any, default: int = 0) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def tokenize(text: str) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer((text or "").lower())]
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    ]


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_post_keyword_rows(
    rows: list[dict[str, str]],
    *,
    min_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global_upvote_values = [parse_float(row.get("upvote_delta_next_snapshot")) for row in rows]
    global_comment_values = [parse_float(row.get("comment_delta_next_snapshot")) for row in rows]
    global_upvote_velocity = [parse_float(row.get("upvote_velocity_per_hour")) for row in rows]
    global_comment_velocity = [parse_float(row.get("comment_velocity_per_hour")) for row in rows]

    global_baseline = {
        "avg_next_upvote_delta": safe_mean(global_upvote_values),
        "avg_next_comment_delta": safe_mean(global_comment_values),
        "avg_upvote_velocity": safe_mean(global_upvote_velocity),
        "avg_comment_velocity": safe_mean(global_comment_velocity),
    }
    strong_upvote_cutoff = global_baseline["avg_next_upvote_delta"]
    strong_comment_cutoff = global_baseline["avg_next_comment_delta"]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        text = f"{row.get('title', '')} {row.get('body', '')}"
        for token in unique_tokens(text):
            grouped[token].append(row)

    keyword_rows: list[dict[str, Any]] = []
    for token, token_rows in grouped.items():
        if len(token_rows) < min_count:
            continue
        upvote_values = [parse_float(row.get("upvote_delta_next_snapshot")) for row in token_rows]
        comment_values = [parse_float(row.get("comment_delta_next_snapshot")) for row in token_rows]
        upvote_velocity_values = [parse_float(row.get("upvote_velocity_per_hour")) for row in token_rows]
        comment_velocity_values = [parse_float(row.get("comment_velocity_per_hour")) for row in token_rows]

        avg_next_upvote_delta = safe_mean(upvote_values)
        avg_next_comment_delta = safe_mean(comment_values)
        avg_upvote_velocity = safe_mean(upvote_velocity_values)
        avg_comment_velocity = safe_mean(comment_velocity_values)

        keyword_rows.append(
            {
                "keyword": token,
                "row_count": len(token_rows),
                "avg_next_upvote_delta": avg_next_upvote_delta,
                "avg_next_comment_delta": avg_next_comment_delta,
                "avg_upvote_velocity": avg_upvote_velocity,
                "avg_comment_velocity": avg_comment_velocity,
                "upvote_lift_vs_global": (
                    avg_next_upvote_delta - global_baseline["avg_next_upvote_delta"]
                ),
                "comment_lift_vs_global": (
                    avg_next_comment_delta - global_baseline["avg_next_comment_delta"]
                ),
                "upvote_lift_ratio_vs_global": safe_ratio(
                    avg_next_upvote_delta,
                    global_baseline["avg_next_upvote_delta"],
                ),
                "comment_lift_ratio_vs_global": safe_ratio(
                    avg_next_comment_delta,
                    global_baseline["avg_next_comment_delta"],
                ),
                "strong_upvote_share": safe_ratio(
                    sum(1 for value in upvote_values if value > strong_upvote_cutoff),
                    len(upvote_values),
                ),
                "strong_comment_share": safe_ratio(
                    sum(1 for value in comment_values if value > strong_comment_cutoff),
                    len(comment_values),
                ),
            }
        )

    keyword_rows.sort(
        key=lambda row: (
            -(row.get("upvote_lift_vs_global") or 0.0),
            -(row.get("comment_lift_vs_global") or 0.0),
            -(row.get("row_count") or 0),
            row.get("keyword", ""),
        )
    )
    return keyword_rows, global_baseline


def build_comment_keyword_rows(
    comment_rows: list[dict[str, str]],
    status_rows: list[dict[str, str]],
    *,
    min_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status_by_key = {
        (row.get("subreddit", ""), row.get("post_id", "")): row
        for row in status_rows
    }

    parent_attention_values = []
    parent_popularity_values = []
    comment_upvote_values = []
    comment_reply_values = []
    alive_values = []
    eligible_comment_rows: list[dict[str, str]] = []

    for row in comment_rows:
        key = (row.get("subreddit", ""), row.get("post_id", ""))
        parent = status_by_key.get(key)
        body = row.get("body", "") or ""
        if parent is None or not body.strip() or body.strip() in {"[deleted]", "[removed]"}:
            continue
        eligible_comment_rows.append(row)
        parent_attention_values.append(parse_float(parent.get("current_attention_score")))
        parent_popularity_values.append(parse_float(parent.get("general_popularity_score")))
        comment_upvote_values.append(parse_float(row.get("upvotes_at_snapshot")))
        comment_reply_values.append(parse_float(row.get("reply_count_at_snapshot")))
        alive_values.append(1.0 if parent.get("latest_activity_state") in {"surging", "alive", "emerging"} else 0.0)

    global_baseline = {
        "avg_parent_current_attention": safe_mean(parent_attention_values),
        "avg_parent_general_popularity": safe_mean(parent_popularity_values),
        "avg_comment_upvotes": safe_mean(comment_upvote_values),
        "avg_comment_replies": safe_mean(comment_reply_values),
        "alive_parent_share": safe_mean(alive_values),
    }

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible_comment_rows:
        for token in unique_tokens(row.get("body", "")):
            grouped[token].append(row)

    keyword_rows: list[dict[str, Any]] = []
    for token, token_rows in grouped.items():
        if len(token_rows) < min_count:
            continue

        parent_attention = []
        parent_popularity = []
        comment_upvotes = []
        comment_replies = []
        alive_parent_flags = []

        for row in token_rows:
            parent = status_by_key[(row.get("subreddit", ""), row.get("post_id", ""))]
            parent_attention.append(parse_float(parent.get("current_attention_score")))
            parent_popularity.append(parse_float(parent.get("general_popularity_score")))
            comment_upvotes.append(parse_float(row.get("upvotes_at_snapshot")))
            comment_replies.append(parse_float(row.get("reply_count_at_snapshot")))
            alive_parent_flags.append(
                1.0 if parent.get("latest_activity_state") in {"surging", "alive", "emerging"} else 0.0
            )

        avg_parent_attention = safe_mean(parent_attention)
        avg_parent_popularity = safe_mean(parent_popularity)
        avg_comment_upvotes = safe_mean(comment_upvotes)
        avg_comment_replies = safe_mean(comment_replies)

        keyword_rows.append(
            {
                "keyword": token,
                "row_count": len(token_rows),
                "avg_parent_current_attention": avg_parent_attention,
                "avg_parent_general_popularity": avg_parent_popularity,
                "avg_comment_upvotes": avg_comment_upvotes,
                "avg_comment_replies": avg_comment_replies,
                "alive_parent_share": safe_mean(alive_parent_flags),
                "attention_lift_vs_global": (
                    avg_parent_attention - global_baseline["avg_parent_current_attention"]
                ),
                "popularity_lift_vs_global": (
                    avg_parent_popularity - global_baseline["avg_parent_general_popularity"]
                ),
                "comment_upvote_lift_vs_global": (
                    avg_comment_upvotes - global_baseline["avg_comment_upvotes"]
                ),
                "comment_reply_lift_vs_global": (
                    avg_comment_replies - global_baseline["avg_comment_replies"]
                ),
            }
        )

    keyword_rows.sort(
        key=lambda row: (
            -(row.get("attention_lift_vs_global") or 0.0),
            -(row.get("popularity_lift_vs_global") or 0.0),
            -(row.get("row_count") or 0),
            row.get("keyword", ""),
        )
    )
    return keyword_rows, global_baseline


def summarize_keywords(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: row.get(metric) or 0.0, reverse=True)
    weakest = sorted(rows, key=lambda row: row.get(metric) or 0.0)
    return {
        "best": ordered[:top_n],
        "worst": weakest[:top_n],
    }


def main() -> None:
    args = parse_args()

    post_rows = load_rows(Path(args.post_input))
    comment_rows = load_rows(Path(args.comment_input))
    status_rows = load_rows(Path(args.status_input))

    post_keyword_rows, post_baseline = build_post_keyword_rows(
        post_rows,
        min_count=args.min_post_keyword_count,
    )
    comment_keyword_rows, comment_baseline = build_comment_keyword_rows(
        comment_rows,
        status_rows,
        min_count=args.min_comment_keyword_count,
    )

    output_dir = Path(args.output_dir)
    post_output = output_dir / "keyword_signal_posts.csv"
    comment_output = output_dir / "keyword_signal_comments.csv"
    summary_output = output_dir / "keyword_signal_summary.json"

    write_csv(post_keyword_rows, post_output)
    write_csv(comment_keyword_rows, comment_output)

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "post_baseline": post_baseline,
        "comment_baseline": comment_baseline,
        "post_keywords_by_upvote_lift": summarize_keywords(
            post_keyword_rows,
            metric="upvote_lift_vs_global",
            top_n=args.top_n,
        ),
        "post_keywords_by_comment_lift": summarize_keywords(
            post_keyword_rows,
            metric="comment_lift_vs_global",
            top_n=args.top_n,
        ),
        "comment_keywords_by_attention_lift": summarize_keywords(
            comment_keyword_rows,
            metric="attention_lift_vs_global",
            top_n=args.top_n,
        ),
        "comment_keywords_by_popularity_lift": summarize_keywords(
            comment_keyword_rows,
            metric="popularity_lift_vs_global",
            top_n=args.top_n,
        ),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Post keyword rows written to {post_output}")
    print(f"Comment keyword rows written to {comment_output}")
    print(f"Keyword summary written to {summary_output}")
    print(f"Post keyword count: {len(post_keyword_rows)}")
    print(f"Comment keyword count: {len(comment_keyword_rows)}")


if __name__ == "__main__":
    main()
