from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REDDIT_COMMENTS_URL_RE = re.compile(
    r"(?:https?://(?:www\.)?reddit\.com)?/r/(?P<subreddit>[^/]+)/comments/(?P<post_id>[A-Za-z0-9_]+)/",
    re.IGNORECASE,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be zero or a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Reddit tracking input JSON from the latest tracking candidate table so the "
            "same promising Reddit posts can be rechecked directly."
        )
    )
    parser.add_argument(
        "--source",
        default="data/history/reddit/tracking_candidates_latest.csv",
        help="Tracking candidate CSV produced by build_reddit_history.py.",
    )
    parser.add_argument(
        "--output",
        default="configs/reddit_track_candidates_latest.json",
        help="Where to write the tracking input JSON.",
    )
    parser.add_argument(
        "--manifest",
        default="data/tracking/reddit_track_candidates_latest_manifest.csv",
        help="Where to write the selected post manifest CSV.",
    )
    parser.add_argument(
        "--max-posts",
        type=positive_int,
        default=20,
        help="Maximum number of post URLs to include overall.",
    )
    parser.add_argument(
        "--per-subreddit-limit",
        type=positive_int,
        default=5,
        help="Maximum number of posts to include per subreddit.",
    )
    parser.add_argument(
        "--max-comments",
        type=non_negative_int,
        default=20,
        help="How many comments to request for each tracked post.",
    )
    parser.add_argument(
        "--max-items",
        type=positive_int,
        help=(
            "Optional explicit maxItems value. If omitted, it is derived from the "
            "selected post count and max-comments."
        ),
    )
    parser.add_argument(
        "--scroll-timeout",
        type=positive_int,
        default=40,
        help="Scroll timeout setting in seconds.",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_reddit_post_target(url: str) -> tuple[str, str]:
    text = clean_text(url)
    if not text:
        return "", ""
    match = REDDIT_COMMENTS_URL_RE.search(text)
    if not match:
        return "", ""
    return clean_text(match.group("subreddit")).lower(), clean_text(match.group("post_id"))


def normalize_post_id(value: str | None) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return text if text.lower().startswith("t3_") else f"t3_{text}"


def canonical_candidate_key(row: dict[str, Any]) -> tuple[str, str] | None:
    subreddit_from_url, post_id_from_url = parse_reddit_post_target(row.get("url") or "")
    subreddit = subreddit_from_url or clean_text(row.get("subreddit")).lower()
    post_id = normalize_post_id(row.get("post_id")) or normalize_post_id(post_id_from_url)
    if not subreddit or not post_id:
        return None
    return subreddit, post_id


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Tracking candidate CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Tracking candidate CSV is empty.")
    required = {"subreddit", "post_id", "url"}
    if not required.issubset(rows[0].keys()):
        joined = ", ".join(sorted(required))
        raise SystemExit(f"Tracking candidate CSV must contain: {joined}")
    return rows


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    priority_rank = {"highest": 0, "high": 1, "medium": 2, "low": 3, "very_low": 4}
    return (
        priority_rank.get((row.get("analysis_priority") or "").strip().lower(), 9),
        parse_int(row.get("focus_rank_in_subreddit")) or 999999,
        -(parse_float(row.get("last_comment_velocity_per_hour")) or 0.0),
        -(parse_float(row.get("last_upvote_velocity_per_hour")) or 0.0),
        -(parse_float(row.get("total_comment_growth")) or 0.0),
        -(parse_float(row.get("total_upvote_growth")) or 0.0),
    )


def select_rows(
    rows: list[dict[str, Any]],
    *,
    max_posts: int,
    per_subreddit_limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=candidate_sort_key)
    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for row in ordered:
        canonical_key = canonical_candidate_key(row)
        if canonical_key is None or canonical_key in seen_keys:
            continue
        subreddit, _post_id = canonical_key
        url = clean_text(row.get("url"))
        if not subreddit or not url:
            continue
        if counts[subreddit] >= per_subreddit_limit:
            continue
        seen_keys.add(canonical_key)
        selected.append(
            {
                **dict(row),
                "subreddit": subreddit,
                "post_id": normalize_post_id(row.get("post_id")) or _post_id,
            }
        )
        counts[subreddit] += 1
        if len(selected) >= max_posts:
            break
    if not selected:
        raise SystemExit("No tracking candidates were selected with the current limits.")
    return selected


def compute_max_items(selected_count: int, max_comments: int, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return max(selected_count, selected_count * (max_comments + 1))


def build_actor_input(
    rows: list[dict[str, Any]],
    *,
    max_comments: int,
    max_items: int,
    scroll_timeout: int,
) -> dict[str, Any]:
    return {
        "startUrls": [{"url": row["url"]} for row in rows],
        "maxItems": max_items,
        "maxPostCount": len(rows),
        "maxComments": max_comments,
        "maxCommunitiesCount": len({(row.get("subreddit") or "").strip().lower() for row in rows}) or 1,
        "maxUserCount": 1,
        "scrollTimeout": scroll_timeout,
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    rows = load_rows(source_path)
    selected = select_rows(
        rows,
        max_posts=args.max_posts,
        per_subreddit_limit=args.per_subreddit_limit,
    )
    max_items = compute_max_items(
        selected_count=len(selected),
        max_comments=args.max_comments,
        explicit=args.max_items,
    )
    payload = build_actor_input(
        selected,
        max_comments=args.max_comments,
        max_items=max_items,
        scroll_timeout=args.scroll_timeout,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_manifest(Path(args.manifest), selected)

    print(f"Selected {len(selected)} tracking candidate post(s) from {source_path}")
    print(f"Actor input written to {output_path}")
    print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
