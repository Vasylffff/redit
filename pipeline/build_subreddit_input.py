from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VALID_LISTINGS = ("hot", "new", "rising", "top")


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
            "Build a Reddit listing config JSON file for scraping a subreddit "
            "listing such as r/Games/new."
        )
    )
    parser.add_argument(
        "subreddit",
        help="Subreddit name without the leading r/, for example: Games",
    )
    parser.add_argument(
        "--listing",
        choices=VALID_LISTINGS,
        default="new",
        help="Subreddit listing to target.",
    )
    parser.add_argument(
        "--top-time",
        choices=("hour", "day", "week", "month", "year", "all"),
        default="week",
        help="Time window for --listing top.",
    )
    parser.add_argument(
        "--max-items",
        type=positive_int,
        default=150,
        help="Maximum item count setting stored in the config.",
    )
    parser.add_argument(
        "--max-post-count",
        type=positive_int,
        default=150,
        help="Maximum post count setting stored in the config.",
    )
    parser.add_argument(
        "--max-comments",
        type=non_negative_int,
        default=0,
        help="How many comments to scrape per post. Use 0 for discovery-only runs.",
    )
    parser.add_argument(
        "--max-communities-count",
        type=positive_int,
        default=1,
        help="Maximum communities count setting stored in the config.",
    )
    parser.add_argument(
        "--max-user-count",
        type=positive_int,
        default=1,
        help="Maximum user count setting stored in the config.",
    )
    parser.add_argument(
        "--scroll-timeout",
        type=positive_int,
        default=40,
        help="Scroll timeout setting in seconds.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. If omitted, a file under configs/ is created.",
    )
    return parser.parse_args()


def normalize_subreddit_name(subreddit: str) -> str:
    normalized = subreddit.strip()
    normalized = re.sub(r"^/?r/", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip("/")
    if not normalized:
        raise SystemExit("Subreddit name must not be empty.")
    return normalized


def build_subreddit_url(subreddit: str, listing: str, top_time: str) -> str:
    base = f"https://www.reddit.com/r/{subreddit}/{listing}/"
    if listing == "top":
        return f"{base}?t={top_time}"
    return base


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    subreddit = normalize_subreddit_name(args.subreddit)
    start_url = build_subreddit_url(subreddit, args.listing, args.top_time)
    payload: dict[str, Any] = {
        "startUrls": [{"url": start_url}],
        "sort": args.listing,
        "maxItems": args.max_items,
        "maxPostCount": args.max_post_count,
        "maxComments": args.max_comments,
        "maxCommunitiesCount": args.max_communities_count,
        "maxUserCount": args.max_user_count,
        "scrollTimeout": args.scroll_timeout,
    }
    return payload, subreddit


def default_output_path(subreddit: str, listing: str) -> Path:
    return Path("configs") / f"reddit_r_{subreddit.lower()}_{listing}.json"


def main() -> None:
    args = parse_args()
    payload, subreddit = build_payload(args)
    output_path = Path(args.output) if args.output else default_output_path(subreddit, args.listing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote Reddit config JSON to {output_path}")
    print(f"Start URL: {payload['startUrls'][0]['url']}")


if __name__ == "__main__":
    main()
