from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_subreddit_input import build_subreddit_url, normalize_subreddit_name


DISCOVERY_VARIANTS = (
    ("new", None, "new"),
    ("hot", None, "hot"),
    ("rising", None, "rising"),
    ("top", "day", "top_day"),
    ("top", "week", "top_week"),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-run Reddit discovery batch across subreddits and listing "
            "variants such as new, hot, rising, top-day, and top-week."
        )
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=["technology", "news", "worldnews", "politics"],
        help="Subreddits to include in the discovery batch.",
    )
    parser.add_argument(
        "--max-items",
        type=positive_int,
        default=100,
        help="Maximum item count value for each config.",
    )
    parser.add_argument(
        "--max-post-count",
        type=positive_int,
        default=100,
        help="Maximum post count value for each config.",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Apify maxComments value for each config. Use 0 for discovery mode.",
    )
    parser.add_argument(
        "--scroll-timeout",
        type=positive_int,
        default=40,
        help="Scroll timeout value in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/discovery_batch",
        help="Directory where the generated config JSON files will be written.",
    )
    parser.add_argument(
        "--manifest",
        default="configs/discovery_batch_manifest.csv",
        help="Manifest CSV listing the generated config files.",
    )
    return parser.parse_args()


def build_payload(
    subreddit: str,
    listing: str,
    top_time: str | None,
    *,
    max_items: int,
    max_post_count: int,
    max_comments: int,
    scroll_timeout: int,
) -> dict:
    url = build_subreddit_url(subreddit, listing, top_time or "week")
    return {
        "startUrls": [{"url": url}],
        "sort": listing,
        "maxItems": max_items,
        "maxPostCount": max_post_count,
        "maxComments": max_comments,
        "maxCommunitiesCount": 1,
        "maxUserCount": 1,
        "scrollTimeout": scroll_timeout,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []

    for subreddit_input in args.subreddits:
        subreddit = normalize_subreddit_name(subreddit_input)
        for listing, top_time, slug in DISCOVERY_VARIANTS:
            payload = build_payload(
                subreddit=subreddit,
                listing=listing,
                top_time=top_time,
                max_items=args.max_items,
                max_post_count=args.max_post_count,
                max_comments=args.max_comments,
                scroll_timeout=args.scroll_timeout,
            )
            filename = f"reddit_r_{subreddit}_{slug}.json"
            path = output_dir / filename
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            manifest_rows.append(
                {
                    "subreddit": subreddit,
                    "listing": listing,
                    "top_time": top_time or "",
                    "config_path": str(path),
                    "start_url": payload["startUrls"][0]["url"],
                    "max_items": args.max_items,
                    "max_post_count": args.max_post_count,
                    "max_comments": args.max_comments,
                }
            )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} config files to {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
