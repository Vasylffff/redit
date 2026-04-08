from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize mixed Reddit JSON snapshot output into posts, comments, and "
            "basic post-level feature tables."
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to the Reddit JSON output file saved from a run.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/reddit_json",
        help="Directory where normalized CSV files will be written.",
    )
    return parser.parse_args()


def load_items(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Input file is not valid JSON: {input_path}") from exc
    if not isinstance(payload, list):
        raise SystemExit("Reddit JSON output must be a JSON array of items.")
    return [item for item in payload if isinstance(item, dict)]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "reddit_json_output"


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def isoformat_or_empty(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    # Some bodies contain doubly escaped HTML entities from the actor output.
    for _ in range(2):
        text = html.unescape(text)
    return text.replace("\r\n", "\n").strip()


def author_name(value: Any) -> str:
    text = clean_text(value)
    return text or "[deleted]"


def extract_domain(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower()


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def build_post_rows(posts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, Any]] = {}

    for post in posts:
        post_created_at = parse_datetime(post.get("createdAt"))
        post_scraped_at = parse_datetime(post.get("scrapedAt"))
        row = {
            "post_id": clean_text(post.get("id")),
            "parsed_post_id": clean_text(post.get("parsedId")),
            "url": clean_text(post.get("url")),
            "link": clean_text(post.get("link")),
            "link_domain": extract_domain(clean_text(post.get("link")) or clean_text(post.get("url"))),
            "author": author_name(post.get("username")),
            "author_id": clean_text(post.get("userId")),
            "community_name": clean_text(post.get("communityName")),
            "subreddit": clean_text(post.get("parsedCommunityName")) or clean_text(post.get("category")),
            "title": clean_text(post.get("title")),
            "body": clean_text(post.get("body")),
            "flair": clean_text(post.get("flair")),
            "created_at": isoformat_or_empty(post_created_at),
            "scraped_at": isoformat_or_empty(post_scraped_at),
            "upvotes": post.get("upVotes"),
            "upvote_ratio": post.get("upVoteRatio"),
            "number_of_comments": post.get("numberOfComments"),
            "is_video": bool(post.get("isVideo", False)),
            "is_ad": bool(post.get("isAd", False)),
            "over_18": bool(post.get("over18", False)),
            "thumbnail_url": clean_text(post.get("thumbnailUrl")),
            "video_url": clean_text(post.get("videoUrl")),
            "image_urls": "|".join(post.get("imageUrls", []) or []),
            "image_count": len(post.get("imageUrls", []) or []),
            "title_length_chars": len(clean_text(post.get("title"))),
            "body_length_chars": len(clean_text(post.get("body"))),
        }
        rows.append(row)
        if row["post_id"]:
            indexed[row["post_id"]] = row

    return rows, indexed


def build_comment_rows(
    comments: list[dict[str, Any]],
    indexed_posts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared_comments: dict[str, dict[str, Any]] = {}

    for comment in comments:
        comment_id = clean_text(comment.get("id"))
        post_id = clean_text(comment.get("postId"))
        created_at = parse_datetime(comment.get("createdAt"))
        scraped_at = parse_datetime(comment.get("scrapedAt"))
        post_row = indexed_posts.get(post_id, {})
        post_created_at = parse_datetime(post_row.get("created_at")) if post_row.get("created_at") else None
        seconds_since_post = None
        if created_at and post_created_at:
            seconds_since_post = (created_at - post_created_at).total_seconds()

        prepared_comments[comment_id] = {
            "comment_id": comment_id,
            "parsed_comment_id": clean_text(comment.get("parsedId")),
            "post_id": post_id,
            "parent_id": clean_text(comment.get("parentId")),
            "url": clean_text(comment.get("url")),
            "author": author_name(comment.get("username")),
            "author_id": clean_text(comment.get("userId")),
            "community_name": clean_text(comment.get("communityName")),
            "subreddit": clean_text(comment.get("category")) or post_row.get("subreddit", ""),
            "body": clean_text(comment.get("body")),
            "created_at": isoformat_or_empty(created_at),
            "scraped_at": isoformat_or_empty(scraped_at),
            "upvotes": comment.get("upVotes"),
            "number_of_replies": comment.get("numberOfreplies"),
            "post_created_at": post_row.get("created_at", ""),
            "seconds_since_post": seconds_since_post,
            "is_top_level": clean_text(comment.get("parentId")) == post_id,
            "body_length_chars": len(clean_text(comment.get("body"))),
        }

    depth_cache: dict[str, int | None] = {}

    def compute_depth(comment_id: str) -> int | None:
        if comment_id in depth_cache:
            return depth_cache[comment_id]

        row = prepared_comments.get(comment_id)
        if row is None:
            depth_cache[comment_id] = None
            return None

        parent_id = row["parent_id"]
        if not parent_id:
            depth_cache[comment_id] = None
            return None
        if parent_id == row["post_id"]:
            depth_cache[comment_id] = 1
            return 1
        if parent_id not in prepared_comments:
            depth_cache[comment_id] = None
            return None

        parent_depth = compute_depth(parent_id)
        depth_cache[comment_id] = None if parent_depth is None else parent_depth + 1
        return depth_cache[comment_id]

    rows: list[dict[str, Any]] = []
    for comment_id, row in prepared_comments.items():
        row["comment_depth"] = compute_depth(comment_id)
        rows.append(row)
    return rows


def build_feature_rows(
    post_rows: list[dict[str, Any]],
    comment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comments_by_post: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comment in comment_rows:
        comments_by_post[comment["post_id"]].append(comment)

    features: list[dict[str, Any]] = []
    for post in post_rows:
        post_comments = comments_by_post.get(post["post_id"], [])
        comment_upvotes = [
            float(comment["upvotes"])
            for comment in post_comments
            if isinstance(comment.get("upvotes"), (int, float))
        ]
        comment_delays = [
            float(comment["seconds_since_post"])
            for comment in post_comments
            if isinstance(comment.get("seconds_since_post"), (int, float))
            and float(comment["seconds_since_post"]) >= 0
        ]
        unique_commenters = {
            comment["author"] for comment in post_comments if comment.get("author")
        }
        max_depth = max(
            (
                int(comment["comment_depth"])
                for comment in post_comments
                if isinstance(comment.get("comment_depth"), int)
            ),
            default=0,
        )
        post_created_at = parse_datetime(post.get("created_at"))
        post_scraped_at = parse_datetime(post.get("scraped_at"))
        hours_until_scrape = None
        if post_created_at and post_scraped_at:
            hours_until_scrape = (post_scraped_at - post_created_at).total_seconds() / 3600

        features.append(
            {
                "post_id": post["post_id"],
                "subreddit": post["subreddit"],
                "author": post["author"],
                "created_at": post["created_at"],
                "scraped_at": post["scraped_at"],
                "hours_until_scrape": hours_until_scrape,
                "title_length_chars": post["title_length_chars"],
                "body_length_chars": post["body_length_chars"],
                "upvotes": post["upvotes"],
                "upvote_ratio": post["upvote_ratio"],
                "number_of_comments": post["number_of_comments"],
                "scraped_comment_count": len(post_comments),
                "scraped_comment_coverage": (
                    len(post_comments) / post["number_of_comments"]
                    if isinstance(post.get("number_of_comments"), (int, float))
                    and post["number_of_comments"]
                    else None
                ),
                "top_level_comment_count": sum(1 for comment in post_comments if comment["is_top_level"]),
                "reply_comment_count": sum(1 for comment in post_comments if not comment["is_top_level"]),
                "author_reply_count": sum(
                    1 for comment in post_comments if comment["author"] == post["author"]
                ),
                "unique_commenter_count": len(unique_commenters),
                "avg_comment_upvotes": safe_mean(comment_upvotes),
                "median_comment_upvotes": safe_median(comment_upvotes),
                "max_comment_depth": max_depth,
                "first_comment_delay_seconds": min(comment_delays) if comment_delays else None,
                "first_hour_comment_count": sum(
                    1 for delay in comment_delays if delay <= 3600
                ),
                "first_day_comment_count": sum(
                    1 for delay in comment_delays if delay <= 86400
                ),
                "has_video": post["is_video"],
                "has_images": post["image_count"] > 0,
                "image_count": post["image_count"],
                "link_domain": post["link_domain"],
            }
        )

    return features


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
    destination: Path,
    input_file: str,
    post_rows: list[dict[str, Any]],
    comment_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_file": str(Path(input_file)),
        "post_count": len(post_rows),
        "comment_count": len(comment_rows),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    items = load_items(args.input_file)

    posts = [item for item in items if clean_text(item.get("dataType")) == "post"]
    comments = [item for item in items if clean_text(item.get("dataType")) == "comment"]

    post_rows, indexed_posts = build_post_rows(posts)
    comment_rows = build_comment_rows(comments, indexed_posts)
    feature_rows = build_feature_rows(post_rows, comment_rows)

    output_base = Path(args.output_dir) / slugify(Path(args.input_file).stem)
    posts_path = output_base / "posts.csv"
    comments_path = output_base / "comments.csv"
    features_path = output_base / "post_features.csv"
    metadata_path = output_base / "metadata.json"

    write_csv(post_rows, posts_path)
    write_csv(comment_rows, comments_path)
    write_csv(feature_rows, features_path)
    write_metadata(metadata_path, args.input_file, post_rows, comment_rows)

    print(f"Saved {len(post_rows)} post row(s) to {posts_path}")
    print(f"Saved {len(comment_rows)} comment row(s) to {comments_path}")
    print(f"Saved {len(feature_rows)} feature row(s) to {features_path}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
