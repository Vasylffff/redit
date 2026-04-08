from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

USER_AGENT = "REditCollector/1.0 (free public JSON path)"
BASE = "https://www.reddit.com"
REQUEST_DELAY = 1.1  # seconds between requests to stay inside rate limit
REQUEST_MAX_ATTEMPTS = 2
REQUEST_RETRY_BACKOFF_SECONDS = 1
MAX_CONSECUTIVE_TARGET_FETCH_FAILURES = 20
MAX_CONSECUTIVE_DNS_FAILURES = 3
REDDIT_POST_RE = re.compile(r"/r/(?P<subreddit>[^/]+)/comments/(?P<post_id>[A-Za-z0-9]+)/")


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
            "Collect recent Reddit submissions and comments using public .json "
            "endpoints with no API key."
        )
    )
    parser.add_argument(
        "subreddits",
        nargs="*",
        help="One or more subreddit names, for example: technology news worldnews",
    )
    parser.add_argument(
        "--post-limit",
        type=positive_int,
        default=100,
        help="How many recent submissions to fetch per subreddit (default 100).",
    )
    parser.add_argument(
        "--comment-limit-per-post",
        type=non_negative_int,
        default=20,
        help="How many comments to keep per submission (default 20, use 0 for none).",
    )
    parser.add_argument(
        "--sort",
        choices=["new", "hot", "top", "rising"],
        default="new",
        help="Listing sort to use when fetching posts (default: new).",
    )
    parser.add_argument(
        "--time-filter",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="day",
        help="Time filter for --sort top (default: day).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Base directory where collected data snapshots will be written.",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "reddit_json"],
        default="csv",
        help=(
            "csv writes submissions/comments CSVs into a timestamped folder. "
            "reddit_json writes mixed post/comment JSON files compatible with the "
            "existing Reddit history pipeline."
        ),
    )
    parser.add_argument(
        "--schedule-name",
        default="",
        help="Optional schedule name to store in metadata.",
    )
    parser.add_argument(
        "--cadence-label",
        default="",
        help="Optional cadence label to store in metadata.",
    )
    parser.add_argument(
        "--scheduled-hour",
        default="",
        help="Optional scheduled hour label to store in metadata.",
    )
    parser.add_argument(
        "--post-urls-file",
        default="",
        help=(
            "Optional text or CSV file of exact Reddit post URLs/permalinks to recheck. "
            "If provided, the collector tracks those posts directly instead of scanning subreddit listings."
        ),
    )
    parser.add_argument(
        "--post-url-column",
        default="url",
        help="Column name to read from --post-urls-file when the file is CSV (default: url).",
    )
    return parser.parse_args()


def _get(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> Any:
    last_error: requests.RequestException | None = None
    for attempt in range(1, REQUEST_MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, timeout=15)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"  Rate limited - sleeping {retry_after}s")
                time.sleep(retry_after)
                response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= REQUEST_MAX_ATTEMPTS:
                break
            backoff_seconds = REQUEST_RETRY_BACKOFF_SECONDS * attempt
            print(
                f"  Request attempt {attempt} failed for {url}; "
                f"retrying in {backoff_seconds}s ({exc})"
            )
            time.sleep(backoff_seconds)
    assert last_error is not None
    raise last_error


def is_dns_resolution_error(exc: requests.RequestException) -> bool:
    text = str(exc).lower()
    return (
        "failed to resolve" in text
        or "getaddrinfo failed" in text
        or "name or service not known" in text
        or "nameresolutionerror" in text
    )


def utc_iso(timestamp: float | None) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("&amp;", "&").strip()


def slugify(value: str) -> str:
    characters = [character if character.isalnum() else "_" for character in value]
    return "".join(characters).strip("_") or "reddit"


def load_post_targets(path: str, url_column: str) -> list[str]:
    input_path = Path(path)
    if not input_path.is_file():
        raise SystemExit(f"Post URL file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            values = []
            for row in reader:
                candidate = normalize_text(row.get(url_column))
                if candidate:
                    values.append(candidate)
            return values

    return [
        normalize_text(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if normalize_text(line)
    ]


def normalize_permalink(url_or_path: str) -> str:
    text = normalize_text(url_or_path)
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        return parsed.path.rstrip("/")
    return text.rstrip("/")


def parse_submission_target(url_or_path: str) -> tuple[str, str, str]:
    permalink = normalize_permalink(url_or_path)
    match = REDDIT_POST_RE.search(permalink)
    if not match:
        raise ValueError(f"Not a recognizable Reddit post permalink: {url_or_path}")
    subreddit = match.group("subreddit")
    post_id = match.group("post_id")
    canonical = f"/r/{subreddit}/comments/{post_id}"
    return subreddit, post_id, canonical


def fetch_posts(
    session: requests.Session,
    subreddit: str,
    sort: str,
    time_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    url = f"{BASE}/r/{subreddit}/{sort}.json"
    params: dict[str, Any] = {"limit": 100, "raw_json": 1}
    if sort == "top":
        params["t"] = time_filter

    posts: list[dict[str, Any]] = []
    after: str | None = None

    while len(posts) < limit:
        if after:
            params["after"] = after

        try:
            data = _get(session, url, params)
        except requests.RequestException as exc:
            print(f"  Request error fetching r/{subreddit}: {exc}")
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            if len(posts) >= limit:
                break
            payload = child.get("data")
            if isinstance(payload, dict):
                posts.append(payload)

        after = data.get("data", {}).get("after")
        if not after:
            break

    return posts


def flatten_comments(node: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return results
    kind = node.get("kind")
    data = node.get("data", {})

    if kind == "t1" and isinstance(data, dict):
        results.append(data)
        replies = data.get("replies")
        if isinstance(replies, dict):
            for child in replies.get("data", {}).get("children", []):
                results.extend(flatten_comments(child))
    elif kind == "Listing":
        for child in data.get("children", []):
            results.extend(flatten_comments(child))

    return results


def fetch_comments(
    session: requests.Session,
    submission_id: str,
    subreddit: str,
    limit: int,
) -> list[dict[str, Any]]:
    url = f"{BASE}/r/{subreddit}/comments/{submission_id}.json"
    try:
        data = _get(session, url, {"raw_json": 1, "limit": limit})
    except requests.RequestException as exc:
        print(f"  Request error fetching comments for {submission_id}: {exc}")
        return []

    if not isinstance(data, list) or len(data) < 2:
        return []

    flat = flatten_comments(data[1])
    return flat[:limit]


def fetch_submission_bundle(
    session: requests.Session,
    url_or_path: str,
    comment_limit: int,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]], str, bool]:
    subreddit, _, canonical = parse_submission_target(url_or_path)
    url = f"{BASE}{canonical}.json"
    params = {"raw_json": 1, "limit": max(1, comment_limit)}
    try:
        data = _get(session, url, params)
    except requests.RequestException as exc:
        print(f"  Request error fetching {canonical}: {exc}")
        return subreddit, None, [], canonical, is_dns_resolution_error(exc)
    if not isinstance(data, list) or len(data) < 1:
        return subreddit, None, [], canonical, False

    post_data: dict[str, Any] | None = None
    post_listing = data[0]
    if isinstance(post_listing, dict):
        children = post_listing.get("data", {}).get("children", [])
        if children and isinstance(children[0], dict):
            payload = children[0].get("data")
            if isinstance(payload, dict):
                post_data = payload

    comments: list[dict[str, Any]] = []
    if comment_limit > 0 and len(data) > 1:
        comments = flatten_comments(data[1])[:comment_limit]

    return subreddit, post_data, comments, canonical, False


def build_submission_row(post: dict[str, Any], subreddit: str) -> dict[str, Any]:
    permalink = normalize_text(post.get("permalink"))
    return {
        "submission_id": normalize_text(post.get("id")),
        "subreddit": subreddit,
        "title": normalize_text(post.get("title")),
        "author": normalize_text(post.get("author")) or "[deleted]",
        "created_utc": post.get("created_utc", ""),
        "created_at": utc_iso(post.get("created_utc")),
        "score": post.get("score", 0),
        "upvote_ratio": post.get("upvote_ratio", ""),
        "num_comments": post.get("num_comments", 0),
        "permalink": f"{BASE}{permalink}" if permalink else "",
        "url": normalize_text(post.get("url")),
        "is_self": bool(post.get("is_self", False)),
        "over_18": bool(post.get("over_18", False)),
        "spoiler": bool(post.get("spoiler", False)),
        "stickied": bool(post.get("stickied", False)),
        "locked": bool(post.get("locked", False)),
        "selftext": normalize_text(post.get("selftext")),
    }


def build_comment_row(
    comment: dict[str, Any],
    submission_id: str,
    subreddit: str,
) -> dict[str, Any]:
    permalink = normalize_text(comment.get("permalink"))
    return {
        "comment_id": normalize_text(comment.get("id")),
        "submission_id": submission_id,
        "subreddit": subreddit,
        "author": normalize_text(comment.get("author")) or "[deleted]",
        "created_utc": comment.get("created_utc", ""),
        "created_at": utc_iso(comment.get("created_utc")),
        "score": comment.get("score", 0),
        "parent_id": normalize_text(comment.get("parent_id")),
        "permalink": f"{BASE}{permalink}" if permalink else "",
        "body": normalize_text(comment.get("body")),
    }


def extract_preview_images(post: dict[str, Any]) -> list[str]:
    preview = post.get("preview")
    if not isinstance(preview, dict):
        return []
    images = preview.get("images")
    if not isinstance(images, list):
        return []
    results: list[str] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        source = image.get("source")
        if not isinstance(source, dict):
            continue
        url = normalize_text(source.get("url"))
        if url:
            results.append(url)
    return results


def extract_video_url(post: dict[str, Any]) -> str:
    media = post.get("media")
    if not isinstance(media, dict):
        return ""
    reddit_video = media.get("reddit_video")
    if not isinstance(reddit_video, dict):
        return ""
    return normalize_text(reddit_video.get("fallback_url"))


def extract_thumbnail_url(post: dict[str, Any]) -> str:
    value = normalize_text(post.get("thumbnail"))
    if value in {"", "self", "default", "nsfw", "spoiler", "image"}:
        return ""
    return value


def build_reddit_json_post_item(post: dict[str, Any], subreddit: str, scraped_at_iso: str) -> dict[str, Any]:
    parsed_id = normalize_text(post.get("id"))
    permalink = normalize_text(post.get("permalink"))
    full_permalink = f"{BASE}{permalink}" if permalink else ""
    url = normalize_text(post.get("url"))
    parsed_url = urlparse(url) if url else None
    link = ""
    if parsed_url and parsed_url.netloc.lower() not in {"www.reddit.com", "reddit.com", ""}:
        link = url

    return {
        "id": f"t3_{parsed_id}" if parsed_id else "",
        "parsedId": parsed_id,
        "url": full_permalink,
        "username": normalize_text(post.get("author")) or "[deleted]",
        "userId": "",
        "title": normalize_text(post.get("title")),
        "communityName": f"r/{subreddit}",
        "parsedCommunityName": subreddit,
        "body": normalize_text(post.get("selftext")),
        "html": normalize_text(post.get("selftext_html")),
        "link": link,
        "numberOfComments": post.get("num_comments", 0),
        "flair": normalize_text(post.get("link_flair_text")),
        "upVotes": post.get("score", 0),
        "upVoteRatio": post.get("upvote_ratio"),
        "isVideo": bool(post.get("is_video", False)),
        "isAd": bool(post.get("promoted", False)),
        "over18": bool(post.get("over_18", False)),
        "videoUrl": extract_video_url(post),
        "thumbnailUrl": extract_thumbnail_url(post),
        "imageUrls": extract_preview_images(post),
        "createdAt": utc_iso(post.get("created_utc")),
        "scrapedAt": scraped_at_iso,
        "dataType": "post",
    }


def build_reddit_json_comment_item(
    comment: dict[str, Any],
    submission_id: str,
    subreddit: str,
    scraped_at_iso: str,
) -> dict[str, Any]:
    parsed_id = normalize_text(comment.get("id"))
    permalink = normalize_text(comment.get("permalink"))
    full_permalink = f"{BASE}{permalink}" if permalink else ""
    return {
        "id": f"t1_{parsed_id}" if parsed_id else "",
        "parsedId": parsed_id,
        "url": full_permalink,
        "postId": f"t3_{submission_id}" if submission_id else "",
        "parentId": normalize_text(comment.get("parent_id")),
        "username": normalize_text(comment.get("author")) or "[deleted]",
        "userId": "",
        "category": subreddit,
        "communityName": f"r/{subreddit}",
        "body": normalize_text(comment.get("body")),
        "createdAt": utc_iso(comment.get("created_utc")),
        "scrapedAt": scraped_at_iso,
        "upVotes": comment.get("score", 0),
        "numberOfreplies": 0,
        "html": "",
        "dataType": "comment",
    }


def write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    destination: Path,
    subreddits: list[str],
    sort: str,
    post_limit: int,
    comment_limit_per_post: int,
    submission_count: int,
    comment_count: int,
) -> None:
    payload = {
        "collected_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": "reddit_public_json",
        "subreddits": subreddits,
        "sort": sort,
        "post_limit_per_subreddit": post_limit,
        "comment_limit_per_post": comment_limit_per_post,
        "submission_count": submission_count,
        "comment_count": comment_count,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_reddit_json_snapshot(
    *,
    output_dir: Path,
    snapshot_name: str,
    subreddit: str,
    sort: str,
    time_filter: str,
    schedule_name: str,
    cadence_label: str,
    scheduled_hour: str,
    items: list[dict[str, Any]],
    post_count: int,
    comment_count: int,
    saved_at_iso: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{snapshot_name}.json"
    metadata_path = output_dir / f"{snapshot_name}_metadata.json"
    output_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    metadata = {
        "source_type": "reddit_public_json",
        "subreddit": subreddit,
        "listing": sort,
        "top_time": time_filter if sort == "top" else "",
        "schedule_name": schedule_name,
        "cadence_label": cadence_label,
        "scheduled_hour": scheduled_hour,
        "saved_at": saved_at_iso,
        "item_count": len(items),
        "post_count": post_count,
        "comment_count": comment_count,
        "input_file": f"free_public_json::{subreddit}::{sort}",
        "output_file": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path


def main() -> None:
    args = parse_args()
    if not args.subreddits and not args.post_urls_file:
        raise SystemExit("Provide subreddit names or --post-urls-file.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    snapshot_name = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / snapshot_name

    all_submissions: list[dict[str, Any]] = []
    all_comments: list[dict[str, Any]] = []
    written_json_outputs: list[tuple[Path, Path, int, int]] = []

    if args.post_urls_file:
        post_targets = load_post_targets(args.post_urls_file, args.post_url_column)
        grouped_posts: dict[str, list[dict[str, Any]]] = {}
        grouped_comments: dict[str, list[dict[str, Any]]] = {}
        grouped_items: dict[str, list[dict[str, Any]]] = {}
        skipped_targets = 0
        consecutive_failed_targets = 0
        consecutive_dns_failures = 0
        dns_failure_detected = False

        print(f"Tracking {len(post_targets)} exact post URL(s) ...")
        for target in post_targets:
            try:
                subreddit, post, comments, canonical, had_dns_failure = fetch_submission_bundle(
                    session,
                    target,
                    comment_limit=args.comment_limit_per_post,
                )
            except ValueError as exc:
                print(f"  Skipping invalid target: {exc}")
                skipped_targets += 1
                consecutive_failed_targets += 1
                continue
            if post is None:
                print(f"  No post data returned for {canonical}")
                skipped_targets += 1
                consecutive_failed_targets += 1
                if had_dns_failure:
                    dns_failure_detected = True
                    consecutive_dns_failures += 1
                else:
                    consecutive_dns_failures = 0
                if consecutive_dns_failures >= MAX_CONSECUTIVE_DNS_FAILURES:
                    print(
                        "  Aborting remaining exact post tracking early after "
                        f"{MAX_CONSECUTIVE_DNS_FAILURES} consecutive DNS resolution failures."
                    )
                    break
                if consecutive_failed_targets >= MAX_CONSECUTIVE_TARGET_FETCH_FAILURES:
                    print(
                        "  Aborting remaining exact post tracking after "
                        f"{MAX_CONSECUTIVE_TARGET_FETCH_FAILURES} consecutive failures."
                    )
                    break
                continue
            consecutive_failed_targets = 0
            consecutive_dns_failures = 0

            scraped_at_iso = datetime.now(tz=timezone.utc).isoformat()
            grouped_posts.setdefault(subreddit, [])
            grouped_comments.setdefault(subreddit, [])
            grouped_items.setdefault(subreddit, [])

            submission_id = normalize_text(post.get("id"))
            submission_row = build_submission_row(post, subreddit)
            all_submissions.append(submission_row)
            grouped_posts[subreddit].append(submission_row)
            grouped_items[subreddit].append(build_reddit_json_post_item(post, subreddit, scraped_at_iso))

            for comment in comments:
                comment_row = build_comment_row(comment, submission_id, subreddit)
                all_comments.append(comment_row)
                grouped_comments[subreddit].append(comment_row)
                grouped_items[subreddit].append(
                    build_reddit_json_comment_item(
                        comment,
                        submission_id,
                        subreddit,
                        scraped_at_iso,
                    )
                )

        if args.output_format == "reddit_json":
            for subreddit, json_items in grouped_items.items():
                file_stem = f"{snapshot_name}_free_public_json_{slugify(subreddit)}_tracked"
                output_path, metadata_path = write_reddit_json_snapshot(
                    output_dir=output_dir,
                    snapshot_name=file_stem,
                    subreddit=subreddit,
                    sort="",
                    time_filter="",
                    schedule_name=args.schedule_name or "free_tracking",
                    cadence_label=args.cadence_label,
                    scheduled_hour=args.scheduled_hour,
                    items=json_items,
                    post_count=len(grouped_posts.get(subreddit, [])),
                    comment_count=len(grouped_comments.get(subreddit, [])),
                    saved_at_iso=datetime.now(tz=timezone.utc).isoformat(),
                )
                written_json_outputs.append(
                    (
                        output_path,
                        metadata_path,
                        len(grouped_posts.get(subreddit, [])),
                        len(grouped_comments.get(subreddit, [])),
                    )
                )
        if skipped_targets:
            print(f"Skipped {skipped_targets} exact post target(s) during tracking.")
        if dns_failure_detected:
            print("Exact post tracking warning: DNS resolution failed for one or more Reddit targets.")
    else:
        for subreddit in args.subreddits:
            print(f"Fetching r/{subreddit} ({args.sort}) ...")
            posts = fetch_posts(
                session,
                subreddit,
                sort=args.sort,
                time_filter=args.time_filter,
                limit=args.post_limit,
            )
            print(f"  {len(posts)} posts fetched")

            scraped_at_iso = datetime.now(tz=timezone.utc).isoformat()
            subreddit_submissions: list[dict[str, Any]] = []
            subreddit_comments: list[dict[str, Any]] = []
            json_items: list[dict[str, Any]] = []

            for post in posts:
                submission_id = normalize_text(post.get("id"))
                submission_row = build_submission_row(post, subreddit)
                all_submissions.append(submission_row)
                subreddit_submissions.append(submission_row)
                json_items.append(build_reddit_json_post_item(post, subreddit, scraped_at_iso))

                if args.comment_limit_per_post > 0 and post.get("num_comments", 0) > 0:
                    raw_comments = fetch_comments(
                        session,
                        submission_id,
                        subreddit,
                        limit=args.comment_limit_per_post,
                    )
                    for comment in raw_comments:
                        comment_row = build_comment_row(comment, submission_id, subreddit)
                        all_comments.append(comment_row)
                        subreddit_comments.append(comment_row)
                        json_items.append(
                            build_reddit_json_comment_item(
                                comment,
                                submission_id,
                                subreddit,
                                scraped_at_iso,
                            )
                        )

            if args.output_format == "reddit_json":
                file_stem = f"{snapshot_name}_free_public_json_{slugify(subreddit)}_{args.sort}"
                output_path, metadata_path = write_reddit_json_snapshot(
                    output_dir=output_dir,
                    snapshot_name=file_stem,
                    subreddit=subreddit,
                    sort=args.sort,
                    time_filter=args.time_filter,
                    schedule_name=args.schedule_name,
                    cadence_label=args.cadence_label,
                    scheduled_hour=args.scheduled_hour,
                    items=json_items,
                    post_count=len(subreddit_submissions),
                    comment_count=len(subreddit_comments),
                    saved_at_iso=scraped_at_iso,
                )
                written_json_outputs.append(
                    (
                        output_path,
                        metadata_path,
                        len(subreddit_submissions),
                        len(subreddit_comments),
                    )
                )

    if args.output_format == "csv":
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_csv(all_submissions, snapshot_dir / "submissions.csv")
        write_csv(all_comments, snapshot_dir / "comments.csv")
        write_metadata(
            destination=snapshot_dir / "metadata.json",
            subreddits=args.subreddits,
            sort=args.sort,
            post_limit=args.post_limit,
            comment_limit_per_post=args.comment_limit_per_post,
            submission_count=len(all_submissions),
            comment_count=len(all_comments),
        )

        print(f"\nSaved {len(all_submissions)} submissions -> {snapshot_dir / 'submissions.csv'}")
        print(f"Saved {len(all_comments)} comments      -> {snapshot_dir / 'comments.csv'}")
        print(f"Metadata -> {snapshot_dir / 'metadata.json'}")
        return

    print("")
    for output_path, metadata_path, submission_count, comment_count in written_json_outputs:
        print(f"Saved {submission_count} submissions -> {output_path}")
        print(f"Saved {comment_count} comments      -> {output_path}")
        print(f"Metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
