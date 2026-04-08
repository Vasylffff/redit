from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect recent Reddit submissions and comments for one or more subreddits "
            "and save them as CSV files."
        )
    )
    parser.add_argument(
        "subreddits",
        nargs="+",
        help="One or more subreddit names, for example: wallstreetbets stocks investing",
    )
    parser.add_argument(
        "--post-limit",
        type=positive_int,
        default=100,
        help="How many recent submissions to fetch per subreddit.",
    )
    parser.add_argument(
        "--comment-limit-per-post",
        type=positive_int,
        default=20,
        help="How many comments to keep per submission after flattening comment threads.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Base directory where timestamped data snapshots will be written.",
    )
    return parser.parse_args()


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'python-dotenv'. Run: py -m pip install -r requirements.txt"
        ) from exc

    load_dotenv()


def load_credentials() -> RedditCredentials:
    required = {
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID", "").strip(),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required environment variables: {joined}. "
            "Copy .env.example to .env and fill in your Reddit API credentials."
        )
    return RedditCredentials(
        client_id=required["REDDIT_CLIENT_ID"],
        client_secret=required["REDDIT_CLIENT_SECRET"],
        user_agent=required["REDDIT_USER_AGENT"],
    )


def create_reddit_client(credentials: RedditCredentials) -> Any:
    try:
        import praw
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'praw'. Run: py -m pip install -r requirements.txt"
        ) from exc

    reddit = praw.Reddit(
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        user_agent=credentials.user_agent,
        check_for_async=False,
    )
    reddit.read_only = True
    return reddit


def utc_iso(timestamp: float | None) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def author_name(author: Any) -> str:
    if author is None:
        return "[deleted]"
    return getattr(author, "name", "[deleted]")


def collect_submissions_and_comments(
    reddit: Any,
    subreddit_name: str,
    post_limit: int,
    comment_limit_per_post: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subreddit = reddit.subreddit(subreddit_name)
    submissions: list[dict[str, Any]] = []
    comments: list[dict[str, Any]] = []

    for submission in subreddit.new(limit=post_limit):
        submissions.append(
            {
                "submission_id": submission.id,
                "subreddit": subreddit_name,
                "title": normalize_text(submission.title),
                "author": author_name(submission.author),
                "created_utc": submission.created_utc,
                "created_at": utc_iso(submission.created_utc),
                "score": submission.score,
                "upvote_ratio": getattr(submission, "upvote_ratio", None),
                "num_comments": submission.num_comments,
                "permalink": f"https://www.reddit.com{submission.permalink}",
                "url": submission.url,
                "is_self": submission.is_self,
                "over_18": submission.over_18,
                "spoiler": submission.spoiler,
                "stickied": submission.stickied,
                "locked": submission.locked,
                "selftext": normalize_text(submission.selftext),
            }
        )

        submission.comments.replace_more(limit=0)
        flat_comments = submission.comments.list()[:comment_limit_per_post]
        for comment in flat_comments:
            comments.append(
                {
                    "comment_id": comment.id,
                    "submission_id": submission.id,
                    "subreddit": subreddit_name,
                    "author": author_name(comment.author),
                    "created_utc": comment.created_utc,
                    "created_at": utc_iso(comment.created_utc),
                    "score": comment.score,
                    "parent_id": comment.parent_id,
                    "permalink": f"https://www.reddit.com{comment.permalink}",
                    "body": normalize_text(comment.body),
                }
            )

    return submissions, comments


def write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    if not rows:
        destination.write_text("", encoding="utf-8")
        return

    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    destination: Path,
    subreddits: list[str],
    post_limit: int,
    comment_limit_per_post: int,
    submission_count: int,
    comment_count: int,
) -> None:
    payload = {
        "collected_at": datetime.now(tz=timezone.utc).isoformat(),
        "subreddits": subreddits,
        "post_limit_per_subreddit": post_limit,
        "comment_limit_per_post": comment_limit_per_post,
        "submission_count": submission_count,
        "comment_count": comment_count,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_environment()
    credentials = load_credentials()
    reddit = create_reddit_client(credentials)

    snapshot_name = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path(args.output_dir) / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    all_submissions: list[dict[str, Any]] = []
    all_comments: list[dict[str, Any]] = []

    for subreddit_name in args.subreddits:
        submissions, comments = collect_submissions_and_comments(
            reddit=reddit,
            subreddit_name=subreddit_name,
            post_limit=args.post_limit,
            comment_limit_per_post=args.comment_limit_per_post,
        )
        all_submissions.extend(submissions)
        all_comments.extend(comments)

    write_csv(all_submissions, snapshot_dir / "submissions.csv")
    write_csv(all_comments, snapshot_dir / "comments.csv")
    write_metadata(
        destination=snapshot_dir / "metadata.json",
        subreddits=args.subreddits,
        post_limit=args.post_limit,
        comment_limit_per_post=args.comment_limit_per_post,
        submission_count=len(all_submissions),
        comment_count=len(all_comments),
    )

    print(f"Saved {len(all_submissions)} submissions to {snapshot_dir / 'submissions.csv'}")
    print(f"Saved {len(all_comments)} comments to {snapshot_dir / 'comments.csv'}")
    print(f"Metadata written to {snapshot_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
