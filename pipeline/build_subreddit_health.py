from __future__ import annotations

"""
build_subreddit_health.py

Reads subreddit_snapshots.csv and post_snapshots.csv and produces:

  subreddit_health_trend.csv   — one row per subreddit per snapshot with
                                 rolling health metrics and a health label
  subreddit_health_latest.csv  — one row per subreddit showing current state,
                                 trend direction, health score, and a forecast

Both files are picked up automatically by export_history_to_sqlite.py.

Health score (0–100) is built from four equally-weighted components:

  upvote_trend     — is average upvotes per post rising or falling?
  comment_trend    — is average comment count per post rising or falling?
  post_volume      — are new posts still arriving at a healthy rate?
  vitality         — what share of posts are NOT in a dead/unknown state?

Labels
  healthy   75–100
  declining 50–74
  critical  25–49
  dead       0–24
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TREND_WINDOW = 5          # snapshots used for slope calculation
MIN_SLOPE_SNAPSHOTS = 2   # need at least this many points for a slope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subreddit health trend and latest tables from history data.")
    parser.add_argument(
        "--history-dir",
        default="data/history/reddit",
        help="Directory containing post_snapshots.csv and subreddit_snapshots.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/history/reddit",
        help="Directory where subreddit_health_*.csv will be written.",
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=TREND_WINDOW,
        help=f"Number of recent snapshots used to compute trend slopes (default {TREND_WINDOW}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def linear_slope(values: list[float]) -> float | None:
    """Least-squares slope — no numpy needed."""
    n = len(values)
    if n < MIN_SLOPE_SNAPSHOTS:
        return None
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def round2(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


# ---------------------------------------------------------------------------
# Health scoring
# ---------------------------------------------------------------------------

def slope_score(slope: float | None, reference: float) -> float:
    """
    Convert a raw slope into a 0–25 component score.
    reference is a typical absolute magnitude for this metric so we can
    normalise — uses the average value across all snapshots as context.
    """
    if slope is None:
        return 12.5   # neutral when we have no data
    if reference <= 0:
        return 12.5
    normalised = slope / reference   # e.g. +0.1 means 10% per-snapshot improvement
    # clamp to [-1, 1] then map to [0, 25]
    clamped = max(-1.0, min(1.0, normalised * 5))
    return round((clamped + 1.0) / 2.0 * 25, 2)


def health_label(score: float) -> str:
    if score >= 75:
        return "healthy"
    if score >= 50:
        return "declining"
    if score >= 25:
        return "critical"
    return "dead"


def forecast(score: float, upvote_slope: float | None, comment_slope: float | None) -> str:
    both_positive = (upvote_slope or 0) > 0 and (comment_slope or 0) > 0
    both_negative = (upvote_slope or 0) < 0 and (comment_slope or 0) < 0

    if score < 25:
        return "at risk of dying"
    if score < 50:
        return "recovering" if both_positive else "at risk"
    if score < 75:
        return "stable" if not both_negative else "declining"
    return "growing" if both_positive else "stable"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_dead_share(post_rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    """
    Returns {(subreddit, snapshot_id): dead_share} where dead_share is the
    fraction of posts in that snapshot that are effectively low-vitality.
    `dead` counts fully; `dying` counts as a half-weight contribution.
    """
    totals: dict[tuple[str, str], int] = defaultdict(int)
    dead: dict[tuple[str, str], float] = defaultdict(float)
    for row in post_rows:
        key = (row.get("subreddit", ""), row.get("snapshot_id", ""))
        totals[key] += 1
        if row.get("activity_state", "") == "dead":
            dead[key] += 1.0
        elif row.get("activity_state", "") == "dying":
            dead[key] += 0.5
    return {
        key: dead[key] / total
        for key, total in totals.items()
        if total > 0
    }


def build_trend_rows(
    subreddit_rows: list[dict[str, str]],
    dead_shares: dict[tuple[str, str], float],
    trend_window: int,
) -> list[dict[str, Any]]:
    # Group by subreddit, sorted by snapshot time
    by_sub: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in subreddit_rows:
        by_sub[row.get("subreddit", "")].append(row)
    for rows in by_sub.values():
        rows.sort(key=lambda r: r.get("snapshot_time_utc", ""))

    trend_rows: list[dict[str, Any]] = []

    for subreddit, rows in sorted(by_sub.items()):
        for i, row in enumerate(rows):
            window = rows[max(0, i - trend_window + 1): i + 1]

            avg_upvotes_series   = [v for r in window if (v := to_float(r.get("average_upvotes"))) is not None]
            avg_comments_series  = [v for r in window if (v := to_float(r.get("average_comment_count"))) is not None]
            new_posts_series     = [v for r in window if (v := to_float(r.get("new_post_count_since_previous_snapshot"))) is not None]
            post_count_series    = [v for r in window if (v := to_float(r.get("post_count_in_snapshot"))) is not None]

            upvote_slope   = linear_slope(avg_upvotes_series)
            comment_slope  = linear_slope(avg_comments_series)
            new_post_slope = linear_slope(new_posts_series)

            ref_upvotes  = (sum(avg_upvotes_series)  / len(avg_upvotes_series))  if avg_upvotes_series  else 1.0
            ref_comments = (sum(avg_comments_series) / len(avg_comments_series)) if avg_comments_series else 1.0
            ref_new_post = (sum(new_posts_series)    / len(new_posts_series))    if new_posts_series    else 1.0

            snapshot_id = row.get("snapshot_id", "")
            dead_share = dead_shares.get((subreddit, snapshot_id))
            vitality = (1.0 - dead_share) * 25 if dead_share is not None else 12.5

            s_upvote  = slope_score(upvote_slope,   ref_upvotes)
            s_comment = slope_score(comment_slope,  ref_comments)
            s_volume  = slope_score(new_post_slope, ref_new_post)
            score     = round(s_upvote + s_comment + s_volume + vitality, 2)

            avg_new_posts = (sum(new_posts_series) / len(new_posts_series)) if new_posts_series else None
            avg_post_count = (sum(post_count_series) / len(post_count_series)) if post_count_series else None

            persisting = to_float(row.get("persisting_post_count_from_previous_snapshot"))
            prev_total = to_float(rows[i - 1].get("post_count_in_snapshot")) if i > 0 else None
            retention_rate = round2(persisting / prev_total) if persisting is not None and prev_total else None

            trend_rows.append({
                "snapshot_id":           snapshot_id,
                "snapshot_time_utc":     row.get("snapshot_time_utc", ""),
                "subreddit":             subreddit,
                "listing_type":          row.get("listing_type", "") or row.get("listing", ""),
                "post_count":            to_int(row.get("post_count_in_snapshot")),
                "new_post_count":        to_int(row.get("new_post_count_since_previous_snapshot")),
                "persisting_post_count": to_int(row.get("persisting_post_count_from_previous_snapshot")),
                "retention_rate":        retention_rate,
                "avg_upvotes":           round2(to_float(row.get("average_upvotes"))),
                "median_upvotes":        round2(to_float(row.get("median_upvotes"))),
                "avg_comments":          round2(to_float(row.get("average_comment_count"))),
                "median_comments":       round2(to_float(row.get("median_comment_count"))),
                "dead_post_share":       round2(dead_share),
                "upvote_slope":          round2(upvote_slope),
                "comment_slope":         round2(comment_slope),
                "new_post_slope":        round2(new_post_slope),
                "avg_upvotes_window":    round2(ref_upvotes)   if avg_upvotes_series  else None,
                "avg_comments_window":   round2(ref_comments)  if avg_comments_series else None,
                "avg_new_posts_window":  round2(avg_new_posts),
                "avg_post_count_window": round2(avg_post_count),
                "score_upvote_trend":    round2(s_upvote),
                "score_comment_trend":   round2(s_comment),
                "score_volume_trend":    round2(s_volume),
                "score_vitality":        round2(vitality),
                "health_score":          score,
                "health_label":          health_label(score),
                "trend_window_size":     len(window),
                "computed_at":           datetime.now(tz=timezone.utc).isoformat(),
            })

    trend_rows.sort(key=lambda r: (r["subreddit"], r["snapshot_time_utc"]))
    return trend_rows


def build_latest_rows(trend_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per subreddit — the most recent snapshot's health state + forecast."""
    latest: dict[str, dict[str, Any]] = {}
    for row in trend_rows:
        sub = row["subreddit"]
        if sub not in latest or row["snapshot_time_utc"] > latest[sub]["snapshot_time_utc"]:
            latest[sub] = row

    result = []
    for sub, row in sorted(latest.items()):
        upvote_slope  = row.get("upvote_slope")
        comment_slope = row.get("comment_slope")
        score         = row["health_score"]
        result.append({
            "subreddit":             sub,
            "health_score":          score,
            "health_label":          row["health_label"],
            "forecast":              forecast(score, upvote_slope, comment_slope),
            "upvote_slope":          upvote_slope,
            "comment_slope":         comment_slope,
            "new_post_slope":        row.get("new_post_slope"),
            "dead_post_share":       row.get("dead_post_share"),
            "retention_rate":        row.get("retention_rate"),
            "avg_upvotes":           row.get("avg_upvotes"),
            "avg_comments":          row.get("avg_comments"),
            "avg_new_posts_window":  row.get("avg_new_posts_window"),
            "latest_snapshot_time":  row["snapshot_time_utc"],
            "latest_snapshot_id":    row["snapshot_id"],
            "computed_at":           row["computed_at"],
        })
    return result


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir)
    output_dir  = Path(args.output_dir)

    print("Loading data ...")
    sub_rows   = load_csv(history_dir / "subreddit_snapshots.csv")
    post_rows  = load_csv(history_dir / "post_snapshots.csv")

    print(f"  {len(sub_rows)} subreddit snapshot rows")
    print(f"  {len(post_rows)} post snapshot rows")

    dead_shares  = build_dead_share(post_rows)
    trend_rows   = build_trend_rows(sub_rows, dead_shares, args.trend_window)
    latest_rows  = build_latest_rows(trend_rows)

    trend_path  = output_dir / "subreddit_health_trend.csv"
    latest_path = output_dir / "subreddit_health_latest.csv"
    write_csv(trend_rows,  trend_path)
    write_csv(latest_rows, latest_path)

    print(f"\nSaved {len(trend_rows)} trend rows  -> {trend_path}")
    print(f"Saved {len(latest_rows)} latest rows -> {latest_path}")
    print("\nCurrent subreddit health:")
    for row in latest_rows:
        print(
            f"  {row['subreddit']:<12} "
            f"score={row['health_score']:>6.1f}  "
            f"label={row['health_label']:<10}  "
            f"forecast={row['forecast']}"
        )


if __name__ == "__main__":
    main()
