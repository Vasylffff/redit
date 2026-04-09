"""
Best Posting Hours Analysis
============================
Which hours produce the most surging/alive posts?
Which hours produce posts that die fast?
Broken down by subreddit and overall.
"""

import csv
import os
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("BEST POSTING HOURS ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get posts with created_at and lifecycle data
    rows = cur.execute("""
        SELECT post_id, subreddit, created_at, latest_activity_state,
               max_upvotes, max_comments, total_upvote_growth,
               observed_hours, general_popularity_score, current_attention_score
        FROM post_lifecycles
        WHERE created_at IS NOT NULL AND created_at != ''
          AND latest_activity_state IS NOT NULL AND latest_activity_state != ''
    """).fetchall()

    print(f"  Loaded {len(rows)} posts with creation time + lifecycle state")

    # Parse hour from created_at
    hour_data = defaultdict(list)       # hour -> [post dicts]
    sub_hour_data = defaultdict(list)   # (subreddit, hour) -> [post dicts]

    for post_id, sub, created_at, state, max_up, max_com, up_growth, obs_hrs, pop_score, att_score in rows:
        try:
            # Parse ISO format
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            hour = dt.hour
        except (ValueError, AttributeError):
            continue

        post = {
            "post_id": post_id,
            "subreddit": sub,
            "hour": hour,
            "state": state,
            "max_upvotes": max_up or 0,
            "max_comments": max_com or 0,
            "upvote_growth": up_growth or 0,
            "observed_hours": obs_hrs or 0,
            "popularity": pop_score or 0,
            "attention": att_score or 0,
        }
        hour_data[hour].append(post)
        sub_hour_data[(sub, hour)].append(post)

    # Overall analysis by hour
    print(f"\n{'=' * 70}")
    print("OVERALL: BEST HOURS TO POST (UTC)")
    print(f"{'=' * 70}")
    print(f"{'Hour':>6} {'Posts':>6} {'Surge%':>8} {'Alive%':>8} {'Dead%':>8} {'Avg Up':>8} {'Avg Pop':>8}")
    print("-" * 70)

    hour_stats = []
    for hour in range(24):
        posts = hour_data.get(hour, [])
        if not posts:
            continue
        n = len(posts)
        surging = sum(1 for p in posts if p["state"] == "surging") / n
        alive = sum(1 for p in posts if p["state"] in ("surging", "alive")) / n
        dead = sum(1 for p in posts if p["state"] in ("dead", "dying")) / n
        avg_up = statistics.mean([p["max_upvotes"] for p in posts])
        avg_pop = statistics.mean([p["popularity"] for p in posts])

        hour_stats.append({
            "hour_utc": hour,
            "post_count": n,
            "surging_pct": round(surging, 4),
            "alive_pct": round(alive, 4),
            "dead_pct": round(dead, 4),
            "avg_max_upvotes": round(avg_up, 1),
            "avg_popularity": round(avg_pop, 2),
        })

        # Highlight best hours
        marker = ""
        if alive > 0.35:
            marker = " <-- GOOD"
        elif dead > 0.60:
            marker = " <-- AVOID"

        print(f"{hour:>4}:00 {n:>6} {surging:>7.1%} {alive:>7.1%} {dead:>7.1%} {avg_up:>8.0f} {avg_pop:>8.1f}{marker}")

    # Find best and worst hours
    if hour_stats:
        best = max(hour_stats, key=lambda x: x["alive_pct"])
        worst = max(hour_stats, key=lambda x: x["dead_pct"])
        print(f"\n  BEST hour:  {best['hour_utc']:02d}:00 UTC ({best['alive_pct']:.1%} alive/surging, avg {best['avg_max_upvotes']:.0f} upvotes)")
        print(f"  WORST hour: {worst['hour_utc']:02d}:00 UTC ({worst['dead_pct']:.1%} dead/dying)")

    # Per-subreddit breakdown
    print(f"\n{'=' * 70}")
    print("PER-SUBREDDIT: TOP 3 HOURS")
    print(f"{'=' * 70}")

    subreddits = sorted(set(p["subreddit"] for posts in hour_data.values() for p in posts))
    sub_hour_stats = []

    for sub in subreddits:
        sub_hours = []
        for hour in range(24):
            posts = sub_hour_data.get((sub, hour), [])
            if len(posts) < 3:
                continue
            n = len(posts)
            alive = sum(1 for p in posts if p["state"] in ("surging", "alive")) / n
            avg_up = statistics.mean([p["max_upvotes"] for p in posts])

            sub_hours.append({
                "subreddit": sub,
                "hour_utc": hour,
                "post_count": n,
                "alive_pct": round(alive, 4),
                "avg_max_upvotes": round(avg_up, 1),
            })
            sub_hour_stats.append(sub_hours[-1])

        if sub_hours:
            top3 = sorted(sub_hours, key=lambda x: -x["alive_pct"])[:3]
            print(f"\n  {sub}:")
            for h in top3:
                print(f"    {h['hour_utc']:02d}:00 UTC - {h['alive_pct']:.1%} alive/surging ({h['post_count']} posts, avg {h['avg_max_upvotes']:.0f} upvotes)")

    # Save CSVs
    path1 = os.path.join(OUT_DIR, "posting_hours_overall.csv")
    if hour_stats:
        with open(path1, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(hour_stats[0].keys()))
            w.writeheader()
            w.writerows(hour_stats)
        print(f"\n  Saved: {path1}")

    path2 = os.path.join(OUT_DIR, "posting_hours_by_subreddit.csv")
    if sub_hour_stats:
        with open(path2, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sub_hour_stats[0].keys()))
            w.writeheader()
            w.writerows(sub_hour_stats)
        print(f"  Saved: {path2}")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
