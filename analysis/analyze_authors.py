"""
Author Analysis
================
Do certain posters consistently produce viral content?
Author success rates, posting patterns, and influence.
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
    print("AUTHOR ANALYSIS")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT author, subreddit, latest_activity_state, max_upvotes, max_comments,
               total_upvote_growth, observed_hours, title
        FROM post_lifecycles
        WHERE author IS NOT NULL AND author != '' AND author != '[deleted]'
          AND latest_activity_state IS NOT NULL AND latest_activity_state != 'unknown'
    """).fetchall()

    print(f"  {len(rows)} posts with known authors")

    author_stats = defaultdict(lambda: {
        "posts": 0, "upvotes": [], "states": defaultdict(int),
        "subreddits": set(), "titles": []
    })

    for author, sub, state, max_up, max_com, growth, hours, title in rows:
        d = author_stats[author]
        d["posts"] += 1
        d["upvotes"].append(max_up or 0)
        d["states"][state] += 1
        d["subreddits"].add(sub)
        d["titles"].append(title or "")

    unique_authors = len(author_stats)
    multi_posters = sum(1 for d in author_stats.values() if d["posts"] >= 3)
    print(f"  {unique_authors} unique authors")
    print(f"  {multi_posters} authors with 3+ posts")

    # Top authors by post count
    print(f"\n{'=' * 70}")
    print("MOST PROLIFIC AUTHORS")
    print(f"{'=' * 70}")
    print(f"  {'Author':<25} {'Posts':>6} {'Med Up':>8} {'Avg Up':>8} {'Surge%':>8} {'Subs':>5}")
    print(f"  {'-' * 65}")

    results = []
    for author, d in author_stats.items():
        if d["posts"] < 3:
            continue
        med_up = statistics.median(d["upvotes"])
        avg_up = statistics.mean(d["upvotes"])
        total = d["posts"]
        surge_pct = (d["states"].get("surging", 0) + d["states"].get("alive", 0)) / total
        dead_pct = (d["states"].get("dead", 0) + d["states"].get("dying", 0)) / total

        results.append({
            "author": author,
            "post_count": total,
            "median_upvotes": round(med_up),
            "avg_upvotes": round(avg_up),
            "total_upvotes": sum(d["upvotes"]),
            "alive_rate": round(surge_pct, 4),
            "dead_rate": round(dead_pct, 4),
            "subreddit_count": len(d["subreddits"]),
            "subreddits": ", ".join(sorted(d["subreddits"])),
        })

    for r in sorted(results, key=lambda x: -x["post_count"])[:20]:
        print(f"  {r['author']:<25} {r['post_count']:>6} {r['median_upvotes']:>8} {r['avg_upvotes']:>8} {r['alive_rate']:>7.0%} {r['subreddit_count']:>5}")

    # Top by influence (total upvotes)
    print(f"\n{'=' * 70}")
    print("MOST INFLUENTIAL AUTHORS (by total upvotes, min 3 posts)")
    print(f"{'=' * 70}")
    for r in sorted(results, key=lambda x: -x["total_upvotes"])[:20]:
        print(f"  {r['author']:<25} total={r['total_upvotes']:>10,}  posts={r['post_count']:>4}  avg={r['avg_upvotes']:>8}  alive={r['alive_rate']:.0%}")

    # Best success rate (min 5 posts)
    print(f"\n{'=' * 70}")
    print("HIGHEST SUCCESS RATE AUTHORS (min 5 posts)")
    print(f"{'=' * 70}")
    high_volume = [r for r in results if r["post_count"] >= 5]
    for r in sorted(high_volume, key=lambda x: -x["alive_rate"])[:15]:
        print(f"  {r['author']:<25} alive={r['alive_rate']:.0%}  posts={r['post_count']}  avg_up={r['avg_upvotes']}")

    # Worst authors
    print(f"\n{'=' * 70}")
    print("LOWEST SUCCESS RATE AUTHORS (min 5 posts)")
    print(f"{'=' * 70}")
    for r in sorted(high_volume, key=lambda x: x["alive_rate"])[:15]:
        print(f"  {r['author']:<25} alive={r['alive_rate']:.0%}  dead={r['dead_rate']:.0%}  posts={r['post_count']}  avg_up={r['avg_upvotes']}")

    # Cross-subreddit posters
    print(f"\n{'=' * 70}")
    print("CROSS-SUBREDDIT POSTERS (post in 3+ subreddits)")
    print(f"{'=' * 70}")
    cross = [r for r in results if r["subreddit_count"] >= 3]
    for r in sorted(cross, key=lambda x: -x["post_count"])[:15]:
        print(f"  {r['author']:<25} subs={r['subreddit_count']}  posts={r['post_count']}  alive={r['alive_rate']:.0%}  [{r['subreddits']}]")

    # Author consistency: do prolific authors have consistent success?
    print(f"\n{'=' * 70}")
    print("AUTHOR CONSISTENCY (min 10 posts)")
    print(f"{'=' * 70}")
    prolific = [r for r in results if r["post_count"] >= 10]
    if prolific:
        avg_alive = statistics.mean([r["alive_rate"] for r in prolific])
        print(f"  Avg alive rate for prolific authors: {avg_alive:.0%}")
        consistent = sum(1 for r in prolific if r["alive_rate"] > 0.3)
        print(f"  Authors with >30% alive rate: {consistent}/{len(prolific)}")
        overall_alive = sum(1 for r in results if r["alive_rate"] > 0) / len(results) if results else 0
        print(f"  For comparison, overall avg alive rate: {statistics.mean([r['alive_rate'] for r in results]):.0%}")

    # Save
    path = os.path.join(OUT_DIR, "author_analysis.csv")
    if results:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(sorted(results, key=lambda x: -x["total_upvotes"]))
        print(f"\n  Saved: {path} ({len(results)} authors)")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
