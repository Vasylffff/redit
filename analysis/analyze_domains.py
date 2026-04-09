"""
Link Domain Analysis
=====================
Which external domains get the most upvotes?
Which domains produce surging posts vs dead posts?
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
    print("LINK DOMAIN ANALYSIS")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT link_domain, subreddit, latest_activity_state, max_upvotes, max_comments,
               total_upvote_growth, observed_hours, title
        FROM post_lifecycles
        WHERE link_domain IS NOT NULL AND link_domain != ''
          AND latest_activity_state IS NOT NULL AND latest_activity_state != 'unknown'
    """).fetchall()

    print(f"  {len(rows)} posts with external links")

    domain_stats = defaultdict(lambda: {"posts": 0, "upvotes": [], "states": defaultdict(int), "subreddits": set()})

    for domain, sub, state, max_up, max_com, growth, hours, title in rows:
        d = domain_stats[domain]
        d["posts"] += 1
        d["upvotes"].append(max_up or 0)
        d["states"][state] += 1
        d["subreddits"].add(sub)

    # Top domains by post count
    print(f"\n{'=' * 70}")
    print("TOP DOMAINS BY POST COUNT")
    print(f"{'=' * 70}")
    print(f"  {'Domain':<35} {'Posts':>6} {'Med Up':>8} {'Avg Up':>8} {'Surge%':>8} {'Dead%':>8}")
    print(f"  {'-' * 80}")

    results = []
    for domain, d in sorted(domain_stats.items(), key=lambda x: -x[1]["posts"]):
        if d["posts"] < 5:
            continue
        med_up = statistics.median(d["upvotes"])
        avg_up = statistics.mean(d["upvotes"])
        total = d["posts"]
        surge_pct = (d["states"].get("surging", 0) + d["states"].get("alive", 0)) / total
        dead_pct = (d["states"].get("dead", 0) + d["states"].get("dying", 0)) / total

        results.append({
            "domain": domain,
            "post_count": total,
            "median_upvotes": round(med_up),
            "avg_upvotes": round(avg_up),
            "alive_rate": round(surge_pct, 4),
            "dead_rate": round(dead_pct, 4),
            "subreddits": len(d["subreddits"]),
        })

    for r in sorted(results, key=lambda x: -x["post_count"])[:30]:
        print(f"  {r['domain']:<35} {r['post_count']:>6} {r['median_upvotes']:>8} {r['avg_upvotes']:>8} {r['alive_rate']:>7.0%} {r['dead_rate']:>7.0%}")

    # Top domains by upvotes
    print(f"\n{'=' * 70}")
    print("TOP DOMAINS BY MEDIAN UPVOTES (min 5 posts)")
    print(f"{'=' * 70}")
    for r in sorted(results, key=lambda x: -x["median_upvotes"])[:20]:
        print(f"  {r['domain']:<35} med={r['median_upvotes']:>8}  avg={r['avg_upvotes']:>8}  posts={r['post_count']}")

    # Best survival rate domains
    print(f"\n{'=' * 70}")
    print("BEST SURVIVAL RATE DOMAINS (min 5 posts)")
    print(f"{'=' * 70}")
    for r in sorted(results, key=lambda x: -x["alive_rate"])[:15]:
        print(f"  {r['domain']:<35} alive={r['alive_rate']:.0%}  dead={r['dead_rate']:.0%}  posts={r['post_count']}  med_up={r['median_upvotes']}")

    # Worst domains (highest death rate)
    print(f"\n{'=' * 70}")
    print("WORST SURVIVAL RATE DOMAINS (min 5 posts)")
    print(f"{'=' * 70}")
    for r in sorted(results, key=lambda x: -x["dead_rate"])[:15]:
        print(f"  {r['domain']:<35} dead={r['dead_rate']:.0%}  alive={r['alive_rate']:.0%}  posts={r['post_count']}  med_up={r['median_upvotes']}")

    # Save
    path = os.path.join(OUT_DIR, "domain_analysis.csv")
    if results:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(sorted(results, key=lambda x: -x["post_count"]))
        print(f"\n  Saved: {path} ({len(results)} domains)")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
