"""
Cross-Subreddit Correlation Analysis
======================================
When one subreddit surges, do others follow?
Track same-story propagation across subreddits.
"""

import csv
import os
import re
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def normalize_title(title):
    """Simplify title for fuzzy matching"""
    t = title.lower().strip()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    # Remove common prefixes
    for prefix in ["breaking ", "update ", "megathread "]:
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t


def title_similarity(t1, t2):
    """Simple word overlap similarity"""
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return 0
    # Remove very common words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "and", "or", "but", "not", "with", "from", "by",
                 "as", "it", "its", "has", "have", "had", "that", "this", "be", "been"}
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    if not words1 or not words2:
        return 0
    overlap = words1 & words2
    return len(overlap) / min(len(words1), len(words2))


def main():
    print("=" * 70)
    print("CROSS-SUBREDDIT CORRELATION ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Activity correlation: when one sub surges, do others?
    print("\n[1/3] Subreddit activity correlation...")

    # Get hourly snapshot counts per subreddit
    snap_rows = cur.execute("""
        SELECT subreddit, snapshot_id,
               SUM(CASE WHEN activity_state IN ('surging','alive') THEN 1 ELSE 0 END) as active_count,
               COUNT(*) as total_count
        FROM post_snapshots
        WHERE activity_state IS NOT NULL AND activity_state != ''
        GROUP BY subreddit, snapshot_id
        ORDER BY snapshot_id
    """).fetchall()

    # Build time series per subreddit
    sub_series = defaultdict(dict)  # sub -> {snap_id: active_ratio}
    for sub, snap_id, active, total in snap_rows:
        if total > 0:
            sub_series[sub][snap_id] = active / total

    # Compute pairwise correlation
    subreddits = sorted(sub_series.keys())
    common_snaps = set.intersection(*[set(sub_series[s].keys()) for s in subreddits]) if subreddits else set()
    print(f"  {len(common_snaps)} common snapshot timepoints across {len(subreddits)} subreddits")

    print(f"\n  Activity Ratio Correlation Matrix:")
    print(f"  {'':>14}", end="")
    for s in subreddits:
        print(f" {s[:11]:>12}", end="")
    print()

    corr_data = []
    if len(common_snaps) >= 5:
        for s1 in subreddits:
            print(f"  {s1:>14}", end="")
            vals1 = [sub_series[s1][snap] for snap in sorted(common_snaps)]
            for s2 in subreddits:
                vals2 = [sub_series[s2][snap] for snap in sorted(common_snaps)]
                # Pearson correlation
                n = len(vals1)
                mean1 = sum(vals1) / n
                mean2 = sum(vals2) / n
                cov = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(vals1, vals2)) / n
                std1 = (sum((v - mean1) ** 2 for v in vals1) / n) ** 0.5
                std2 = (sum((v - mean2) ** 2 for v in vals2) / n) ** 0.5
                corr = cov / (std1 * std2) if std1 > 0 and std2 > 0 else 0
                print(f" {corr:>12.3f}", end="")
                if s1 != s2:
                    corr_data.append({
                        "subreddit_1": s1, "subreddit_2": s2,
                        "correlation": round(corr, 4), "datapoints": n
                    })
            print()
    else:
        print("  Not enough common snapshots for correlation")

    # 2. Same-story propagation
    print(f"\n[2/3] Same-story detection across subreddits...")

    post_rows = cur.execute("""
        SELECT post_id, subreddit, title, created_at, max_upvotes, latest_activity_state
        FROM post_lifecycles
        WHERE title IS NOT NULL AND title != ''
          AND created_at IS NOT NULL AND created_at != ''
    """).fetchall()

    # Group by subreddit, normalize titles
    posts_by_sub = defaultdict(list)
    for pid, sub, title, created_at, max_up, state in post_rows:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        posts_by_sub[sub].append({
            "post_id": pid, "subreddit": sub, "title": title,
            "norm_title": normalize_title(title),
            "created_at": dt, "max_upvotes": max_up or 0, "state": state or ""
        })

    # Find cross-posted stories (same topic in 2+ subreddits)
    cross_posts = []
    subs = list(posts_by_sub.keys())

    for i, s1 in enumerate(subs):
        for s2 in subs[i+1:]:
            for p1 in posts_by_sub[s1]:
                for p2 in posts_by_sub[s2]:
                    sim = title_similarity(p1["norm_title"], p2["norm_title"])
                    if sim >= 0.5:  # at least 50% word overlap
                        time_diff = abs((p1["created_at"] - p2["created_at"]).total_seconds()) / 3600
                        if time_diff <= 48:  # within 48 hours
                            cross_posts.append({
                                "title_1": p1["title"][:80],
                                "subreddit_1": s1,
                                "title_2": p2["title"][:80],
                                "subreddit_2": s2,
                                "similarity": round(sim, 2),
                                "time_diff_hours": round(time_diff, 1),
                                "first_posted": s1 if p1["created_at"] < p2["created_at"] else s2,
                                "upvotes_1": p1["max_upvotes"],
                                "upvotes_2": p2["max_upvotes"],
                                "state_1": p1["state"],
                                "state_2": p2["state"],
                            })

    # Deduplicate and sort by similarity
    seen = set()
    unique_cross = []
    for cp in sorted(cross_posts, key=lambda x: -x["similarity"]):
        key = tuple(sorted([cp["title_1"][:30], cp["title_2"][:30]]))
        if key not in seen:
            seen.add(key)
            unique_cross.append(cp)

    print(f"  Found {len(unique_cross)} cross-posted stories")

    if unique_cross:
        print(f"\n  Top cross-posted stories:")
        for cp in unique_cross[:15]:
            print(f"\n    [{cp['subreddit_1']}] {cp['title_1'][:60]}...")
            print(f"    [{cp['subreddit_2']}] {cp['title_2'][:60]}...")
            print(f"    Similarity: {cp['similarity']:.0%} | Time gap: {cp['time_diff_hours']:.1f}h | First in: {cp['first_posted']}")
            print(f"    Upvotes: {cp['upvotes_1']} vs {cp['upvotes_2']} | States: {cp['state_1']} vs {cp['state_2']}")

    # 3. Propagation patterns
    print(f"\n[3/3] Propagation patterns...")

    if unique_cross:
        # Which subreddit gets stories first?
        first_counts = defaultdict(int)
        for cp in unique_cross:
            first_counts[cp["first_posted"]] += 1

        print(f"\n  Which subreddit breaks stories first?")
        for sub, count in sorted(first_counts.items(), key=lambda x: -x[1]):
            print(f"    {sub}: {count} times first")

        # Average time lag
        time_lags = [cp["time_diff_hours"] for cp in unique_cross if cp["time_diff_hours"] > 0]
        if time_lags:
            print(f"\n  Average propagation time: {statistics.mean(time_lags):.1f} hours")
            print(f"  Median propagation time: {statistics.median(time_lags):.1f} hours")

        # Does first-to-post get more upvotes?
        first_wins = 0
        total_comps = 0
        for cp in unique_cross:
            if cp["first_posted"] == cp["subreddit_1"]:
                if cp["upvotes_1"] > cp["upvotes_2"]:
                    first_wins += 1
            else:
                if cp["upvotes_2"] > cp["upvotes_1"]:
                    first_wins += 1
            total_comps += 1

        if total_comps > 0:
            print(f"\n  First-to-post gets more upvotes: {first_wins}/{total_comps} ({first_wins/total_comps:.0%})")

    # Save results
    path1 = os.path.join(OUT_DIR, "cross_subreddit_correlation.csv")
    if corr_data:
        with open(path1, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(corr_data[0].keys()))
            w.writeheader()
            w.writerows(corr_data)
        print(f"\n  Saved: {path1}")

    path2 = os.path.join(OUT_DIR, "cross_posted_stories.csv")
    if unique_cross:
        with open(path2, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(unique_cross[0].keys()))
            w.writeheader()
            w.writerows(unique_cross)
        print(f"  Saved: {path2}")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
