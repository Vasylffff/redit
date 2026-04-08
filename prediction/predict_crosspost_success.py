"""
Cross-Posting Success Predictor
================================
If a story is in subreddit A, will it succeed in subreddit B?
Uses our 1305 detected cross-posts to build empirical rules.
"""

import csv
import os
import sqlite3
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def normalize_title(title):
    t = title.lower().strip()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def title_similarity(t1, t2):
    words1 = set(t1.split())
    words2 = set(t2.split())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "and", "or", "but", "not", "with", "from", "by",
                 "as", "it", "its", "has", "have", "had", "that", "this", "be", "been"}
    words1 -= stopwords
    words2 -= stopwords
    if not words1 or not words2:
        return 0
    overlap = words1 & words2
    return len(overlap) / min(len(words1), len(words2))


def main():
    print("=" * 70)
    print("CROSS-POSTING SUCCESS PREDICTOR")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # Load all posts
    rows = conn.execute("""
        SELECT post_id, subreddit, title, created_at, max_upvotes, max_comments,
               latest_activity_state, total_upvote_growth
        FROM post_lifecycles
        WHERE title IS NOT NULL AND title != ''
          AND created_at IS NOT NULL AND created_at != ''
          AND latest_activity_state IS NOT NULL
    """).fetchall()

    posts_by_sub = defaultdict(list)
    for pid, sub, title, created, max_up, max_com, state, growth in rows:
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        posts_by_sub[sub].append({
            "post_id": pid, "subreddit": sub, "title": title,
            "norm_title": normalize_title(title), "created_at": dt,
            "max_upvotes": max_up or 0, "max_comments": max_com or 0,
            "state": state, "growth": growth or 0,
        })

    print(f"  {sum(len(v) for v in posts_by_sub.values())} posts across {len(posts_by_sub)} subreddits")

    # Find all cross-posts
    cross_posts = []
    subs = list(posts_by_sub.keys())
    for i, s1 in enumerate(subs):
        for s2 in subs[i+1:]:
            for p1 in posts_by_sub[s1]:
                for p2 in posts_by_sub[s2]:
                    sim = title_similarity(p1["norm_title"], p2["norm_title"])
                    if sim >= 0.5:
                        time_diff = (p2["created_at"] - p1["created_at"]).total_seconds() / 3600
                        if abs(time_diff) <= 48:
                            first = p1 if p1["created_at"] <= p2["created_at"] else p2
                            second = p2 if p1["created_at"] <= p2["created_at"] else p1
                            cross_posts.append({
                                "first_sub": first["subreddit"],
                                "second_sub": second["subreddit"],
                                "first_upvotes": first["max_upvotes"],
                                "second_upvotes": second["max_upvotes"],
                                "first_state": first["state"],
                                "second_state": second["state"],
                                "time_lag_hours": abs(time_diff),
                                "similarity": sim,
                                "title": first["title"][:80],
                                "first_alive": first["state"] in ("surging", "alive"),
                                "second_alive": second["state"] in ("surging", "alive"),
                            })

    # Deduplicate
    seen = set()
    unique = []
    for cp in cross_posts:
        key = (cp["title"][:30], cp["first_sub"], cp["second_sub"])
        if key not in seen:
            seen.add(key)
            unique.append(cp)

    print(f"  {len(unique)} cross-posted stories found")

    # Build transfer matrix: if alive in sub A, what's chance of alive in sub B?
    print(f"\n{'=' * 70}")
    print("CROSS-POST TRANSFER MATRIX")
    print("(If a story is alive in row-subreddit, what % alive in column-subreddit?)")
    print(f"{'=' * 70}")

    transfer = defaultdict(lambda: defaultdict(lambda: {"total": 0, "alive": 0, "upvotes": []}))

    for cp in unique:
        # First -> Second direction
        transfer[cp["first_sub"]][cp["second_sub"]]["total"] += 1
        if cp["second_alive"]:
            transfer[cp["first_sub"]][cp["second_sub"]]["alive"] += 1
        transfer[cp["first_sub"]][cp["second_sub"]]["upvotes"].append(cp["second_upvotes"])

        # Also track: if first was alive, does second succeed?
        if cp["first_alive"]:
            key = f"{cp['first_sub']}_alive"
            transfer[key][cp["second_sub"]]["total"] += 1
            if cp["second_alive"]:
                transfer[key][cp["second_sub"]]["alive"] += 1

    all_subs = sorted(set(s for cp in unique for s in [cp["first_sub"], cp["second_sub"]]))
    print(f"\n  {'From / To':<20}", end="")
    for s in all_subs:
        print(f" {s[:10]:>11}", end="")
    print(f" {'Total':>8}")
    print(f"  {'-' * (20 + 12 * len(all_subs) + 8)}")

    for s1 in all_subs:
        print(f"  {s1:<20}", end="")
        row_total = 0
        for s2 in all_subs:
            d = transfer[s1][s2]
            if d["total"] > 0:
                rate = d["alive"] / d["total"]
                print(f" {rate:>10.0%}", end=" ")
                row_total += d["total"]
            else:
                print(f" {'':>11}", end="")
        print(f" {row_total:>8}")

    # Upvote ratio: when cross-posted, how much does the second post get?
    print(f"\n{'=' * 70}")
    print("UPVOTE TRANSFER RATIO")
    print("(Second post gets X% of first post's upvotes)")
    print(f"{'=' * 70}")

    for s1 in all_subs:
        for s2 in all_subs:
            if s1 == s2:
                continue
            pairs = [(cp["first_upvotes"], cp["second_upvotes"])
                     for cp in unique
                     if cp["first_sub"] == s1 and cp["second_sub"] == s2
                     and cp["first_upvotes"] > 0]
            if len(pairs) < 5:
                continue
            ratios = [s / f for f, s in pairs if f > 0]
            med_ratio = statistics.median(ratios)
            avg_second = statistics.mean([s for f, s in pairs])
            print(f"  {s1} -> {s2}: median {med_ratio:.1%} of original upvotes (n={len(pairs)}, avg second={avg_second:.0f})")

    # Time lag effect
    print(f"\n{'=' * 70}")
    print("TIME LAG EFFECT ON SUCCESS")
    print("(Does posting faster = better outcome?)")
    print(f"{'=' * 70}")

    lag_buckets = [(0, 1, "< 1h"), (1, 4, "1-4h"), (4, 12, "4-12h"), (12, 48, "12-48h")]
    print(f"  {'Lag':<12} {'Posts':>6} {'2nd Alive%':>12} {'Med 2nd Up':>12} {'Avg 2nd Up':>12}")
    print(f"  {'-' * 55}")

    for lo, hi, label in lag_buckets:
        bucket = [cp for cp in unique if lo <= cp["time_lag_hours"] < hi]
        if not bucket:
            continue
        alive_rate = sum(1 for cp in bucket if cp["second_alive"]) / len(bucket)
        med_up = statistics.median([cp["second_upvotes"] for cp in bucket])
        avg_up = statistics.mean([cp["second_upvotes"] for cp in bucket])
        print(f"  {label:<12} {len(bucket):>6} {alive_rate:>11.0%} {med_up:>12.0f} {avg_up:>12.0f}")

    # Prediction: given a story alive in sub A, predict success in sub B
    print(f"\n{'=' * 70}")
    print("CROSS-POST PREDICTIONS")
    print("(If a story is surging/alive in sub A, should you post it in sub B?)")
    print(f"{'=' * 70}")

    predictions = []
    for s1 in all_subs:
        alive_key = f"{s1}_alive"
        for s2 in all_subs:
            if s1 == s2:
                continue
            d = transfer[alive_key][s2]
            d_all = transfer[s1][s2]
            if d["total"] >= 3:
                rate = d["alive"] / d["total"]
                recommendation = "YES - post it" if rate > 0.3 else ("MAYBE" if rate > 0.15 else "NO - skip it")
                med_up = statistics.median(d_all["upvotes"]) if d_all["upvotes"] else 0
                print(f"  {s1} (alive) -> {s2}: {rate:.0%} success ({d['total']} examples) -> {recommendation} (median {med_up:.0f} up)")
                predictions.append({
                    "source_sub": s1,
                    "target_sub": s2,
                    "success_rate": round(rate, 4),
                    "sample_size": d["total"],
                    "recommendation": recommendation,
                    "median_upvotes": med_up,
                })

    # Save
    path = os.path.join(OUT_DIR, "crosspost_predictions.csv")
    if predictions:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
            w.writeheader()
            w.writerows(predictions)
        print(f"\n  Saved: {path}")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
