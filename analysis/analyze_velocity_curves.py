"""
Upvote Velocity Curves Analysis
================================
Compare how fast posts gain upvotes across subreddits.
Track velocity over post age to find growth patterns.
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
    print("UPVOTE VELOCITY CURVES ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get all post snapshots with age and upvotes
    rows = cur.execute("""
        SELECT p.subreddit, p.post_id, p.age_minutes_at_snapshot, p.upvotes_at_snapshot, p.comment_count_at_snapshot,
               l.latest_activity_state, l.max_upvotes
        FROM post_snapshots p
        LEFT JOIN post_lifecycles l ON p.post_id = l.post_id
        WHERE p.age_minutes_at_snapshot IS NOT NULL AND p.upvotes_at_snapshot IS NOT NULL
        ORDER BY p.post_id, p.age_minutes_at_snapshot
    """).fetchall()

    print(f"  Loaded {len(rows)} snapshot observations")

    # Bucket by age (in hours)
    age_buckets = [0.5, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72]
    bucket_labels = ["0-30m", "30m-1h", "1-2h", "2-3h", "3-4h", "4-6h",
                     "6-8h", "8-12h", "12-18h", "18-24h", "24-36h", "36-48h", "48-72h"]

    def get_bucket(age_min):
        age_hrs = age_min / 60
        for i, b in enumerate(age_buckets):
            if age_hrs <= b:
                return i
        return len(age_buckets)

    # Aggregate by subreddit + age bucket
    sub_bucket = defaultdict(lambda: defaultdict(list))  # sub -> bucket_idx -> [upvotes]
    state_bucket = defaultdict(lambda: defaultdict(list))  # state -> bucket_idx -> [upvotes]

    # Track velocity per post (upvotes gained between snapshots)
    post_snapshots = defaultdict(list)
    post_meta = {}

    for sub, pid, age_min, upvotes, comments, state, max_up in rows:
        bucket = get_bucket(age_min)
        if bucket < len(bucket_labels):
            sub_bucket[sub][bucket].append(upvotes)
            if state:
                state_bucket[state][bucket].append(upvotes)

        post_snapshots[pid].append((age_min, upvotes, comments))
        if pid not in post_meta:
            post_meta[pid] = {"subreddit": sub, "state": state, "max_upvotes": max_up or 0}

    # Compute velocity curves per subreddit
    print(f"\n{'=' * 70}")
    print("UPVOTE VELOCITY BY SUBREDDIT (median upvotes at each age)")
    print(f"{'=' * 70}")
    print(f"{'Age Bucket':<12}", end="")
    subreddits = sorted(sub_bucket.keys())
    for sub in subreddits:
        print(f" {sub:>12}", end="")
    print()
    print("-" * (12 + 13 * len(subreddits)))

    curve_data = []
    for i, label in enumerate(bucket_labels):
        row = {"age_bucket": label}
        print(f"{label:<12}", end="")
        for sub in subreddits:
            vals = sub_bucket[sub].get(i, [])
            if vals:
                med = statistics.median(vals)
                row[f"{sub}_median_upvotes"] = round(med, 0)
                row[f"{sub}_count"] = len(vals)
                print(f" {med:>12.0f}", end="")
            else:
                row[f"{sub}_median_upvotes"] = ""
                row[f"{sub}_count"] = 0
                print(f" {'':>12}", end="")
        print()
        curve_data.append(row)

    # Velocity by lifecycle state
    print(f"\n{'=' * 70}")
    print("UPVOTE VELOCITY BY LIFECYCLE STATE (median upvotes at each age)")
    print(f"{'=' * 70}")
    state_order = ["surging", "alive", "emerging", "cooling", "dying", "dead"]
    print(f"{'Age Bucket':<12}", end="")
    for state in state_order:
        print(f" {state:>12}", end="")
    print()
    print("-" * (12 + 13 * len(state_order)))

    for i, label in enumerate(bucket_labels):
        print(f"{label:<12}", end="")
        for state in state_order:
            vals = state_bucket[state].get(i, [])
            if vals:
                med = statistics.median(vals)
                print(f" {med:>12.0f}", end="")
            else:
                print(f" {'':>12}", end="")
        print()

    # Compute per-post velocity (upvotes/hour between first and last snapshot)
    print(f"\n{'=' * 70}")
    print("AVERAGE VELOCITY BY SUBREDDIT (upvotes/hour)")
    print(f"{'=' * 70}")

    sub_velocities = defaultdict(list)
    state_velocities = defaultdict(list)

    for pid, snaps in post_snapshots.items():
        if len(snaps) < 2:
            continue
        snaps_sorted = sorted(snaps, key=lambda x: x[0])
        first_age, first_up, _ = snaps_sorted[0]
        last_age, last_up, _ = snaps_sorted[-1]
        hours = (last_age - first_age) / 60
        if hours < 0.1:
            continue
        vel = (last_up - first_up) / hours
        meta = post_meta[pid]
        sub_velocities[meta["subreddit"]].append(vel)
        if meta["state"]:
            state_velocities[meta["state"]].append(vel)

    print(f"{'Subreddit':<15} {'Posts':>6} {'Median vel':>12} {'Mean vel':>12} {'P90 vel':>12}")
    print("-" * 60)

    velocity_stats = []
    for sub in subreddits:
        vels = sub_velocities[sub]
        if not vels:
            continue
        vels_sorted = sorted(vels)
        med = statistics.median(vels)
        mean = statistics.mean(vels)
        p90 = vels_sorted[int(len(vels) * 0.9)]
        print(f"{sub:<15} {len(vels):>6} {med:>12.1f} {mean:>12.1f} {p90:>12.1f}")
        velocity_stats.append({
            "subreddit": sub,
            "post_count": len(vels),
            "median_velocity": round(med, 2),
            "mean_velocity": round(mean, 2),
            "p90_velocity": round(p90, 2),
        })

    # State velocity comparison
    print(f"\n{'State':<15} {'Posts':>6} {'Median vel':>12} {'Mean vel':>12}")
    print("-" * 50)
    for state in state_order:
        vels = state_velocities.get(state, [])
        if not vels:
            continue
        med = statistics.median(vels)
        mean = statistics.mean(vels)
        print(f"{state:<15} {len(vels):>6} {med:>12.1f} {mean:>12.1f}")

    # Early velocity as predictor
    print(f"\n{'=' * 70}")
    print("EARLY VELOCITY AS PREDICTOR")
    print("(Posts that gain X upvotes in first hour -> what state do they end up in?)")
    print(f"{'=' * 70}")

    early_vel_states = defaultdict(lambda: defaultdict(int))
    for pid, snaps in post_snapshots.items():
        snaps_sorted = sorted(snaps, key=lambda x: x[0])
        # Find upvotes at ~1 hour
        first_up = snaps_sorted[0][1]
        hour_up = None
        for age_min, up, _ in snaps_sorted:
            if age_min >= 30:
                hour_up = up
                break
        if hour_up is None:
            continue

        growth = hour_up - first_up
        state = post_meta[pid].get("state", "")
        if not state:
            continue

        if growth >= 500:
            bucket = "500+"
        elif growth >= 100:
            bucket = "100-500"
        elif growth >= 50:
            bucket = "50-100"
        elif growth >= 10:
            bucket = "10-50"
        else:
            bucket = "0-10"

        early_vel_states[bucket][state] += 1

    print(f"{'Early growth':<14} {'surging':>8} {'alive':>8} {'cooling':>8} {'dying':>8} {'dead':>8}")
    print("-" * 60)
    for bucket in ["500+", "100-500", "50-100", "10-50", "0-10"]:
        states = early_vel_states.get(bucket, {})
        total = sum(states.values())
        if total == 0:
            continue
        print(f"{bucket:<14}", end="")
        for s in ["surging", "alive", "cooling", "dying", "dead"]:
            pct = states.get(s, 0) / total if total else 0
            print(f" {pct:>7.0%}", end="")
        print(f"  (n={total})")

    # Save
    path1 = os.path.join(OUT_DIR, "velocity_curves.csv")
    if curve_data:
        with open(path1, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(curve_data[0].keys()))
            w.writeheader()
            w.writerows(curve_data)
        print(f"\n  Saved: {path1}")

    path2 = os.path.join(OUT_DIR, "velocity_by_subreddit.csv")
    if velocity_stats:
        with open(path2, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(velocity_stats[0].keys()))
            w.writeheader()
            w.writerows(velocity_stats)
        print(f"  Saved: {path2}")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
