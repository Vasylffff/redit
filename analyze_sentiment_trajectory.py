"""
Sentiment Trajectory Analysis
==============================
Track how sentiment CHANGES over a post's life.
- Does sentiment flip from positive to negative (or vice versa)?
- Does a sentiment flip predict post death?
- What's the typical sentiment arc for surging vs dying posts?
"""

import csv
import os
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except ImportError:
    print("ERROR: pip install vaderSentiment")
    raise

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_comment_timeseries(conn):
    """Load comments grouped by (post_id, snapshot_id) with timestamps"""
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT c.post_id, c.snapshot_id, c.snapshot_time_utc, c.body,
               c.upvotes_at_snapshot, c.age_minutes_at_snapshot
        FROM comment_snapshots c
        WHERE c.body IS NOT NULL
          AND c.body != ''
          AND c.body != '[deleted]'
          AND c.body != '[removed]'
        ORDER BY c.post_id, c.snapshot_time_utc
    """).fetchall()

    # Group by post_id -> snapshot_id
    post_snaps = defaultdict(lambda: defaultdict(list))
    snap_times = {}  # snapshot_id -> time string

    for pid, snap_id, snap_time, body, upvotes, age_min in rows:
        score = _VADER.polarity_scores(body)["compound"]
        weight = max(1.0, upvotes) if upvotes else 1.0
        post_snaps[pid][snap_id].append({
            "score": score,
            "weight": weight,
            "age_min": age_min or 0,
        })
        snap_times[snap_id] = snap_time

    print(f"  Loaded comments for {len(post_snaps)} posts across {len(snap_times)} snapshots")
    return post_snaps, snap_times


def compute_snapshot_sentiment(comments):
    """Aggregate sentiment for a list of comments in one snapshot"""
    if not comments:
        return None
    scores = [c["score"] for c in comments]
    weights = [c["weight"] for c in comments]
    total_w = sum(weights)

    return {
        "count": len(comments),
        "mean": statistics.mean(scores),
        "weighted": sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0,
        "pos_share": sum(1 for s in scores if s > 0.05) / len(scores),
        "neg_share": sum(1 for s in scores if s < -0.05) / len(scores),
        "variance": statistics.variance(scores) if len(scores) > 1 else 0,
    }


def build_trajectories(post_snaps, snap_times, lifecycles):
    """Build sentiment trajectory for each post (ordered by snapshot time)"""
    trajectories = []

    for pid, snap_dict in post_snaps.items():
        if len(snap_dict) < 2:
            continue

        # Sort snapshots by time
        sorted_snaps = sorted(snap_dict.items(),
                              key=lambda x: snap_times.get(x[0], ""))

        timeline = []
        for snap_id, comments in sorted_snaps:
            agg = compute_snapshot_sentiment(comments)
            if agg:
                timeline.append({
                    "snapshot_id": snap_id,
                    "time": snap_times.get(snap_id, ""),
                    **agg,
                })

        if len(timeline) < 2:
            continue

        # Compute trajectory features
        first = timeline[0]
        last = timeline[-1]
        mid_idx = len(timeline) // 2
        mid = timeline[mid_idx]

        sentiment_change = last["weighted"] - first["weighted"]
        sentiment_slope = sentiment_change / len(timeline) if len(timeline) > 1 else 0
        variance_change = last["variance"] - first["variance"]
        neg_share_change = last["neg_share"] - first["neg_share"]

        # Detect flips
        started_positive = first["weighted"] > 0.05
        started_negative = first["weighted"] < -0.05
        ended_positive = last["weighted"] > 0.05
        ended_negative = last["weighted"] < -0.05

        flip = "none"
        if started_positive and ended_negative:
            flip = "pos_to_neg"
        elif started_negative and ended_positive:
            flip = "neg_to_pos"
        elif started_positive and ended_positive:
            flip = "stayed_positive"
        elif started_negative and ended_negative:
            flip = "stayed_negative"
        else:
            flip = "neutral"

        # Peak and trough
        all_weighted = [t["weighted"] for t in timeline]
        peak_sent = max(all_weighted)
        trough_sent = min(all_weighted)
        volatility = peak_sent - trough_sent

        lc = lifecycles.get(pid, {})

        traj = {
            "post_id": pid,
            "subreddit": lc.get("subreddit", ""),
            "state": lc.get("state", ""),
            "snapshot_count": len(timeline),
            "first_sentiment": round(first["weighted"], 4),
            "mid_sentiment": round(mid["weighted"], 4),
            "last_sentiment": round(last["weighted"], 4),
            "sentiment_change": round(sentiment_change, 4),
            "sentiment_slope": round(sentiment_slope, 4),
            "variance_change": round(variance_change, 4),
            "neg_share_change": round(neg_share_change, 4),
            "flip_type": flip,
            "peak_sentiment": round(peak_sent, 4),
            "trough_sentiment": round(trough_sent, 4),
            "volatility": round(volatility, 4),
            "first_comment_count": first["count"],
            "last_comment_count": last["count"],
            "max_upvotes": lc.get("max_upvotes", 0),
        }
        trajectories.append(traj)

    return trajectories


def analyze_trajectories(trajectories):
    """Analyze sentiment trajectory patterns"""
    print(f"\n{'=' * 70}")
    print("SENTIMENT TRAJECTORY PATTERNS")
    print(f"{'=' * 70}")
    print(f"  Posts with multi-snapshot sentiment: {len(trajectories)}")

    # Group by flip type
    flip_counts = defaultdict(int)
    flip_states = defaultdict(lambda: defaultdict(int))
    for t in trajectories:
        flip_counts[t["flip_type"]] += 1
        if t["state"]:
            flip_states[t["flip_type"]][t["state"]] += 1

    print(f"\n  Sentiment flip distribution:")
    for flip, count in sorted(flip_counts.items(), key=lambda x: -x[1]):
        pct = count / len(trajectories)
        print(f"    {flip:<20} {count:>5} ({pct:.1%})")

    # Key question: does flipping predict death?
    print(f"\n{'=' * 70}")
    print("DOES A SENTIMENT FLIP PREDICT POST DEATH?")
    print(f"{'=' * 70}")
    print(f"  {'Flip type':<20} {'surging':>8} {'alive':>8} {'cooling':>8} {'dying':>8} {'dead':>8}")
    print(f"  {'-'*65}")

    state_order = ["surging", "alive", "cooling", "dying", "dead"]
    for flip in ["pos_to_neg", "neg_to_pos", "stayed_positive", "stayed_negative", "neutral"]:
        states = flip_states.get(flip, {})
        total = sum(states.values())
        if total < 5:
            continue
        print(f"  {flip:<20}", end="")
        for s in state_order:
            pct = states.get(s, 0) / total if total else 0
            print(f" {pct:>7.0%}", end="")
        print(f"  (n={total})")

    # Sentiment slope vs outcome
    print(f"\n{'=' * 70}")
    print("SENTIMENT SLOPE vs OUTCOME")
    print("(negative slope = sentiment getting worse over time)")
    print(f"{'=' * 70}")

    state_slopes = defaultdict(list)
    for t in trajectories:
        if t["state"] and t["state"] != "unknown":
            state_slopes[t["state"]].append(t["sentiment_slope"])

    print(f"  {'State':<15} {'Posts':>6} {'Avg slope':>12} {'Med slope':>12}")
    print(f"  {'-'*50}")
    for state in ["surging", "alive", "emerging", "cooling", "dying", "dead"]:
        slopes = state_slopes.get(state, [])
        if not slopes:
            continue
        avg = statistics.mean(slopes)
        med = statistics.median(slopes)
        direction = "worsening" if avg < -0.005 else ("improving" if avg > 0.005 else "stable")
        print(f"  {state:<15} {len(slopes):>6} {avg:>+12.4f} {med:>+12.4f}  ({direction})")

    # Volatility vs outcome
    print(f"\n{'=' * 70}")
    print("SENTIMENT VOLATILITY vs OUTCOME")
    print("(high volatility = big swings between positive and negative)")
    print(f"{'=' * 70}")

    state_vol = defaultdict(list)
    for t in trajectories:
        if t["state"] and t["state"] != "unknown":
            state_vol[t["state"]].append(t["volatility"])

    print(f"  {'State':<15} {'Avg volatility':>15} {'Med volatility':>15}")
    print(f"  {'-'*50}")
    for state in ["surging", "alive", "emerging", "cooling", "dying", "dead"]:
        vols = state_vol.get(state, [])
        if not vols:
            continue
        print(f"  {state:<15} {statistics.mean(vols):>15.4f} {statistics.median(vols):>15.4f}")

    return trajectories


def build_trajectory_classifier(trajectories):
    """Train classifier using trajectory features to predict post survival"""
    if not _HAS_SKLEARN:
        print("\n  Skipping classifier (no scikit-learn)")
        return

    usable = [t for t in trajectories
              if t["state"] in ("surging", "alive", "cooling", "dying", "dead")
              and t["snapshot_count"] >= 3]

    if len(usable) < 30:
        print(f"\n  Not enough data for classifier ({len(usable)} posts, need 30+)")
        return

    feature_names = [
        "first_sentiment", "mid_sentiment", "last_sentiment",
        "sentiment_change", "sentiment_slope",
        "variance_change", "neg_share_change",
        "volatility", "first_comment_count", "last_comment_count"
    ]

    X = [[t[f] for f in feature_names] for t in usable]
    # Binary: 1 = alive (surging/alive), 0 = dying (cooling/dying/dead)
    y = [1 if t["state"] in ("surging", "alive") else 0 for t in usable]

    pos = sum(y)
    neg = len(y) - pos

    print(f"\n{'=' * 70}")
    print("TRAJECTORY-BASED SURVIVAL CLASSIFIER")
    print(f"{'=' * 70}")
    print(f"  Training: {len(usable)} posts ({pos} alive, {neg} dying/dead)")

    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)

    cv_folds = min(5, min(pos, neg))
    if cv_folds >= 2:
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
        print(f"  Cross-validation accuracy ({cv_folds}-fold): {scores.mean():.2%} (+/- {scores.std():.2%})")

    clf.fit(X, y)
    print(f"\n  Feature importance (what predicts survival):")
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.01:
            bar = "#" * int(imp * 40)
            print(f"    {name:<25} {imp:.3f}  {bar}")


def main():
    print("=" * 70)
    print("SENTIMENT TRAJECTORY ANALYSIS")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    print("\n[1/4] Loading and scoring comments by snapshot...")
    post_snaps, snap_times = load_comment_timeseries(conn)

    print("\n[2/4] Loading lifecycle data...")
    cur = conn.cursor()
    lc_rows = cur.execute("""
        SELECT post_id, subreddit, latest_activity_state, max_upvotes
        FROM post_lifecycles
    """).fetchall()
    lifecycles = {r[0]: {"subreddit": r[1], "state": r[2], "max_upvotes": r[3] or 0}
                  for r in lc_rows}
    print(f"  {len(lifecycles)} lifecycles loaded")

    print("\n[3/4] Building sentiment trajectories...")
    trajectories = build_trajectories(post_snaps, snap_times, lifecycles)
    print(f"  {len(trajectories)} posts with multi-point sentiment trajectories")

    print("\n[4/4] Analyzing patterns...")
    trajectories = analyze_trajectories(trajectories)
    build_trajectory_classifier(trajectories)

    # Save
    path = os.path.join(OUT_DIR, "sentiment_trajectories.csv")
    if trajectories:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trajectories[0].keys()))
            w.writeheader()
            w.writerows(trajectories)
        print(f"\n  Saved: {path} ({len(trajectories)} rows)")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
