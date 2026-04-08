"""
Subreddit Direction Predictor
==============================
Predict whether a subreddit is trending up or down based on:
- Health metrics (upvotes, comments, dead share, retention)
- Sentiment trends (are comments getting more positive/negative?)
- Activity patterns (post volume, engagement rates)

Uses rolling windows to detect momentum shifts.
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
    raise ImportError("pip install vaderSentiment")

try:
    from sklearn.linear_model import LinearRegression
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_trend_slope(values):
    """Simple linear regression slope over a series of values"""
    if len(values) < 3:
        return 0
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0


def compute_momentum(values, window=5):
    """Compare recent window average to overall average"""
    if len(values) < window + 2:
        return 0
    recent = statistics.mean(values[-window:])
    overall = statistics.mean(values)
    if overall == 0:
        return 0
    return (recent - overall) / abs(overall)


def main():
    print("=" * 70)
    print("SUBREDDIT DIRECTION PREDICTOR")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Load health trend data per subreddit
    subreddits = ["Games", "news", "politics", "technology", "worldnews"]
    sub_trends = {}

    for sub in subreddits:
        rows = cur.execute("""
            SELECT snapshot_time_utc, post_count, new_post_count, persisting_post_count,
                   retention_rate, avg_upvotes, median_upvotes, avg_comments,
                   median_comments, dead_post_share
            FROM subreddit_health_trend
            WHERE subreddit = ?
            ORDER BY snapshot_time_utc
        """, (sub,)).fetchall()
        sub_trends[sub] = rows

    print(f"  Loaded trend data: {', '.join(f'{s}={len(v)}' for s, v in sub_trends.items())}")

    # Load sentiment per subreddit over time (from comment snapshots)
    print("\n  Computing sentiment trends per subreddit...")
    sub_sentiment_series = defaultdict(list)  # sub -> [(snap_time, avg_sentiment)]

    snap_sentiments = cur.execute("""
        SELECT c.subreddit, c.snapshot_id, c.body, c.upvotes_at_snapshot
        FROM comment_snapshots c
        WHERE c.body IS NOT NULL AND c.body != ''
          AND c.body != '[deleted]' AND c.body != '[removed]'
        ORDER BY c.subreddit, c.snapshot_id
    """).fetchall()

    # Group by (subreddit, snapshot_id)
    snap_groups = defaultdict(list)
    for sub, snap_id, body, upvotes in snap_sentiments:
        score = _VADER.polarity_scores(body)["compound"]
        weight = max(1, upvotes) if upvotes else 1
        snap_groups[(sub, snap_id)].append((score, weight))

    for (sub, snap_id), comments in snap_groups.items():
        total_w = sum(w for s, w in comments)
        weighted_sent = sum(s * w for s, w in comments) / total_w if total_w > 0 else 0
        sub_sentiment_series[sub].append({
            "snapshot_id": snap_id,
            "weighted_sentiment": weighted_sent,
            "comment_count": len(comments),
            "pos_share": sum(1 for s, w in comments if s > 0.05) / len(comments),
            "neg_share": sum(1 for s, w in comments if s < -0.05) / len(comments),
        })

    print(f"  Sentiment snapshots: {', '.join(f'{s}={len(v)}' for s, v in sub_sentiment_series.items())}")

    # Analyze each subreddit
    results = []
    for sub in subreddits:
        print(f"\n{'=' * 70}")
        print(f"  {sub.upper()}")
        print(f"{'=' * 70}")

        trend = sub_trends.get(sub, [])
        if len(trend) < 10:
            print(f"  Not enough trend data ({len(trend)} points)")
            continue

        # Extract time series
        upvotes_series = [r[5] for r in trend if r[5] is not None and isinstance(r[5], (int, float))]
        comments_series = [r[7] for r in trend if r[7] is not None and isinstance(r[7], (int, float))]
        dead_series = [r[9] for r in trend if r[9] is not None and isinstance(r[9], (int, float))]
        retention_series = [r[4] for r in trend if r[4] is not None and isinstance(r[4], (int, float))]
        post_count_series = [r[1] for r in trend if r[1] is not None and isinstance(r[1], (int, float))]

        # Health metric trends
        upvote_slope = compute_trend_slope(upvotes_series) if upvotes_series else 0
        comment_slope = compute_trend_slope(comments_series) if comments_series else 0
        dead_slope = compute_trend_slope(dead_series) if dead_series else 0
        retention_slope = compute_trend_slope(retention_series) if retention_series else 0

        # Momentum (recent vs overall)
        upvote_momentum = compute_momentum(upvotes_series) if len(upvotes_series) > 7 else 0
        comment_momentum = compute_momentum(comments_series) if len(comments_series) > 7 else 0
        dead_momentum = compute_momentum(dead_series) if len(dead_series) > 7 else 0

        # Sentiment trends
        sent_data = sub_sentiment_series.get(sub, [])
        sent_series = [s["weighted_sentiment"] for s in sent_data]
        neg_series = [s["neg_share"] for s in sent_data]

        sent_slope = compute_trend_slope(sent_series) if len(sent_series) > 5 else 0
        sent_momentum = compute_momentum(sent_series) if len(sent_series) > 7 else 0
        neg_slope = compute_trend_slope(neg_series) if len(neg_series) > 5 else 0

        # Current values (last 5 snapshots)
        recent_upvotes = statistics.mean(upvotes_series[-5:]) if upvotes_series else 0
        recent_dead = statistics.mean(dead_series[-5:]) if dead_series else 0
        recent_sentiment = statistics.mean(sent_series[-5:]) if sent_series else 0
        recent_neg = statistics.mean(neg_series[-5:]) if neg_series else 0

        # Composite direction score (-100 to +100)
        # Positive = subreddit is improving, Negative = declining
        direction_score = 0

        # Upvote trend (weight: 25)
        if upvote_momentum > 0.1:
            direction_score += 25
        elif upvote_momentum > 0:
            direction_score += 10
        elif upvote_momentum < -0.1:
            direction_score -= 25
        else:
            direction_score -= 10

        # Dead post share trend (weight: 25, inverted)
        if dead_momentum < -0.1:
            direction_score += 25  # fewer dead = good
        elif dead_momentum < 0:
            direction_score += 10
        elif dead_momentum > 0.1:
            direction_score -= 25
        else:
            direction_score -= 10

        # Sentiment trend (weight: 25)
        if sent_slope > 0.001:
            direction_score += 25
        elif sent_slope > 0:
            direction_score += 10
        elif sent_slope < -0.001:
            direction_score -= 25
        else:
            direction_score -= 10

        # Comment engagement (weight: 25)
        if comment_momentum > 0.1:
            direction_score += 25
        elif comment_momentum > 0:
            direction_score += 10
        elif comment_momentum < -0.1:
            direction_score -= 25
        else:
            direction_score -= 10

        # Direction label
        if direction_score >= 50:
            direction = "STRONG UPTREND"
        elif direction_score >= 20:
            direction = "MILD UPTREND"
        elif direction_score >= -20:
            direction = "STABLE/MIXED"
        elif direction_score >= -50:
            direction = "MILD DECLINE"
        else:
            direction = "STRONG DECLINE"

        print(f"\n  Direction Score: {direction_score:+d}/100  -->  {direction}")
        print(f"\n  Metrics breakdown:")
        print(f"    Upvote momentum:    {upvote_momentum:+.3f}  ({'rising' if upvote_momentum > 0 else 'falling'})")
        print(f"    Dead post momentum: {dead_momentum:+.3f}  ({'worse' if dead_momentum > 0 else 'better'})")
        print(f"    Sentiment slope:    {sent_slope:+.6f}  ({'improving' if sent_slope > 0 else 'worsening'})")
        print(f"    Comment momentum:   {comment_momentum:+.3f}  ({'growing' if comment_momentum > 0 else 'shrinking'})")
        print(f"\n  Current snapshot (last 5 avg):")
        print(f"    Avg upvotes:    {recent_upvotes:,.0f}")
        print(f"    Dead share:     {recent_dead:.1%}")
        print(f"    Sentiment:      {recent_sentiment:+.3f}")
        print(f"    Negative share: {recent_neg:.1%}")

        # Forecast: where is this going in 24h?
        if len(upvotes_series) > 10:
            # Simple linear extrapolation
            forecast_up = recent_upvotes + upvote_slope * 24
            forecast_dead = max(0, min(1, recent_dead + dead_slope * 24))
            forecast_sent = recent_sentiment + sent_slope * 24

            print(f"\n  24-hour forecast (linear extrapolation):")
            print(f"    Upvotes:   {recent_upvotes:,.0f} -> {forecast_up:,.0f}")
            print(f"    Dead share: {recent_dead:.1%} -> {forecast_dead:.1%}")
            print(f"    Sentiment: {recent_sentiment:+.3f} -> {forecast_sent:+.3f}")

        result = {
            "subreddit": sub,
            "direction_score": direction_score,
            "direction_label": direction,
            "upvote_momentum": round(upvote_momentum, 4),
            "dead_momentum": round(dead_momentum, 4),
            "sentiment_slope": round(sent_slope, 6),
            "comment_momentum": round(comment_momentum, 4),
            "recent_avg_upvotes": round(recent_upvotes, 0),
            "recent_dead_share": round(recent_dead, 4),
            "recent_sentiment": round(recent_sentiment, 4),
            "recent_neg_share": round(recent_neg, 4),
            "datapoints": len(trend),
        }
        results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: SUBREDDIT DIRECTION")
    print(f"{'=' * 70}")
    print(f"  {'Subreddit':<15} {'Score':>8} {'Direction':<20} {'Upvote Mom':>12} {'Sentiment':>10}")
    print(f"  {'-' * 70}")
    for r in sorted(results, key=lambda x: -x["direction_score"]):
        print(f"  {r['subreddit']:<15} {r['direction_score']:>+8} {r['direction_label']:<20} {r['upvote_momentum']:>+12.3f} {r['recent_sentiment']:>+10.3f}")

    # Save
    path = os.path.join(OUT_DIR, "subreddit_direction.csv")
    if results:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\n  Saved: {path}")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
