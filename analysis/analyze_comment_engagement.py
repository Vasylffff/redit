"""
Comment Engagement Analysis
============================
Deep analysis of comment upvotes as engagement signals.
- Top-comment dominance: is one comment getting all upvotes?
- High-upvote comment sentiment vs low-upvote
- Comment upvote concentration (Gini-like) as predictor
- Combine with sentiment for improved survival prediction
"""

import csv
import math
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
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def gini_coefficient(values):
    """Compute Gini coefficient (0=equal, 1=one comment has all upvotes)"""
    if not values or len(values) < 2:
        return 0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0
    cum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    return cum / (n * total)


def main():
    print("=" * 70)
    print("COMMENT ENGAGEMENT ANALYSIS")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Load all comments with upvotes
    rows = cur.execute("""
        SELECT c.post_id, c.body, c.upvotes_at_snapshot, c.reply_count_at_snapshot,
               c.is_top_level_comment, c.age_minutes_at_snapshot
        FROM comment_snapshots c
        WHERE c.body IS NOT NULL AND c.body != ''
          AND c.body != '[deleted]' AND c.body != '[removed]'
          AND c.upvotes_at_snapshot IS NOT NULL
    """).fetchall()
    print(f"  Loaded {len(rows)} comments")

    # Load lifecycles
    lc_rows = cur.execute("""
        SELECT post_id, subreddit, latest_activity_state, max_upvotes, max_comments
        FROM post_lifecycles
    """).fetchall()
    lifecycles = {r[0]: {"subreddit": r[1], "state": r[2], "max_upvotes": r[3] or 0, "max_comments": r[4] or 0}
                  for r in lc_rows}

    # Build per-post comment engagement profiles
    post_comments = defaultdict(list)
    for pid, body, upvotes, replies, is_top, age_min in rows:
        score = _VADER.polarity_scores(body)["compound"]
        post_comments[pid].append({
            "body": body,
            "upvotes": upvotes or 0,
            "replies": replies or 0,
            "is_top_level": is_top,
            "age_min": age_min or 0,
            "sentiment": score,
        })

    print(f"  {len(post_comments)} posts with comments")

    # Compute engagement features per post
    post_features = []
    for pid, comments in post_comments.items():
        lc = lifecycles.get(pid, {})
        state = lc.get("state", "")
        if not state or state == "unknown":
            continue

        upvotes_list = [c["upvotes"] for c in comments]
        sentiments = [c["sentiment"] for c in comments]

        # Basic stats
        total_comment_upvotes = sum(upvotes_list)
        max_comment_upvotes = max(upvotes_list) if upvotes_list else 0
        mean_comment_upvotes = statistics.mean(upvotes_list) if upvotes_list else 0
        median_comment_upvotes = statistics.median(upvotes_list) if upvotes_list else 0

        # Upvote concentration
        upvote_gini = gini_coefficient(upvotes_list)
        top_comment_share = max_comment_upvotes / total_comment_upvotes if total_comment_upvotes > 0 else 0

        # Sentiment of top-upvoted comments (community-endorsed opinion)
        sorted_by_upvotes = sorted(comments, key=lambda c: -c["upvotes"])
        top_3 = sorted_by_upvotes[:3]
        top_3_sentiment = statistics.mean([c["sentiment"] for c in top_3]) if top_3 else 0

        # Sentiment of low-upvoted comments (minority opinion)
        bottom_half = sorted_by_upvotes[len(sorted_by_upvotes)//2:]
        bottom_sentiment = statistics.mean([c["sentiment"] for c in bottom_half]) if bottom_half else 0

        # Sentiment gap: do popular and unpopular comments disagree?
        sentiment_gap = top_3_sentiment - bottom_sentiment

        # Early vs late comment sentiment
        sorted_by_age = sorted(comments, key=lambda c: c["age_min"])
        early = sorted_by_age[:max(1, len(sorted_by_age)//3)]
        late = sorted_by_age[2*len(sorted_by_age)//3:]
        early_sentiment = statistics.mean([c["sentiment"] for c in early]) if early else 0
        late_sentiment = statistics.mean([c["sentiment"] for c in late]) if late else 0

        # Overall sentiment
        all_sentiment = statistics.mean(sentiments) if sentiments else 0
        weighted_sentiment = (sum(c["sentiment"] * max(1, c["upvotes"]) for c in comments) /
                              sum(max(1, c["upvotes"]) for c in comments)) if comments else 0

        # Top-level vs reply sentiment
        top_level = [c for c in comments if c["is_top_level"]]
        replies = [c for c in comments if not c["is_top_level"]]
        top_level_sentiment = statistics.mean([c["sentiment"] for c in top_level]) if top_level else 0
        reply_sentiment = statistics.mean([c["sentiment"] for c in replies]) if replies else 0

        feat = {
            "post_id": pid,
            "subreddit": lc.get("subreddit", ""),
            "state": state,
            "comment_count": len(comments),
            "total_comment_upvotes": total_comment_upvotes,
            "max_comment_upvotes": max_comment_upvotes,
            "mean_comment_upvotes": round(mean_comment_upvotes, 1),
            "median_comment_upvotes": median_comment_upvotes,
            "upvote_gini": round(upvote_gini, 4),
            "top_comment_share": round(top_comment_share, 4),
            "top3_sentiment": round(top_3_sentiment, 4),
            "bottom_half_sentiment": round(bottom_sentiment, 4),
            "sentiment_gap": round(sentiment_gap, 4),
            "early_sentiment": round(early_sentiment, 4),
            "late_sentiment": round(late_sentiment, 4),
            "all_sentiment": round(all_sentiment, 4),
            "weighted_sentiment": round(weighted_sentiment, 4),
            "top_level_sentiment": round(top_level_sentiment, 4),
            "reply_sentiment": round(reply_sentiment, 4),
            "max_post_upvotes": lc.get("max_upvotes", 0),
        }
        post_features.append(feat)

    print(f"  {len(post_features)} posts with engagement profiles")

    # Analysis 1: How do engagement features differ by state?
    print(f"\n{'=' * 70}")
    print("COMMENT ENGAGEMENT BY LIFECYCLE STATE")
    print(f"{'=' * 70}")

    state_order = ["surging", "alive", "cooling", "dying", "dead"]
    metrics = ["mean_comment_upvotes", "upvote_gini", "top3_sentiment",
               "sentiment_gap", "weighted_sentiment"]

    print(f"  {'Metric':<25}", end="")
    for s in state_order:
        print(f" {s:>10}", end="")
    print()
    print(f"  {'-' * 75}")

    state_groups = defaultdict(list)
    for f in post_features:
        state_groups[f["state"]].append(f)

    for metric in metrics:
        print(f"  {metric:<25}", end="")
        for s in state_order:
            vals = [f[metric] for f in state_groups.get(s, [])]
            if vals:
                print(f" {statistics.mean(vals):>10.3f}", end="")
            else:
                print(f" {'':>10}", end="")
        print()

    # Analysis 2: Top-endorsed sentiment vs overall
    print(f"\n{'=' * 70}")
    print("COMMUNITY-ENDORSED vs OVERALL SENTIMENT")
    print("(Do high-upvote comments agree with the crowd?)")
    print(f"{'=' * 70}")

    for s in state_order:
        posts = state_groups.get(s, [])
        if not posts:
            continue
        top3_avg = statistics.mean([f["top3_sentiment"] for f in posts])
        all_avg = statistics.mean([f["all_sentiment"] for f in posts])
        gap_avg = statistics.mean([f["sentiment_gap"] for f in posts])
        print(f"  {s:<12} Top-3 endorsed: {top3_avg:+.3f}  |  Overall: {all_avg:+.3f}  |  Gap: {gap_avg:+.3f}")

    # Analysis 3: Early vs late sentiment shift
    print(f"\n{'=' * 70}")
    print("EARLY vs LATE COMMENT SENTIMENT")
    print("(Does the tone change as more people pile on?)")
    print(f"{'=' * 70}")

    for s in state_order:
        posts = state_groups.get(s, [])
        if not posts:
            continue
        early_avg = statistics.mean([f["early_sentiment"] for f in posts])
        late_avg = statistics.mean([f["late_sentiment"] for f in posts])
        shift = late_avg - early_avg
        direction = "worsening" if shift < -0.01 else ("improving" if shift > 0.01 else "stable")
        print(f"  {s:<12} Early: {early_avg:+.3f}  Late: {late_avg:+.3f}  Shift: {shift:+.3f} ({direction})")

    # Analysis 4: Upvote concentration
    print(f"\n{'=' * 70}")
    print("UPVOTE CONCENTRATION (Gini coefficient)")
    print("(0=equal distribution, 1=one comment dominates)")
    print(f"{'=' * 70}")

    for s in state_order:
        posts = state_groups.get(s, [])
        if not posts:
            continue
        gini_avg = statistics.mean([f["upvote_gini"] for f in posts])
        share_avg = statistics.mean([f["top_comment_share"] for f in posts])
        print(f"  {s:<12} Gini: {gini_avg:.3f}  |  Top comment gets {share_avg:.1%} of all upvotes")

    # Build classifier with all features
    if _HAS_SKLEARN:
        print(f"\n{'=' * 70}")
        print("ENHANCED SURVIVAL CLASSIFIER")
        print("(using comment engagement + sentiment features)")
        print(f"{'=' * 70}")

        usable = [f for f in post_features
                   if f["state"] in ("surging", "alive", "cooling", "dying", "dead")
                   and f["comment_count"] >= 3]

        feature_names = [
            "comment_count", "total_comment_upvotes", "max_comment_upvotes",
            "mean_comment_upvotes", "upvote_gini", "top_comment_share",
            "top3_sentiment", "bottom_half_sentiment", "sentiment_gap",
            "early_sentiment", "late_sentiment", "weighted_sentiment",
        ]

        X = [[f[fn] for fn in feature_names] for f in usable]
        y = [1 if f["state"] in ("surging", "alive") else 0 for f in usable]

        pos = sum(y)
        neg = len(y) - pos
        print(f"  Training: {len(usable)} posts ({pos} alive, {neg} dying/dead)")

        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
        cv_folds = min(5, min(pos, neg))
        if cv_folds >= 2:
            scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
            print(f"  Cross-validation accuracy ({cv_folds}-fold): {scores.mean():.2%} (+/- {scores.std():.2%})")

        clf.fit(X, y)
        print(f"\n  Feature importance:")
        for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
            if imp > 0.005:
                bar = "#" * int(imp * 40)
                print(f"    {name:<28} {imp:.3f}  {bar}")

    # Save
    path = os.path.join(OUT_DIR, "comment_engagement.csv")
    if post_features:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(post_features[0].keys()))
            w.writeheader()
            w.writerows(post_features)
        print(f"\n  Saved: {path} ({len(post_features)} rows)")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
