"""
Title Style Performance Analysis
==================================
Do questions, numbers, emotional words, or certain patterns perform better?
"""

import csv
import os
import re
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except ImportError:
    _VADER = None

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def classify_title(title):
    """Classify title into style categories"""
    t = title.lower().strip()
    features = {}

    features["is_question"] = "?" in title
    features["has_number"] = bool(re.search(r'\d', title))
    features["has_quote"] = '"' in title or "'" in title or "\u2018" in title or "\u201c" in title
    features["is_breaking"] = any(w in t for w in ["breaking", "just in", "live:", "update:"])
    features["has_colon"] = ":" in title
    features["is_opinion"] = any(w in t for w in ["opinion", "editorial", "column", "commentary"])
    features["has_negative_word"] = any(w in t.split() for w in [
        "killed", "dies", "dead", "death", "crash", "attack", "war", "bomb",
        "fire", "shooting", "murder", "arrest", "guilty", "fraud", "ban",
        "crisis", "collapse", "destroy", "threat", "fear", "worst"
    ])
    features["has_positive_word"] = any(w in t.split() for w in [
        "wins", "success", "record", "breakthrough", "best", "amazing",
        "free", "new", "launch", "announces", "reveals", "historic"
    ])
    features["has_shock_word"] = any(w in t for w in [
        "shocking", "unbelievable", "insane", "incredible", "stunning",
        "explosive", "bombshell", "devastating", "unprecedented"
    ])
    features["word_count"] = len(title.split())
    features["char_count"] = len(title)
    features["all_caps_words"] = sum(1 for w in title.split() if w.isupper() and len(w) > 1)

    # Title sentiment
    if _VADER:
        features["title_sentiment"] = _VADER.polarity_scores(title)["compound"]
    else:
        features["title_sentiment"] = 0

    return features


def main():
    print("=" * 70)
    print("TITLE STYLE PERFORMANCE ANALYSIS")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT title, subreddit, latest_activity_state, max_upvotes, max_comments,
               observed_hours
        FROM post_lifecycles
        WHERE title IS NOT NULL AND title != ''
          AND latest_activity_state IS NOT NULL AND latest_activity_state != 'unknown'
    """).fetchall()

    print(f"  {len(rows)} posts analyzed")

    # Classify all titles
    data = []
    for title, sub, state, max_up, max_com, obs_hrs in rows:
        features = classify_title(title)
        features["subreddit"] = sub
        features["state"] = state
        features["max_upvotes"] = max_up or 0
        features["max_comments"] = max_com or 0
        features["alive"] = state in ("surging", "alive")
        data.append(features)

    # Analyze each style feature
    bool_features = [
        ("is_question", "Question titles (?)"),
        ("has_number", "Titles with numbers"),
        ("has_quote", "Titles with quotes"),
        ("is_breaking", "Breaking news titles"),
        ("has_colon", "Titles with colon (:)"),
        ("has_negative_word", "Negative word titles"),
        ("has_positive_word", "Positive word titles"),
        ("has_shock_word", "Shock/sensational words"),
    ]

    print(f"\n{'=' * 70}")
    print("TITLE STYLE vs PERFORMANCE")
    print(f"{'=' * 70}")
    print(f"  {'Feature':<30} {'With':>6} {'W/o':>6} {'Alive% With':>13} {'Alive% W/o':>13} {'Med Up With':>13} {'Med Up W/o':>13}")
    print(f"  {'-' * 100}")

    style_results = []
    for feat_key, feat_name in bool_features:
        with_feat = [d for d in data if d[feat_key]]
        without_feat = [d for d in data if not d[feat_key]]

        if not with_feat or not without_feat:
            continue

        alive_with = sum(1 for d in with_feat if d["alive"]) / len(with_feat)
        alive_without = sum(1 for d in without_feat if d["alive"]) / len(without_feat)
        med_with = statistics.median([d["max_upvotes"] for d in with_feat])
        med_without = statistics.median([d["max_upvotes"] for d in without_feat])

        diff = alive_with - alive_without
        marker = " +" if diff > 0.03 else (" -" if diff < -0.03 else "  ")

        print(f"  {feat_name:<30} {len(with_feat):>6} {len(without_feat):>6} {alive_with:>12.1%} {alive_without:>12.1%} {med_with:>13.0f} {med_without:>13.0f}{marker}")

        style_results.append({
            "feature": feat_name,
            "count_with": len(with_feat),
            "count_without": len(without_feat),
            "alive_rate_with": round(alive_with, 4),
            "alive_rate_without": round(alive_without, 4),
            "median_upvotes_with": med_with,
            "median_upvotes_without": med_without,
            "advantage": round(diff, 4),
        })

    # Title length analysis
    print(f"\n{'=' * 70}")
    print("TITLE LENGTH vs PERFORMANCE")
    print(f"{'=' * 70}")

    length_buckets = [(0, 50, "Short (<50 chars)"), (50, 80, "Medium (50-80)"),
                      (80, 120, "Long (80-120)"), (120, 300, "Very long (120+)")]

    print(f"  {'Length':<25} {'Posts':>6} {'Alive%':>8} {'Med Up':>8}")
    print(f"  {'-' * 50}")
    for lo, hi, label in length_buckets:
        bucket = [d for d in data if lo <= d["char_count"] < hi]
        if not bucket:
            continue
        alive = sum(1 for d in bucket if d["alive"]) / len(bucket)
        med_up = statistics.median([d["max_upvotes"] for d in bucket])
        print(f"  {label:<25} {len(bucket):>6} {alive:>7.1%} {med_up:>8.0f}")

    # Title sentiment vs performance
    print(f"\n{'=' * 70}")
    print("TITLE SENTIMENT vs PERFORMANCE")
    print(f"{'=' * 70}")

    sent_buckets = [(-1, -0.3, "Very negative"), (-0.3, -0.05, "Mildly negative"),
                    (-0.05, 0.05, "Neutral"), (0.05, 0.3, "Mildly positive"),
                    (0.3, 1.01, "Very positive")]

    print(f"  {'Sentiment':<20} {'Posts':>6} {'Alive%':>8} {'Med Up':>8} {'Avg Up':>8}")
    print(f"  {'-' * 55}")
    for lo, hi, label in sent_buckets:
        bucket = [d for d in data if lo <= d["title_sentiment"] < hi]
        if not bucket:
            continue
        alive = sum(1 for d in bucket if d["alive"]) / len(bucket)
        med_up = statistics.median([d["max_upvotes"] for d in bucket])
        avg_up = statistics.mean([d["max_upvotes"] for d in bucket])
        print(f"  {label:<20} {len(bucket):>6} {alive:>7.1%} {med_up:>8.0f} {avg_up:>8.0f}")

    # Per subreddit: what style works best?
    print(f"\n{'=' * 70}")
    print("BEST TITLE STYLE PER SUBREDDIT")
    print(f"{'=' * 70}")

    for sub in sorted(set(d["subreddit"] for d in data)):
        sub_data = [d for d in data if d["subreddit"] == sub]
        best_feat = None
        best_diff = -1

        for feat_key, feat_name in bool_features:
            with_f = [d for d in sub_data if d[feat_key]]
            without_f = [d for d in sub_data if not d[feat_key]]
            if len(with_f) < 5 or len(without_f) < 5:
                continue
            alive_w = sum(1 for d in with_f if d["alive"]) / len(with_f)
            alive_wo = sum(1 for d in without_f if d["alive"]) / len(without_f)
            diff = alive_w - alive_wo
            if diff > best_diff:
                best_diff = diff
                best_feat = feat_name
                best_with = alive_w
                best_count = len(with_f)

        if best_feat and best_diff > 0:
            print(f"  {sub}: {best_feat} -> {best_with:.0%} alive ({best_count} posts, +{best_diff:.0%} vs without)")
        else:
            print(f"  {sub}: No clear winning style")

    # Save
    path = os.path.join(OUT_DIR, "title_style_analysis.csv")
    if style_results:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(style_results[0].keys()))
            w.writeheader()
            w.writerows(style_results)
        print(f"\n  Saved: {path}")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
