"""
Mood Predictor — Sentiment Trajectory Analysis
================================================
1. Correlate early comment sentiment with post lifecycle outcomes
   (do negative posts die faster? do positive ones surge longer?)
2. Build a classifier: given a post's current sentiment profile → predict mood direction
3. Output charts + summary CSV

Uses VADER for comment scoring, lifecycle states from history.db
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
    from sklearn.metrics import classification_report
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("WARNING: scikit-learn not available, skipping classifier")

# ── paths ──
PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_comment_sentiment(conn):
    """Score every comment with VADER, return {post_id: [(score, upvotes, body_len)]}"""
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT post_id, body, upvotes_at_snapshot, body_length_chars
        FROM comment_snapshots
        WHERE body IS NOT NULL
          AND body != ''
          AND body != '[deleted]'
          AND body != '[removed]'
    """).fetchall()

    post_comments = defaultdict(list)
    for post_id, body, upvotes, body_len in rows:
        score = _VADER.polarity_scores(body)["compound"]
        weight = max(1.0, upvotes) if upvotes else 1.0
        post_comments[post_id].append((score, weight, body_len or 0))

    print(f"  Scored {sum(len(v) for v in post_comments.values())} comments across {len(post_comments)} posts")
    return post_comments


def load_lifecycles(conn):
    """Load post lifecycle data: {post_id: {state, states_seen, velocities, ...}}"""
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT post_id, subreddit, latest_activity_state, activity_states_seen,
               last_upvote_velocity_per_hour, last_comment_velocity_per_hour,
               first_upvotes, last_upvotes, max_upvotes,
               first_comments, last_comments, max_comments,
               total_upvote_growth, total_comment_growth,
               observed_hours, snapshot_count, is_currently_dead
        FROM post_lifecycles
    """).fetchall()

    lifecycles = {}
    for r in rows:
        lifecycles[r[0]] = {
            "post_id": r[0],
            "subreddit": r[1],
            "state": r[2],
            "states_seen": r[3],
            "upvote_vel": r[4] or 0,
            "comment_vel": r[5] or 0,
            "first_upvotes": r[6] or 0,
            "last_upvotes": r[7] or 0,
            "max_upvotes": r[8] or 0,
            "first_comments": r[9] or 0,
            "last_comments": r[10] or 0,
            "max_comments": r[11] or 0,
            "upvote_growth": r[12] or 0,
            "comment_growth": r[13] or 0,
            "observed_hours": r[14] or 0,
            "snapshot_count": r[15] or 0,
            "is_dead": r[16] or 0,
        }
    print(f"  Loaded {len(lifecycles)} post lifecycles")
    return lifecycles


def compute_post_sentiment(comments_list):
    """Given [(score, weight, body_len), ...] → aggregated sentiment dict"""
    if not comments_list:
        return None
    scores = [s for s, w, bl in comments_list]
    weights = [w for s, w, bl in comments_list]

    mean_sent = statistics.mean(scores)
    weighted_sent = sum(s * w for s, w, bl in comments_list) / sum(weights) if sum(weights) > 0 else 0
    pos_share = sum(1 for s in scores if s > 0.05) / len(scores)
    neg_share = sum(1 for s in scores if s < -0.05) / len(scores)
    variance = statistics.variance(scores) if len(scores) > 1 else 0

    return {
        "comment_count": len(comments_list),
        "sentiment_mean": round(mean_sent, 4),
        "sentiment_weighted": round(weighted_sent, 4),
        "positive_share": round(pos_share, 4),
        "negative_share": round(neg_share, 4),
        "sentiment_variance": round(variance, 4),
        "avg_body_len": round(statistics.mean([bl for s, w, bl in comments_list]), 1),
    }


# ── MOOD LABELS ──
# Map lifecycle states to mood outcomes for prediction
STATE_TO_MOOD = {
    "surging": "positive",
    "alive": "positive",
    "emerging": "neutral",
    "cooling": "negative",
    "dying": "negative",
    "dead": "negative",
    "unknown": "neutral",
}


def analyze_sentiment_vs_lifecycle(post_sentiments, lifecycles):
    """Core analysis: correlate sentiment with lifecycle outcomes"""
    # Group by lifecycle state
    state_sentiments = defaultdict(list)
    mood_data = []

    for post_id, sent in post_sentiments.items():
        if post_id not in lifecycles:
            continue
        lc = lifecycles[post_id]
        state = lc["state"]
        if not state:
            continue

        state_sentiments[state].append(sent)
        mood = STATE_TO_MOOD.get(state, "neutral")
        mood_data.append({
            "post_id": post_id,
            "subreddit": lc["subreddit"],
            "state": state,
            "mood": mood,
            **sent,
            "upvote_vel": lc["upvote_vel"],
            "comment_vel": lc["comment_vel"],
            "upvote_growth": lc["upvote_growth"],
            "comment_growth": lc["comment_growth"],
            "observed_hours": lc["observed_hours"],
            "is_dead": lc["is_dead"],
        })

    # Print correlation table
    print("\n" + "=" * 70)
    print("SENTIMENT vs LIFECYCLE STATE CORRELATION")
    print("=" * 70)
    print(f"{'State':<12} {'Posts':>5} {'Avg Sent':>10} {'Wgt Sent':>10} {'Pos%':>8} {'Neg%':>8} {'Variance':>10}")
    print("-" * 70)

    state_order = ["surging", "alive", "emerging", "cooling", "dying", "dead", "unknown"]
    for state in state_order:
        sents = state_sentiments.get(state, [])
        if not sents:
            continue
        n = len(sents)
        avg_s = statistics.mean([s["sentiment_mean"] for s in sents])
        avg_w = statistics.mean([s["sentiment_weighted"] for s in sents])
        avg_p = statistics.mean([s["positive_share"] for s in sents])
        avg_n = statistics.mean([s["negative_share"] for s in sents])
        avg_v = statistics.mean([s["sentiment_variance"] for s in sents])
        print(f"{state:<12} {n:>5} {avg_s:>10.4f} {avg_w:>10.4f} {avg_p:>8.2%} {avg_n:>8.2%} {avg_v:>10.4f}")

    # Key finding: do positive posts survive longer?
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    alive_posts = [s for pid, s in post_sentiments.items()
                   if pid in lifecycles and lifecycles[pid]["state"] in ("surging", "alive")]
    dead_posts = [s for pid, s in post_sentiments.items()
                  if pid in lifecycles and lifecycles[pid]["state"] in ("dying", "dead")]

    if alive_posts and dead_posts:
        alive_avg = statistics.mean([s["sentiment_mean"] for s in alive_posts])
        dead_avg = statistics.mean([s["sentiment_mean"] for s in dead_posts])
        diff = alive_avg - dead_avg
        print(f"  Alive/surging posts avg sentiment: {alive_avg:+.4f}  (n={len(alive_posts)})")
        print(f"  Dying/dead posts avg sentiment:    {dead_avg:+.4f}  (n={len(dead_posts)})")
        print(f"  Difference:                        {diff:+.4f}")
        if diff > 0.05:
            print("  --> POSITIVE correlation: happier comments = longer-living posts")
        elif diff < -0.05:
            print("  --> NEGATIVE correlation: angrier comments = longer-living posts (controversial!)")
        else:
            print("  --> WEAK/NO correlation between sentiment and post survival")

        # Variance comparison
        alive_var = statistics.mean([s["sentiment_variance"] for s in alive_posts])
        dead_var = statistics.mean([s["sentiment_variance"] for s in dead_posts])
        print(f"\n  Alive/surging sentiment variance: {alive_var:.4f}")
        print(f"  Dying/dead sentiment variance:    {dead_var:.4f}")
        if alive_var > dead_var + 0.02:
            print("  --> Living posts have MORE polarized discussions (controversy keeps them alive)")
        elif dead_var > alive_var + 0.02:
            print("  --> Dead posts have MORE polarized discussions (toxicity kills engagement)")
        else:
            print("  --> Similar variance levels")

    return mood_data


def build_mood_classifier(mood_data):
    """Train a simple decision tree: sentiment features → predict mood (positive/negative/neutral)"""
    if not _HAS_SKLEARN:
        print("\n  Skipping classifier (scikit-learn not installed)")
        return None

    # Filter to posts with enough comments
    usable = [d for d in mood_data if d["comment_count"] >= 3 and d["mood"] != "neutral"]
    if len(usable) < 20:
        print(f"\n  Not enough data for classifier ({len(usable)} usable posts, need 20+)")
        return None

    # Features: sentiment_mean, sentiment_weighted, positive_share, negative_share, variance, comment_count
    feature_names = ["sentiment_mean", "sentiment_weighted", "positive_share",
                     "negative_share", "sentiment_variance", "comment_count", "avg_body_len"]
    X = [[d[f] for f in feature_names] for d in usable]
    y = [1 if d["mood"] == "positive" else 0 for d in usable]

    pos_count = sum(y)
    neg_count = len(y) - pos_count
    print(f"\n{'=' * 70}")
    print("MOOD CLASSIFIER (Decision Tree)")
    print(f"{'=' * 70}")
    print(f"  Training on {len(usable)} posts: {pos_count} positive, {neg_count} negative")

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)

    # Cross-validation
    cv_folds = min(5, min(pos_count, neg_count))
    if cv_folds >= 2:
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
        print(f"  Cross-validation accuracy ({cv_folds}-fold): {scores.mean():.2%} (+/- {scores.std():.2%})")
    else:
        print("  Not enough data for cross-validation")

    # Train on all data and show feature importance
    clf.fit(X, y)
    print(f"\n  Feature importance:")
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.01:
            bar = "#" * int(imp * 40)
            print(f"    {name:<25} {imp:.3f}  {bar}")

    # Show rules
    print(f"\n  Prediction rules (tree depth={clf.get_depth()}):")
    print(f"    If a new post's comments have:")
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])[:3]:
        if imp > 0.05:
            print(f"      - High {name} -> more likely to {('stay alive' if name.startswith('positive') or name == 'sentiment_mean' else 'impact mood')}")

    return clf


def predict_current_mood(conn, post_sentiments, lifecycles, classifier=None):
    """For currently active posts, predict their mood trajectory"""
    active = {pid: lc for pid, lc in lifecycles.items()
              if lc["state"] in ("surging", "alive", "emerging") and pid in post_sentiments}

    if not active:
        print("\n  No active posts with sentiment data to predict")
        return []

    print(f"\n{'=' * 70}")
    print("MOOD PREDICTIONS FOR ACTIVE POSTS")
    print(f"{'=' * 70}")

    predictions = []
    for post_id, lc in sorted(active.items(), key=lambda x: -x[1]["upvote_vel"])[:20]:
        sent = post_sentiments[post_id]
        # Simple rule-based prediction
        mood_score = sent["sentiment_weighted"]
        if mood_score > 0.15:
            mood_pred = "STRONGLY POSITIVE - likely to keep surging"
        elif mood_score > 0.05:
            mood_pred = "MILDLY POSITIVE - should stay alive"
        elif mood_score < -0.15:
            mood_pred = "STRONGLY NEGATIVE - may die fast or go viral (controversial)"
        elif mood_score < -0.05:
            mood_pred = "MILDLY NEGATIVE - could cool down"
        else:
            mood_pred = "NEUTRAL - standard trajectory"

        # Variance tells us about controversy
        controversy = ""
        if sent["sentiment_variance"] > 0.3:
            controversy = " [HIGH CONTROVERSY]"
        elif sent["sentiment_variance"] > 0.15:
            controversy = " [moderate debate]"

        pred = {
            "post_id": post_id,
            "subreddit": lc["subreddit"],
            "state": lc["state"],
            "sentiment": mood_score,
            "prediction": mood_pred,
            "controversy": controversy.strip(),
            "comments": sent["comment_count"],
            "pos_share": sent["positive_share"],
            "neg_share": sent["negative_share"],
            "upvote_vel": lc["upvote_vel"],
        }
        predictions.append(pred)

        title = conn.execute("SELECT title FROM post_lifecycles WHERE post_id = ?", (post_id,)).fetchone()
        title_text = (title[0][:60] + "...") if title and len(title[0]) > 60 else (title[0] if title else "?")
        print(f"\n  [{lc['subreddit']}] {title_text}")
        print(f"    State: {lc['state']}  |  Upvote vel: {lc['upvote_vel']:.1f}/hr  |  Comments: {sent['comment_count']}")
        print(f"    Sentiment: {mood_score:+.3f}  |  Pos: {sent['positive_share']:.0%}  Neg: {sent['negative_share']:.0%}{controversy}")
        print(f"    --> {mood_pred}")

    return predictions


def save_results(mood_data, predictions):
    """Save analysis results to CSV"""
    # Full correlation data
    corr_path = os.path.join(OUT_DIR, "mood_correlation.csv")
    if mood_data:
        fields = list(mood_data[0].keys())
        with open(corr_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(mood_data)
        print(f"\n  Saved correlation data: {corr_path} ({len(mood_data)} rows)")

    # Predictions for active posts
    pred_path = os.path.join(OUT_DIR, "mood_predictions.csv")
    if predictions:
        fields = list(predictions[0].keys())
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(predictions)
        print(f"  Saved predictions: {pred_path} ({len(predictions)} rows)")


def main():
    print("=" * 70)
    print("MOOD PREDICTOR - Sentiment Trajectory Analysis")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    print("\n[1/5] Scoring comments with VADER...")
    post_comments = load_comment_sentiment(conn)

    print("\n[2/5] Loading lifecycle data...")
    lifecycles = load_lifecycles(conn)

    # Compute per-post sentiment aggregates
    print("\n[3/5] Computing per-post sentiment profiles...")
    post_sentiments = {}
    for post_id, comments in post_comments.items():
        agg = compute_post_sentiment(comments)
        if agg:
            post_sentiments[post_id] = agg
    print(f"  {len(post_sentiments)} posts with sentiment profiles")

    # Correlation analysis
    print("\n[4/5] Analyzing sentiment vs lifecycle correlation...")
    mood_data = analyze_sentiment_vs_lifecycle(post_sentiments, lifecycles)

    # Build classifier
    classifier = build_mood_classifier(mood_data)

    # Predict active posts
    print("\n[5/5] Predicting mood for active posts...")
    predictions = predict_current_mood(conn, post_sentiments, lifecycles, classifier)

    # Save
    save_results(mood_data, predictions)

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
