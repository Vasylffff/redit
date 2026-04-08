"""
Post Outcome Predictor
=======================
Given a post's early stats -> predict:
1. Will it POP or FLOP? (probability)
2. Rough peak upvotes range
3. How long it stays alive
4. Hour-by-hour trajectory using state transitions
5. Growth multiplier
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
    _VADER = None

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Empirical data tables (built from our 5000+ posts)
# Growth multipliers: (early_upvote_bucket) -> median multiplier
GROWTH_MULT = {
    "Games":      {5: 7.2, 30: 4.7, 100: 1.1, 500: 1.0},
    "news":       {5: 111.2, 30: 29.5, 100: 8.0, 500: 1.1},
    "politics":   {5: 6.3, 30: 4.0, 100: 5.9, 500: 1.3},
    "technology":  {5: 19.0, 30: 6.4, 100: 6.9, 500: 1.0},
    "worldnews":  {5: 17.4, 30: 6.3, 100: 7.1, 500: 1.0},
}

# Alive duration median (hours)
ALIVE_HOURS = {
    "Games": 49, "news": 42, "politics": 11, "technology": 28, "worldnews": 22,
}

# Peak timing median (hours)
PEAK_HOURS = {
    "Games": 70, "news": 96, "politics": 13, "technology": 34, "worldnews": 25,
}

# State transition matrix (from our 116K transitions)
TRANSITIONS = {
    "surging": {"surging": 0.75, "alive": 0.13, "cooling": 0.08, "dying": 0.02, "dead": 0.01},
    "alive":   {"surging": 0.03, "alive": 0.63, "cooling": 0.11, "dying": 0.11, "dead": 0.13},
    "cooling": {"surging": 0.12, "alive": 0.39, "cooling": 0.34, "dying": 0.06, "dead": 0.07},
    "dying":   {"surging": 0.02, "alive": 0.44, "cooling": 0.08, "dying": 0.44, "dead": 0.02},
    "dead":    {"surging": 0.01, "alive": 0.42, "cooling": 0.09, "dying": 0.00, "dead": 0.48},
}


def get_growth_multiplier(subreddit, early_upvotes):
    """Lookup empirical growth multiplier"""
    mults = GROWTH_MULT.get(subreddit, GROWTH_MULT.get("news"))
    thresholds = sorted(mults.keys())
    for t in thresholds:
        if early_upvotes <= t:
            return mults[t]
    return mults[thresholds[-1]]


def predict_trajectory(start_state, steps=12):
    """Simulate state trajectory using Markov transitions"""
    states = ["surging", "alive", "cooling", "dying", "dead"]
    # Start with probability 1.0 in start_state
    probs = {s: 0.0 for s in states}
    probs[start_state] = 1.0

    trajectory = [dict(probs)]
    for step in range(steps):
        new_probs = {s: 0.0 for s in states}
        for from_state in states:
            if probs[from_state] < 0.001:
                continue
            trans = TRANSITIONS.get(from_state, {})
            for to_state in states:
                new_probs[to_state] += probs[from_state] * trans.get(to_state, 0)
        probs = new_probs
        trajectory.append(dict(probs))

    return trajectory


def classify_early_state(early_upvotes, early_comments, subreddit):
    """Estimate starting state from early metrics"""
    # Rough thresholds based on our data
    if early_upvotes >= 200:
        return "surging"
    elif early_upvotes >= 50:
        return "alive"
    elif early_upvotes >= 10:
        return "cooling"  # could go either way
    else:
        return "dying"


def predict_single_post(subreddit, early_upvotes, early_comments, comment_sentiment=None):
    """Full prediction for one post"""
    result = {}

    # 1. Starting state
    start_state = classify_early_state(early_upvotes, early_comments, subreddit)
    result["start_state"] = start_state

    # 2. Pop or flop probability
    traj = predict_trajectory(start_state, steps=12)
    # After 6 steps (~6-12 hours), what's probability of alive/surging?
    mid = traj[6]
    pop_prob = mid["surging"] + mid["alive"]
    flop_prob = mid["dying"] + mid["dead"]
    result["pop_probability"] = round(pop_prob, 3)
    result["flop_probability"] = round(flop_prob, 3)

    # 3. Growth multiplier and peak estimate
    mult = get_growth_multiplier(subreddit, early_upvotes)
    estimated_peak = round(early_upvotes * mult)
    result["growth_multiplier"] = mult
    result["estimated_peak_upvotes"] = estimated_peak
    # Range (rough: 0.3x to 3x of estimate)
    result["peak_range_low"] = round(estimated_peak * 0.3)
    result["peak_range_high"] = round(estimated_peak * 3)

    # 4. Alive duration
    alive_hrs = ALIVE_HOURS.get(subreddit, 24)
    peak_hrs = PEAK_HOURS.get(subreddit, 24)
    # Adjust by state
    if start_state == "surging":
        alive_hrs = round(alive_hrs * 1.5)
        peak_hrs = round(peak_hrs * 0.8)
    elif start_state == "dying":
        alive_hrs = round(alive_hrs * 0.3)
        peak_hrs = round(peak_hrs * 0.5)
    result["estimated_alive_hours"] = alive_hrs
    result["estimated_peak_hour"] = peak_hrs

    # 5. Sentiment adjustment
    if comment_sentiment is not None:
        if comment_sentiment < -0.15:
            result["sentiment_effect"] = "controversial - may boost engagement"
            result["pop_probability"] = min(0.99, result["pop_probability"] * 1.1)
        elif comment_sentiment > 0.15:
            result["sentiment_effect"] = "positive - standard trajectory"
        else:
            result["sentiment_effect"] = "neutral"
    else:
        result["sentiment_effect"] = "no data"

    # 6. Hour-by-hour trajectory
    result["trajectory"] = traj

    return result


def format_prediction(subreddit, early_upvotes, early_comments, comment_sentiment=None):
    """Pretty print a prediction"""
    pred = predict_single_post(subreddit, early_upvotes, early_comments, comment_sentiment)

    print(f"\n  {'=' * 60}")
    print(f"  POST PREDICTION: r/{subreddit}")
    print(f"  Early signal: {early_upvotes} upvotes, {early_comments} comments")
    if comment_sentiment is not None:
        print(f"  Comment sentiment: {comment_sentiment:+.3f}")
    print(f"  {'=' * 60}")

    # Verdict
    pop = pred["pop_probability"]
    if pop >= 0.6:
        verdict = "LIKELY TO POP"
        icon = "[++]"
    elif pop >= 0.4:
        verdict = "COIN FLIP"
        icon = "[+-]"
    elif pop >= 0.2:
        verdict = "PROBABLY FLOPS"
        icon = "[--]"
    else:
        verdict = "DEAD ON ARRIVAL"
        icon = "[XX]"

    print(f"\n  {icon} Verdict: {verdict}")
    print(f"      Rise chance: {pop:.0%}  |  Flop chance: {pred['flop_probability']:.0%}")

    print(f"\n  Peak estimate: ~{pred['estimated_peak_upvotes']:,} upvotes")
    print(f"      Range: {pred['peak_range_low']:,} - {pred['peak_range_high']:,}")
    print(f"      Growth: {pred['growth_multiplier']:.1f}x from current")

    print(f"\n  Timing:")
    print(f"      Stays alive: ~{pred['estimated_alive_hours']}h")
    print(f"      Peaks around: ~{pred['estimated_peak_hour']}h")

    if pred["sentiment_effect"] != "no data":
        print(f"\n  Sentiment: {pred['sentiment_effect']}")

    # Trajectory
    traj = pred["trajectory"]
    states = ["surging", "alive", "cooling", "dying", "dead"]
    hours_per_step = 2  # rough mapping

    print(f"\n  Hour-by-hour trajectory:")
    print(f"  {'Hour':<8} {'surging':>8} {'alive':>8} {'cooling':>8} {'dying':>8} {'dead':>8}  dominant")
    print(f"  {'-' * 65}")
    for i, step in enumerate(traj[:10]):
        hour = i * hours_per_step
        dominant = max(step, key=step.get)
        print(f"  {hour:>4}h   ", end="")
        for s in states:
            print(f" {step[s]:>7.0%}", end="")
        print(f"  {dominant}")

    return pred


def main():
    print("=" * 70)
    print("POST OUTCOME PREDICTOR")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # Load current active posts and predict their future
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get currently surging/alive posts
    active = cur.execute("""
        SELECT l.post_id, l.subreddit, l.title, l.latest_activity_state,
               l.last_upvotes, l.last_comments, l.max_upvotes,
               l.last_upvote_velocity_per_hour, l.observed_hours
        FROM post_lifecycles l
        WHERE l.latest_activity_state IN ('surging', 'alive', 'emerging')
        ORDER BY l.last_upvote_velocity_per_hour DESC
        LIMIT 15
    """).fetchall()

    print(f"\n  Predicting top {len(active)} active posts...")

    # Get comment sentiment for these posts
    post_sentiments = {}
    if _VADER:
        for post_id, *_ in active:
            comments = cur.execute("""
                SELECT body, upvotes_at_snapshot FROM comment_snapshots
                WHERE post_id = ? AND body IS NOT NULL AND body != ''
                  AND body != '[deleted]' AND body != '[removed]'
            """, (post_id,)).fetchall()
            if comments:
                scores = []
                weights = []
                for body, up in comments:
                    s = _VADER.polarity_scores(body)["compound"]
                    w = max(1, up) if up else 1
                    scores.append(s)
                    weights.append(w)
                total_w = sum(weights)
                post_sentiments[post_id] = sum(s*w for s,w in zip(scores, weights)) / total_w if total_w > 0 else 0

    predictions = []
    for post_id, sub, title, state, upvotes, comments, max_up, vel, obs_hrs in active:
        title_short = (title[:55] + "...") if title and len(title) > 55 else (title or "?")
        print(f"\n  [{sub}] {title_short}")
        upvotes = float(upvotes or 0)
        comments = float(comments or 0)
        vel = float(vel or 0)
        print(f"  Current: {state} | {upvotes:.0f} upvotes | {comments:.0f} comments | {vel:.0f}/hr velocity")

        sent = post_sentiments.get(post_id)
        pred = format_prediction(sub, int(upvotes), int(comments), sent)

        predictions.append({
            "post_id": post_id,
            "subreddit": sub,
            "title": title_short,
            "current_state": state,
            "current_upvotes": upvotes,
            "current_comments": comments,
            "velocity": vel,
            "pop_probability": pred["pop_probability"],
            "flop_probability": pred["flop_probability"],
            "estimated_peak": pred["estimated_peak_upvotes"],
            "peak_range_low": pred["peak_range_low"],
            "peak_range_high": pred["peak_range_high"],
            "growth_multiplier": pred["growth_multiplier"],
            "alive_hours": pred["estimated_alive_hours"],
            "peak_hour": pred["estimated_peak_hour"],
            "sentiment_effect": pred["sentiment_effect"],
        })

    # Also run example predictions for the report
    print(f"\n\n{'=' * 70}")
    print("EXAMPLE PREDICTIONS (for report)")
    print(f"{'=' * 70}")

    examples = [
        ("news", 5, 0, None, "Fresh post, no traction yet"),
        ("news", 50, 10, 0.1, "Moderate start, positive comments"),
        ("news", 500, 80, -0.3, "Hot start, angry comments"),
        ("politics", 10, 3, -0.2, "Typical politics post, slightly negative"),
        ("politics", 200, 40, -0.5, "Controversial politics post"),
        ("technology", 20, 5, 0.2, "Small tech post, positive reaction"),
        ("technology", 300, 50, -0.1, "Trending tech post"),
        ("worldnews", 100, 20, -0.2, "Breaking news, negative sentiment"),
        ("worldnews", 1000, 200, -0.4, "Major story, very negative"),
    ]

    for sub, up, com, sent, desc in examples:
        print(f"\n  --- {desc} ---")
        format_prediction(sub, up, com, sent)

    # Save
    path = os.path.join(OUT_DIR, "post_outcome_predictions.csv")
    if predictions:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
            w.writeheader()
            w.writerows(predictions)
        print(f"\n\n  Saved: {path} ({len(predictions)} predictions)")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
