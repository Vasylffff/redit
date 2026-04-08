"""Predict topic trajectory using post-level quality signals."""
import sqlite3, re, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during'}

conn = sqlite3.connect(DB_PATH, timeout=30)

rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit,
           l.last_upvote_velocity_per_hour
    FROM post_snapshots p
    JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

# Comment quality per post
comment_rows = conn.execute("""
    SELECT post_id, upvotes_at_snapshot, body
    FROM comment_snapshots
    WHERE body IS NOT NULL AND body != '' AND body != '[deleted]' AND body != '[removed]'
""").fetchall()

post_comments = defaultdict(list)
for pid, cup, body in comment_rows:
    score = vader.polarity_scores(body)["compound"]
    post_comments[pid].append({"upvotes": cup or 0, "sentiment": score})

def gini(vals):
    if not vals or len(vals) < 2:
        return 0
    s = sorted(vals)
    n = len(s)
    t = sum(s)
    if t == 0:
        return 0
    return sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(s)) / (n * t)

post_quality = {}
for pid, comments in post_comments.items():
    ups = [c["upvotes"] for c in comments]
    sents = [c["sentiment"] for c in comments]
    total_w = sum(max(1, u) for u in ups)
    post_quality[pid] = {
        "gini": gini(ups),
        "sentiment": sum(s * max(1, u) for s, u in zip(sents, ups)) / max(1, total_w),
        "comment_count": len(comments),
        "neg_share": sum(1 for s in sents if s < -0.05) / max(1, len(sents)),
    }

print("Scored quality for %d posts" % len(post_quality))

# Daily topic metrics with quality
day_kw = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "surging": 0, "alive": 0, "dead": 0,
    "ginis": [], "sentiments": [], "velocities": [], "neg_shares": []
}))

seen = set()
for pid, title, day, max_up, state, sub, vel in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = re.findall(r"[a-z]+", title.lower())
    for w in set(words):
        if len(w) > 3 and w not in STOPWORDS:
            d = day_kw[day][w]
            d["posts"] += 1
            d["total_up"] += (max_up or 0)
            if state == "surging":
                d["surging"] += 1
            elif state == "alive":
                d["alive"] += 1
            elif state in ("dead", "dying"):
                d["dead"] += 1
            if vel:
                d["velocities"].append(vel)
            pq = post_quality.get(pid)
            if pq:
                d["ginis"].append(pq["gini"])
                d["sentiments"].append(pq["sentiment"])
                d["neg_shares"].append(pq["neg_share"])

days = sorted(day_kw.keys())
kw_series = defaultdict(list)
for day in days:
    for kw, d in day_kw[day].items():
        if d["posts"] < 3:
            continue
        n = d["posts"]
        kw_series[kw].append({
            "day": day, "posts": n, "total_up": d["total_up"],
            "surge_rate": d["surging"] / n,
            "alive_rate": (d["surging"] + d["alive"]) / n,
            "dead_rate": d["dead"] / n,
            "avg_gini": float(np.mean(d["ginis"])) if d["ginis"] else 0,
            "avg_sentiment": float(np.mean(d["sentiments"])) if d["sentiments"] else 0,
            "avg_velocity": float(np.mean(d["velocities"])) if d["velocities"] else 0,
            "avg_neg_share": float(np.mean(d["neg_shares"])) if d["neg_shares"] else 0,
        })

good = {kw: sorted(s, key=lambda x: x["day"]) for kw, s in kw_series.items() if len(s) >= 5}
print("Topics with 5+ days: %d" % len(good))


def basic_feats(w):
    posts = [x["posts"] for x in w]
    total = [x["total_up"] for x in w]
    return [posts[-1], posts[-2], posts[-3], total[-1], total[-2],
            (posts[-1] - posts[0]) / max(1, posts[0]), sum(posts) / 3]


def enhanced_feats(w):
    f = basic_feats(w)
    f.extend([
        w[-1]["surge_rate"],
        w[-1]["alive_rate"],
        w[-1]["dead_rate"],
        w[-1]["avg_gini"],
        w[-1]["avg_sentiment"],
        w[-1]["avg_velocity"],
        w[-1]["avg_neg_share"],
        w[-1]["surge_rate"] - w[0]["surge_rate"],
        w[-1]["avg_gini"] - w[0]["avg_gini"],
        w[-1]["avg_sentiment"] - w[0]["avg_sentiment"],
    ])
    return f


print("\n" + "=" * 70)
print("BASIC vs ENHANCED TOPIC PREDICTION")
print("=" * 70)

for days_ahead in [1, 2, 3]:
    X_basic, X_enh, y = [], [], []
    for kw, series in good.items():
        for i in range(3, len(series) - days_ahead + 1):
            w = series[i - 3:i]
            target = series[i + days_ahead - 1]
            X_basic.append(basic_feats(w))
            X_enh.append(enhanced_feats(w))
            y.append(target["total_up"])

    X_b = np.array(X_basic)
    X_e = np.array(X_enh)
    y = np.array(y)

    rf_b = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_e = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)

    r2_b = cross_val_score(rf_b, X_b, y, cv=5, scoring="r2")
    r2_e = cross_val_score(rf_e, X_e, y, cv=5, scoring="r2")

    diff = r2_e.mean() - r2_b.mean()
    print("  %dd: Basic R2=%.3f  Enhanced R2=%.3f  Improvement: %+.3f" % (
        days_ahead, r2_b.mean(), r2_e.mean(), diff))

# Feature importance
print("\n" + "=" * 70)
print("WHAT DRIVES TOPIC POPULARITY? (Feature Importance)")
print("=" * 70)

X_all, y_all = [], []
for kw, series in good.items():
    for i in range(3, len(series)):
        w = series[i - 3:i]
        X_all.append(enhanced_feats(w))
        y_all.append(series[i]["total_up"])

rf_final = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_final.fit(np.array(X_all), np.array(y_all))

feat_names = [
    "posts_yesterday", "posts_2d_ago", "posts_3d_ago",
    "upvotes_yesterday", "upvotes_2d_ago",
    "post_momentum", "avg_daily_posts",
    "surge_rate", "alive_rate", "dead_rate",
    "comment_gini", "comment_sentiment", "avg_velocity",
    "negative_share", "surge_trend", "gini_trend", "sentiment_trend",
]

print("  %-25s %10s" % ("Feature", "Importance"))
print("  " + "-" * 50)
for name, imp in sorted(zip(feat_names, rf_final.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.005:
        bar = "#" * int(imp * 40)
        print("  %-25s %.3f  %s" % (name, imp, bar))

# Topic predictions with quality context
print("\n" + "=" * 70)
print("TOPIC QUALITY ANALYSIS & PREDICTIONS")
print("=" * 70)

for kw in ["trump", "iran", "war", "ukraine", "russia", "israel", "game",
           "military", "hormuz", "china", "court", "bondi"]:
    series = good.get(kw)
    if not series or len(series) < 4:
        continue
    latest = series[-1]
    prev = series[-2]

    # Predict next day
    w = series[-3:]
    feat = np.array([enhanced_feats(w)])
    pred_up = rf_final.predict(feat)[0]
    pct_change = (pred_up - latest["total_up"]) / max(1, latest["total_up"])

    print("\n  %s:" % kw.upper())
    print("    Posts: %d | Upvotes: %dK | Surge: %.0f%% | Gini: %.2f | Sentiment: %+.2f | Neg: %.0f%%" % (
        latest["posts"], latest["total_up"] / 1000, latest["surge_rate"] * 100,
        latest["avg_gini"], latest["avg_sentiment"], latest["avg_neg_share"] * 100))

    if latest["surge_rate"] > 0.3 and latest["avg_gini"] > 0.5:
        quality = "HIGH QUALITY - surging posts with focused discussion"
        direction = "RISING"
    elif latest["surge_rate"] > 0.15 and latest["avg_sentiment"] < -0.1:
        quality = "CONTROVERSIAL - active but negative"
        direction = "VOLATILE"
    elif latest["dead_rate"] > 0.5:
        quality = "LOW QUALITY - most posts dying"
        direction = "FADING"
    elif latest["alive_rate"] > 0.5:
        quality = "HEALTHY - majority of posts alive"
        direction = "STABLE"
    else:
        quality = "MIXED - no clear signal"
        direction = "UNCERTAIN"

    print("    Quality: %s" % quality)
    print("    Predicted upvotes tomorrow: %dK (%+.0f%%)" % (pred_up / 1000, pct_change * 100))
    print("    Direction: %s" % direction)

conn.close()
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
