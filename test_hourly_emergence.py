"""Test if quality signals help predict keyword explosion at different observation windows."""
import sqlite3, re, os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www'}

conn = sqlite3.connect(DB_PATH, timeout=30)

# Use daily data but track CUMULATIVE history
rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit,
           l.last_upvote_velocity_per_hour
    FROM post_snapshots p
    JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

comment_rows = conn.execute("""
    SELECT post_id, upvotes_at_snapshot, body
    FROM comment_snapshots
    WHERE body IS NOT NULL AND body != '' AND body != '[deleted]' AND body != '[removed]'
""").fetchall()

post_comments = defaultdict(list)
for pid, cup, body in comment_rows:
    post_comments[pid].append({"upvotes": cup or 0, "sentiment": vader.polarity_scores(body)["compound"]})

def gini(vals):
    if not vals or len(vals) < 2:
        return 0
    s = sorted(vals)
    n = len(s)
    t = sum(s)
    if t == 0:
        return 0
    return sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(s)) / (n * t)

# Build per-day keyword data
day_kw = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "alive": 0, "surging": 0, "subs": set(),
    "ginis": [], "sentiments": [], "velocities": [], "post_ids": []
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
            if state in ("surging", "alive"):
                d["alive"] += 1
            if state == "surging":
                d["surging"] += 1
            d["subs"].add(sub)
            if vel:
                d["velocities"].append(vel)
            d["post_ids"].append(pid)
            pc = post_comments.get(pid)
            if pc:
                ups = [c["upvotes"] for c in pc]
                d["ginis"].append(gini(ups))
                d["sentiments"].append(np.mean([c["sentiment"] for c in pc]))

days = sorted(day_kw.keys())
print("Days: %d" % len(days))

# Test with CUMULATIVE observation windows
# After seeing a keyword for N days with cumulative X posts, will it explode?
for min_days_observed in [2, 3, 4, 5]:
    print("\n" + "=" * 70)
    print("KEYWORDS OBSERVED %d+ DAYS, still small (cumulative 3-15 posts)" % min_days_observed)
    print("Does quality predict explosion in next 3 days?")
    print("=" * 70)

    X_simple, X_enhanced, y = [], [], []
    all_kws = set(w for d in day_kw.values() for w in d)

    for kw in all_kws:
        kw_days = []
        for day in days:
            d = day_kw[day].get(kw)
            if d and d["posts"] > 0:
                kw_days.append((day, d))

        # For each observation point after min_days_observed
        for obs_idx in range(min_days_observed - 1, len(kw_days)):
            day_idx = days.index(kw_days[obs_idx][0])

            # Need future data
            if day_idx + 3 >= len(days):
                continue

            # Cumulative stats up to this point
            history = kw_days[:obs_idx + 1]
            cum_posts = sum(d["posts"] for _, d in history)

            # Only look at still-small keywords (3-15 cumulative posts)
            if cum_posts < 3 or cum_posts > 15:
                continue

            cum_up = sum(d["total_up"] for _, d in history)
            cum_alive = sum(d["alive"] for _, d in history)
            cum_surging = sum(d["surging"] for _, d in history)
            cum_subs = set()
            for _, d in history:
                cum_subs.update(d["subs"])

            days_seen = len(history)
            avg_up_per_post = cum_up / max(1, cum_posts)
            alive_rate = cum_alive / max(1, cum_posts)
            surge_rate = cum_surging / max(1, cum_posts)
            posts_per_day = cum_posts / max(1, days_seen)

            # Quality from all observed days
            all_ginis = [g for _, d in history for g in d["ginis"]]
            all_sents = [s for _, d in history for s in d["sentiments"]]
            all_vels = [v for _, d in history for v in d["velocities"]]

            avg_gini = np.mean(all_ginis) if all_ginis else 0
            avg_sent = np.mean(all_sents) if all_sents else 0
            avg_vel = np.mean(all_vels) if all_vels else 0

            # Recent trend (last 2 days)
            if len(history) >= 2:
                recent_growth = (history[-1][1]["posts"] - history[-2][1]["posts"]) / max(1, history[-2][1]["posts"])
            else:
                recent_growth = 0

            # Target: does it explode in next 3 days? (any day > 2x cumulative average)
            future_max = 0
            for j in range(day_idx + 1, min(day_idx + 4, len(days))):
                fd = day_kw[days[j]].get(kw)
                if fd:
                    future_max = max(future_max, fd["posts"])
            exploded = 1 if future_max > posts_per_day * 3 else 0

            X_simple.append([cum_posts, cum_up, len(cum_subs), days_seen, posts_per_day])
            X_enhanced.append([
                cum_posts, cum_up, len(cum_subs), days_seen, posts_per_day,
                avg_gini, avg_sent, avg_vel, alive_rate, surge_rate,
                avg_up_per_post, recent_growth,
            ])
            y.append(exploded)

    if len(y) < 100:
        print("Not enough data (%d)" % len(y))
        continue

    X_s = np.array(X_simple)
    X_e = np.array(X_enhanced)
    y = np.array(y)

    print("Samples: %d  Exploded: %d (%.1f%%)" % (len(y), sum(y), sum(y) / len(y) * 100))

    rf_s = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_e = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    roc_s = cross_val_score(rf_s, X_s, y, cv=5, scoring="roc_auc")
    roc_e = cross_val_score(rf_e, X_e, y, cv=5, scoring="roc_auc")

    print("Simple (counts):    ROC = %.3f" % roc_s.mean())
    print("Enhanced (quality): ROC = %.3f" % roc_e.mean())
    diff = roc_e.mean() - roc_s.mean()
    print("Improvement:        %+.3f %s" % (diff, "***" if diff > 0.02 else ("*" if diff > 0.01 else "")))

    # Show what quality features matter
    rf_e.fit(X_e, y)
    feat_names = ["cum_posts", "cum_up", "subs", "days_seen", "posts_per_day",
                  "gini", "sentiment", "velocity", "alive_rate", "surge_rate",
                  "avg_up_per_post", "recent_growth"]
    print("Top features:")
    for name, imp in sorted(zip(feat_names, rf_e.feature_importances_), key=lambda x: -x[1])[:6]:
        bar = "#" * int(imp * 30)
        print("  %-18s %.3f  %s" % (name, imp, bar))

conn.close()
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
