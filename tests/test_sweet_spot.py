"""Test ratio features for 5-15 post keywords."""
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

rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit, l.last_upvote_velocity_per_hour
    FROM post_snapshots p JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

comment_rows = conn.execute("""
    SELECT post_id, upvotes_at_snapshot, body FROM comment_snapshots
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

day_kw = defaultdict(lambda: defaultdict(list))
seen = set()
for pid, title, day, max_up, state, sub, vel in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    pc = post_comments.get(pid, [])
    pg = gini([c["upvotes"] for c in pc]) if len(pc) >= 2 else 0
    ps = np.mean([c["sentiment"] for c in pc]) if pc else 0
    for w in set(re.findall(r"[a-z]+", title.lower())):
        if len(w) > 3 and w not in STOPWORDS:
            day_kw[day][w].append({
                "up": max_up or 0, "state": state, "sub": sub, "vel": vel or 0,
                "gini": pg, "sent": ps, "coms": len(pc),
                "surging": 1 if state == "surging" else 0,
                "alive": 1 if state in ("surging", "alive") else 0,
            })

days = sorted(day_kw.keys())

X_simple, X_ratios, y = [], [], []

for kw in set(w for d in day_kw.values() for w in d):
    for i in range(len(days) - 3):
        posts = day_kw[days[i]].get(kw, [])
        n = len(posts)
        if n < 5 or n > 15:
            continue

        max_future = max(len(day_kw[days[j]].get(kw, [])) for j in range(i + 1, min(i + 4, len(days))))
        grew = 1 if max_future > n * 1.5 else 0

        prev = sum(1 for j in range(i) if day_kw[days[j]].get(kw))
        total_up = sum(p["up"] for p in posts)
        subs = len(set(p["sub"] for p in posts))

        X_simple.append([n, total_up, subs, prev])

        surging_pct = sum(p["surging"] for p in posts) / n
        alive_pct = sum(p["alive"] for p in posts) / n
        dead_pct = sum(1 for p in posts if p["state"] in ("dead", "dying")) / n

        ups = sorted([p["up"] for p in posts])
        top_post_share = ups[-1] / max(1, total_up)
        top3_share = sum(ups[-3:]) / max(1, total_up) if n >= 3 else 1
        up_ratio = ups[-1] / max(1, ups[0]) if ups[0] > 0 else ups[-1]

        vels = [p["vel"] for p in posts]
        has_fast = 1 if max(vels) > 100 else 0
        vel_spread = max(vels) - min(vels)

        coms_posts = [p for p in posts if p["coms"] > 0]
        high_gini_pct = sum(1 for p in coms_posts if p["gini"] > 0.5) / max(1, len(coms_posts)) if coms_posts else 0
        neg_sent_pct = sum(1 for p in coms_posts if p["sent"] < -0.1) / max(1, len(coms_posts)) if coms_posts else 0

        X_ratios.append([
            n, total_up, subs, prev,
            surging_pct, alive_pct, dead_pct,
            top_post_share, top3_share, up_ratio,
            has_fast, vel_spread,
            high_gini_pct, neg_sent_pct,
        ])
        y.append(grew)

Xs = np.array(X_simple)
Xr = np.array(X_ratios)
ya = np.array(y)

print("5-15 POSTS SWEET SPOT: RATIO FEATURES")
print("=" * 60)
print("Samples: %d  Grew: %d (%.1f%%)" % (len(y), sum(y), sum(y) / len(y) * 100))

roc_s = cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), Xs, ya, cv=5, scoring="roc_auc")
roc_r = cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), Xr, ya, cv=5, scoring="roc_auc")

print("Simple (counts):   ROC = %.3f" % roc_s.mean())
print("Ratio features:    ROC = %.3f" % roc_r.mean())
print("Improvement:       %+.3f" % (roc_r.mean() - roc_s.mean()))

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(Xr, ya)
feat_names = [
    "posts", "total_up", "subs", "prev_days",
    "surging_pct", "alive_pct", "dead_pct",
    "top_post_share", "top3_share", "up_ratio",
    "has_fast_post", "vel_spread",
    "high_gini_pct", "neg_sentiment_pct",
]

print("\nFeature importance:")
for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.01:
        bar = "#" * int(imp * 30)
        print("  %-20s %.3f  %s" % (name, imp, bar))

print("\nPROFILE: Growing vs Not-Growing (5-15 posts)")
print("=" * 60)
grew_idx = [i for i, v in enumerate(y) if v == 1]
not_idx = [i for i, v in enumerate(y) if v == 0]

for j, name in enumerate(feat_names):
    gv = [X_ratios[i][j] for i in grew_idx]
    nv = [X_ratios[i][j] for i in not_idx]
    if gv and nv:
        g_mean = np.mean(gv)
        n_mean = np.mean(nv)
        diff = g_mean - n_mean
        pct_diff = diff / max(abs(n_mean), 0.001) * 100
        star = " <--" if abs(pct_diff) > 10 else ""
        print("  %-20s Grew: %8.2f  Didnt: %8.2f  %+.0f%%%s" % (name, g_mean, n_mean, pct_diff, star))

conn.close()
print("\n" + "=" * 60)
print("DONE")
