"""Test if best-post features predict keyword explosion better than averages."""
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
           l.max_upvotes, l.latest_activity_state, l.subreddit,
           l.last_upvote_velocity_per_hour, l.max_comments
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

# Build per-keyword per-day with individual post features
day_kw_posts = defaultdict(lambda: defaultdict(list))
seen = set()

for pid, title, day, max_up, state, sub, vel, max_com in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)

    title_sent = vader.polarity_scores(title)["compound"]
    has_shock = 1 if any(w in title.lower() for w in ["shocking", "unprecedented", "breaking", "explosive", "stunning"]) else 0
    has_question = 1 if "?" in title else 0

    pc = post_comments.get(pid, [])
    if pc:
        com_ups = [c["upvotes"] for c in pc]
        post_gini = gini(com_ups)
        post_sent = np.mean([c["sentiment"] for c in pc])
        post_neg = sum(1 for c in pc if c["sentiment"] < -0.05) / max(1, len(pc))
        post_com_count = len(pc)
    else:
        post_gini = 0
        post_sent = 0
        post_neg = 0
        post_com_count = 0

    post_data = {
        "up": max_up or 0, "state": state, "sub": sub,
        "vel": vel or 0, "comments": max_com or 0,
        "gini": post_gini, "comment_sent": post_sent, "neg_share": post_neg,
        "com_count": post_com_count, "title_sent": title_sent,
        "has_shock": has_shock, "has_question": has_question,
        "surging": 1 if state == "surging" else 0,
        "alive": 1 if state in ("surging", "alive") else 0,
        "title_len": len(title),
    }

    words = re.findall(r"[a-z]+", title.lower())
    for w in set(words):
        if len(w) > 3 and w not in STOPWORDS:
            day_kw_posts[day][w].append(post_data)

days = sorted(day_kw_posts.keys())
print("Days: %d" % len(days))

# Build 4 feature sets
X_simple, X_avg, X_best, X_all, y = [], [], [], [], []

for kw in set(w for d in day_kw_posts.values() for w in d):
    for i in range(len(days) - 3):
        posts = day_kw_posts[days[i]].get(kw, [])
        if not posts or len(posts) < 1 or len(posts) > 5:
            continue

        max_future = max(
            len(day_kw_posts[days[j]].get(kw, []))
            for j in range(i + 1, min(i + 4, len(days)))
        )
        exploded = 1 if max_future >= 10 else 0
        prev = sum(1 for j in range(i) if day_kw_posts[days[j]].get(kw))

        n = len(posts)
        total_up = sum(p["up"] for p in posts)
        subs = len(set(p["sub"] for p in posts))

        # Simple
        X_simple.append([n, total_up, subs, prev])

        # Average
        has_comments = [p for p in posts if p["com_count"] > 0]
        avg_vel = np.mean([p["vel"] for p in posts])
        avg_gini = np.mean([p["gini"] for p in has_comments]) if has_comments else 0
        avg_sent = np.mean([p["comment_sent"] for p in has_comments]) if has_comments else 0
        avg_title_sent = np.mean([p["title_sent"] for p in posts])
        avg_alive = np.mean([p["alive"] for p in posts])

        X_avg.append([n, total_up, subs, prev, avg_vel, avg_gini, avg_sent, avg_title_sent, avg_alive])

        # Best post
        best = max(posts, key=lambda p: p["up"])
        best_vel = max(p["vel"] for p in posts)
        best_gini = max((p["gini"] for p in posts), default=0)
        best_com = max(p["com_count"] for p in posts)
        best_surging = max(p["surging"] for p in posts)
        neg_sents = [p["comment_sent"] for p in has_comments]
        most_neg = min(neg_sents) if neg_sents else 0

        X_best.append([n, total_up, subs, prev,
                       best["up"], best_vel, best_gini, best_com,
                       best["title_sent"], best_surging, most_neg])

        # All combined
        up_spread = max(p["up"] for p in posts) - min(p["up"] for p in posts)
        any_shock = max(p["has_shock"] for p in posts)
        any_question = max(p["has_question"] for p in posts)

        X_all.append([n, total_up, subs, prev,
                      avg_vel, avg_gini, avg_sent, avg_title_sent, avg_alive,
                      best["up"], best_vel, best_gini, best_com,
                      best["title_sent"], best_surging, most_neg,
                      up_spread, any_shock, any_question,
                      np.mean([p["title_len"] for p in posts])])

        y.append(exploded)

print("Samples: %d  Exploded: %d (%.1f%%)" % (len(y), sum(y), sum(y)/len(y)*100))
print()

print("=" * 70)
print("COMPARISON: Simple vs Average vs Best-Post vs All")
print("=" * 70)

for name, X_arr in [("Simple (counts only)", X_simple),
                     ("Average post features", X_avg),
                     ("Best post features", X_best),
                     ("All combined", X_all)]:
    X = np.array(X_arr)
    ya = np.array(y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    roc = cross_val_score(rf, X, ya, cv=5, scoring="roc_auc")
    print("  %-25s ROC = %.3f (+/-%.3f)" % (name, roc.mean(), roc.std()))

# Feature importance for best-post model
print()
print("BEST-POST model features:")
rf_best = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_best.fit(np.array(X_best), np.array(y))
feat_best = ["posts", "total_up", "subs", "prev_days",
             "best_up", "best_vel", "best_gini", "best_comments",
             "best_title_sent", "any_surging", "most_neg_sent"]
for name, imp in sorted(zip(feat_best, rf_best.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.01:
        bar = "#" * int(imp * 30)
        print("  %-20s %.3f  %s" % (name, imp, bar))

# Feature importance for ALL model
print()
print("ALL COMBINED features:")
rf_all = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_all.fit(np.array(X_all), np.array(y))
feat_all = ["posts", "total_up", "subs", "prev_days",
            "avg_vel", "avg_gini", "avg_sent", "avg_title_sent", "avg_alive",
            "best_up", "best_vel", "best_gini", "best_comments",
            "best_title_sent", "any_surging", "most_neg_sent",
            "upvote_spread", "has_shock", "has_question", "avg_title_len"]
for name, imp in sorted(zip(feat_all, rf_all.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.01:
        bar = "#" * int(imp * 30)
        print("  %-20s %.3f  %s" % (name, imp, bar))

conn.close()
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
