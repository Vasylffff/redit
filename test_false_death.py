"""Predict: when a topic drops, is it truly dead or just a gap?

When a topic goes from 2+ posts to <2 posts, predict if it will revive.
Features: how big was it, how fast did it drop, how much engagement, etc.
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [],
    "velocities": [], "comment_vels": [], "titles": []
}))
seen = set()
for pid, title, day, max_up, sub, max_com, vel, cvel in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = sorted(set(w for w in re.findall(r"[a-z]+", title.lower())
                       if len(w) > 4 and w not in STOPWORDS))
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            pair = words[i] + "+" + words[j]
            d = day_pair[day][pair]
            d["posts"] += 1
            d["total_up"] += (max_up or 0)
            d["subs"].add(sub)
            d["comments"].append(max_com or 0)
            d["velocities"].append(vel or 0)
            d["comment_vels"].append(cvel or 0)
            if len(d["titles"]) < 2:
                d["titles"].append(title[:80].encode("ascii", "replace").decode())

days = sorted(day_pair.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))


# Find all drop events and build features
X_tr, y_tr, X_te, y_te, info_te = [], [], [], [], []

all_pairs = set()
for d in days:
    all_pairs.update(day_pair[d].keys())

for pair in all_pairs:
    counts = [day_pair[d].get(pair, {}).get("posts", 0) for d in days]
    total = sum(counts)
    if total < 5:
        continue

    for i in range(1, len(counts)):
        # Drop event: was 2+ posts, now <2
        if counts[i - 1] < 2 or counts[i] >= 2:
            continue

        # Will it revive in next 3 days?
        revives = 0
        for j in range(i + 1, min(i + 4, len(counts))):
            if counts[j] >= 2:
                revives = 1
                break

        # Need at least 3 days after drop to know the answer
        if i + 3 >= len(counts):
            continue

        # Features from the day BEFORE the drop (last healthy day)
        prev_day = days[i - 1]
        d = day_pair[prev_day].get(pair, {})
        if not d or d["posts"] < 1:
            continue

        vels = d.get("velocities", [])
        cvels = d.get("comment_vels", [])
        coms = d.get("comments", [])
        bc = max(coms) if coms else 0
        ac = sum(coms) / max(1, len(coms))
        bv = max(vels) if vels else 0
        av = sum(vels) / max(1, len(vels))
        bcv = max(cvels) if cvels else 0

        # Historical features
        peak_so_far = max(counts[:i])
        days_alive = sum(1 for c in counts[:i] if c >= 2)
        avg_posts = np.mean([c for c in counts[:i] if c >= 1]) if any(c >= 1 for c in counts[:i]) else 0

        # How sharp was the drop?
        if i >= 2 and counts[i - 2] > 0:
            trend_2d = counts[i - 1] / max(1, counts[i - 2])
        else:
            trend_2d = 1.0

        feats = [
            d["posts"],               # posts on last healthy day
            d["total_up"],            # upvotes on last healthy day
            len(d.get("subs", set())), # subreddits on last healthy day
            bc, ac,                    # comment signals
            bv, av, bcv,              # velocity signals
            peak_so_far,              # how big this topic got
            days_alive,               # how many days it was active
            avg_posts,                # average daily posts when active
            trend_2d,                 # was it already declining?
            d["total_up"] / max(1, d["posts"]),  # up per post
        ]

        phase = "train" if days[i] < days[split] else "test"
        if phase == "train":
            X_tr.append(feats)
            y_tr.append(revives)
        else:
            X_te.append(feats)
            y_te.append(revives)
            info_te.append({
                "pair": pair, "drop_day": days[i],
                "prev_posts": d["posts"], "peak": peak_so_far,
                "days_alive": days_alive, "up": d["total_up"],
                "revives": revives, "trend": trend_2d,
                "title": d["titles"][0] if d["titles"] else "?",
            })

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_te, y_te = np.array(X_te), np.array(y_te)

feat_names = ["last_posts", "last_upvotes", "last_subs", "best_comments",
              "avg_comments", "best_velocity", "avg_velocity", "best_comment_vel",
              "peak_posts", "days_alive", "avg_daily_posts", "2day_trend", "up_per_post"]

print("\n" + "=" * 70)
print("  PREDICT: TRUE DEATH vs FALSE DEATH (will it revive?)")
print("=" * 70)
print("\nTrain: %d drops, %d revive (%.1f%%)" % (len(y_tr), sum(y_tr), sum(y_tr) / max(1, len(y_tr)) * 100))
print("Test:  %d drops, %d revive (%.1f%%)" % (len(y_te), sum(y_te), sum(y_te) / max(1, len(y_te)) * 100))

if sum(y_tr) >= 10 and sum(y_te) > 0 and sum(y_te) < len(y_te):
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, probs)

    print("\nROC AUC: %.3f" % roc)

    print("\nFeature importance:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))

    for i, inf in enumerate(info_te):
        inf["prob"] = probs[i]

    # Correct: model said revive, it revived
    print("\nCORRECT REVIVAL PREDICTIONS (said it will come back, it did):")
    print("%-25s %5s %5s %5s %5s %5s %s" % (
        "Topic", "Last", "Peak", "Days", "Trend", "Prob", "Title"))
    print("-" * 90)
    correct_revive = sorted([p for p in info_te if p["revives"] and p["prob"] > 0.3],
                            key=lambda x: -x["prob"])
    for p in correct_revive[:15]:
        print("%-25s %5d %5d %5d %4.1fx %4.0f%%  %s" % (
            p["pair"], p["prev_posts"], p["peak"], p["days_alive"],
            p["trend"], p["prob"] * 100, p["title"][:35]))

    # Correct: model said truly dead, it was
    print("\nCORRECT DEATH PREDICTIONS (said it's dead, it was):")
    correct_dead = sorted([p for p in info_te if not p["revives"] and p["prob"] < 0.2],
                          key=lambda x: x["prob"])
    for p in correct_dead[:10]:
        print("%-25s %5d %5d %5d %4.1fx %4.0f%%  %s" % (
            p["pair"], p["prev_posts"], p["peak"], p["days_alive"],
            p["trend"], p["prob"] * 100, p["title"][:35]))

    # False deaths: model said dead but it revived (missed revival)
    print("\nMISSED REVIVALS (model said dead, but it came back):")
    missed = sorted([p for p in info_te if p["revives"] and p["prob"] < 0.15],
                    key=lambda x: -x["peak"])
    for p in missed[:10]:
        print("%-25s %5d %5d %5d %4.1fx %4.0f%%  %s" % (
            p["pair"], p["prev_posts"], p["peak"], p["days_alive"],
            p["trend"], p["prob"] * 100, p["title"][:35]))

    # Practical: at different thresholds, how well can we separate?
    print("\n" + "=" * 70)
    print("  PRACTICAL FILTERING")
    print("=" * 70)
    print("\n  If model says revival probability > threshold, keep watching:")
    print("  %-12s %8s %8s %8s %8s" % ("Threshold", "Flagged", "Correct", "Precis.", "Recall"))
    print("  " + "-" * 50)
    total_revives = sum(y_te)
    for thresh in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        flagged = sum(1 for p in info_te if p["prob"] > thresh)
        correct = sum(1 for p in info_te if p["prob"] > thresh and p["revives"])
        precision = correct / max(1, flagged) * 100
        recall = correct / max(1, total_revives) * 100
        print("  >%-10.0f%% %8d %8d %7.0f%% %7.0f%%" % (
            thresh * 100, flagged, correct, precision, recall))

conn.close()
print("\nDONE")
