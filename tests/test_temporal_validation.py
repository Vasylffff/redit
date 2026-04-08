"""Temporal validation: train on days 1-8, test on days 9-12."""
import sqlite3, re, os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)

rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit,
           l.last_upvote_velocity_per_hour, l.max_comments
    FROM post_snapshots p JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

day_kw_count = defaultdict(lambda: defaultdict(int))
day_kw_posts = defaultdict(lambda: defaultdict(list))
seen = set()

for pid, title, day, max_up, state, sub, vel, comments in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = set(w for w in re.findall(r"[a-z]+", title.lower()) if len(w) > 4 and w not in STOPWORDS)
    for w in words:
        day_kw_count[day][w] += 1
        day_kw_posts[day][w].append({
            "up": max_up or 0, "vel": vel or 0,
            "comments": comments or 0, "state": state, "title": title,
        })

days = sorted(day_kw_count.keys())
split = 8
print("Train: %s to %s (%d days)" % (days[0], days[split-1], split))
print("Test:  %s to %s (%d days)" % (days[split], days[-1], len(days) - split))

# Build training data
def build_data(day_range):
    X, y, info = [], [], []
    for day in day_range:
        day_idx = days.index(day)
        for kw in day_kw_count[day]:
            count = day_kw_count[day][kw]
            if count < 1 or count > 3:
                continue
            if day_idx + 3 >= len(days):
                continue
            posts = day_kw_posts[day][kw]
            best = max(posts, key=lambda p: p["up"])
            max_future = max(day_kw_count[days[j]].get(kw, 0) for j in range(day_idx + 1, min(day_idx + 4, len(days))))
            grew = 1 if max_future >= 5 else 0
            X.append([count, best["up"], best["vel"], best["comments"]])
            y.append(grew)
            info.append({"kw": kw, "day": day, "count": count, "up": best["up"],
                         "comments": best["comments"], "grew": grew,
                         "future_peak": max_future, "title": best["title"][:60]})
    return np.array(X), np.array(y), info

X_train, y_train, _ = build_data(days[:split])
X_test, y_test, test_info = build_data(days[split:])

print("\nTrain: %d samples, %d grew (%.1f%%)" % (len(y_train), sum(y_train), sum(y_train)/len(y_train)*100))
print("Test:  %d samples, %d grew (%.1f%%)" % (len(y_test), sum(y_test), sum(y_test)/len(y_test)*100))

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

test_probs = rf.predict_proba(X_test)[:, 1]
test_roc = roc_auc_score(y_test, test_probs)
print("\n" + "=" * 70)
print("TEMPORAL VALIDATION ROC AUC: %.3f" % test_roc)
print("(trained on past, tested on future)")
print("=" * 70)

# Add probabilities to test info
for i, info in enumerate(test_info):
    info["prob"] = test_probs[i]

# Top predictions
print("\nTOP PREDICTIONS (model thinks will grow):")
print("%-20s %5s %6s %6s %5s %8s %s" % ("Keyword", "Posts", "UpK", "Coms", "Prob", "Actually", "Title"))
print("-" * 90)

sorted_test = sorted(test_info, key=lambda x: -x["prob"])
for p in sorted_test[:20]:
    result = "GREW->%d" % p["future_peak"] if p["grew"] else "no"
    print("%-20s %5d %6dK %6d %4.0f%% %8s %s" % (
        p["kw"], p["count"], p["up"] // 1000, p["comments"],
        p["prob"] * 100, result, p["title"][:45]))

# Bottom predictions (model thinks won't grow)
print("\nBOTTOM PREDICTIONS (model thinks won't grow):")
for p in sorted_test[-10:]:
    result = "GREW->%d" % p["future_peak"] if p["grew"] else "no"
    print("%-20s %5d %6dK %6d %4.0f%% %8s %s" % (
        p["kw"], p["count"], p["up"] // 1000, p["comments"],
        p["prob"] * 100, result, p["title"][:45]))

# Hits and misses
hits = sum(1 for p in test_info if (p["prob"] > 0.1 and p["grew"]) or (p["prob"] <= 0.1 and not p["grew"]))
print("\nHit rate (threshold 10%%): %d/%d = %.1f%%" % (hits, len(test_info), hits/len(test_info)*100))

# Show missed topics (grew but low probability)
missed = [p for p in test_info if p["grew"] and p["prob"] < 0.1]
if missed:
    print("\nMISSED TOPICS (grew but model gave low probability):")
    for p in sorted(missed, key=lambda x: -x["future_peak"])[:10]:
        print("  %-20s %dK up, %d coms, prob=%.0f%%, grew to %d" % (
            p["kw"], p["up"]//1000, p["comments"], p["prob"]*100, p["future_peak"]))

conn.close()
print("\nDONE")
