"""Temporal validation with co-occurrence pairs (real topics).
Two tiers:
  Tier 1 (emergence):  catch at 1-3 posts, predict growth to 5+
  Tier 2 (escalation): catch at 4-7 posts, predict growth to 10+
"""
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

day_pair_data = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [], "titles": []
}))
seen = set()

for pid, title, day, max_up, state, sub, vel, comments in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = sorted(set(w for w in re.findall(r"[a-z]+", title.lower()) if len(w) > 4 and w not in STOPWORDS))
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            pair = words[i] + "+" + words[j]
            d = day_pair_data[day][pair]
            d["posts"] += 1
            d["total_up"] += (max_up or 0)
            d["subs"].add(sub)
            d["comments"].append(comments or 0)
            if len(d["titles"]) < 2:
                d["titles"].append(title[:70])

days = sorted(day_pair_data.keys())
split = 8

print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))


def build_data(day_range, min_posts=1, max_posts=3, growth_target=5):
    X, y, info = [], [], []
    for day in day_range:
        day_idx = days.index(day)
        if day_idx + 3 >= len(days):
            continue
        for pair, d in day_pair_data[day].items():
            if d["posts"] < min_posts or d["posts"] > max_posts:
                continue
            max_future = max(
                day_pair_data[days[j]].get(pair, {}).get("posts", 0)
                for j in range(day_idx + 1, min(day_idx + 4, len(days)))
            )
            grew = 1 if max_future >= growth_target else 0
            best_comments = max(d["comments"]) if d["comments"] else 0
            avg_comments = sum(d["comments"]) / max(1, len(d["comments"]))

            X.append([
                d["posts"],
                d["total_up"],
                len(d["subs"]),
                best_comments,
                avg_comments,
                d["total_up"] / max(1, d["posts"]),
            ])
            y.append(grew)
            info.append({
                "pair": pair, "day": day, "posts": d["posts"],
                "up": d["total_up"], "comments": best_comments,
                "subs": len(d["subs"]), "grew": grew,
                "future_peak": max_future,
                "title": d["titles"][0] if d["titles"] else "?",
            })
    return np.array(X), np.array(y), info


def run_tier(tier_name, min_posts, max_posts, growth_target):
    print("\n" + "#" * 70)
    print("  TIER: %s  (catch at %d-%d posts, predict growth to %d+)" % (
        tier_name, min_posts, max_posts, growth_target))
    print("#" * 70)

    X_train, y_train, _ = build_data(days[:split], min_posts, max_posts, growth_target)
    X_test, y_test, test_info = build_data(days[split:], min_posts, max_posts, growth_target)

    if len(y_train) == 0 or len(y_test) == 0:
        print("  Not enough data for this tier (train=%d, test=%d)" % (len(y_train), len(y_test)))
        return None

    print("\nTrain: %d pairs, %d grew (%.2f%%)" % (len(y_train), sum(y_train), sum(y_train) / len(y_train) * 100))
    print("Test:  %d pairs, %d grew (%.2f%%)" % (len(y_test), sum(y_test), sum(y_test) / len(y_test) * 100))

    if sum(y_train) < 5 or sum(y_test) == 0:
        print("  Too few positive examples to train/evaluate (train_pos=%d, test_pos=%d)" % (
            sum(y_train), sum(y_test)))
        return None

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    test_probs = rf.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, test_probs)

    feature_names = ["posts", "total_upvotes", "subreddit_count", "best_comments", "avg_comments", "upvotes_per_post"]
    importances = rf.feature_importances_

    print("\n" + "=" * 70)
    print("TEMPORAL VALIDATION — %s" % tier_name)
    print("ROC AUC: %.3f" % test_roc)
    print("=" * 70)

    print("\nFeature importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print("  %-20s %.1f%%" % (name, imp * 100))

    for i, info in enumerate(test_info):
        info["prob"] = test_probs[i]

    print("\nCORRECT PREDICTIONS (model said grow, it grew):")
    print("%-30s %5s %6s %6s %5s %6s %s" % ("Topic", "Posts", "UpK", "Coms", "Prob", "Peak", "Title"))
    print("-" * 100)
    correct = sorted([p for p in test_info if p["grew"] and p["prob"] > 0.05], key=lambda x: -x["prob"])
    for p in correct[:15]:
        print("%-30s %5d %6dK %6d %4.0f%% %6d %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["comments"],
            p["prob"] * 100, p["future_peak"], p["title"][:40]))

    print("\nFALSE ALARMS (model said grow, didn't):")
    false_alarms = sorted([p for p in test_info if not p["grew"] and p["prob"] > 0.1], key=lambda x: -x["prob"])
    for p in false_alarms[:10]:
        print("%-30s %5d %6dK %6d %4.0f%% %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["comments"],
            p["prob"] * 100, p["title"][:45]))

    print("\nMISSED (grew but low probability):")
    missed = sorted([p for p in test_info if p["grew"] and p["prob"] < 0.05], key=lambda x: -x["future_peak"])
    for p in missed[:10]:
        print("%-30s %5d %6dK %6d %4.0f%% grew->%d %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["comments"],
            p["prob"] * 100, p["future_peak"], p["title"][:35]))

    return test_roc


# --- Run both tiers ---
roc_tier1 = run_tier("EMERGENCE (1-3 -> 5+)", min_posts=1, max_posts=3, growth_target=5)
roc_tier2 = run_tier("ESCALATION (4-7 -> 10+)", min_posts=4, max_posts=7, growth_target=10)

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
if roc_tier1 is not None:
    print("  Tier 1 -- Emergence  (1-3 -> 5+):  ROC AUC = %.3f" % roc_tier1)
if roc_tier2 is not None:
    print("  Tier 2 -- Escalation (4-7 -> 10+): ROC AUC = %.3f" % roc_tier2)
if roc_tier1 is None and roc_tier2 is None:
    print("  Not enough data for either tier")
print("=" * 70)

conn.close()
print("\nDONE")
