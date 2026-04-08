"""Predict which subreddit a topic will spread to next.

If a word pair appears in subreddit A, will it appear in subreddit B tomorrow?
Uses temporal validation: train on days 1-8, test on days 9-12.
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}
SUBREDDITS = ["news", "politics", "worldnews", "technology", "Games"]

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

# Build: per day, per pair, per subreddit stats
# day -> pair -> sub -> {posts, upvotes, comments, velocity}
day_pair_sub = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "comments": [], "velocities": []
})))
# Also track which subs a pair appears in per day
day_pair_subs = defaultdict(lambda: defaultdict(set))

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
            d = day_pair_sub[day][pair][sub]
            d["posts"] += 1
            d["total_up"] += (max_up or 0)
            d["comments"].append(max_com or 0)
            d["velocities"].append(vel or 0)
            day_pair_subs[day][pair].add(sub)

days = sorted(day_pair_sub.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))


# ================================================================
# TASK 1: Will this topic spread to a NEW subreddit tomorrow?
# ================================================================
print("\n" + "=" * 70)
print("  TASK 1: Will a topic spread to a new subreddit?")
print("  (topic in N subs today -> appears in N+1 subs tomorrow)")
print("=" * 70)

X_tr, y_tr, X_te, y_te, info_te = [], [], [], [], []

for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
    for day in day_range:
        di = days.index(day)
        if di + 1 >= len(days):
            continue
        tomorrow = days[di + 1]
        for pair in day_pair_subs[day]:
            today_subs = day_pair_subs[day][pair]
            tomorrow_subs = day_pair_subs[tomorrow].get(pair, set())
            new_subs = tomorrow_subs - today_subs
            spread = 1 if len(new_subs) > 0 else 0

            # Only consider pairs in 1-3 subs (room to spread)
            if len(today_subs) < 1 or len(today_subs) >= 4:
                continue

            # Features from all subs combined
            total_posts = sum(day_pair_sub[day][pair][s]["posts"] for s in today_subs)
            total_up = sum(day_pair_sub[day][pair][s]["total_up"] for s in today_subs)
            all_comments = []
            all_vels = []
            for s in today_subs:
                all_comments.extend(day_pair_sub[day][pair][s]["comments"])
                all_vels.extend(day_pair_sub[day][pair][s]["velocities"])

            best_com = max(all_comments) if all_comments else 0
            avg_com = sum(all_comments) / max(1, len(all_comments))
            best_vel = max(all_vels) if all_vels else 0
            avg_vel = sum(all_vels) / max(1, len(all_vels))

            feats = [
                total_posts,
                total_up,
                len(today_subs),
                best_com,
                avg_com,
                total_up / max(1, total_posts),
                best_vel,
                avg_vel,
            ]

            if phase == "train":
                X_tr.append(feats)
                y_tr.append(spread)
            else:
                X_te.append(feats)
                y_te.append(spread)
                info_te.append({
                    "pair": pair, "day": day, "posts": total_posts,
                    "up": total_up, "subs": len(today_subs),
                    "spread": spread, "new_subs": new_subs,
                    "today_subs": today_subs,
                })

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_te, y_te = np.array(X_te), np.array(y_te)

print("\nTrain: %d pairs, %d spread (%.2f%%)" % (len(y_tr), sum(y_tr), sum(y_tr) / max(1, len(y_tr)) * 100))
print("Test:  %d pairs, %d spread (%.2f%%)" % (len(y_te), sum(y_te), sum(y_te) / max(1, len(y_te)) * 100))

if sum(y_tr) >= 10 and sum(y_te) > 0:
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, probs)

    feat_names = ["posts", "total_upvotes", "current_subs", "best_comments",
                  "avg_comments", "up_per_post", "best_velocity", "avg_velocity"]
    print("\nROC AUC: %.3f" % roc)
    print("\nFeature importance:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        print("  %-20s %.1f%%" % (name, imp * 100))

    for i, inf in enumerate(info_te):
        inf["prob"] = probs[i]

    print("\nCORRECT -- predicted spread, it spread:")
    print("%-25s %5s %6s %4s %5s  %-20s -> %s" % (
        "Topic", "Posts", "UpK", "Subs", "Prob", "Was in", "Spread to"))
    print("-" * 100)
    correct = sorted([p for p in info_te if p["spread"] and p["prob"] > 0.1], key=lambda x: -x["prob"])
    for p in correct[:15]:
        print("%-25s %5d %6dK %4d %4.0f%%  %-20s -> %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["subs"],
            p["prob"] * 100,
            ",".join(sorted(p["today_subs"])),
            ",".join(sorted(p["new_subs"]))))

    print("\nFALSE ALARMS -- predicted spread, didn't:")
    false = sorted([p for p in info_te if not p["spread"] and p["prob"] > 0.2], key=lambda x: -x["prob"])
    for p in false[:10]:
        print("%-25s %5d %6dK %4d %4.0f%%  in: %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["subs"],
            p["prob"] * 100, ",".join(sorted(p["today_subs"]))))


# ================================================================
# TASK 2: WHICH subreddit will it spread to?
# ================================================================
print("\n" + "=" * 70)
print("  TASK 2: WHICH subreddit will it spread to?")
print("  (per-target-subreddit prediction)")
print("=" * 70)

for target_sub in SUBREDDITS:
    X_tr2, y_tr2, X_te2, y_te2 = [], [], [], []

    for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
        for day in day_range:
            di = days.index(day)
            if di + 1 >= len(days):
                continue
            tomorrow = days[di + 1]
            for pair in day_pair_subs[day]:
                today_subs = day_pair_subs[day][pair]
                # Only pairs NOT already in target sub
                if target_sub in today_subs:
                    continue
                if len(today_subs) < 1:
                    continue

                appeared = 1 if target_sub in day_pair_subs[tomorrow].get(pair, set()) else 0

                total_posts = sum(day_pair_sub[day][pair][s]["posts"] for s in today_subs)
                total_up = sum(day_pair_sub[day][pair][s]["total_up"] for s in today_subs)
                all_comments = []
                all_vels = []
                for s in today_subs:
                    all_comments.extend(day_pair_sub[day][pair][s]["comments"])
                    all_vels.extend(day_pair_sub[day][pair][s]["velocities"])

                best_com = max(all_comments) if all_comments else 0
                best_vel = max(all_vels) if all_vels else 0

                # Which subs is it in? (one-hot for the other 4)
                sub_flags = [1 if s in today_subs else 0 for s in SUBREDDITS if s != target_sub]

                feats = [total_posts, total_up, len(today_subs), best_com,
                         total_up / max(1, total_posts), best_vel] + sub_flags

                if phase == "train":
                    X_tr2.append(feats)
                    y_tr2.append(appeared)
                else:
                    X_te2.append(feats)
                    y_te2.append(appeared)

    X_tr2, y_tr2 = np.array(X_tr2), np.array(y_tr2)
    X_te2, y_te2 = np.array(X_te2), np.array(y_te2)

    if sum(y_tr2) < 10 or sum(y_te2) == 0:
        print("  r/%-12s: not enough data (train_pos=%d, test_pos=%d)" % (
            target_sub, sum(y_tr2), sum(y_te2)))
        continue

    rf2 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf2.fit(X_tr2, y_tr2)
    roc2 = roc_auc_score(y_te2, rf2.predict_proba(X_te2)[:, 1])
    print("  r/%-12s: ROC AUC = %.3f  (test: %d pos / %d total, %.2f%%)" % (
        target_sub, roc2, sum(y_te2), len(y_te2), sum(y_te2) / len(y_te2) * 100))


# ================================================================
# TASK 3: Historical spread patterns
# ================================================================
print("\n" + "=" * 70)
print("  TASK 3: Historical spread patterns")
print("  (which subreddit breaks stories first?)")
print("=" * 70)

first_sub = defaultdict(lambda: defaultdict(int))
spread_pairs = defaultdict(int)
spread_time = defaultdict(list)

for pair in set().union(*[day_pair_subs[d].keys() for d in days]):
    pair_history = []
    for day in days:
        if pair in day_pair_subs[day]:
            for s in day_pair_subs[day][pair]:
                pair_history.append((day, s))

    if not pair_history:
        continue

    seen_subs = set()
    first_day = pair_history[0][0]
    for day, sub in sorted(pair_history):
        if sub not in seen_subs:
            if len(seen_subs) == 0:
                first_sub["first"][sub] += 1
            else:
                for prev in seen_subs:
                    spread_pairs["%s->%s" % (prev, sub)] += 1
                di_first = days.index(first_day)
                di_now = days.index(day)
                spread_time[sub].append(di_now - di_first)
            seen_subs.add(sub)

print("\nWho breaks stories first:")
for sub, count in sorted(first_sub["first"].items(), key=lambda x: -x[1]):
    print("  r/%-12s %d times" % (sub, count))

print("\nTop spread routes:")
for route, count in sorted(spread_pairs.items(), key=lambda x: -x[1])[:15]:
    print("  %-25s %d times" % (route, count))

print("\nAvg days until topic reaches subreddit (from first appearance):")
for sub in SUBREDDITS:
    if spread_time[sub]:
        print("  r/%-12s avg %.1f days (n=%d)" % (
            sub, np.mean(spread_time[sub]), len(spread_time[sub])))

conn.close()
print("\nDONE")
