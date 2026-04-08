"""Detect when a topic is dying before it actually drops.

Questions:
  1) From the peak day, how many days until the topic dies (< 2 posts)?
  2) Can we detect a topic is dying BEFORE the drop happens?
  3) What signals predict topic death?
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [], "max_single": 0,
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
            d["max_single"] = max(d["max_single"], max_up or 0)
            d["velocities"].append(vel or 0)
            d["comment_vels"].append(cvel or 0)
            if len(d["titles"]) < 2:
                d["titles"].append(title[:80])

days = sorted(day_pair.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))


# Build topic trajectories
pair_trajectories = defaultdict(dict)
for day in days:
    for pair, d in day_pair[day].items():
        pair_trajectories[pair][day] = d


# ================================================================
# TASK 1: From any observation day, will this topic die tomorrow?
# (die = drops to < 2 posts or disappears)
# ================================================================
print("\n" + "=" * 70)
print("  TASK 1: Will this topic die tomorrow?")
print("  (from any day of observation, predict next day < 2 posts)")
print("=" * 70)

X_tr, y_tr, X_te, y_te, info_te = [], [], [], [], []

for pair, daily in pair_trajectories.items():
    pair_days = sorted(daily.keys())
    if len(pair_days) < 2:
        continue
    total = sum(daily[d]["posts"] for d in pair_days)
    if total < 5:
        continue

    for idx in range(len(pair_days) - 1):
        day = pair_days[idx]
        next_day = pair_days[idx + 1]
        di = days.index(day)

        d = daily[day]
        d_next = daily[next_day]

        # Also check if pair simply doesn't appear next day
        next_day_global = days[di + 1] if di + 1 < len(days) else None
        if next_day_global and next_day_global != next_day:
            # Pair skipped a day = effectively dead
            dies_tomorrow = 1
        else:
            dies_tomorrow = 1 if d_next["posts"] < 2 else 0

        # Previous day data (if exists)
        if idx > 0:
            prev = daily[pair_days[idx - 1]]
            post_trend = d["posts"] / max(1, prev["posts"])
            up_trend = d["total_up"] / max(1, prev["total_up"])
        else:
            post_trend = 1.0
            up_trend = 1.0

        vels = d["velocities"]
        cvels = d["comment_vels"]
        bc = max(d["comments"]) if d["comments"] else 0
        avg_com = sum(d["comments"]) / max(1, len(d["comments"]))
        best_vel = max(vels) if vels else 0
        avg_vel = sum(vels) / max(1, len(vels))
        best_cvel = max(cvels) if cvels else 0

        # Days since first appearance
        age = idx

        feats = [
            d["posts"],
            d["total_up"],
            len(d["subs"]),
            bc,
            avg_com,
            d["total_up"] / max(1, d["posts"]),
            best_vel,
            avg_vel,
            best_cvel,
            post_trend,
            up_trend,
            age,
            d["max_single"],
        ]

        phase = "train" if day < days[split] else "test"
        if phase == "train":
            X_tr.append(feats)
            y_tr.append(dies_tomorrow)
        else:
            X_te.append(feats)
            y_te.append(dies_tomorrow)
            info_te.append({
                "pair": pair, "day": day, "posts": d["posts"],
                "up": d["total_up"], "subs": len(d["subs"]),
                "dies": dies_tomorrow, "age": age,
                "post_trend": post_trend, "best_vel": best_vel,
                "title": d["titles"][0] if d["titles"] else "?",
            })

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_te, y_te = np.array(X_te), np.array(y_te)

print("\nTrain: %d observations, %d die (%.1f%%)" % (
    len(y_tr), sum(y_tr), sum(y_tr) / max(1, len(y_tr)) * 100))
print("Test:  %d observations, %d die (%.1f%%)" % (
    len(y_te), sum(y_te), sum(y_te) / max(1, len(y_te)) * 100))

feat_names = ["posts", "total_upvotes", "subs", "best_comments", "avg_comments",
              "up_per_post", "best_velocity", "avg_velocity", "best_comment_vel",
              "post_trend", "upvote_trend", "age_days", "max_single_up"]

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

    # Show topics the model correctly caught dying
    print("\nCORRECT DEATH PREDICTIONS (said dying, actually died):")
    print("%-25s %5s %6s %4s %5s %5s %s" % (
        "Topic", "Posts", "UpK", "Age", "Trend", "Prob", "Title"))
    print("-" * 90)
    correct_death = sorted([p for p in info_te if p["dies"] and p["prob"] > 0.5],
                           key=lambda x: -x["prob"])
    for p in correct_death[:15]:
        print("%-25s %5d %6dK %4d %4.1fx %4.0f%%  %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["age"],
            p["post_trend"], p["prob"] * 100, p["title"][:35]))

    # Topics the model said are still alive (and were)
    print("\nCORRECT SURVIVAL (said alive, actually survived):")
    correct_alive = sorted([p for p in info_te if not p["dies"] and p["prob"] < 0.5],
                           key=lambda x: x["prob"])
    for p in correct_alive[:15]:
        print("%-25s %5d %6dK %4d %4.1fx %4.0f%%  %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["age"],
            p["post_trend"], p["prob"] * 100, p["title"][:35]))

    # Early warning: topics that are still big but model says dying
    print("\nEARLY WARNING (still has posts but model predicts death):")
    early_warn = sorted([p for p in info_te if p["dies"] and p["prob"] > 0.5 and p["posts"] >= 3],
                        key=lambda x: -x["posts"])
    for p in early_warn[:15]:
        print("%-25s %5d %6dK %4d %4.1fx %4.0f%%  %s" % (
            p["pair"], p["posts"], p["up"] // 1000, p["age"],
            p["post_trend"], p["prob"] * 100, p["title"][:35]))


# ================================================================
# TASK 2: Days to death regression
# ================================================================
print("\n" + "=" * 70)
print("  TASK 2: How many days until topic dies?")
print("  (from peak day, how long until < 2 posts)")
print("=" * 70)

train_recs, test_recs = [], []
for pair, daily in pair_trajectories.items():
    pair_days = sorted(daily.keys())
    if len(pair_days) < 3:
        continue
    counts = [daily[d]["posts"] for d in pair_days]
    total = sum(counts)
    if total < 5:
        continue

    peak_idx = np.argmax(counts)
    peak_day = pair_days[peak_idx]
    peak_d = daily[peak_day]

    # Find death day (first day after peak with < 2 posts)
    death_offset = None
    for k in range(peak_idx + 1, len(pair_days)):
        if counts[k] < 2:
            death_offset = k - peak_idx
            break
    if death_offset is None:
        death_offset = len(pair_days) - peak_idx  # hasn't died yet, use remaining

    vels = peak_d["velocities"]
    cvels = peak_d["comment_vels"]
    bc = max(peak_d["comments"]) if peak_d["comments"] else 0

    rec = {
        "pair": pair,
        "peak_day": peak_day,
        "peak_posts": counts[peak_idx],
        "peak_up": peak_d["total_up"],
        "peak_subs": len(peak_d["subs"]),
        "peak_best_com": bc,
        "peak_best_vel": max(vels) if vels else 0,
        "peak_avg_vel": sum(vels) / max(1, len(vels)),
        "peak_best_cvel": max(cvels) if cvels else 0,
        "days_to_death": death_offset,
        "title": peak_d["titles"][0] if peak_d["titles"] else "?",
    }

    if peak_day < days[split]:
        train_recs.append(rec)
    else:
        test_recs.append(rec)

print("Train: %d topics, Test: %d topics" % (len(train_recs), len(test_recs)))

if train_recs and test_recs:
    def peak_features(r):
        return [r["peak_posts"], r["peak_up"], r["peak_subs"], r["peak_best_com"],
                r["peak_up"] / max(1, r["peak_posts"]), r["peak_best_vel"],
                r["peak_avg_vel"], r["peak_best_cvel"]]

    X_tr_r = np.array([peak_features(r) for r in train_recs])
    y_tr_r = np.array([r["days_to_death"] for r in train_recs])
    X_te_r = np.array([peak_features(r) for r in test_recs])
    y_te_r = np.array([r["days_to_death"] for r in test_recs])

    rf_r = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf_r.fit(X_tr_r, y_tr_r)
    preds_r = rf_r.predict(X_te_r)

    r2 = r2_score(y_te_r, preds_r)
    mae = mean_absolute_error(y_te_r, preds_r)
    print("\nR2: %.3f" % r2)
    print("MAE: %.1f days" % mae)
    print("Test range: %d to %d days (mean %.1f)" % (
        min(y_te_r), max(y_te_r), np.mean(y_te_r)))

    peak_feat_names = ["peak_posts", "peak_upvotes", "peak_subs", "peak_best_comments",
                       "peak_up_per_post", "peak_best_vel", "peak_avg_vel", "peak_best_cvel"]
    print("\nFeature importance:")
    for name, imp in sorted(zip(peak_feat_names, rf_r.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))

    print("\nBiggest topics -- days to death:")
    print("%-25s %5s %6s %6s %6s %s" % ("Topic", "Peak", "Actual", "Pred", "Error", "Title"))
    print("-" * 90)
    indexed = sorted(zip(test_recs, preds_r), key=lambda x: -x[0]["peak_posts"])
    for r, p in indexed[:20]:
        print("%-25s %5d %4d d %4.1f d %+4.1f d %s" % (
            r["pair"], r["peak_posts"], r["days_to_death"], p,
            p - r["days_to_death"], r["title"][:35]))

conn.close()
print("\nDONE")
