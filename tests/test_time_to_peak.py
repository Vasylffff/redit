"""Predict how long until a topic peaks.

For each word pair, track its daily post count trajectory.
From the first day, predict which day it will hit its maximum.
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

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


# Build topic trajectories: for each pair, its daily post counts
pair_trajectories = defaultdict(dict)
for day in days:
    for pair, d in day_pair[day].items():
        pair_trajectories[pair][day] = d

# For each pair, find: first day, peak day, days to peak
topic_records = []
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
    peak_count = counts[peak_idx]
    days_to_peak = peak_idx  # 0 = peaks on first day

    # Features from first day
    first = daily[pair_days[0]]
    vels = first["velocities"]
    cvels = first["comment_vels"]
    bc = max(first["comments"]) if first["comments"] else 0

    # Features from first 2 days (if available)
    if len(pair_days) >= 2:
        second = daily[pair_days[1]]
        growth_d1_d2 = second["posts"] / max(1, first["posts"])
        up_growth = second["total_up"] / max(1, first["total_up"])
    else:
        growth_d1_d2 = 1.0
        up_growth = 1.0

    topic_records.append({
        "pair": pair,
        "first_day": pair_days[0],
        "peak_day": peak_day,
        "days_to_peak": days_to_peak,
        "peak_count": peak_count,
        "total_days": len(pair_days),
        "total_posts": total,
        # Day 1 features
        "d1_posts": first["posts"],
        "d1_up": first["total_up"],
        "d1_subs": len(first["subs"]),
        "d1_best_com": bc,
        "d1_max_single": first["max_single"],
        "d1_best_vel": max(vels) if vels else 0,
        "d1_avg_vel": sum(vels) / max(1, len(vels)),
        "d1_best_cvel": max(cvels) if cvels else 0,
        # Growth features
        "growth_d1_d2": growth_d1_d2,
        "up_growth_d1_d2": up_growth,
        "title": first["titles"][0] if first["titles"] else "?",
    })

print("Topics with 5+ total posts and 3+ days: %d" % len(topic_records))

# Split by first_day
train_recs = [r for r in topic_records if r["first_day"] < days[split]]
test_recs = [r for r in topic_records if r["first_day"] >= days[split]]
print("Train: %d, Test: %d" % (len(train_recs), len(test_recs)))


def make_features(r):
    return [r["d1_posts"], r["d1_up"], r["d1_subs"], r["d1_best_com"],
            r["d1_max_single"], r["d1_best_vel"], r["d1_avg_vel"],
            r["d1_best_cvel"], r["growth_d1_d2"], r["up_growth_d1_d2"]]


feat_names = ["d1_posts", "d1_upvotes", "d1_subs", "d1_best_comments",
              "d1_max_single_up", "d1_best_velocity", "d1_avg_velocity",
              "d1_best_comment_vel", "post_growth_d1d2", "upvote_growth_d1d2"]


# ================================================================
# REGRESSION: Predict days to peak
# ================================================================
print("\n" + "=" * 70)
print("  REGRESSION: How many days until peak?")
print("=" * 70)

X_tr = np.array([make_features(r) for r in train_recs])
y_tr = np.array([r["days_to_peak"] for r in train_recs])
X_te = np.array([make_features(r) for r in test_recs])
y_te = np.array([r["days_to_peak"] for r in test_recs])

if len(X_tr) > 0 and len(X_te) > 0:
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)

    r2 = r2_score(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    print("\nR2: %.3f" % r2)
    print("MAE: %.1f days" % mae)
    print("Test range: %d to %d days (mean %.1f, median %.1f)" % (
        min(y_te), max(y_te), np.mean(y_te), np.median(y_te)))

    print("\nFeature importance:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))

    print("\nPredictions for test topics:")
    print("%-25s %6s %6s %6s %5s %s" % ("Topic", "Actual", "Pred", "Error", "Peak", "Title"))
    print("-" * 95)
    test_indexed = sorted(zip(test_recs, preds), key=lambda x: -x[0]["peak_count"])
    for r, p in test_indexed[:25]:
        print("%-25s %4d d %4.1f d %+4.1f d %5d %s" % (
            r["pair"], r["days_to_peak"], p, p - r["days_to_peak"],
            r["peak_count"], r["title"][:35]))


# ================================================================
# CLASSIFICATION: Will it peak today (day 0) or later?
# ================================================================
print("\n" + "=" * 70)
print("  CLASSIFICATION: Will it peak TODAY or grow more?")
print("  (peaks_today=0 means it will grow more)")
print("=" * 70)

y_tr_cls = (np.array([r["days_to_peak"] for r in train_recs]) == 0).astype(int)
y_te_cls = (np.array([r["days_to_peak"] for r in test_recs]) == 0).astype(int)

if len(X_tr) > 0 and sum(y_tr_cls) > 5 and sum(y_te_cls) > 0 and sum(y_te_cls) < len(y_te_cls):
    rf_cls = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_cls.fit(X_tr, y_tr_cls)
    probs = rf_cls.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te_cls, probs)

    print("\nROC AUC: %.3f" % roc)
    print("Peaks on day 0: train=%d/%d (%.0f%%), test=%d/%d (%.0f%%)" % (
        sum(y_tr_cls), len(y_tr_cls), sum(y_tr_cls) / len(y_tr_cls) * 100,
        sum(y_te_cls), len(y_te_cls), sum(y_te_cls) / len(y_te_cls) * 100))

    print("\nFeature importance:")
    for name, imp in sorted(zip(feat_names, rf_cls.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))


# ================================================================
# CLASSIFICATION: Has it peaked already? (from day 2 observation)
# ================================================================
print("\n" + "=" * 70)
print("  CLASSIFICATION: Observing at day 2 -- has it peaked already?")
print("  (useful: should I keep watching this topic?)")
print("=" * 70)

# Rebuild features using day 1+2 data
X_tr2, y_tr2, X_te2, y_te2, info2 = [], [], [], [], []

for phase, recs in [("train", train_recs), ("test", test_recs)]:
    for r in recs:
        if r["total_days"] < 3:
            continue
        pair_daily = pair_trajectories[r["pair"]]
        pair_days = sorted(pair_daily.keys())
        if len(pair_days) < 3:
            continue

        d1 = pair_daily[pair_days[0]]
        d2 = pair_daily[pair_days[1]]

        # Has it peaked by day 1 (i.e., day 2+ will be lower)?
        peaked = 1 if r["days_to_peak"] <= 1 else 0

        d1_posts = d1["posts"]
        d2_posts = d2["posts"]
        d1_up = d1["total_up"]
        d2_up = d2["total_up"]
        d1_vel = max(d1["velocities"]) if d1["velocities"] else 0
        d2_vel = max(d2["velocities"]) if d2["velocities"] else 0
        d1_com = max(d1["comments"]) if d1["comments"] else 0
        d2_com = max(d2["comments"]) if d2["comments"] else 0

        feats = [
            d1_posts, d2_posts,
            d2_posts / max(1, d1_posts),  # post growth
            d1_up, d2_up,
            d2_up / max(1, d1_up),  # upvote growth
            d1_vel, d2_vel,
            d2_vel / max(1, d1_vel) if d1_vel > 0 else 0,  # velocity change
            d1_com, d2_com,
            len(d1["subs"]), len(d2["subs"]),
        ]

        if phase == "train":
            X_tr2.append(feats)
            y_tr2.append(peaked)
        else:
            X_te2.append(feats)
            y_te2.append(peaked)
            info2.append({"pair": r["pair"], "peaked": peaked,
                          "days_to_peak": r["days_to_peak"],
                          "peak_count": r["peak_count"],
                          "d1": d1_posts, "d2": d2_posts,
                          "title": r["title"]})

X_tr2, y_tr2 = np.array(X_tr2), np.array(y_tr2)
X_te2, y_te2 = np.array(X_te2), np.array(y_te2)

if len(X_tr2) > 0 and sum(y_tr2) > 5 and sum(y_te2) > 0 and sum(y_te2) < len(y_te2):
    rf2 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf2.fit(X_tr2, y_tr2)
    probs2 = rf2.predict_proba(X_te2)[:, 1]
    roc2 = roc_auc_score(y_te2, probs2)

    feat_names2 = ["d1_posts", "d2_posts", "post_growth", "d1_up", "d2_up",
                   "up_growth", "d1_vel", "d2_vel", "vel_change",
                   "d1_comments", "d2_comments", "d1_subs", "d2_subs"]

    print("\nROC AUC: %.3f" % roc2)
    print("Already peaked: train=%d/%d (%.0f%%), test=%d/%d (%.0f%%)" % (
        sum(y_tr2), len(y_tr2), sum(y_tr2) / len(y_tr2) * 100,
        sum(y_te2), len(y_te2), sum(y_te2) / len(y_te2) * 100))

    print("\nFeature importance:")
    for name, imp in sorted(zip(feat_names2, rf2.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))

    for i, inf in enumerate(info2):
        inf["prob"] = probs2[i]

    print("\nSTILL GROWING (model says not peaked yet, correct):")
    print("%-25s %4s %4s %6s %5s %s" % ("Topic", "D1", "D2", "Peak@", "Prob", "Title"))
    print("-" * 80)
    still_growing = sorted([p for p in info2 if not p["peaked"] and p["prob"] < 0.5],
                           key=lambda x: x["prob"])
    for p in still_growing[:10]:
        print("%-25s %4d %4d  day %d %4.0f%%  %s" % (
            p["pair"], p["d1"], p["d2"], p["days_to_peak"],
            p["prob"] * 100, p["title"][:35]))

    print("\nALREADY PEAKED (model says peaked, correct):")
    peaked_correct = sorted([p for p in info2 if p["peaked"] and p["prob"] > 0.5],
                            key=lambda x: -x["prob"])
    for p in peaked_correct[:10]:
        print("%-25s %4d %4d  day %d %4.0f%%  %s" % (
            p["pair"], p["d1"], p["d2"], p["days_to_peak"],
            p["prob"] * 100, p["title"][:35]))

conn.close()
print("\nDONE")
