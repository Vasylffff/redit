"""Test different definitions of 'dead' for topics.

Current definition (<2 posts for 1 day) has 17% revival rate = too aggressive.
Try stricter definitions and see which one:
  1) Has lowest false-death (revival) rate
  2) Makes days-to-death regression actually work
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments, p.upvote_velocity_per_hour
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [], "velocities": []
}))
seen = set()
for pid, title, day, max_up, sub, max_com, vel in rows:
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

days = sorted(day_pair.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))

all_pairs = set()
for d in days:
    all_pairs.update(day_pair[d].keys())
big_pairs = [p for p in all_pairs
             if sum(day_pair[d].get(p, {}).get("posts", 0) for d in days) >= 5]
print("Topics with 5+ total posts: %d" % len(big_pairs))


# ================================================================
# TEST DEATH DEFINITIONS
# ================================================================
print("\n" + "=" * 70)
print("  TESTING DEATH DEFINITIONS — revival rates")
print("=" * 70)

print("\n  %-50s %8s %8s %8s" % ("Definition", "Deaths", "Revived", "Rate"))
print("  " + "-" * 80)

for def_name, window in [
    ("1 day <2 posts (current)", 1),
    ("2 consecutive days <2 posts", 2),
    ("3 consecutive days <2 posts", 3),
]:
    total_deaths = 0
    revivals = 0
    for pair in big_pairs:
        counts = [day_pair[d].get(pair, {}).get("posts", 0) for d in days]
        for i in range(len(counts) - window + 1):
            is_dead = all(counts[i + k] < 2 for k in range(window))
            if not is_dead:
                continue
            # Check it was alive before
            if i == 0 or counts[i - 1] < 2:
                continue
            total_deaths += 1
            # Check revival in next 3 days after the window
            end = i + window
            revived = any(counts[j] >= 2 for j in range(end, min(end + 3, len(counts))))
            if revived:
                revivals += 1
    rate = revivals / max(1, total_deaths) * 100
    print("  %-50s %8d %8d %7.1f%%" % (def_name, total_deaths, revivals, rate))


# ================================================================
# USE 2 CONSECUTIVE DAYS — good balance
# PREDICT: from peak, days until confirmed death (2 consecutive <2)
# ================================================================
print("\n" + "=" * 70)
print("  DEATH = 2 consecutive days <2 posts")
print("  Predict: days from peak to confirmed death")
print("=" * 70)

train_recs, test_recs = [], []
for pair in big_pairs:
    counts = [day_pair[d].get(pair, {}).get("posts", 0) for d in days]
    peak_idx = int(np.argmax(counts))
    peak_count = counts[peak_idx]
    if peak_count < 3:
        continue

    # Find confirmed death: 2 consecutive days <2
    death_idx = None
    for i in range(peak_idx, len(counts) - 1):
        if counts[i] < 2 and counts[i + 1] < 2:
            death_idx = i
            break

    if death_idx is None:
        continue

    days_to_death = death_idx - peak_idx

    peak_d = day_pair[days[peak_idx]].get(pair, {})
    vels = peak_d.get("velocities", [])
    coms = peak_d.get("comments", [])
    bc = max(coms) if coms else 0
    bv = max(vels) if vels else 0
    av = sum(vels) / max(1, len(vels))

    # Post-peak decline rate (if we have day after peak)
    if peak_idx + 1 < len(counts) and peak_count > 0:
        decline_rate = counts[peak_idx + 1] / peak_count
    else:
        decline_rate = 0

    rec = {
        "pair": pair, "peak_day": days[peak_idx],
        "peak_posts": peak_count,
        "peak_up": peak_d.get("total_up", 0),
        "peak_subs": len(peak_d.get("subs", set())),
        "peak_best_com": bc,
        "peak_best_vel": bv,
        "peak_avg_vel": av,
        "decline_rate": decline_rate,
        "days_to_death": days_to_death,
        "counts": counts,
    }

    if days[peak_idx] < days[split]:
        train_recs.append(rec)
    else:
        test_recs.append(rec)

print("Train: %d, Test: %d" % (len(train_recs), len(test_recs)))


def feats(r):
    return [r["peak_posts"], r["peak_up"], r["peak_subs"],
            r["peak_best_com"], r["peak_up"] / max(1, r["peak_posts"]),
            r["peak_best_vel"], r["peak_avg_vel"], r["decline_rate"]]


feat_names = ["peak_posts", "peak_upvotes", "peak_subs", "peak_best_comments",
              "peak_up_per_post", "peak_best_vel", "peak_avg_vel", "decline_rate_d1"]

if train_recs and test_recs:
    X_tr = np.array([feats(r) for r in train_recs])
    y_tr = np.array([r["days_to_death"] for r in train_recs])
    X_te = np.array([feats(r) for r in test_recs])
    y_te = np.array([r["days_to_death"] for r in test_recs])

    print("\nDays to death: min=%d max=%d mean=%.1f median=%.1f" % (
        min(y_te), max(y_te), np.mean(y_te), np.median(y_te)))

    # Regression
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)

    r2 = r2_score(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    print("\nREGRESSION:")
    print("  R2:  %.3f" % r2)
    print("  MAE: %.1f days" % mae)

    print("\n  Feature importance:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        print("    %-22s %.1f%%" % (name, imp * 100))

    # Show predictions
    print("\n  Biggest topics:")
    print("  %-25s %5s %6s %6s %6s" % ("Topic", "Peak", "Actual", "Pred", "Error"))
    print("  " + "-" * 60)
    indexed = sorted(zip(test_recs, preds), key=lambda x: -x[0]["peak_posts"])
    for r, p in indexed[:15]:
        print("  %-25s %5d %4d d %4.1f d %+4.1f d" % (
            r["pair"], r["peak_posts"], r["days_to_death"], p, p - r["days_to_death"]))

    # Classification: quick death (0-1) vs slow (2+)
    print("\n  CLASSIFICATION: Quick death (0-1d) vs slow (2+d)")
    y_tr_cls = (y_tr <= 1).astype(int)
    y_te_cls = (y_te <= 1).astype(int)

    if sum(y_tr_cls) > 5 and 0 < sum(y_te_cls) < len(y_te_cls):
        rf_cls = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        rf_cls.fit(X_tr, y_tr_cls)
        roc = roc_auc_score(y_te_cls, rf_cls.predict_proba(X_te)[:, 1])
        print("  ROC AUC: %.3f" % roc)
        print("  Quick: train=%d/%d (%.0f%%), test=%d/%d (%.0f%%)" % (
            sum(y_tr_cls), len(y_tr_cls), sum(y_tr_cls) / len(y_tr_cls) * 100,
            sum(y_te_cls), len(y_te_cls), sum(y_te_cls) / len(y_te_cls) * 100))

        print("\n  Feature importance (quick vs slow):")
        for name, imp in sorted(zip(feat_names, rf_cls.feature_importances_), key=lambda x: -x[1]):
            print("    %-22s %.1f%%" % (name, imp * 100))

    # Classification: dies within 3 days vs survives longer
    print("\n  CLASSIFICATION: Dies within 3 days vs survives 4+")
    y_tr_3 = (y_tr <= 3).astype(int)
    y_te_3 = (y_te <= 3).astype(int)

    if sum(y_tr_3) > 5 and 0 < sum(y_te_3) < len(y_te_3):
        rf_3 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        rf_3.fit(X_tr, y_tr_3)
        roc_3 = roc_auc_score(y_te_3, rf_3.predict_proba(X_te)[:, 1])
        print("  ROC AUC: %.3f" % roc_3)
        print("  Dies <=3d: train=%d/%d (%.0f%%), test=%d/%d (%.0f%%)" % (
            sum(y_tr_3), len(y_tr_3), sum(y_tr_3) / len(y_tr_3) * 100,
            sum(y_te_3), len(y_te_3), sum(y_te_3) / len(y_te_3) * 100))

conn.close()
print("\nDONE")
