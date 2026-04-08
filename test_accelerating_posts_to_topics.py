"""Find small posts that are accelerating and check if their keywords become topics."""
import sqlite3, re, os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)

# Get post timelines - need multiple snapshots to measure acceleration
rows = conn.execute("""
    SELECT p.post_id, p.title, p.subreddit,
           substr(p.snapshot_time_utc, 1, 10) as day,
           p.age_minutes_at_snapshot, p.upvotes_at_snapshot,
           p.comment_count_at_snapshot, p.upvote_velocity_per_hour,
           p.activity_state
    FROM post_snapshots p
    WHERE p.title IS NOT NULL
      AND p.age_minutes_at_snapshot IS NOT NULL
      AND p.activity_state IS NOT NULL
    ORDER BY p.post_id, p.age_minutes_at_snapshot
""").fetchall()

# Build per-post timelines
post_tl = defaultdict(list)
post_meta = {}
for pid, title, sub, day, age, up, com, vel, state in rows:
    post_tl[pid].append({
        "age_min": age, "up": up or 0, "com": com or 0,
        "vel": vel or 0, "state": state,
    })
    if pid not in post_meta:
        post_meta[pid] = {"title": title, "sub": sub, "day": day}

print("Posts with timelines: %d" % len(post_tl))

# Identify ACCELERATING small posts:
# - Currently small (under 500 upvotes)
# - But velocity is INCREASING between snapshots
# - Or state is improving (dying->alive, alive->surging)
accelerating_posts = {}
for pid, tl in post_tl.items():
    if len(tl) < 2:
        continue
    st = sorted(tl, key=lambda x: x["age_min"])
    latest = st[-1]
    prev = st[-2]

    # Small post (under 500 upvotes)
    if latest["up"] > 500:
        continue

    # Acceleration signals
    vel_increase = latest["vel"] - prev["vel"]
    up_accel = (latest["up"] - prev["up"]) / max(1, (latest["age_min"] - prev["age_min"]) / 60)
    state_rank = {"dead": 0, "dying": 1, "cooling": 2, "alive": 3, "surging": 4}
    state_improved = state_rank.get(latest["state"], 0) > state_rank.get(prev["state"], 0)
    is_accelerating = vel_increase > 0 or state_improved

    accelerating_posts[pid] = {
        "up": latest["up"],
        "com": latest["com"],
        "vel": latest["vel"],
        "vel_increase": vel_increase,
        "up_accel": up_accel,
        "state_improved": 1 if state_improved else 0,
        "is_accelerating": 1 if is_accelerating else 0,
        "state": latest["state"],
        "snapshots": len(st),
    }

print("Small posts (<500 up) with 2+ snapshots: %d" % len(accelerating_posts))
print("Of those, accelerating: %d (%.0f%%)" % (
    sum(1 for p in accelerating_posts.values() if p["is_accelerating"]),
    sum(1 for p in accelerating_posts.values() if p["is_accelerating"]) / max(1, len(accelerating_posts)) * 100))

# Build keyword data: which keywords have accelerating posts?
day_kw = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "accel_posts": 0, "total_up": 0,
    "best_vel_increase": 0, "best_up_accel": 0,
    "any_state_improved": 0, "best_comments": 0,
    "subs": set(), "avg_vel": [],
}))

seen = set()
for pid, meta in post_meta.items():
    key = (pid, meta["day"])
    if key in seen:
        continue
    seen.add(key)
    ap = accelerating_posts.get(pid)
    if not ap:
        continue
    words = set(w for w in re.findall(r"[a-z]+", meta["title"].lower()) if len(w) > 4 and w not in STOPWORDS)
    for w in words:
        d = day_kw[meta["day"]][w]
        d["posts"] += 1
        d["total_up"] += ap["up"]
        d["subs"].add(meta["sub"])
        d["avg_vel"].append(ap["vel"])
        d["best_comments"] = max(d["best_comments"], ap["com"])
        if ap["is_accelerating"]:
            d["accel_posts"] += 1
            d["best_vel_increase"] = max(d["best_vel_increase"], ap["vel_increase"])
            d["best_up_accel"] = max(d["best_up_accel"], ap["up_accel"])
        if ap["state_improved"]:
            d["any_state_improved"] = 1

days = sorted(day_kw.keys())
split = len(days) - 4

print("Days: %d, split at %d" % (len(days), split))


def build(day_range):
    X_count, X_accel, y, info = [], [], [], []
    for day in day_range:
        di = days.index(day)
        if di + 3 >= len(days):
            continue
        for kw, d in day_kw[day].items():
            n = d["posts"]
            if n < 1 or n > 5:
                continue
            max_future = 0
            for j in range(di + 1, min(di + 4, len(days))):
                fk = day_kw[days[j]].get(kw)
                if fk:
                    max_future = max(max_future, fk["posts"])
            grew = 1 if max_future >= 5 else 0

            subs = len(d["subs"])
            X_count.append([n, d["total_up"], subs])

            accel_ratio = d["accel_posts"] / max(1, n)
            avg_v = np.mean(d["avg_vel"]) if d["avg_vel"] else 0

            X_accel.append([
                n, d["total_up"], subs,
                d["accel_posts"],           # how many posts are accelerating
                accel_ratio,                # what % of posts accelerating
                d["best_vel_increase"],     # strongest velocity increase
                d["best_up_accel"],         # strongest upvote acceleration
                d["any_state_improved"],    # any post improved state
                d["best_comments"],         # most commented post
                avg_v,                      # average velocity
            ])
            y.append(grew)
            info.append({
                "kw": kw, "day": day, "posts": n, "up": d["total_up"],
                "accel": d["accel_posts"], "accel_ratio": accel_ratio,
                "vel_inc": d["best_vel_increase"], "state_up": d["any_state_improved"],
                "grew": grew, "future": max_future,
            })
    return np.array(X_count), np.array(X_accel), np.array(y), info


Xc_tr, Xa_tr, y_tr, _ = build(days[:split])
Xc_te, Xa_te, y_te, test_info = build(days[split:])

print("Train: %d, %d grew (%.1f%%)" % (len(y_tr), sum(y_tr), sum(y_tr) / len(y_tr) * 100))
print("Test: %d, %d grew (%.1f%%)" % (len(y_te), sum(y_te), sum(y_te) / len(y_te) * 100))

if sum(y_te) == 0:
    print("No growth events in test!")
    conn.close()
    exit()

rf_c = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_a = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_c.fit(Xc_tr, y_tr)
rf_a.fit(Xa_tr, y_tr)

probs_c = rf_c.predict_proba(Xc_te)[:, 1]
probs_a = rf_a.predict_proba(Xa_te)[:, 1]

roc_c = roc_auc_score(y_te, probs_c)
roc_a = roc_auc_score(y_te, probs_a)

print("\n" + "=" * 60)
print("SMALL ACCELERATING POSTS -> TOPIC PREDICTION")
print("=" * 60)
print("Counts only:           ROC = %.3f" % roc_c)
print("+ Acceleration signals: ROC = %.3f" % roc_a)
print("Improvement:            %+.3f" % (roc_a - roc_c))

# Detection rates
growers = [(i, p) for i, p in enumerate(test_info) if p["grew"]]
for thresh in [0.05, 0.10, 0.20]:
    cc = sum(1 for i, p in growers if probs_c[i] > thresh)
    ca = sum(1 for i, p in growers if probs_a[i] > thresh)
    print("  >%.0f%%: Counts=%d/%d  Accel=%d/%d" % (thresh * 100, cc, len(growers), ca, len(growers)))

# Feature importance
fn = ["posts", "total_up", "subs", "accel_posts", "accel_ratio",
      "best_vel_increase", "best_up_accel", "state_improved", "best_comments", "avg_velocity"]
print("\nFeatures:")
for name, imp in sorted(zip(fn, rf_a.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.01:
        bar = "#" * int(imp * 30)
        print("  %-22s %.3f %s" % (name, imp, bar))

# Profile
print("\nProfile (growing vs not):")
grower_info = [p for p in test_info if p["grew"]]
non_info = [p for p in test_info if not p["grew"]]
for metric in ["posts", "up", "accel", "accel_ratio", "vel_inc", "state_up"]:
    g = np.mean([p[metric] for p in grower_info])
    n = np.mean([p[metric] for p in non_info])
    ratio = g / max(0.001, n)
    print("  %-18s Grew: %8.2f  Didnt: %8.2f  %.1fx" % (metric, g, n, ratio))

# Show examples
print("\nEXAMPLES - keywords with accelerating posts that grew:")
for i, p in enumerate(test_info):
    p["prob_a"] = probs_a[i]
accel_growers = [p for p in test_info if p["grew"] and p["accel"] > 0]
for p in sorted(accel_growers, key=lambda x: -x["prob_a"])[:15]:
    print("  %-20s accel=%d/%d vel_inc=%.0f prob=%.0f%% grew->%d" % (
        p["kw"], p["accel"], p["posts"], p["vel_inc"], p["prob_a"] * 100, p["future"]))

conn.close()
print("\nDONE")
