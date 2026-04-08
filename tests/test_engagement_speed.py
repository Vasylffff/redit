"""Test engagement speed and acceleration as topic emergence predictors."""
import sqlite3, re, os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)

# Get ALL snapshots per post (multiple time points = speed + acceleration)
rows = conn.execute("""
    SELECT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           p.age_minutes_at_snapshot, p.upvotes_at_snapshot,
           p.comment_count_at_snapshot, p.upvote_velocity_per_hour,
           l.subreddit, l.max_upvotes
    FROM post_snapshots p
    JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND p.age_minutes_at_snapshot IS NOT NULL
    ORDER BY p.post_id, p.age_minutes_at_snapshot
""").fetchall()

# Build per-post timeline
post_timeline = defaultdict(list)
post_meta = {}
for pid, title, day, age, up, com, vel, sub, max_up in rows:
    post_timeline[pid].append({
        "age_min": age, "up": up or 0, "com": com or 0, "vel": vel or 0,
    })
    if pid not in post_meta:
        post_meta[pid] = {"title": title, "day": day, "sub": sub, "max_up": max_up or 0}

# Compute speed and acceleration for each post
post_features = {}
for pid, timeline in post_timeline.items():
    if len(timeline) < 2:
        continue
    st = sorted(timeline, key=lambda x: x["age_min"])

    # Speed at first snapshot
    first = st[0]
    first_age_h = max(0.1, first["age_min"] / 60)
    up_speed = first["up"] / first_age_h
    com_speed = first["com"] / first_age_h

    # Acceleration: speed change between first and second snapshot
    if len(st) >= 2:
        second = st[1]
        dt = max(0.1, (second["age_min"] - first["age_min"]) / 60)
        up_accel = (second["up"] - first["up"]) / dt - up_speed  # change in speed
        com_accel = (second["com"] - first["com"]) / dt - com_speed
        vel_at_2 = second["vel"]
    else:
        up_accel = 0
        com_accel = 0
        vel_at_2 = first["vel"]

    # Velocity acceleration
    if len(st) >= 2:
        vel_accel = (st[1]["vel"] - st[0]["vel"])
    else:
        vel_accel = 0

    post_features[pid] = {
        "up_speed": up_speed,
        "com_speed": com_speed,
        "up_accel": up_accel,
        "com_accel": com_accel,
        "vel_accel": vel_accel,
        "first_vel": first["vel"],
        "first_up": first["up"],
        "first_com": first["com"],
        "first_age": first["age_min"],
        "snapshots": len(st),
    }

print("Posts with speed/acceleration: %d" % len(post_features))

# Build keyword data with speed features
day_kw = defaultdict(lambda: defaultdict(list))
for pid, pf in post_features.items():
    meta = post_meta.get(pid)
    if not meta:
        continue
    words = set(w for w in re.findall(r"[a-z]+", meta["title"].lower()) if len(w) > 4 and w not in STOPWORDS)
    for w in words:
        day_kw[meta["day"]][w].append({**pf, "sub": meta["sub"], "max_up": meta["max_up"]})

days = sorted(day_kw.keys())
split = 8

print("Days: %d, split at %d" % (len(days), split))


def build_data(day_range):
    X_count, X_speed, X_accel, y, info = [], [], [], [], []
    for day in day_range:
        day_idx = days.index(day)
        if day_idx + 3 >= len(days):
            continue
        for kw, posts in day_kw[day].items():
            n = len(posts)
            if n < 1 or n > 3:
                continue
            max_future = max(len(day_kw[days[j]].get(kw, [])) for j in range(day_idx + 1, min(day_idx + 4, len(days))))
            grew = 1 if max_future >= 5 else 0

            total_up = sum(p["max_up"] for p in posts)
            subs = len(set(p["sub"] for p in posts))

            # Counts only
            X_count.append([n, total_up, subs])

            # + Speed
            best_up_speed = max(p["up_speed"] for p in posts)
            best_com_speed = max(p["com_speed"] for p in posts)
            best_vel = max(p["first_vel"] for p in posts)
            best_com = max(p["first_com"] for p in posts)

            X_speed.append([
                n, total_up, subs,
                best_up_speed, best_com_speed, best_vel, best_com,
            ])

            # + Acceleration
            best_up_accel = max(p["up_accel"] for p in posts)
            best_com_accel = max(p["com_accel"] for p in posts)
            best_vel_accel = max(p["vel_accel"] for p in posts)

            X_accel.append([
                n, total_up, subs,
                best_up_speed, best_com_speed, best_vel, best_com,
                best_up_accel, best_com_accel, best_vel_accel,
            ])

            y.append(grew)
            info.append({
                "kw": kw, "day": day, "posts": n, "up": total_up,
                "up_speed": best_up_speed, "com_speed": best_com_speed,
                "up_accel": best_up_accel, "vel_accel": best_vel_accel,
                "grew": grew, "future": max_future,
            })
    return np.array(X_count), np.array(X_speed), np.array(X_accel), np.array(y), info


Xc_tr, Xs_tr, Xa_tr, y_tr, _ = build_data(days[:split])
Xc_te, Xs_te, Xa_te, y_te, test_info = build_data(days[split:])

print("Train: %d, %d grew" % (len(y_tr), sum(y_tr)))
print("Test: %d, %d grew" % (len(y_te), sum(y_te)))

if sum(y_te) == 0:
    print("No growth events in test set!")
    conn.close()
    exit()

# Train and compare
models = [
    ("Counts only", Xc_tr, Xc_te),
    ("+ Speed", Xs_tr, Xs_te),
    ("+ Speed + Acceleration", Xa_tr, Xa_te),
]

print("\n" + "=" * 60)
print("TEMPORAL VALIDATION: COUNTS vs SPEED vs ACCELERATION")
print("=" * 60)

best_probs = None
best_name = ""
best_roc = 0

for name, X_tr, X_te in models:
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, probs)
    print("  %-30s ROC = %.3f" % (name, roc))

    if roc > best_roc:
        best_roc = roc
        best_probs = probs
        best_name = name

    # Detection rate
    growers = [(i, p) for i, p in enumerate(test_info) if p["grew"]]
    caught_5 = sum(1 for i, p in growers if probs[i] > 0.05)
    caught_10 = sum(1 for i, p in growers if probs[i] > 0.10)
    print("  %-30s Caught >5%%: %d/%d  >10%%: %d/%d" % ("", caught_5, len(growers), caught_10, len(growers)))

# Feature importance for acceleration model
print("\n" + "=" * 60)
print("ACCELERATION MODEL FEATURES:")
print("=" * 60)
rf_a = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_a.fit(Xa_tr, y_tr)
fn = ["posts", "total_up", "subs", "up_speed", "com_speed", "velocity", "comments",
      "up_acceleration", "com_acceleration", "vel_acceleration"]
for name, imp in sorted(zip(fn, rf_a.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.01:
        bar = "#" * int(imp * 30)
        print("  %-22s %.3f %s" % (name, imp, bar))

# Profile: growers vs non-growers speed and acceleration
print("\n" + "=" * 60)
print("SPEED/ACCELERATION PROFILE:")
print("=" * 60)

grower_info = [p for p in test_info if p["grew"]]
non_info = [p for p in test_info if not p["grew"]]

for metric in ["up_speed", "com_speed", "up_accel", "vel_accel"]:
    g = np.mean([p[metric] for p in grower_info]) if grower_info else 0
    n = np.mean([p[metric] for p in non_info]) if non_info else 0
    ratio = g / max(0.001, n)
    print("  %-15s Grew: %8.1f  Didnt: %8.1f  Ratio: %.1fx" % (metric, g, n, ratio))

conn.close()
print("\nDONE")
