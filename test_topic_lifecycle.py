"""Topic lifecycle states — like post states but for topics.

Assign each topic a daily state based on its trajectory:
  surging  — posts increasing fast (> 1.5x previous day)
  growing  — posts increasing (> 1.0x)
  stable   — roughly same posts
  cooling  — posts declining
  dying    — posts dropped significantly (< 0.5x)
  dead     — disappeared or < 2 posts

Then predict: what state will this topic be in TOMORROW?
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

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
                d["titles"].append(title[:80])

days = sorted(day_pair.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))

STATE_NAMES = ["surging", "growing", "stable", "cooling", "dying", "dead"]
STATE_MAP = {s: i for i, s in enumerate(STATE_NAMES)}


def assign_state(today_posts, yesterday_posts):
    """Assign topic state based on post count change."""
    if today_posts < 2:
        return "dead"
    if yesterday_posts == 0:
        return "surging"  # brand new with 2+ posts
    ratio = today_posts / yesterday_posts
    if ratio > 1.5:
        return "surging"
    elif ratio > 1.05:
        return "growing"
    elif ratio > 0.8:
        return "stable"
    elif ratio > 0.5:
        return "cooling"
    else:
        return "dying"


# Build topic trajectories with states
pair_trajectories = defaultdict(list)
for pair in set().union(*[day_pair[d].keys() for d in days]):
    for di, day in enumerate(days):
        d = day_pair[day].get(pair)
        posts = d["posts"] if d else 0

        # Previous day
        if di > 0:
            prev = day_pair[days[di - 1]].get(pair)
            prev_posts = prev["posts"] if prev else 0
        else:
            prev_posts = 0

        state = assign_state(posts, prev_posts)

        if d and posts >= 1:
            vels = d["velocities"]
            cvels = d["comment_vels"]
            bc = max(d["comments"]) if d["comments"] else 0
            avg_com = sum(d["comments"]) / max(1, len(d["comments"]))
            best_vel = max(vels) if vels else 0
            avg_vel = sum(vels) / max(1, len(vels))
            best_cvel = max(cvels) if cvels else 0

            pair_trajectories[pair].append({
                "day": day, "posts": posts, "prev_posts": prev_posts,
                "state": state, "total_up": d["total_up"],
                "subs": len(d["subs"]), "best_com": bc, "avg_com": avg_com,
                "best_vel": best_vel, "avg_vel": avg_vel, "best_cvel": best_cvel,
                "title": d["titles"][0] if d["titles"] else "?",
            })

# Filter to topics with enough history
good_pairs = {p: traj for p, traj in pair_trajectories.items()
              if len(traj) >= 3 and sum(t["posts"] for t in traj) >= 5}
print("Topics with 3+ days and 5+ total posts: %d" % len(good_pairs))


# ================================================================
# STATE DISTRIBUTION
# ================================================================
print("\n" + "=" * 70)
print("  TOPIC STATE DISTRIBUTION")
print("=" * 70)

all_states = []
for pair, traj in good_pairs.items():
    for t in traj:
        all_states.append(t["state"])

print("\nOverall distribution:")
for s in STATE_NAMES:
    count = sum(1 for x in all_states if x == s)
    print("  %-10s %6d  (%.1f%%)" % (s, count, count / len(all_states) * 100))


# ================================================================
# TRANSITION MATRIX
# ================================================================
print("\n" + "=" * 70)
print("  TOPIC STATE TRANSITION MATRIX")
print("  (today's state -> tomorrow's state)")
print("=" * 70)

transitions = defaultdict(lambda: defaultdict(int))
for pair, traj in good_pairs.items():
    for i in range(len(traj) - 1):
        if traj[i + 1]["day"] == days[days.index(traj[i]["day"]) + 1]:
            transitions[traj[i]["state"]][traj[i + 1]["state"]] += 1

print("\n%-10s" % "From\\To", end="")
for s in STATE_NAMES:
    print(" %8s" % s, end="")
print("  %8s" % "Total")
print("-" * 75)

for from_s in STATE_NAMES:
    total = sum(transitions[from_s].values())
    if total == 0:
        continue
    print("%-10s" % from_s, end="")
    for to_s in STATE_NAMES:
        count = transitions[from_s][to_s]
        pct = count / total * 100 if total > 0 else 0
        print(" %7.0f%%" % pct, end="")
    print("  %8d" % total)


# ================================================================
# PREDICT NEXT STATE
# ================================================================
print("\n" + "=" * 70)
print("  PREDICT NEXT TOPIC STATE")
print("=" * 70)

X_tr, y_tr, X_te, y_te, info_te = [], [], [], [], []

for pair, traj in good_pairs.items():
    for i in range(1, len(traj) - 1):
        curr = traj[i]
        prev = traj[i - 1]
        nxt = traj[i + 1]
        day = curr["day"]
        next_day = nxt["day"]

        # Only consecutive days
        di = days.index(day)
        if di + 1 >= len(days) or days[di + 1] != next_day:
            continue

        post_trend = curr["posts"] / max(1, prev["posts"])
        up_trend = curr["total_up"] / max(1, prev["total_up"])

        feats = [
            curr["posts"], curr["total_up"], curr["subs"],
            curr["best_com"], curr["avg_com"],
            curr["total_up"] / max(1, curr["posts"]),
            curr["best_vel"], curr["avg_vel"], curr["best_cvel"],
            post_trend, up_trend,
            STATE_MAP[curr["state"]],
            STATE_MAP[prev["state"]],
        ]

        target = STATE_MAP[nxt["state"]]

        if day < days[split]:
            X_tr.append(feats)
            y_tr.append(target)
        else:
            X_te.append(feats)
            y_te.append(target)
            info_te.append({
                "pair": pair, "day": day,
                "curr_state": curr["state"], "next_state": nxt["state"],
                "posts": curr["posts"], "up": curr["total_up"],
                "title": curr["title"],
            })

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_te, y_te = np.array(X_te), np.array(y_te)

print("\nTrain: %d, Test: %d" % (len(y_tr), len(y_te)))

feat_names = ["posts", "total_upvotes", "subs", "best_comments", "avg_comments",
              "up_per_post", "best_velocity", "avg_velocity", "best_comment_vel",
              "post_trend", "upvote_trend", "current_state", "previous_state"]

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_tr, y_tr)
preds = rf.predict(X_te)
probs = rf.predict_proba(X_te)

# Per-state accuracy
print("\nOverall accuracy: %.1f%%" % (sum(preds == y_te) / len(y_te) * 100))

present_states = sorted(set(y_te) | set(y_tr))
present_names = [STATE_NAMES[i] for i in present_states]
print("\nPer-state results:")
for si in present_states:
    s = STATE_NAMES[si]
    mask_actual = y_te == si
    mask_pred = preds == si
    tp = sum(mask_actual & mask_pred)
    total_actual = sum(mask_actual)
    total_pred = sum(mask_pred)
    if total_actual > 0:
        recall = tp / total_actual * 100
    else:
        recall = 0
    if total_pred > 0:
        precision = tp / total_pred * 100
    else:
        precision = 0
    print("  %-10s actual=%4d  predicted=%4d  precision=%.0f%%  recall=%.0f%%" % (
        s, total_actual, total_pred, precision, recall))

# Per-state ROC (one-vs-rest)
print("\nPer-state ROC AUC (one-vs-rest):")
for si in present_states:
    s = STATE_NAMES[si]
    binary = (y_te == si).astype(int)
    if sum(binary) > 0 and sum(binary) < len(binary) and si < probs.shape[1]:
        roc = roc_auc_score(binary, probs[:, si])
        print("  %-10s ROC AUC = %.3f  (n=%d)" % (s, roc, sum(binary)))

print("\nFeature importance:")
for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
    print("  %-22s %.1f%%" % (name, imp * 100))


# ================================================================
# SPECIFIC: Detect cooling/dying BEFORE it happens
# ================================================================
print("\n" + "=" * 70)
print("  EARLY DECLINE DETECTION")
print("  Can we spot cooling/dying while topic is still surging/growing?")
print("=" * 70)

# Binary: topic is currently surging/growing, will it be cooling/dying/dead tomorrow?
X_tr_d, y_tr_d, X_te_d, y_te_d, info_d = [], [], [], [], []

for pair, traj in good_pairs.items():
    for i in range(1, len(traj) - 1):
        curr = traj[i]
        prev = traj[i - 1]
        nxt = traj[i + 1]
        day = curr["day"]
        next_day = nxt["day"]

        di = days.index(day)
        if di + 1 >= len(days) or days[di + 1] != next_day:
            continue

        # Only topics currently in good shape
        if curr["state"] not in ("surging", "growing", "stable"):
            continue

        will_decline = 1 if nxt["state"] in ("cooling", "dying", "dead") else 0

        post_trend = curr["posts"] / max(1, prev["posts"])
        up_trend = curr["total_up"] / max(1, prev["total_up"])

        feats = [
            curr["posts"], curr["total_up"], curr["subs"],
            curr["best_com"], curr["avg_com"],
            curr["total_up"] / max(1, curr["posts"]),
            curr["best_vel"], curr["avg_vel"], curr["best_cvel"],
            post_trend, up_trend,
        ]

        if day < days[split]:
            X_tr_d.append(feats)
            y_tr_d.append(will_decline)
        else:
            X_te_d.append(feats)
            y_te_d.append(will_decline)
            info_d.append({
                "pair": pair, "day": day,
                "curr_state": curr["state"], "next_state": nxt["state"],
                "posts": curr["posts"], "up": curr["total_up"],
                "post_trend": post_trend,
                "title": curr["title"],
            })

X_tr_d, y_tr_d = np.array(X_tr_d), np.array(y_tr_d)
X_te_d, y_te_d = np.array(X_te_d), np.array(y_te_d)

print("\nTrain: %d obs, %d will decline (%.1f%%)" % (
    len(y_tr_d), sum(y_tr_d), sum(y_tr_d) / max(1, len(y_tr_d)) * 100))
print("Test:  %d obs, %d will decline (%.1f%%)" % (
    len(y_te_d), sum(y_te_d), sum(y_te_d) / max(1, len(y_te_d)) * 100))

if sum(y_tr_d) >= 10 and sum(y_te_d) > 0 and sum(y_te_d) < len(y_te_d):
    rf_d = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_d.fit(X_tr_d, y_tr_d)
    probs_d = rf_d.predict_proba(X_te_d)[:, 1]
    roc_d = roc_auc_score(y_te_d, probs_d)

    feat_names_d = ["posts", "total_upvotes", "subs", "best_comments", "avg_comments",
                    "up_per_post", "best_velocity", "avg_velocity", "best_comment_vel",
                    "post_trend", "upvote_trend"]

    print("\nROC AUC: %.3f" % roc_d)
    print("\nFeature importance:")
    for name, imp in sorted(zip(feat_names_d, rf_d.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))

    for i, inf in enumerate(info_d):
        inf["prob"] = probs_d[i]

    print("\nEARLY WARNINGS (topic looks healthy but model says decline coming):")
    print("%-25s %5s %6s %-10s %-10s %5s %s" % (
        "Topic", "Posts", "UpK", "Now", "Tomorrow", "Prob", "Title"))
    print("-" * 95)
    warnings = sorted([p for p in info_d if p["prob"] > 0.5 and y_te_d[info_d.index(p)]],
                      key=lambda x: -x["prob"])
    for p in warnings[:15]:
        print("%-25s %5d %6dK %-10s %-10s %4.0f%%  %s" % (
            p["pair"], p["posts"], p["up"] // 1000,
            p["curr_state"], p["next_state"],
            p["prob"] * 100, p["title"][:35]))

conn.close()
print("\nDONE")
