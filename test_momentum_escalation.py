"""Test: can we predict topic escalation from early momentum?

If a topic has 3-6 posts and those posts are rising fast (high velocity,
surging/alive states), does that predict it will hit 10+ posts?

Two-tier comparison:
  A) Count-only features (baseline, same as test_temporal_pairs tier 2)
  B) Count + momentum features (velocity, alive ratio, surging ratio)
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

# Get post-level data with velocity and state
rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           p.upvotes_at_snapshot, p.upvote_velocity_per_hour, p.comment_velocity_per_hour,
           p.activity_state, l.max_upvotes, l.subreddit, l.max_comments
    FROM post_snapshots p JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

# Build per-day per-pair data WITH velocity/state signals
day_pair_data = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [],
    "velocities": [], "comment_vels": [], "states": [], "titles": [],
    "max_single_up": 0
}))
seen = set()

for pid, title, day, up, vel, cvel, state, max_up, sub, max_com in rows:
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
            d["comments"].append(max_com or 0)
            d["velocities"].append(vel or 0)
            d["comment_vels"].append(cvel or 0)
            d["states"].append(state or "dead")
            d["max_single_up"] = max(d["max_single_up"], max_up or 0)
            if len(d["titles"]) < 2:
                d["titles"].append(title[:70])

days = sorted(day_pair_data.keys())
split = 8

print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))


def build_data(day_range, min_posts=3, max_posts=6, growth_target=10, use_momentum=True):
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

            vels = d["velocities"]
            cvels = d["comment_vels"]
            states = d["states"]
            alive_count = sum(1 for s in states if s in ("surging", "alive"))
            surging_count = sum(1 for s in states if s == "surging")
            dead_count = sum(1 for s in states if s in ("dead", "dying"))

            best_vel = max(vels) if vels else 0
            avg_vel = sum(vels) / max(1, len(vels))
            best_cvel = max(cvels) if cvels else 0
            avg_cvel = sum(cvels) / max(1, len(cvels))
            best_comments = max(d["comments"]) if d["comments"] else 0
            avg_comments = sum(d["comments"]) / max(1, len(d["comments"]))

            # Base features (count-only)
            features = [
                d["posts"],
                d["total_up"],
                len(d["subs"]),
                best_comments,
                avg_comments,
                d["total_up"] / max(1, d["posts"]),
            ]

            if use_momentum:
                features.extend([
                    best_vel,
                    avg_vel,
                    best_cvel,
                    avg_cvel,
                    alive_count / max(1, d["posts"]),
                    surging_count / max(1, d["posts"]),
                    dead_count / max(1, d["posts"]),
                    d["max_single_up"],
                ])

            X.append(features)
            y.append(grew)
            info.append({
                "pair": pair, "day": day, "posts": d["posts"],
                "up": d["total_up"], "comments": best_comments,
                "subs": len(d["subs"]), "grew": grew,
                "future_peak": max_future,
                "alive_ratio": alive_count / max(1, d["posts"]),
                "surging_ratio": surging_count / max(1, d["posts"]),
                "best_vel": best_vel,
                "title": d["titles"][0] if d["titles"] else "?",
            })
    return np.array(X), np.array(y), info


# --- A) Baseline: count-only features ---
print("\n" + "#" * 70)
print("  A) BASELINE -- count features only (3-6 posts -> 10+)")
print("#" * 70)

X_tr_a, y_tr_a, _ = build_data(days[:split], use_momentum=False)
X_te_a, y_te_a, info_a = build_data(days[split:], use_momentum=False)

print("\nTrain: %d pairs, %d grew (%.2f%%)" % (len(y_tr_a), sum(y_tr_a), sum(y_tr_a) / max(1, len(y_tr_a)) * 100))
print("Test:  %d pairs, %d grew (%.2f%%)" % (len(y_te_a), sum(y_te_a), sum(y_te_a) / max(1, len(y_te_a)) * 100))

roc_a = None
if sum(y_tr_a) >= 5 and sum(y_te_a) > 0:
    rf_a = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf_a.fit(X_tr_a, y_tr_a)
    probs_a = rf_a.predict_proba(X_te_a)[:, 1]
    roc_a = roc_auc_score(y_te_a, probs_a)
    print("\nROC AUC (baseline): %.3f" % roc_a)

    names_a = ["posts", "total_upvotes", "subreddit_count", "best_comments", "avg_comments", "upvotes_per_post"]
    print("\nFeature importance:")
    for name, imp in sorted(zip(names_a, rf_a.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))


# --- B) With momentum features ---
print("\n" + "#" * 70)
print("  B) MOMENTUM -- count + velocity/state features (3-6 posts -> 10+)")
print("#" * 70)

X_tr_b, y_tr_b, _ = build_data(days[:split], use_momentum=True)
X_te_b, y_te_b, info_b = build_data(days[split:], use_momentum=True)

print("\nTrain: %d pairs, %d grew (%.2f%%)" % (len(y_tr_b), sum(y_tr_b), sum(y_tr_b) / max(1, len(y_tr_b)) * 100))
print("Test:  %d pairs, %d grew (%.2f%%)" % (len(y_te_b), sum(y_te_b), sum(y_te_b) / max(1, len(y_te_b)) * 100))

roc_b = None
if sum(y_tr_b) >= 5 and sum(y_te_b) > 0:
    rf_b = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_b.fit(X_tr_b, y_tr_b)
    probs_b = rf_b.predict_proba(X_te_b)[:, 1]
    roc_b = roc_auc_score(y_te_b, probs_b)
    print("\nROC AUC (momentum): %.3f" % roc_b)

    names_b = ["posts", "total_upvotes", "subreddit_count", "best_comments", "avg_comments",
               "upvotes_per_post", "best_velocity", "avg_velocity", "best_comment_vel",
               "avg_comment_vel", "alive_ratio", "surging_ratio", "dead_ratio", "max_single_post_up"]
    print("\nFeature importance:")
    for name, imp in sorted(zip(names_b, rf_b.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))

    for i, inf in enumerate(info_b):
        inf["prob"] = probs_b[i]

    print("\nFAST-RISING TOPICS THAT GREW (correct):")
    print("%-28s %5s %6s %5s %5s %5s %6s %s" % ("Topic", "Posts", "UpK", "Alive", "Surge", "Prob", "Peak", "Title"))
    print("-" * 105)
    correct = sorted([p for p in info_b if p["grew"] and p["prob"] > 0.03], key=lambda x: -x["prob"])
    for p in correct[:15]:
        print("%-28s %5d %6dK %4.0f%% %5.0f%% %4.0f%% %6d %s" % (
            p["pair"], p["posts"], p["up"] // 1000,
            p["alive_ratio"] * 100, p["surging_ratio"] * 100,
            p["prob"] * 100, p["future_peak"], p["title"][:35]))

    print("\nFALSE ALARMS (model said grow, didn't):")
    false_alarms = sorted([p for p in info_b if not p["grew"] and p["prob"] > 0.1], key=lambda x: -x["prob"])
    for p in false_alarms[:10]:
        print("%-28s %5d %6dK %4.0f%% %5.0f%% %4.0f%% %s" % (
            p["pair"], p["posts"], p["up"] // 1000,
            p["alive_ratio"] * 100, p["surging_ratio"] * 100,
            p["prob"] * 100, p["title"][:40]))

    print("\nMISSED (grew but low probability):")
    missed = sorted([p for p in info_b if p["grew"] and p["prob"] <= 0.03], key=lambda x: -x["future_peak"])
    for p in missed[:10]:
        print("%-28s %5d %6dK %4.0f%% %5.0f%% %4.0f%% grew->%d %s" % (
            p["pair"], p["posts"], p["up"] // 1000,
            p["alive_ratio"] * 100, p["surging_ratio"] * 100,
            p["prob"] * 100, p["future_peak"], p["title"][:30]))

    # Momentum signal comparison
    print("\n" + "=" * 70)
    print("  MOMENTUM SIGNAL CHECK")
    print("=" * 70)
    growing = [p for p in info_b if p["grew"]]
    not_growing = [p for p in info_b if not p["grew"]]
    if growing and not_growing:
        print("                        Growing    Not growing    Ratio")
        print("  Alive ratio:          %5.0f%%       %5.0f%%         %.1fx" % (
            np.mean([p["alive_ratio"] for p in growing]) * 100,
            np.mean([p["alive_ratio"] for p in not_growing]) * 100,
            np.mean([p["alive_ratio"] for p in growing]) / max(0.01, np.mean([p["alive_ratio"] for p in not_growing]))))
        print("  Surging ratio:        %5.0f%%       %5.0f%%         %.1fx" % (
            np.mean([p["surging_ratio"] for p in growing]) * 100,
            np.mean([p["surging_ratio"] for p in not_growing]) * 100,
            np.mean([p["surging_ratio"] for p in growing]) / max(0.01, np.mean([p["surging_ratio"] for p in not_growing]))))
        print("  Best velocity:        %5.0f        %5.0f          %.1fx" % (
            np.mean([p["best_vel"] for p in growing]),
            np.mean([p["best_vel"] for p in not_growing]),
            np.mean([p["best_vel"] for p in growing]) / max(0.01, np.mean([p["best_vel"] for p in not_growing]))))


# --- Summary ---
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
if roc_a is not None:
    print("  A) Baseline (count only):   ROC AUC = %.3f" % roc_a)
if roc_b is not None:
    print("  B) With momentum:           ROC AUC = %.3f" % roc_b)
if roc_a and roc_b:
    diff = roc_b - roc_a
    print("  Momentum improvement:       %+.3f" % diff)
print("=" * 70)

conn.close()
print("\nDONE")
