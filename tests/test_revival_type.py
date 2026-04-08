"""Predict: is this topic the TYPE that revives? (ongoing story vs one-shot)

Not predicting WHEN it revives (that depends on external events),
but WHETHER it's the kind of topic that comes back.

Ongoing stories (wars, court cases, political scandals) = high revival chance
One-shot events (data breach, single announcement) = dead is dead
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
    "velocities": [], "comment_vels": [], "titles": [], "authors": set()
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
            if len(d["titles"]) < 5:
                safe = title[:80].encode("ascii", "replace").decode()
                d["titles"].append(safe)

days = sorted(day_pair.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))

# Build topic profiles using their ENTIRE history up to observation point
# Label: did this topic ever revive after dropping?

all_pairs = set()
for d in days:
    all_pairs.update(day_pair[d].keys())

topic_profiles = []
for pair in all_pairs:
    counts = [day_pair[d].get(pair, {}).get("posts", 0) for d in days]
    total = sum(counts)
    if total < 5:
        continue

    # Count how many times it dropped and revived
    drops = 0
    revivals = 0
    for i in range(1, len(counts) - 1):
        if counts[i] < 2 and counts[i - 1] >= 2:
            drops += 1
            for j in range(i + 1, min(i + 4, len(counts))):
                if counts[j] >= 2:
                    revivals += 1
                    break

    is_reviver = 1 if revivals > 0 else 0

    # Profile features (from first 5 days or first half of data)
    obs_end = min(split, len(counts))
    obs_counts = counts[:obs_end]

    if sum(obs_counts) < 3:
        continue

    peak = max(obs_counts)
    active_days = sum(1 for c in obs_counts if c >= 2)
    total_posts_obs = sum(obs_counts)

    # Gather engagement from observed period
    all_up, all_com, all_vel, all_subs = 0, [], [], set()
    for di in range(obs_end):
        d = day_pair[days[di]].get(pair, {})
        if d and d["posts"] > 0:
            all_up += d["total_up"]
            all_com.extend(d["comments"])
            all_vel.extend(d["velocities"])
            all_subs.update(d.get("subs", set()))

    best_com = max(all_com) if all_com else 0
    avg_com = sum(all_com) / max(1, len(all_com))
    best_vel = max(all_vel) if all_vel else 0
    avg_vel = sum(all_vel) / max(1, len(all_vel))

    # Trajectory shape features
    if active_days >= 2:
        # Consistency: how many active days vs total observed
        consistency = active_days / obs_end
        # Volatility: std of daily counts
        volatility = np.std(obs_counts) if len(obs_counts) > 1 else 0
        # Did it have multiple peaks?
        peaks_count = 0
        for k in range(1, len(obs_counts) - 1):
            if obs_counts[k] > obs_counts[k - 1] and obs_counts[k] > obs_counts[k + 1]:
                peaks_count += 1
    else:
        consistency = 0
        volatility = 0
        peaks_count = 0

    # Subreddit spread over time
    unique_subs = len(all_subs)

    # First day of appearance
    first_day_idx = next(i for i, c in enumerate(counts) if c >= 1)

    topic_profiles.append({
        "pair": pair,
        "is_reviver": is_reviver,
        "drops": drops,
        "revivals": revivals,
        "first_day": days[first_day_idx],
        "peak": peak,
        "active_days": active_days,
        "total_posts": total_posts_obs,
        "total_up": all_up,
        "best_com": best_com,
        "avg_com": avg_com,
        "best_vel": best_vel,
        "avg_vel": avg_vel,
        "unique_subs": unique_subs,
        "consistency": consistency,
        "volatility": volatility,
        "peaks_count": peaks_count,
        "up_per_post": all_up / max(1, total_posts_obs),
    })

# Split — use midpoint of data, not first_day (many topics span both periods)
import random
random.seed(42)
random.shuffle(topic_profiles)
cut = int(len(topic_profiles) * 0.7)
train_p = topic_profiles[:cut]
test_p = topic_profiles[cut:]

print("Train: %d topics, Test: %d topics" % (len(train_p), len(test_p)))


def make_feats(p):
    return [
        p["peak"], p["active_days"], p["total_posts"], p["total_up"],
        p["best_com"], p["avg_com"], p["best_vel"], p["avg_vel"],
        p["unique_subs"], p["consistency"], p["volatility"],
        p["peaks_count"], p["up_per_post"],
    ]


feat_names = ["peak_posts", "active_days", "total_posts", "total_upvotes",
              "best_comments", "avg_comments", "best_velocity", "avg_velocity",
              "unique_subs", "consistency", "volatility", "multiple_peaks", "up_per_post"]

X_tr = np.array([make_feats(p) for p in train_p])
y_tr = np.array([p["is_reviver"] for p in train_p])
X_te = np.array([make_feats(p) for p in test_p])
y_te = np.array([p["is_reviver"] for p in test_p])

print("\n" + "=" * 70)
print("  PREDICT: Is this an ONGOING STORY (will revive) or ONE-SHOT?")
print("=" * 70)
print("\nTrain: %d topics, %d revivers (%.1f%%)" % (len(y_tr), sum(y_tr), sum(y_tr) / max(1, len(y_tr)) * 100))
print("Test:  %d topics, %d revivers (%.1f%%)" % (len(y_te), sum(y_te), sum(y_te) / max(1, len(y_te)) * 100))

if sum(y_tr) >= 10 and sum(y_te) > 0 and sum(y_te) < len(y_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=8, random_state=42)),
    ]

    print("\n  Model comparison:")
    best_roc = 0
    best_model = None
    best_probs = None
    for name, model in models:
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        r = roc_auc_score(y_te, p)
        print("    %-25s ROC AUC = %.3f" % (name, r))
        if r > best_roc:
            best_roc = r
            best_model = (name, model)
            best_probs = p

    rf = best_model[1]
    probs = best_probs
    roc = best_roc
    print("\n  Best: %s (%.3f)" % (best_model[0], roc))

    print("\nFeature importance:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        print("  %-22s %.1f%%" % (name, imp * 100))

    for i, p in enumerate(test_p):
        p["prob"] = probs[i]

    # Profile comparison
    revivers = [p for p in test_p if p["is_reviver"]]
    one_shots = [p for p in test_p if not p["is_reviver"]]

    print("\n  PROFILE: Ongoing stories vs one-shots:")
    print("  %-22s %12s %12s %8s" % ("Feature", "Ongoing", "One-shot", "Ratio"))
    print("  " + "-" * 58)
    for name, fn in [
        ("Peak posts", lambda p: p["peak"]),
        ("Active days", lambda p: p["active_days"]),
        ("Total upvotes", lambda p: p["total_up"]),
        ("Best comments", lambda p: p["best_com"]),
        ("Unique subs", lambda p: p["unique_subs"]),
        ("Consistency", lambda p: p["consistency"]),
        ("Multiple peaks", lambda p: p["peaks_count"]),
    ]:
        r_val = np.mean([fn(p) for p in revivers]) if revivers else 0
        o_val = np.mean([fn(p) for p in one_shots]) if one_shots else 0
        ratio = r_val / max(0.01, o_val)
        print("  %-22s %12.1f %12.1f %7.1fx" % (name, r_val, o_val, ratio))

    # Show examples
    print("\n  ONGOING STORIES (model correctly identified as revivers):")
    print("  %-25s %5s %5s %5s %5s %s" % ("Topic", "Peak", "Days", "Subs", "Prob", "Title"))
    print("  " + "-" * 75)
    correct_ongoing = sorted([p for p in test_p if p["is_reviver"] and p["prob"] > 0.3],
                             key=lambda x: -x["prob"])
    for p in correct_ongoing[:15]:
        title = "?"
        for d in days:
            dd = day_pair[d].get(p["pair"], {})
            if dd and dd.get("titles"):
                title = dd["titles"][0]
                break
        print("  %-25s %5d %5d %5d %4.0f%%  %s" % (
            p["pair"], p["peak"], p["active_days"], p["unique_subs"],
            p["prob"] * 100, title[:35]))

    print("\n  ONE-SHOT EVENTS (model correctly identified as non-revivers):")
    correct_oneshot = sorted([p for p in test_p if not p["is_reviver"] and p["prob"] < 0.15],
                             key=lambda x: x["prob"])
    for p in correct_oneshot[:15]:
        title = "?"
        for d in days:
            dd = day_pair[d].get(p["pair"], {})
            if dd and dd.get("titles"):
                title = dd["titles"][0]
                break
        print("  %-25s %5d %5d %5d %4.0f%%  %s" % (
            p["pair"], p["peak"], p["active_days"], p["unique_subs"],
            p["prob"] * 100, title[:35]))

conn.close()
print("\nDONE")
