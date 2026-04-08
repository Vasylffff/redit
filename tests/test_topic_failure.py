"""Predict which topics will NOT grow — dead on arrival.

The flip side of emergence detection: instead of catching winners,
identify losers early so you can ignore them.
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

feat_names = ["posts", "total_upvotes", "subs", "best_comments", "avg_comments",
              "up_per_post", "best_velocity", "avg_velocity", "best_comment_vel"]


def make_feats(d):
    bc = max(d["comments"]) if d["comments"] else 0
    ac = sum(d["comments"]) / max(1, len(d["comments"]))
    vels = d["velocities"]
    cvels = d["comment_vels"]
    return [
        d["posts"], d["total_up"], len(d["subs"]), bc, ac,
        d["total_up"] / max(1, d["posts"]),
        max(vels) if vels else 0,
        sum(vels) / max(1, len(vels)),
        max(cvels) if cvels else 0,
    ]


# ================================================================
# TEST MULTIPLE THRESHOLDS
# ================================================================
print("\n" + "=" * 70)
print("  PREDICTING TOPIC FAILURE AT DIFFERENT THRESHOLDS")
print("=" * 70)

for threshold, label in [(3, "never reach 3+ posts"), (5, "never reach 5+ posts")]:
    X_tr, y_tr, X_te, y_te, info_te = [], [], [], [], []

    for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
        for day in day_range:
            di = days.index(day)
            if di + 3 >= len(days):
                continue
            for pair, d in day_pair[day].items():
                if d["posts"] < 1 or d["posts"] > 3:
                    continue
                future = [day_pair[days[j]].get(pair, {}).get("posts", 0)
                          for j in range(di + 1, min(di + 4, len(days)))]
                peak = max(future) if future else 0
                fails = 1 if peak < threshold else 0

                feats = make_feats(d)
                if phase == "train":
                    X_tr.append(feats)
                    y_tr.append(fails)
                else:
                    X_te.append(feats)
                    y_te.append(fails)
                    info_te.append({
                        "pair": pair, "posts": d["posts"], "up": d["total_up"],
                        "peak": peak, "fails": fails,
                        "best_vel": max(d["velocities"]) if d["velocities"] else 0,
                        "title": d["titles"][0] if d["titles"] else "?",
                    })

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_te, y_te = np.array(X_te), np.array(y_te)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, probs)

    print("\n  --- %s ---" % label)
    print("  Train: %d pairs, %d fail (%.1f%%)" % (len(y_tr), sum(y_tr), sum(y_tr) / len(y_tr) * 100))
    print("  Test:  %d pairs, %d fail (%.1f%%)" % (len(y_te), sum(y_te), sum(y_te) / len(y_te) * 100))
    print("  ROC AUC: %.3f" % roc)

    print("  Feature importance:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        print("    %-22s %.1f%%" % (name, imp * 100))

    for i, inf in enumerate(info_te):
        inf["prob"] = probs[i]

    # Profile comparison
    dead = [p for p in info_te if p["fails"]]
    grew = [p for p in info_te if not p["fails"]]

    if dead and grew:
        print("\n  Profile comparison:")
        print("  %-22s %10s %10s %8s" % ("Signal", "Fails", "Grows", "Ratio"))
        print("  " + "-" * 55)
        for name, fn in [
            ("Avg upvotes", lambda p: p["up"]),
            ("Avg best velocity", lambda p: p["best_vel"]),
            ("Avg posts day 1", lambda p: p["posts"]),
        ]:
            d_val = np.mean([fn(p) for p in dead])
            g_val = np.mean([fn(p) for p in grew])
            ratio = g_val / max(0.01, d_val)
            print("  %-22s %10.0f %10.0f %7.1fx" % (name, d_val, g_val, ratio))

    # Surprises: model said dead but it grew
    print("\n  SURPRISES (model said fail >80%%, but it grew):")
    print("  %-25s %5s %6s %5s %5s %s" % ("Topic", "Posts", "UpK", "Prob", "Peak", "Title"))
    print("  " + "-" * 85)
    surprises = sorted([p for p in info_te if p["prob"] > 0.8 and not p["fails"]],
                       key=lambda x: -x["peak"])
    for p in surprises[:10]:
        print("  %-25s %5d %6dK %4.0f%% %5d  %s" % (
            p["pair"], p["posts"], p["up"] // 1000,
            p["prob"] * 100, p["peak"], p["title"][:35]))

    # How many can we filter out?
    print("\n  FILTERING POWER (at different confidence levels):")
    for conf in [0.9, 0.95, 0.99]:
        flagged = sum(1 for p in info_te if p["prob"] > conf)
        flagged_correct = sum(1 for p in info_te if p["prob"] > conf and p["fails"])
        flagged_wrong = flagged - flagged_correct
        total_fails = sum(y_te)
        if flagged > 0:
            precision = flagged_correct / flagged * 100
            recall = flagged_correct / max(1, total_fails) * 100
            print("    >%.0f%% confidence: flag %d (%.1f%% of all), precision=%.0f%%, recall=%.0f%%, wrong=%d" % (
                conf * 100, flagged, flagged / len(y_te) * 100,
                precision, recall, flagged_wrong))

conn.close()
print("\nDONE")
