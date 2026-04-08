"""Test: can we predict HOW viral a topic will get?

Three approaches:
  1) Size buckets: small / medium / big / viral
  2) Log-scale regression: predict log(peak) so model isn't scared of big numbers
  3) Growth multiplier: will it 2x? 3x? 5x?
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [], "titles": [], "max_single": 0
}))
seen = set()
for pid, title, day, max_up, sub, max_com in rows:
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
            if len(d["titles"]) < 3:
                d["titles"].append(title[:80])

days = sorted(day_pair.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))


def make_features(d):
    bc = max(d["comments"]) if d["comments"] else 0
    ac = sum(d["comments"]) / max(1, len(d["comments"]))
    return [d["posts"], d["total_up"], len(d["subs"]), bc, ac,
            d["total_up"] / max(1, d["posts"]), d["max_single"]]


def get_peak(day, pair):
    di = days.index(day)
    future = [day_pair[days[j]].get(pair, {}).get("posts", 0)
              for j in range(di + 1, min(di + 4, len(days)))]
    return max(future) if future else 0


# ================================================================
# APPROACH 1: Size buckets
# ================================================================
print("\n" + "=" * 70)
print("  APPROACH 1: Predict SIZE BUCKET (small / medium / big / viral)")
print("=" * 70)

bucket_names = ["small (1-3)", "medium (4-7)", "big (8-15)", "viral (16+)"]


def get_bucket(peak):
    if peak <= 3:
        return 0
    if peak <= 7:
        return 1
    if peak <= 15:
        return 2
    return 3


X_tr, y_tr, X_te, y_te, info_te = [], [], [], [], []
for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
    for day in day_range:
        di = days.index(day)
        if di + 3 >= len(days):
            continue
        for pair, d in day_pair[day].items():
            if d["posts"] < 1 or d["posts"] > 3:
                continue
            peak = get_peak(day, pair)
            bucket = get_bucket(peak)
            feats = make_features(d)
            if phase == "train":
                X_tr.append(feats)
                y_tr.append(bucket)
            else:
                X_te.append(feats)
                y_te.append(bucket)
                info_te.append({"pair": pair, "posts": d["posts"], "up": d["total_up"],
                                "peak": peak, "bucket": bucket,
                                "title": d["titles"][0] if d["titles"] else "?"})

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_te, y_te = np.array(X_te), np.array(y_te)
print("Train: %d, Test: %d" % (len(y_tr), len(y_te)))
for b in range(4):
    print("  %s: train=%d test=%d" % (bucket_names[b], sum(y_tr == b), sum(y_te == b)))

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_tr, y_tr)
preds = rf.predict(X_te)
probs = rf.predict_proba(X_te)

print("\nAccuracy: %.1f%%" % (sum(preds == y_te) / len(y_te) * 100))

print("\nPer-bucket ROC AUC:")
for b in range(4):
    if sum(y_te == b) > 0 and sum(y_te == b) < len(y_te) and b < probs.shape[1]:
        roc = roc_auc_score((y_te == b).astype(int), probs[:, b])
        print("  %s: %.3f" % (bucket_names[b], roc))

print("\nBIG/VIRAL TOPICS -- predicted bucket:")
print("%-28s %5s %5s %-15s %-15s %s" % ("Topic", "Posts", "Peak", "Actual", "Predicted", "Title"))
print("-" * 100)
for i, inf in enumerate(info_te):
    inf["pred_bucket"] = preds[i]
big = sorted([p for p in info_te if p["bucket"] >= 2], key=lambda x: -x["peak"])
for p in big[:15]:
    print("%-28s %5d %5d %-15s %-15s %s" % (
        p["pair"], p["posts"], p["peak"], bucket_names[p["bucket"]],
        bucket_names[p["pred_bucket"]], p["title"][:35]))


# ================================================================
# APPROACH 2: Log-scale regression
# ================================================================
print("\n" + "=" * 70)
print("  APPROACH 2: Log-scale regression")
print("=" * 70)

X_tr2, y_tr2, X_te2, y_te2, info_te2 = [], [], [], [], []
for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
    for day in day_range:
        di = days.index(day)
        if di + 3 >= len(days):
            continue
        for pair, d in day_pair[day].items():
            if d["posts"] < 1 or d["posts"] > 3:
                continue
            peak = get_peak(day, pair)
            if peak < 1:
                continue
            feats = make_features(d)
            if phase == "train":
                X_tr2.append(feats)
                y_tr2.append(np.log1p(peak))
            else:
                X_te2.append(feats)
                y_te2.append(peak)
                info_te2.append({"pair": pair, "posts": d["posts"], "up": d["total_up"],
                                 "peak": peak, "title": d["titles"][0] if d["titles"] else "?"})

X_tr2, y_tr2 = np.array(X_tr2), np.array(y_tr2)
X_te2, y_te2 = np.array(X_te2), np.array(y_te2)

rf2 = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf2.fit(X_tr2, y_tr2)
preds2 = np.expm1(rf2.predict(X_te2))

r2 = r2_score(y_te2, preds2)
mae = mean_absolute_error(y_te2, preds2)
print("R2:  %.3f" % r2)
print("MAE: %.1f posts" % mae)

info_arr = list(zip(info_te2, preds2))
info_arr.sort(key=lambda x: -x[0]["peak"])
print("\nBIG TOPICS -- actual vs predicted:")
print("%-28s %5s %5s %6s %s" % ("Topic", "Posts", "Peak", "Pred", "Title"))
print("-" * 90)
for inf, pred in info_arr[:15]:
    print("%-28s %5d %5d %6.1f %s" % (inf["pair"], inf["posts"], inf["peak"], pred, inf["title"][:40]))

big2 = [(inf, pred) for inf, pred in info_arr if inf["peak"] >= 10]
if big2:
    print("\nFor 10+ topics: avg actual=%.0f, avg predicted=%.1f, MAE=%.1f" % (
        np.mean([x[0]["peak"] for x in big2]),
        np.mean([x[1] for x in big2]),
        mean_absolute_error([x[0]["peak"] for x in big2], [x[1] for x in big2])))


# ================================================================
# APPROACH 3: Growth multiplier
# ================================================================
print("\n" + "=" * 70)
print("  APPROACH 3: Growth multiplier (will it 2x? 3x? 5x?)")
print("=" * 70)

for mult_name, mult in [("2x", 2), ("3x", 3), ("5x", 5)]:
    X_tr3, y_tr3, X_te3, y_te3 = [], [], [], []
    for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
        for day in day_range:
            di = days.index(day)
            if di + 3 >= len(days):
                continue
            for pair, d in day_pair[day].items():
                if d["posts"] < 2 or d["posts"] > 5:
                    continue
                peak = get_peak(day, pair)
                grew = 1 if peak >= d["posts"] * mult else 0
                feats = make_features(d)
                if phase == "train":
                    X_tr3.append(feats)
                    y_tr3.append(grew)
                else:
                    X_te3.append(feats)
                    y_te3.append(grew)

    X_tr3, y_tr3 = np.array(X_tr3), np.array(y_tr3)
    X_te3, y_te3 = np.array(X_te3), np.array(y_te3)
    if sum(y_tr3) < 5 or sum(y_te3) == 0:
        print("  %s: not enough data (train_pos=%d, test_pos=%d)" % (mult_name, sum(y_tr3), sum(y_te3)))
        continue
    rf3 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf3.fit(X_tr3, y_tr3)
    roc3 = roc_auc_score(y_te3, rf3.predict_proba(X_te3)[:, 1])
    print("  Will it %s?  ROC AUC = %.3f  (test: %d pos / %d total, %.1f%% positive)" % (
        mult_name, roc3, sum(y_te3), len(y_te3), sum(y_te3) / len(y_te3) * 100))

conn.close()
print("\nDONE")
