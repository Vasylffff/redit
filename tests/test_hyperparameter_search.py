"""Quick hyperparameter search across top tasks and models."""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [], "velocities": []
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

days = sorted(day_pair.keys())
split = 8


def make_feats(d):
    bc = max(d["comments"]) if d["comments"] else 0
    ac = sum(d["comments"]) / max(1, len(d["comments"]))
    vels = d["velocities"]
    return [d["posts"], d["total_up"], len(d["subs"]), bc, ac,
            d["total_up"] / max(1, d["posts"]),
            max(vels) if vels else 0,
            sum(vels) / max(1, len(vels))]


# Build emergence task (biggest dataset, most important)
print("Building emergence dataset...")
X_tr, y_tr, X_te, y_te = [], [], [], []
for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
    for day in day_range:
        di = days.index(day)
        if di + 3 >= len(days):
            continue
        for pair, d in day_pair[day].items():
            if d["posts"] < 1 or d["posts"] > 3:
                continue
            peak = max(day_pair[days[j]].get(pair, {}).get("posts", 0)
                       for j in range(di + 1, min(di + 4, len(days))))
            grew = 1 if peak >= 5 else 0
            feats = make_feats(d)
            if phase == "train":
                X_tr.append(feats)
                y_tr.append(grew)
            else:
                X_te.append(feats)
                y_te.append(grew)

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_te, y_te = np.array(X_te), np.array(y_te)
# Subsample training data for speed (GBM is slow on 200K)
np.random.seed(42)
if len(X_tr) > 30000:
    idx = np.random.choice(len(X_tr), 30000, replace=False)
    X_tr, y_tr = X_tr[idx], y_tr[idx]
print("Train: %d, Test: %d (pos: %d)" % (len(y_tr), len(y_te), sum(y_te)))

# Scale for LogReg
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# ================================================================
# GRID SEARCH
# ================================================================
print("\n" + "=" * 70)
print("  HYPERPARAMETER SEARCH — Emergence Detection")
print("=" * 70)

configs = [
    # Random Forest variations
    ("RF depth=4", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)),
    ("RF depth=6", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ("RF depth=8", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ("RF depth=12", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)),
    ("RF depth=None", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)),
    ("RF n=50", RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
    ("RF n=100", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
    ("RF n=500", RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42)),
    ("RF msl=1", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=1, random_state=42)),
    ("RF msl=5", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)),
    ("RF msl=20", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42)),
    ("RF msl=50", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=50, random_state=42)),

    # Extra Trees variations
    ("ET depth=6", ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ("ET depth=8", ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ("ET depth=12", ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42)),
    ("ET depth=None", ExtraTreesClassifier(n_estimators=200, max_depth=None, random_state=42)),

    # Gradient Boosting variations
    ("GBM lr=0.01", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.01, random_state=42)),
    ("GBM lr=0.05", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ("GBM lr=0.1", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)),
    ("GBM lr=0.2", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.2, random_state=42)),
    ("GBM d=2 lr=0.1", GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.1, random_state=42)),
    ("GBM d=6 lr=0.1", GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)),
    ("GBM n=300 lr=0.05", GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)),
    ("GBM sub=0.8", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)),

    # Decision Tree variations
    ("DT depth=3", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ("DT depth=5", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("DT depth=8", DecisionTreeClassifier(max_depth=8, random_state=42)),
    ("DT depth=12", DecisionTreeClassifier(max_depth=12, random_state=42)),
    ("DT entropy d=8", DecisionTreeClassifier(max_depth=8, criterion="entropy", random_state=42)),
]

# LogReg needs scaled data — handle separately
logreg_configs = [
    ("LR C=0.01", LogisticRegression(C=0.01, max_iter=3000, random_state=42)),
    ("LR C=0.1", LogisticRegression(C=0.1, max_iter=3000, random_state=42)),
    ("LR C=1.0", LogisticRegression(C=1.0, max_iter=3000, random_state=42)),
    ("LR C=10", LogisticRegression(C=10, max_iter=3000, random_state=42)),
    ("LR C=100", LogisticRegression(C=100, max_iter=3000, random_state=42)),
    ("LR L1 C=1", LogisticRegression(C=1.0, penalty="l1", solver="saga", max_iter=3000, random_state=42)),
    ("LR L1 C=10", LogisticRegression(C=10, penalty="l1", solver="saga", max_iter=3000, random_state=42)),
]

results = []

for name, model in configs:
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, probs)
    results.append((name, roc))

for name, model in logreg_configs:
    model.fit(X_tr_scaled, y_tr)
    probs = model.predict_proba(X_te_scaled)[:, 1]
    roc = roc_auc_score(y_te, probs)
    results.append((name, roc))

# Sort by ROC
results.sort(key=lambda x: -x[1])

print("\n  %-25s %10s" % ("Configuration", "ROC AUC"))
print("  " + "-" * 38)
for name, roc in results:
    marker = " <-- BEST" if roc == results[0][1] else ""
    print("  %-25s %10.3f%s" % (name, roc, marker))

# Summary by model family
print("\n" + "=" * 70)
print("  BEST PER MODEL FAMILY")
print("=" * 70)
families = {"RF": [], "ET": [], "GBM": [], "DT": [], "LR": []}
for name, roc in results:
    for prefix in families:
        if name.startswith(prefix):
            families[prefix].append((name, roc))
            break

print("\n  %-12s %-25s %10s %10s" % ("Family", "Best Config", "Default", "Tuned"))
print("  " + "-" * 60)
defaults = {"RF": None, "ET": None, "GBM": None, "DT": None, "LR": None}
for name, roc in results:
    if name == "RF depth=8" and defaults["RF"] is None:
        defaults["RF"] = roc
    elif name == "ET depth=8" and defaults["ET"] is None:
        defaults["ET"] = roc
    elif name == "GBM lr=0.1" and defaults["GBM"] is None:
        defaults["GBM"] = roc
    elif name == "DT depth=8" and defaults["DT"] is None:
        defaults["DT"] = roc
    elif name == "LR C=1.0" and defaults["LR"] is None:
        defaults["LR"] = roc

for prefix in ["RF", "ET", "GBM", "LR", "DT"]:
    if families[prefix]:
        best = max(families[prefix], key=lambda x: x[1])
        default = defaults.get(prefix, 0)
        improvement = best[1] - default if default else 0
        print("  %-12s %-25s %10.3f %9.3f (%+.3f)" % (
            prefix, best[0], default or 0, best[1], improvement))

conn.close()
print("\nDONE")
