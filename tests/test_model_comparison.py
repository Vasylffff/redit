"""Compare all classifiers across all topic-level prediction tasks.

Tests: Random Forest, Extra Trees, Gradient Boosting, Logistic Regression,
       Decision Tree across every task we've built.
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour,
       p.activity_state
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

# Build pair data
day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [],
    "velocities": [], "comment_vels": [], "titles": [], "states": []
}))
day_pair_subs = defaultdict(lambda: defaultdict(set))
seen = set()
for pid, title, day, max_up, sub, max_com, vel, cvel, state in rows:
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
            d["states"].append(state or "dead")
            if len(d["titles"]) < 2:
                d["titles"].append(title[:80])
            day_pair_subs[day][pair].add(sub)

days = sorted(day_pair.keys())
split = 8
print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))
print()

MODELS = [
    ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ("Extra Trees", lambda: ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ("Grad. Boosting", lambda: GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ("Logistic Reg.", lambda: LogisticRegression(max_iter=2000, random_state=42)),
    ("Decision Tree", lambda: DecisionTreeClassifier(max_depth=8, random_state=42)),
]


def run_comparison(task_name, X_tr, y_tr, X_te, y_te):
    """Run all models on a task, return dict of ROCs."""
    results = {}
    for name, model_fn in MODELS:
        try:
            model = model_fn()
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            roc = roc_auc_score(y_te, probs)
            results[name] = roc
        except Exception as e:
            results[name] = None
    return results


def make_base_feats(d):
    bc = max(d["comments"]) if d["comments"] else 0
    ac = sum(d["comments"]) / max(1, len(d["comments"]))
    vels = d["velocities"]
    return [d["posts"], d["total_up"], len(d["subs"]), bc, ac,
            d["total_up"] / max(1, d["posts"]),
            max(vels) if vels else 0,
            sum(vels) / max(1, len(vels))]


all_results = {}

# ================================================================
# TASK 1: Emergence detection (1-3 -> 5+)
# ================================================================
print("Building Task 1: Emergence detection (1-3 -> 5+)...")
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
            feats = make_base_feats(d)
            if phase == "train":
                X_tr.append(feats); y_tr.append(grew)
            else:
                X_te.append(feats); y_te.append(grew)
X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
if sum(y_tr) >= 5 and sum(y_te) > 0:
    all_results["Emergence (1-3->5+)"] = run_comparison("Emergence", X_tr, y_tr, X_te, y_te)

# ================================================================
# TASK 2: Escalation (4-7 -> 10+)
# ================================================================
print("Building Task 2: Escalation (4-7 -> 10+)...")
X_tr, y_tr, X_te, y_te = [], [], [], []
for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
    for day in day_range:
        di = days.index(day)
        if di + 3 >= len(days):
            continue
        for pair, d in day_pair[day].items():
            if d["posts"] < 4 or d["posts"] > 7:
                continue
            peak = max(day_pair[days[j]].get(pair, {}).get("posts", 0)
                       for j in range(di + 1, min(di + 4, len(days))))
            grew = 1 if peak >= 10 else 0
            feats = make_base_feats(d)
            if phase == "train":
                X_tr.append(feats); y_tr.append(grew)
            else:
                X_te.append(feats); y_te.append(grew)
X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
if sum(y_tr) >= 5 and sum(y_te) > 0:
    all_results["Escalation (4-7->10+)"] = run_comparison("Escalation", X_tr, y_tr, X_te, y_te)

# ================================================================
# TASK 3: Failure filter (won't reach 5+)
# ================================================================
print("Building Task 3: Failure filter...")
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
            fails = 1 if peak < 5 else 0
            feats = make_base_feats(d)
            if phase == "train":
                X_tr.append(feats); y_tr.append(fails)
            else:
                X_te.append(feats); y_te.append(fails)
X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
if sum(y_tr) >= 5 and 0 < sum(y_te) < len(y_te):
    all_results["Failure filter"] = run_comparison("Failure", X_tr, y_tr, X_te, y_te)

# ================================================================
# TASK 4: Will it die tomorrow?
# ================================================================
print("Building Task 4: Topic death tomorrow...")
pair_trajectories = defaultdict(dict)
for day in days:
    for pair, d in day_pair[day].items():
        pair_trajectories[pair][day] = d

X_tr, y_tr, X_te, y_te = [], [], [], []
for pair, daily in pair_trajectories.items():
    pair_days = sorted(daily.keys())
    if len(pair_days) < 2 or sum(daily[d]["posts"] for d in pair_days) < 5:
        continue
    for idx in range(len(pair_days) - 1):
        day = pair_days[idx]
        di = days.index(day)
        next_global = days[di + 1] if di + 1 < len(days) else None
        next_pair = pair_days[idx + 1]
        if next_global and next_global != next_pair:
            dies = 1
        else:
            dies = 1 if daily[next_pair]["posts"] < 2 else 0
        d = daily[day]
        if idx > 0:
            prev = daily[pair_days[idx - 1]]
            pt = d["posts"] / max(1, prev["posts"])
        else:
            pt = 1.0
        feats = make_base_feats(d) + [pt, idx]
        phase = "train" if day < days[split] else "test"
        if phase == "train":
            X_tr.append(feats); y_tr.append(dies)
        else:
            X_te.append(feats); y_te.append(dies)
X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
if sum(y_tr) >= 10 and 0 < sum(y_te) < len(y_te):
    all_results["Death tomorrow"] = run_comparison("Death", X_tr, y_tr, X_te, y_te)

# ================================================================
# TASK 5: Subreddit spread
# ================================================================
print("Building Task 5: Subreddit spread...")
X_tr, y_tr, X_te, y_te = [], [], [], []
for phase, day_range in [("train", days[:split]), ("test", days[split:])]:
    for day in day_range:
        di = days.index(day)
        if di + 1 >= len(days):
            continue
        tomorrow = days[di + 1]
        for pair in day_pair_subs[day]:
            today_subs = day_pair_subs[day][pair]
            tmrw_subs = day_pair_subs[tomorrow].get(pair, set())
            spread = 1 if len(tmrw_subs - today_subs) > 0 else 0
            if len(today_subs) < 1 or len(today_subs) >= 4:
                continue
            d = day_pair[day][pair]
            feats = make_base_feats(d)
            if phase == "train":
                X_tr.append(feats); y_tr.append(spread)
            else:
                X_te.append(feats); y_te.append(spread)
X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
if sum(y_tr) >= 10 and 0 < sum(y_te) < len(y_te):
    all_results["Subreddit spread"] = run_comparison("Spread", X_tr, y_tr, X_te, y_te)

# ================================================================
# TASK 6: Still growing or peaked?
# ================================================================
print("Building Task 6: Peaked or growing...")
X_tr, y_tr, X_te, y_te = [], [], [], []
for pair, daily in pair_trajectories.items():
    pair_days = sorted(daily.keys())
    if len(pair_days) < 3 or sum(daily[d]["posts"] for d in pair_days) < 5:
        continue
    counts = [daily[d]["posts"] for d in pair_days]
    peak_idx = int(np.argmax(counts))
    for idx in range(len(pair_days)):
        if idx < 1:
            continue
        d = daily[pair_days[idx]]
        prev = daily[pair_days[idx - 1]]
        peaks_today = 1 if idx >= peak_idx else 0
        # Growth features
        if idx >= 2:
            d2 = daily[pair_days[idx - 2]]
            growth = d["posts"] / max(1, prev["posts"])
            up_growth = d["total_up"] / max(1, prev["total_up"])
        else:
            growth = 1.0
            up_growth = 1.0
        feats = make_base_feats(d) + [growth, up_growth]
        phase = "train" if pair_days[idx] < days[split] else "test"
        if phase == "train":
            X_tr.append(feats); y_tr.append(peaks_today)
        else:
            X_te.append(feats); y_te.append(peaks_today)
X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
if sum(y_tr) >= 10 and 0 < sum(y_te) < len(y_te):
    all_results["Peaked or growing"] = run_comparison("Peaked", X_tr, y_tr, X_te, y_te)

# ================================================================
# TASK 7: Quick death vs slow death
# ================================================================
print("Building Task 7: Quick death vs slow death...")
X_tr, y_tr, X_te, y_te = [], [], [], []
all_pairs = set()
for d in days:
    all_pairs.update(day_pair[d].keys())
for pair in all_pairs:
    counts = [day_pair[d].get(pair, {}).get("posts", 0) for d in days]
    peak_idx = int(np.argmax(counts))
    peak_count = counts[peak_idx]
    if peak_count < 3:
        continue
    death_idx = None
    for i in range(peak_idx, len(counts) - 1):
        if counts[i] < 2 and counts[i + 1] < 2:
            death_idx = i
            break
    if death_idx is None:
        continue
    dtd = death_idx - peak_idx
    quick = 1 if dtd <= 1 else 0
    peak_d = day_pair[days[peak_idx]].get(pair, {})
    if not peak_d or peak_d["posts"] < 1:
        continue
    decline = counts[peak_idx + 1] / peak_count if peak_idx + 1 < len(counts) and peak_count > 0 else 0
    feats = make_base_feats(peak_d) + [decline]
    phase = "train" if days[peak_idx] < days[split] else "test"
    if phase == "train":
        X_tr.append(feats); y_tr.append(quick)
    else:
        X_te.append(feats); y_te.append(quick)
X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
if sum(y_tr) >= 5 and 0 < sum(y_te) < len(y_te):
    all_results["Quick vs slow death"] = run_comparison("QuickDeath", X_tr, y_tr, X_te, y_te)

# ================================================================
# PRINT RESULTS TABLE
# ================================================================
conn.close()

print("\n" + "=" * 90)
print("  MODEL COMPARISON ACROSS ALL TASKS (ROC AUC)")
print("=" * 90)

model_names = [n for n, _ in MODELS]
header = "%-25s" % "Task"
for n in model_names:
    header += " %14s" % n
print(header)
print("-" * 90)

# Track wins
wins = defaultdict(int)
for task, results in sorted(all_results.items()):
    row = "%-25s" % task
    best = max(v for v in results.values() if v is not None)
    for n in model_names:
        v = results.get(n)
        if v is None:
            row += " %14s" % "FAIL"
        elif v == best:
            row += " %13.3f*" % v
            wins[n] += 1
        else:
            row += " %14.3f" % v
    print(row)

print("-" * 90)
print("%-25s" % "WINS", end="")
for n in model_names:
    print(" %14d" % wins[n], end="")
print()
print("=" * 90)
print("* = best model for this task")
print("\nDONE")
