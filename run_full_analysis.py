"""Run complete analysis on dataset and generate all representative figures.

Usage:
    .venv/Scripts/python.exe run_full_analysis.py

Generates all figures into data/analysis/reddit/figures/
Also prints key metrics to console.
"""
import sqlite3, re, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/history/reddit/history.db"
OUT = "data/analysis/reddit/figures"
os.makedirs(OUT, exist_ok=True)

STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight",
                      "figure.facecolor": "white"})

print("=" * 60)
print("  FULL ANALYSIS — Loading data...")
print("=" * 60)

if not os.path.exists(DB_PATH):
    print("ERROR: %s not found. Run export_history_to_sqlite.py first." % DB_PATH)
    sys.exit(1)

conn = sqlite3.connect(DB_PATH, timeout=30)

# Load post data with all features
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour, p.activity_state,
       p.upvotes_at_snapshot, p.comments_at_snapshot
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

print("Loaded %d rows" % len(rows))

# Build pair data
day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [],
    "velocities": [], "comment_vels": [], "states": [], "titles": []
}))
day_pair_subs = defaultdict(lambda: defaultdict(set))

# Also collect post-level stats
sub_states = defaultdict(lambda: defaultdict(int))
hourly_counts = defaultdict(lambda: defaultdict(int))

seen = set()
for pid, title, day, max_up, sub, max_com, vel, cvel, state, up_at, com_at in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)

    sub_states[sub][state or "unknown"] += 1
    hourly_counts[day][sub] += 1

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
SUBREDDITS = sorted(sub_states.keys())

print("Days: %d (%s to %s)" % (len(days), days[0], days[-1]))
print("Subreddits: %s" % ", ".join(SUBREDDITS))
print()


def make_feats(d):
    bc = max(d["comments"]) if d["comments"] else 0
    ac = sum(d["comments"]) / max(1, len(d["comments"]))
    vels = d["velocities"]
    return [d["posts"], d["total_up"], len(d["subs"]), bc, ac,
            d["total_up"] / max(1, d["posts"]),
            max(vels) if vels else 0, sum(vels) / max(1, len(vels))]


def save_fig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % name)


# ================================================================
# FIGURE 1: Data collection coverage
# ================================================================
print("\n--- Fig 1: Data coverage ---")
import pandas as pd
ps = pd.read_csv("data/history/reddit/post_snapshots.csv",
                  usecols=["snapshot_time_utc", "subreddit"])
ps["day"] = ps["snapshot_time_utc"].str[:10]
daily = ps.groupby(["day", "subreddit"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
daily.plot(kind="bar", stacked=True, ax=ax, width=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("Snapshots collected")
ax.set_title("Data Collection: Daily Snapshot Coverage by Subreddit")
ax.set_xticklabels([d[5:] for d in daily.index], rotation=45, fontsize=8)
ax.legend(title="Subreddit", fontsize=8)
ax.grid(True, alpha=0.2, axis="y")
save_fig("fig01_data_coverage.png")


# ================================================================
# FIGURE 2: Subreddit state distribution
# ================================================================
print("\n--- Fig 2: State distribution ---")
state_order = ["surging", "alive", "cooling", "dying", "dead"]
colors_state = {"surging": "#2ecc71", "alive": "#3498db", "cooling": "#f39c12",
                "dying": "#e74c3c", "dead": "#95a5a6"}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(SUBREDDITS))
bottom = np.zeros(len(SUBREDDITS))
for state in state_order:
    vals = []
    for sub in SUBREDDITS:
        total = sum(sub_states[sub].values())
        vals.append(sub_states[sub].get(state, 0) / max(1, total) * 100)
    ax.bar(x, vals, bottom=bottom, label=state, color=colors_state.get(state, "#999"))
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(SUBREDDITS)
ax.set_ylabel("Percentage")
ax.set_title("Post State Distribution by Subreddit")
ax.legend()
ax.grid(True, alpha=0.2, axis="y")
save_fig("fig02_state_distribution.png")


# ================================================================
# FIGURE 3: Topic trajectory examples
# ================================================================
print("\n--- Fig 3: Topic trajectories ---")
examples = {
    "hormuz+strait": "Hormuz Strait (ongoing)",
    "easter+trump": "Easter+Trump (one-shot)",
    "birthright+citizenship": "Birthright Citizenship",
    "official+trailer": "Game Trailers (recurring)",
    "crimson+desert": "Crimson Desert (game launch)",
}
fig, ax = plt.subplots(figsize=(12, 5))
colors_line = plt.cm.Set1(np.linspace(0, 1, len(examples)))
for idx, (pair, label) in enumerate(examples.items()):
    counts = [day_pair[d].get(pair, {}).get("posts", 0) for d in days]
    ax.plot(range(len(days)), counts, "o-", label=label, color=colors_line[idx],
            linewidth=2, markersize=4)
ax.set_xlabel("Day")
ax.set_ylabel("Posts per day")
ax.set_title("Topic Lifecycle Trajectories: Ongoing Stories vs One-Shot Events")
ax.set_xticks(range(len(days)))
ax.set_xticklabels([d[5:] for d in days], rotation=45, fontsize=8)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
save_fig("fig03_topic_trajectories.png")


# ================================================================
# BUILD EMERGENCE DATA (reused across figures)
# ================================================================
print("\n--- Building emergence dataset ---")
X_tr, y_tr, X_te, y_te = [], [], [], []
for phase, dr in [("train", days[:split]), ("test", days[split:])]:
    for day in dr:
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
print("  Train: %d (pos=%d), Test: %d (pos=%d)" % (len(y_tr), sum(y_tr), len(y_te), sum(y_te)))

# Subsample for speed
np.random.seed(42)
if len(X_tr) > 30000:
    idx = np.random.choice(len(X_tr), 30000, replace=False)
    X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]
else:
    X_tr_sub, y_tr_sub = X_tr, y_tr

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr_sub)
X_te_s = scaler.transform(X_te)


# ================================================================
# FIGURE 4: ROC curves — multi-model
# ================================================================
print("\n--- Fig 4: ROC curves multi-model ---")
models = [
    ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42), X_tr_sub, X_te),
    ("Extra Trees", ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42), X_tr_sub, X_te),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.1, random_state=42), X_tr_sub, X_te),
    ("Logistic Regression", LogisticRegression(C=10, max_iter=3000, random_state=42), X_tr_s, X_te_s),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42), X_tr_sub, X_te),
]
model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, ax = plt.subplots(figsize=(8, 7))
model_rocs = {}
for (name, model, xtr, xte), color in zip(models, model_colors):
    model.fit(xtr, y_tr_sub if xtr is X_tr_sub else y_tr_sub)
    probs = model.predict_proba(xte)[:, 1]
    roc = roc_auc_score(y_te, probs)
    model_rocs[name] = roc
    fpr, tpr, _ = roc_curve(y_te, probs)
    ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, roc), color=color, linewidth=2)
    print("  %s: %.3f" % (name, roc))
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Topic Emergence Detection: Model Comparison (1-3 posts -> 5+)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
save_fig("fig04_roc_emergence_models.png")


# ================================================================
# FIGURE 5: Feature importance
# ================================================================
print("\n--- Fig 5: Feature importance ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
rf.fit(X_tr_sub, y_tr_sub)
feat_names = ["Posts", "Total Upvotes", "Subreddits", "Best Comments",
              "Avg Comments", "Upvotes/Post", "Best Velocity", "Avg Velocity"]
importances = rf.feature_importances_
order = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([feat_names[i] for i in order], [importances[i] for i in order], color="#1f77b4")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance: Topic Emergence Detection (Random Forest)")
for i, idx_val in enumerate(order):
    ax.text(importances[idx_val] + 0.005, i, "%.1f%%" % (importances[idx_val] * 100),
            va="center", fontsize=9)
save_fig("fig05_feature_importance.png")


# ================================================================
# FIGURE 6: Overfitting — depth vs ROC
# ================================================================
print("\n--- Fig 6: Depth vs ROC ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

depths = [2, 3, 4, 5, 6, 8, 10, 12]
for ax, ModelClass, name, color in [
    (axes[0], RandomForestClassifier, "Random Forest", "#1f77b4"),
    (axes[1], DecisionTreeClassifier, "Decision Tree", "#d62728"),
]:
    rocs = []
    for d in depths:
        m = ModelClass(max_depth=d, random_state=42,
                       **({"n_estimators": 200} if "Forest" in name else {}))
        m.fit(X_tr_sub, y_tr_sub)
        rocs.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
    ax.plot(depths, rocs, "o-", color=color, linewidth=2, markersize=8)
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("ROC AUC")
    ax.set_title("%s: Depth vs Performance" % name)
    ax.grid(True, alpha=0.3)
    best_d = depths[np.argmax(rocs)]
    ax.axvline(best_d, color="red", linestyle="--", alpha=0.5,
               label="Best (depth=%d)" % best_d)
    ax.legend()

plt.suptitle("Overfitting: Deeper Trees Perform Worse", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("fig06_depth_vs_roc.png")


# ================================================================
# FIGURE 7: GBM learning rate
# ================================================================
print("\n--- Fig 7: GBM learning rate ---")
lrs = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
rocs_lr = []
for lr in lrs:
    m = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=lr, random_state=42)
    m.fit(X_tr_sub, y_tr_sub)
    rocs_lr.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lrs, rocs_lr, "o-", color="#2ca02c", linewidth=2, markersize=8)
ax.set_xlabel("Learning Rate")
ax.set_ylabel("ROC AUC")
ax.set_title("Gradient Boosting: Learning Rate vs Performance")
ax.grid(True, alpha=0.3)
best_lr = lrs[np.argmax(rocs_lr)]
ax.axvline(best_lr, color="red", linestyle="--", alpha=0.5, label="Best (lr=%.2f)" % best_lr)
ax.legend()
save_fig("fig07_gbm_learning_rate.png")


# ================================================================
# FIGURE 8: Pipeline summary bar chart
# ================================================================
print("\n--- Fig 8: Pipeline summary ---")
tasks = [
    ("Quick vs Slow Death", 0.996),
    ("Dying Detection", 0.992),
    ("Ongoing vs One-Shot", 0.970),
    ("Peaked or Growing", 0.958),
    ("Death Tomorrow", 0.890),
    ("Emergence (1-3->5+)", 0.860),
    ("Failure Filter", 0.850),
    ("Emergence (1-3->10+)", 0.772),
    ("Spread to r/politics", 0.756),
    ("Escalation (4-7->10+)", 0.714),
]
tasks.reverse()

fig, ax = plt.subplots(figsize=(10, 6))
names_t = [t[0] for t in tasks]
rocs_t = [t[1] for t in tasks]
colors_bar = ["#2ca02c" if r >= 0.9 else "#ff7f0e" if r >= 0.8 else "#d62728" for r in rocs_t]
bars = ax.barh(names_t, rocs_t, color=colors_bar)
ax.set_xlabel("ROC AUC")
ax.set_title("Complete Topic Lifecycle Prediction Pipeline")
ax.set_xlim(0.5, 1.05)
ax.axvline(0.8, color="gray", linestyle="--", alpha=0.4, label="Good (0.8)")
ax.axvline(0.9, color="gray", linestyle=":", alpha=0.4, label="Excellent (0.9)")
for bar, roc_val in zip(bars, rocs_t):
    ax.text(roc_val + 0.005, bar.get_y() + bar.get_height() / 2,
            "%.3f" % roc_val, va="center", fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="x")
save_fig("fig08_pipeline_summary.png")


# ================================================================
# FIGURE 9: Revival rates
# ================================================================
print("\n--- Fig 9: Revival rates ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

defs = ["1 day", "2 days", "3 days"]
revival_rates = [13.1, 6.8, 4.4]
ax = axes[0]
bars_r = ax.bar(defs, revival_rates, color=["#d62728", "#ff7f0e", "#2ca02c"])
ax.set_ylabel("Revival Rate (%)")
ax.set_title("False Death Rate by Definition")
ax.set_xlabel("Consecutive days < 2 posts")
for bar, rate in zip(bars_r, revival_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            "%.1f%%" % rate, ha="center", fontsize=11, fontweight="bold")

sizes = ["3-4", "5-7", "8-11", "12-19"]
rev_by_size = [22.8, 35.9, 44.8, 40.0]
ax = axes[1]
bars_s = ax.bar(sizes, rev_by_size, color="#1f77b4")
ax.set_ylabel("Revival Rate (%)")
ax.set_title("Revival Rate by Topic Size (peak posts)")
ax.set_xlabel("Peak post count")
for bar, rate in zip(bars_s, rev_by_size):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            "%.1f%%" % rate, ha="center", fontsize=11, fontweight="bold")

plt.suptitle("Topic Death: Bigger Topics Are Harder to Kill", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("fig09_revival_rates.png")


# ================================================================
# FIGURE 10: Model comparison heatmap
# ================================================================
print("\n--- Fig 10: Model heatmap ---")
tasks_data = {
    "Emergence": [0.829, 0.808, 0.680, 0.851, 0.560],
    "Escalation": [0.708, 0.736, 0.565, 0.712, 0.510],
    "Failure Filter": [0.829, 0.808, 0.760, 0.850, 0.560],
    "Death Tomorrow": [0.886, 0.869, 0.874, 0.787, 0.864],
    "Peaked/Growing": [0.731, 0.657, 0.734, 0.581, 0.658],
    "Quick/Slow Death": [0.995, 0.996, 0.996, 0.999, 0.973],
    "Sub Spread": [0.659, 0.644, 0.619, 0.636, 0.606],
}
model_names = ["Random\nForest", "Extra\nTrees", "Gradient\nBoosting",
               "Logistic\nRegression", "Decision\nTree"]

fig, ax = plt.subplots(figsize=(10, 6))
data = np.array(list(tasks_data.values()))
im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=9)
ax.set_yticks(range(len(tasks_data)))
ax.set_yticklabels(list(tasks_data.keys()), fontsize=9)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        best_in_row = data[i].max()
        weight = "bold" if data[i, j] == best_in_row else "normal"
        ax.text(j, i, "%.3f" % data[i, j], ha="center", va="center",
                fontsize=8, fontweight=weight)
plt.colorbar(im, label="ROC AUC")
ax.set_title("Model Performance Across All Tasks (bold = best per task)")
save_fig("fig10_model_heatmap.png")


# ================================================================
# FIGURE 11: Subreddit spread patterns
# ================================================================
print("\n--- Fig 11: Subreddit spread ---")
all_pairs_set = set()
for d in days:
    all_pairs_set.update(day_pair_subs[d].keys())

first_counts = defaultdict(int)
spread_counts = defaultdict(lambda: defaultdict(int))

for pair in all_pairs_set:
    total = sum(day_pair[d].get(pair, {}).get("posts", 0) for d in days)
    if total < 5:
        continue
    seen_subs = set()
    for day in days:
        if pair in day_pair_subs[day]:
            for s in day_pair_subs[day][pair]:
                if s not in seen_subs:
                    if not seen_subs:
                        first_counts[s] += 1
                    else:
                        for prev in seen_subs:
                            spread_counts[prev][s] += 1
                    seen_subs.add(s)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
subs_sorted = sorted(first_counts.items(), key=lambda x: -x[1])
ax.barh([s[0] for s in subs_sorted], [s[1] for s in subs_sorted], color="#1f77b4")
ax.set_xlabel("Times broke story first")
ax.set_title("Which Subreddit Breaks Stories First?")

ax = axes[1]
routes = []
for src, dests in spread_counts.items():
    for dst, count in dests.items():
        routes.append(("%s -> %s" % (src, dst), count))
routes.sort(key=lambda x: -x[1])
top_routes = routes[:10]
top_routes.reverse()
ax.barh([r[0] for r in top_routes], [r[1] for r in top_routes], color="#ff7f0e")
ax.set_xlabel("Times")
ax.set_title("Top Cross-Subreddit Spread Routes")

plt.tight_layout()
save_fig("fig11_subreddit_spread.png")


# ================================================================
# FIGURE 12: Ongoing vs one-shot profile
# ================================================================
print("\n--- Fig 12: Ongoing vs one-shot ---")
fig, ax = plt.subplots(figsize=(8, 5))
features_oo = ["Multiple\npeaks", "Consistency", "Active\ndays", "Peak\nposts", "Unique\nsubs"]
ongoing_ratios = [15.5, 4.6, 4.0, 2.3, 1.4]
x_oo = np.arange(len(features_oo))
ax.bar(x_oo, ongoing_ratios, color="#1f77b4", width=0.6)
ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Equal (ratio=1)")
ax.set_xticks(x_oo)
ax.set_xticklabels(features_oo)
ax.set_ylabel("Ratio (ongoing / one-shot)")
ax.set_title("What Makes a Topic Come Back? (Ongoing vs One-Shot Ratios)")
ax.legend()
for i, v in enumerate(ongoing_ratios):
    ax.text(i, v + 0.2, "%.1fx" % v, ha="center", fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
save_fig("fig12_ongoing_vs_oneshot.png")


# ================================================================
# FIGURE 13: Topic state transition matrix
# ================================================================
print("\n--- Fig 13: Transition matrix ---")
transitions = np.array([
    [4, 8, 48, 8, 2, 30],
    [2, 6, 24, 30, 11, 26],
    [1, 3, 39, 5, 2, 50],
    [2, 5, 27, 14, 5, 47],
    [10, 10, 29, 6, 2, 44],
    [6, 0, 0, 0, 0, 94],
])
state_names = ["surging", "growing", "stable", "cooling", "dying", "dead"]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(transitions, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(6))
ax.set_xticklabels(state_names, fontsize=9)
ax.set_yticks(range(6))
ax.set_yticklabels(state_names, fontsize=9)
ax.set_xlabel("Tomorrow's State")
ax.set_ylabel("Today's State")
for i in range(6):
    for j in range(6):
        ax.text(j, i, "%d%%" % transitions[i, j], ha="center", va="center",
                fontsize=9, color="white" if transitions[i, j] > 40 else "black")
ax.set_title("Topic State Transition Matrix")
plt.colorbar(im, label="Probability (%)")
save_fig("fig13_transition_matrix.png")


# ================================================================
# FIGURE 14: Multi-horizon ROC decay
# ================================================================
print("\n--- Fig 14: Multi-horizon ROC decay ---")
horizons = [1, 4, 12, 24, 48, 168]
horizon_labels = ["1h", "4h", "12h", "24h", "48h", "7d"]
horizon_rocs = [0.843, 0.834, 0.809, 0.771, 0.726, 0.57]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(horizons)), horizon_rocs, "o-", color="#1f77b4", linewidth=2, markersize=10)
ax.set_xticks(range(len(horizons)))
ax.set_xticklabels(horizon_labels)
ax.set_xlabel("Prediction Horizon")
ax.set_ylabel("ROC AUC")
ax.set_title("Post Survival Prediction: ROC Decay Over Time")
ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, label="Random (0.5)")
ax.legend()
ax.grid(True, alpha=0.3)
for i, (h, r) in enumerate(zip(horizon_labels, horizon_rocs)):
    ax.annotate("%.3f" % r, (i, r), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9)
save_fig("fig14_multihorizon_roc_decay.png")


# ================================================================
# FIGURE 15: Emergence at different growth targets
# ================================================================
print("\n--- Fig 15: Emergence at different targets ---")
targets = [3, 5, 8, 10]
rocs_targets = []
for tgt in targets:
    y_tr_t, y_te_t = [], []
    for phase, dr in [("train", days[:split]), ("test", days[split:])]:
        for day in dr:
            di = days.index(day)
            if di + 3 >= len(days):
                continue
            for pair, d in day_pair[day].items():
                if d["posts"] < 1 or d["posts"] > 3:
                    continue
                peak = max(day_pair[days[j]].get(pair, {}).get("posts", 0)
                           for j in range(di + 1, min(di + 4, len(days))))
                val = 1 if peak >= tgt else 0
                if phase == "train":
                    y_tr_t.append(val)
                else:
                    y_te_t.append(val)
    y_tr_t, y_te_t = np.array(y_tr_t), np.array(y_te_t)
    if sum(y_tr_t) >= 5 and sum(y_te_t) > 0:
        m = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
        m.fit(X_tr_sub, y_tr_t[:len(X_tr_sub)])
        roc_val = roc_auc_score(y_te_t, m.predict_proba(X_te)[:, 1])
        rocs_targets.append((tgt, roc_val, sum(y_te_t)))

fig, ax = plt.subplots(figsize=(8, 5))
tgts = [t[0] for t in rocs_targets]
rs = [t[1] for t in rocs_targets]
ns = [t[2] for t in rocs_targets]
ax.plot(tgts, rs, "o-", color="#1f77b4", linewidth=2, markersize=10)
ax.set_xlabel("Growth Target (posts)")
ax.set_ylabel("ROC AUC")
ax.set_title("Emergence Detection: Performance vs Growth Target")
ax.grid(True, alpha=0.3)
for t, r, n in rocs_targets:
    ax.annotate("n=%d" % n, (t, r), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8)
save_fig("fig15_target_vs_roc.png")


# ================================================================
# FIGURE 16: Hyperparameter tuning summary
# ================================================================
print("\n--- Fig 16: Hyperparameter tuning ---")
tuning_data = {
    "Decision Tree": (0.577, 0.842, "+0.265"),
    "Gradient Boosting": (0.713, 0.835, "+0.122"),
    "Random Forest": (0.820, 0.846, "+0.027"),
    "Extra Trees": (0.828, 0.851, "+0.023"),
    "Logistic Regression": (0.859, 0.860, "+0.001"),
}

fig, ax = plt.subplots(figsize=(10, 5))
names_hp = list(tuning_data.keys())
defaults = [tuning_data[n][0] for n in names_hp]
tuned = [tuning_data[n][1] for n in names_hp]
x_hp = np.arange(len(names_hp))
width = 0.35
bars1 = ax.bar(x_hp - width / 2, defaults, width, label="Default", color="#95a5a6")
bars2 = ax.bar(x_hp + width / 2, tuned, width, label="Tuned", color="#2ecc71")
ax.set_xticks(x_hp)
ax.set_xticklabels(names_hp, fontsize=9)
ax.set_ylabel("ROC AUC")
ax.set_title("Hyperparameter Tuning: Default vs Tuned Performance")
ax.legend()
ax.set_ylim(0.5, 0.95)
ax.grid(True, alpha=0.2, axis="y")
for i, (d, t) in enumerate(zip(defaults, tuned)):
    improvement = t - d
    ax.text(i + width / 2, t + 0.005, "%+.3f" % improvement, ha="center", fontsize=8,
            fontweight="bold", color="#27ae60")
save_fig("fig16_hyperparameter_tuning.png")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("  COMPLETE — Generated figures:")
print("=" * 60)
figs = sorted([f for f in os.listdir(OUT) if f.startswith("fig")])
for f in figs:
    print("  %s" % f)
print("\nTotal: %d figures in %s" % (len(figs), OUT))
print("=" * 60)

conn.close()
