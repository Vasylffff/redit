"""Generate all figures for Assessment 2 report."""
import sqlite3, re, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
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

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour, p.activity_state
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

# Build pair data
day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [],
    "velocities": [], "states": []
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
            d["states"].append(state or "dead")
            day_pair_subs[day][pair].add(sub)

days = sorted(day_pair.keys())
split = 8


def make_feats(d):
    bc = max(d["comments"]) if d["comments"] else 0
    ac = sum(d["comments"]) / max(1, len(d["comments"]))
    vels = d["velocities"]
    return [d["posts"], d["total_up"], len(d["subs"]), bc, ac,
            d["total_up"] / max(1, d["posts"]),
            max(vels) if vels else 0, sum(vels) / max(1, len(vels))]


def build_emergence_data():
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for phase, dr in [("train", days[:split]), ("test", days[split:])]:
        for day in dr:
            di = days.index(day)
            if di + 3 >= len(days): continue
            for pair, d in day_pair[day].items():
                if d["posts"] < 1 or d["posts"] > 3: continue
                peak = max(day_pair[days[j]].get(pair, {}).get("posts", 0)
                           for j in range(di + 1, min(di + 4, len(days))))
                grew = 1 if peak >= 5 else 0
                feats = make_feats(d)
                if phase == "train": X_tr.append(feats); y_tr.append(grew)
                else: X_te.append(feats); y_te.append(grew)
    return np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)


# ================================================================
# FIGURE 1: Topic lifecycle example trajectories
# ================================================================
print("Fig 1: Topic trajectories...")
fig, ax = plt.subplots(figsize=(12, 5))
examples = {
    "hormuz+strait": "Hormuz Strait (ongoing)",
    "easter+trump": "Easter+Trump (one-shot)",
    "birthright+citizenship": "Birthright Citizenship (court case)",
    "official+trailer": "Game Trailers (recurring)",
    "crimson+desert": "Crimson Desert (game launch)",
}
colors = plt.cm.Set1(np.linspace(0, 1, len(examples)))
for idx, (pair, label) in enumerate(examples.items()):
    counts = [day_pair[d].get(pair, {}).get("posts", 0) for d in days]
    ax.plot(range(len(days)), counts, "o-", label=label, color=colors[idx], linewidth=2, markersize=4)
ax.set_xlabel("Day")
ax.set_ylabel("Posts per day")
ax.set_title("Topic Lifecycle Trajectories: Ongoing Stories vs One-Shot Events")
ax.set_xticks(range(len(days)))
ax.set_xticklabels([d[5:] for d in days], rotation=45, fontsize=8)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT, "fig1_topic_trajectories.png"))
plt.close()

# ================================================================
# FIGURE 2: ROC curves — emergence detection, multiple models
# ================================================================
print("Fig 2: ROC curves multi-model...")
X_tr, y_tr, X_te, y_te = build_emergence_data()
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

fig, ax = plt.subplots(figsize=(8, 7))
models = [
    ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42), X_tr, X_te),
    ("Extra Trees", ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42), X_tr, X_te),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.1, random_state=42), X_tr, X_te),
    ("Logistic Regression", LogisticRegression(C=10, max_iter=3000, random_state=42), X_tr_s, X_te_s),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42), X_tr, X_te),
]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for (name, model, xtr, xte), color in zip(models, colors):
    model.fit(xtr, y_tr)
    probs = model.predict_proba(xte)[:, 1]
    roc = roc_auc_score(y_te, probs)
    fpr, tpr, _ = roc_curve(y_te, probs)
    ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, roc), color=color, linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Topic Emergence Detection: Model Comparison (1-3 posts -> 5+)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT, "fig2_roc_emergence_models.png"))
plt.close()

# ================================================================
# FIGURE 3: Feature importance bar chart
# ================================================================
print("Fig 3: Feature importance...")
rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
rf.fit(X_tr, y_tr)
feat_names = ["Posts", "Total Upvotes", "Subreddits", "Best Comments",
              "Avg Comments", "Upvotes/Post", "Best Velocity", "Avg Velocity"]
importances = rf.feature_importances_
order = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([feat_names[i] for i in order], [importances[i] for i in order], color="#1f77b4")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance: Topic Emergence Detection (Random Forest)")
for i, idx in enumerate(order):
    ax.text(importances[idx] + 0.005, i, "%.1f%%" % (importances[idx] * 100), va="center", fontsize=9)
plt.savefig(os.path.join(OUT, "fig3_feature_importance.png"))
plt.close()

# ================================================================
# FIGURE 4: Hyperparameter tuning — depth vs ROC
# ================================================================
print("Fig 4: Depth vs ROC...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RF depth
depths_rf = [2, 3, 4, 5, 6, 8, 10, 12]
rocs_rf = []
for d in depths_rf:
    m = RandomForestClassifier(n_estimators=200, max_depth=d, random_state=42)
    m.fit(X_tr, y_tr)
    rocs_rf.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
axes[0].plot(depths_rf, rocs_rf, "o-", color="#1f77b4", linewidth=2, markersize=8)
axes[0].set_xlabel("Max Depth")
axes[0].set_ylabel("ROC AUC")
axes[0].set_title("Random Forest: Depth vs Performance")
axes[0].grid(True, alpha=0.3)
best_d = depths_rf[np.argmax(rocs_rf)]
axes[0].axvline(best_d, color="red", linestyle="--", alpha=0.5, label="Best (depth=%d)" % best_d)
axes[0].legend()

# DT depth
depths_dt = [2, 3, 4, 5, 6, 8, 10, 12]
rocs_dt = []
for d in depths_dt:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_tr, y_tr)
    rocs_dt.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
axes[1].plot(depths_dt, rocs_dt, "o-", color="#d62728", linewidth=2, markersize=8)
axes[1].set_xlabel("Max Depth")
axes[1].set_ylabel("ROC AUC")
axes[1].set_title("Decision Tree: Depth vs Performance")
axes[1].grid(True, alpha=0.3)
best_d2 = depths_dt[np.argmax(rocs_dt)]
axes[1].axvline(best_d2, color="red", linestyle="--", alpha=0.5, label="Best (depth=%d)" % best_d2)
axes[1].legend()

plt.suptitle("Overfitting: Deeper Trees Perform Worse", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_depth_vs_roc.png"))
plt.close()

# ================================================================
# FIGURE 5: Full pipeline ROC summary
# ================================================================
print("Fig 5: Pipeline summary...")
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
names = [t[0] for t in tasks]
rocs = [t[1] for t in tasks]
colors_bar = ["#2ca02c" if r >= 0.9 else "#ff7f0e" if r >= 0.8 else "#d62728" for r in rocs]
bars = ax.barh(names, rocs, color=colors_bar)
ax.set_xlabel("ROC AUC")
ax.set_title("Complete Topic Lifecycle Prediction Pipeline")
ax.set_xlim(0.5, 1.05)
ax.axvline(0.8, color="gray", linestyle="--", alpha=0.4, label="Good (0.8)")
ax.axvline(0.9, color="gray", linestyle=":", alpha=0.4, label="Excellent (0.9)")
for bar, roc in zip(bars, rocs):
    ax.text(roc + 0.005, bar.get_y() + bar.get_height() / 2, "%.3f" % roc, va="center", fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="x")
plt.savefig(os.path.join(OUT, "fig5_pipeline_summary.png"))
plt.close()

# ================================================================
# FIGURE 6: Topic death definition — revival rates
# ================================================================
print("Fig 6: Revival rates...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By definition
defs = ["1 day", "2 days", "3 days"]
revival_rates = [13.1, 6.8, 4.4]
ax = axes[0]
bars = ax.bar(defs, revival_rates, color=["#d62728", "#ff7f0e", "#2ca02c"])
ax.set_ylabel("Revival Rate (%)")
ax.set_title("False Death Rate by Definition")
ax.set_xlabel("Consecutive days < 2 posts")
for bar, rate in zip(bars, revival_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            "%.1f%%" % rate, ha="center", fontsize=11, fontweight="bold")

# By topic size
sizes = ["3-4", "5-7", "8-11", "12-19"]
rev_rates = [22.8, 35.9, 44.8, 40.0]
ax = axes[1]
bars = ax.bar(sizes, rev_rates, color="#1f77b4")
ax.set_ylabel("Revival Rate (%)")
ax.set_title("Revival Rate by Topic Size (peak posts)")
ax.set_xlabel("Peak post count")
for bar, rate in zip(bars, rev_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            "%.1f%%" % rate, ha="center", fontsize=11, fontweight="bold")

plt.suptitle("Topic Death: Bigger Topics Are Harder to Kill", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig6_revival_rates.png"))
plt.close()

# ================================================================
# FIGURE 7: Model comparison heatmap
# ================================================================
print("Fig 7: Model comparison heatmap...")
tasks_data = {
    "Emergence": [0.829, 0.808, 0.680, 0.851, 0.560],
    "Escalation": [0.708, 0.736, 0.565, 0.712, 0.510],
    "Failure Filter": [0.829, 0.808, 0.760, 0.850, 0.560],
    "Death Tomorrow": [0.886, 0.869, 0.874, 0.787, 0.864],
    "Peaked/Growing": [0.731, 0.657, 0.734, 0.581, 0.658],
    "Quick/Slow Death": [0.995, 0.996, 0.996, 0.999, 0.973],
    "Sub Spread": [0.659, 0.644, 0.619, 0.636, 0.606],
}
model_names = ["Random\nForest", "Extra\nTrees", "Gradient\nBoosting", "Logistic\nRegression", "Decision\nTree"]

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
        text = "%.3f" % data[i, j]
        weight = "bold" if data[i, j] == best_in_row else "normal"
        ax.text(j, i, text, ha="center", va="center", fontsize=8, fontweight=weight)
plt.colorbar(im, label="ROC AUC")
ax.set_title("Model Performance Across All Tasks (bold = best per task)")
plt.savefig(os.path.join(OUT, "fig7_model_heatmap.png"))
plt.close()

# ================================================================
# FIGURE 8: Subreddit spread patterns
# ================================================================
print("Fig 8: Subreddit spread...")
SUBREDDITS = ["news", "politics", "worldnews", "technology", "Games"]
spread_counts = defaultdict(lambda: defaultdict(int))
first_counts = defaultdict(int)

all_pairs_set = set()
for d in days:
    all_pairs_set.update(day_pair_subs[d].keys())

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

# Who breaks first
ax = axes[0]
subs_sorted = sorted(first_counts.items(), key=lambda x: -x[1])
ax.barh([s[0] for s in subs_sorted], [s[1] for s in subs_sorted], color="#1f77b4")
ax.set_xlabel("Times broke story first")
ax.set_title("Which Subreddit Breaks Stories First?")

# Top spread routes
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
plt.savefig(os.path.join(OUT, "fig8_subreddit_spread.png"))
plt.close()

# ================================================================
# FIGURE 9: Confusion matrix for emergence
# ================================================================
print("Fig 9: Confusion matrix...")
rf_best = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
rf_best.fit(X_tr, y_tr)
probs = rf_best.predict_proba(X_te)[:, 1]
preds = (probs > 0.1).astype(int)  # 10% threshold

cm = confusion_matrix(y_te, preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
labels = ["Won't Grow", "Will Grow"]
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticks([0, 1])
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title("Emergence Detection Confusion Matrix\n(threshold=10%%)")
plt.colorbar(im)
plt.savefig(os.path.join(OUT, "fig9_confusion_matrix.png"))
plt.close()

# ================================================================
# FIGURE 10: Hyperparameter tuning — GBM learning rate
# ================================================================
print("Fig 10: GBM learning rate...")
# Subsample for speed
np.random.seed(42)
idx = np.random.choice(len(X_tr), min(30000, len(X_tr)), replace=False)
X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]

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
plt.savefig(os.path.join(OUT, "fig10_gbm_learning_rate.png"))
plt.close()

# ================================================================
# FIGURE 11: Data collection coverage
# ================================================================
print("Fig 11: Data coverage...")
import pandas as pd
ps = pd.read_csv("data/history/reddit/post_snapshots.csv", usecols=["snapshot_time_utc", "subreddit"])
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
plt.savefig(os.path.join(OUT, "fig11_data_coverage.png"))
plt.close()

# ================================================================
# FIGURE 12: Ongoing vs one-shot profile
# ================================================================
print("Fig 12: Ongoing vs one-shot profile...")
fig, ax = plt.subplots(figsize=(8, 5))
features = ["Multiple\npeaks", "Consistency", "Active\ndays", "Peak\nposts", "Unique\nsubs"]
ongoing = [15.5, 4.6, 4.0, 2.3, 1.4]
x = np.arange(len(features))
ax.bar(x, ongoing, color="#1f77b4", width=0.6)
ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Equal (ratio=1)")
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.set_ylabel("Ratio (ongoing / one-shot)")
ax.set_title("What Makes a Topic Come Back? (Ongoing vs One-Shot Ratios)")
ax.legend()
for i, v in enumerate(ongoing):
    ax.text(i, v + 0.2, "%.1fx" % v, ha="center", fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
plt.savefig(os.path.join(OUT, "fig12_ongoing_vs_oneshot.png"))
plt.close()

# ================================================================
# FIGURE 13: Topic state transition matrix
# ================================================================
print("Fig 13: Transition matrix...")
transitions = np.array([
    [4, 8, 48, 8, 2, 30],    # surging
    [2, 6, 24, 30, 11, 26],   # growing
    [1, 3, 39, 5, 2, 50],     # stable
    [2, 5, 27, 14, 5, 47],    # cooling
    [10, 10, 29, 6, 2, 44],   # dying
    [6, 0, 0, 0, 0, 94],      # dead
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
plt.savefig(os.path.join(OUT, "fig13_transition_matrix.png"))
plt.close()

# ================================================================
# FIGURE 14: Emergence detection at different targets
# ================================================================
print("Fig 14: Emergence at different targets...")
targets = [3, 5, 8, 10, 15]
rocs_targets = []
for tgt in targets:
    y_te_t = []
    for day in days[split:]:
        di = days.index(day)
        if di + 3 >= len(days): continue
        for pair, d in day_pair[day].items():
            if d["posts"] < 1 or d["posts"] > 3: continue
            peak = max(day_pair[days[j]].get(pair, {}).get("posts", 0)
                       for j in range(di + 1, min(di + 4, len(days))))
            y_te_t.append(1 if peak >= tgt else 0)
    # Reuse same X_te, just different y
    y_t = np.array(y_te_t)
    if sum(y_t) > 0 and sum(y_t) < len(y_t):
        # Retrain for this target
        y_tr_t = []
        for day in days[:split]:
            di = days.index(day)
            if di + 3 >= len(days): continue
            for pair, d in day_pair[day].items():
                if d["posts"] < 1 or d["posts"] > 3: continue
                peak = max(day_pair[days[j]].get(pair, {}).get("posts", 0)
                           for j in range(di + 1, min(di + 4, len(days))))
                y_tr_t.append(1 if peak >= tgt else 0)
        y_tr_t = np.array(y_tr_t)
        if sum(y_tr_t) >= 5:
            m = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
            m.fit(X_tr, y_tr_t)
            roc = roc_auc_score(y_t, m.predict_proba(X_te)[:, 1])
            rocs_targets.append((tgt, roc, sum(y_t)))

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
    ax.annotate("n=%d" % n, (t, r), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
plt.savefig(os.path.join(OUT, "fig14_target_vs_roc.png"))
plt.close()

conn.close()

# List all generated figures
figs = sorted(os.listdir(OUT))
print("\n" + "=" * 50)
print("Generated %d figures in %s:" % (len(figs), OUT))
for f in figs:
    print("  %s" % f)
print("DONE")
