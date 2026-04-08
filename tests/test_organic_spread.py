"""Test organic spread signals: unique authors, time concentration, subreddit mix, angle diversity."""
import sqlite3, re, os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)

rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, p.snapshot_time_utc,
           substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit,
           l.author, l.last_upvote_velocity_per_hour,
           l.max_comments, l.created_at
    FROM post_snapshots p
    JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

# Build keyword data with organic spread features
day_kw = defaultdict(lambda: defaultdict(lambda: {
    "posts": [], "authors": set(), "subs": set(), "times": [], "titles": []
}))
seen = set()

for pid, title, snap_time, day, max_up, state, sub, author, vel, comments, created in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = set(w for w in re.findall(r"[a-z]+", title.lower()) if len(w) > 4 and w not in STOPWORDS)
    for w in words:
        d = day_kw[day][w]
        d["posts"].append({
            "pid": pid, "up": max_up or 0, "vel": vel or 0,
            "comments": comments or 0, "state": state, "title": title,
            "created": created or "",
        })
        if author and author != "[deleted]":
            d["authors"].add(author)
        d["subs"].add(sub)
        d["titles"].append(title)
        if created:
            d["times"].append(created)

days = sorted(day_kw.keys())
split = 8

print("Days: %d (train: %s-%s, test: %s-%s)" % (
    len(days), days[0], days[split-1], days[split], days[-1]))


def title_diversity(titles):
    """How different are the titles? More diverse = story has multiple angles."""
    if len(titles) < 2:
        return 0
    # Simple: count unique words across all titles vs total words
    all_words = []
    per_title_words = []
    for t in titles:
        words = set(re.findall(r"[a-z]+", t.lower())) - STOPWORDS
        all_words.extend(words)
        per_title_words.append(words)
    if not all_words:
        return 0
    # Jaccard diversity: avg pairwise dissimilarity
    pairs = 0
    total_div = 0
    for i in range(len(per_title_words)):
        for j in range(i+1, len(per_title_words)):
            a, b = per_title_words[i], per_title_words[j]
            if a | b:
                total_div += 1 - len(a & b) / len(a | b)
                pairs += 1
    return total_div / max(1, pairs)


def time_concentration(times):
    """How concentrated in time are the posts? Smaller = more concentrated."""
    if len(times) < 2:
        return 0
    # Parse and find time spread in hours
    parsed = []
    for t in times:
        try:
            # Just use the hour part for rough estimate
            parts = t.split("T")
            if len(parts) >= 2:
                hour = int(parts[1][:2])
                parsed.append(hour)
        except (ValueError, IndexError):
            pass
    if len(parsed) < 2:
        return 0
    return max(parsed) - min(parsed)


def build_data(day_range):
    X_count, X_organic, y, info = [], [], [], []
    for day in day_range:
        day_idx = days.index(day)
        if day_idx + 3 >= len(days):
            continue
        for kw, d in day_kw[day].items():
            n = len(d["posts"])
            if n < 1 or n > 5:
                continue
            max_future = max(
                len(day_kw[days[j]].get(kw, {}).get("posts", []))
                for j in range(day_idx + 1, min(day_idx + 4, len(days)))
            )
            grew = 1 if max_future >= 5 else 0

            total_up = sum(p["up"] for p in d["posts"])
            num_subs = len(d["subs"])

            # Simple counts
            X_count.append([n, total_up, num_subs])

            # Organic spread features
            unique_authors = len(d["authors"])
            author_ratio = unique_authors / max(1, n)  # 1.0 = all different authors
            sub_list = list(d["subs"])
            has_news_and_politics = 1 if ("news" in sub_list or "worldnews" in sub_list) and "politics" in sub_list else 0
            has_tech_crossover = 1 if "technology" in sub_list and len(sub_list) > 1 else 0
            time_spread = time_concentration(d["times"])
            time_concentrated = 1 if time_spread <= 2 and n >= 2 else 0  # posts within 2 hours
            title_div = title_diversity(d["titles"])
            best_comments = max(p["comments"] for p in d["posts"])
            best_up = max(p["up"] for p in d["posts"])
            any_surging = max(1 if p["state"] == "surging" else 0 for p in d["posts"])

            X_organic.append([
                n, total_up, num_subs,
                unique_authors,         # different people posting
                author_ratio,           # ratio of unique authors
                has_news_and_politics,  # crosses news/politics boundary
                has_tech_crossover,     # tech + another sub
                time_concentrated,      # posts clustered in time
                time_spread,            # hours between first and last
                title_div,              # different angles on same topic
                best_comments,          # most commented post
                best_up,                # most upvoted post
                any_surging,            # any post surging
            ])
            y.append(grew)
            info.append({
                "kw": kw, "day": day, "posts": n, "up": total_up,
                "authors": unique_authors, "author_ratio": author_ratio,
                "subs": num_subs, "time_conc": time_concentrated,
                "title_div": title_div, "best_com": best_comments,
                "grew": grew, "future": max_future,
            })
    return np.array(X_count), np.array(X_organic), np.array(y), info


Xc_tr, Xo_tr, y_tr, _ = build_data(days[:split])
Xc_te, Xo_te, y_te, test_info = build_data(days[split:])

print("Train: %d, %d grew (%.1f%%)" % (len(y_tr), sum(y_tr), sum(y_tr)/len(y_tr)*100))
print("Test: %d, %d grew (%.1f%%)" % (len(y_te), sum(y_te), sum(y_te)/len(y_te)*100))

if sum(y_te) == 0:
    print("No growth events!")
    conn.close()
    exit()

rf_c = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_o = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_c.fit(Xc_tr, y_tr)
rf_o.fit(Xo_tr, y_tr)

probs_c = rf_c.predict_proba(Xc_te)[:, 1]
probs_o = rf_o.predict_proba(Xo_te)[:, 1]

roc_c = roc_auc_score(y_te, probs_c)
roc_o = roc_auc_score(y_te, probs_o)

print("\n" + "=" * 60)
print("TEMPORAL VALIDATION: COUNTS vs ORGANIC SPREAD")
print("=" * 60)
print("  Counts only:     ROC = %.3f" % roc_c)
print("  + Organic spread: ROC = %.3f" % roc_o)
print("  Improvement:      %+.3f" % (roc_o - roc_c))

# Detection rates
growers = [(i, p) for i, p in enumerate(test_info) if p["grew"]]
caught_c = sum(1 for i, p in growers if probs_c[i] > 0.05)
caught_o = sum(1 for i, p in growers if probs_o[i] > 0.05)
caught_c10 = sum(1 for i, p in growers if probs_c[i] > 0.10)
caught_o10 = sum(1 for i, p in growers if probs_o[i] > 0.10)

print("\n  Detection >5%%:  Counts=%d/%d  Organic=%d/%d" % (caught_c, len(growers), caught_o, len(growers)))
print("  Detection >10%%: Counts=%d/%d  Organic=%d/%d" % (caught_c10, len(growers), caught_o10, len(growers)))

# Feature importance
print("\n" + "=" * 60)
print("ORGANIC SPREAD FEATURES:")
print("=" * 60)
fn = ["posts", "total_up", "subs", "unique_authors", "author_ratio",
      "news_politics_cross", "tech_crossover", "time_concentrated",
      "time_spread", "title_diversity", "best_comments", "best_upvotes", "any_surging"]
for name, imp in sorted(zip(fn, rf_o.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.01:
        bar = "#" * int(imp * 30)
        print("  %-22s %.3f %s" % (name, imp, bar))

# Profile
print("\n" + "=" * 60)
print("ORGANIC SPREAD PROFILE:")
print("=" * 60)
grower_info = [p for p in test_info if p["grew"]]
non_info = [p for p in test_info if not p["grew"]]

for metric in ["posts", "up", "authors", "author_ratio", "subs", "time_conc", "title_div", "best_com"]:
    g = np.mean([p[metric] for p in grower_info])
    n = np.mean([p[metric] for p in non_info])
    ratio = g / max(0.001, n)
    print("  %-18s Grew: %8.2f  Didnt: %8.2f  Ratio: %.1fx" % (metric, g, n, ratio))

# Show what organic model catches
print("\n" + "=" * 60)
print("WHAT ORGANIC MODEL CATCHES:")
print("=" * 60)
for i, p in enumerate(test_info):
    p["prob_c"] = probs_c[i]
    p["prob_o"] = probs_o[i]

# Topics caught by organic but missed by counts
organic_wins = [p for p in test_info if p["grew"] and p["prob_o"] > p["prob_c"] * 1.5]
if organic_wins:
    print("Topics where organic model beat counts:")
    for p in sorted(organic_wins, key=lambda x: -x["prob_o"])[:10]:
        print("  %-15s authors=%d subs=%d coms=%d  organic=%.0f%% counts=%.0f%% grew->%d" % (
            p["kw"], p["authors"], p["subs"], p["best_com"],
            p["prob_o"]*100, p["prob_c"]*100, p["future"]))

conn.close()
print("\nDONE")
