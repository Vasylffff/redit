"""Test established virality models on co-occurrence pairs.

Szabo-Huberman log-linear: log(peak) = a * log(early) + b
Tested at two observation windows:
  - Early: 1-3 posts (minimal info)
  - Established: 4-8 posts (more signal)
"""
import sqlite3, re, numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
from numpy.linalg import lstsq

DB_PATH = "data/history/reddit/history.db"
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www','report','reports'}

conn = sqlite3.connect(DB_PATH, timeout=30)
rows = conn.execute("""SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc,1,10) as day,
       l.max_upvotes, l.subreddit, l.max_comments,
       p.upvote_velocity_per_hour, p.comment_velocity_per_hour
FROM post_snapshots p JOIN post_lifecycles l ON p.post_id=l.post_id
WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL""").fetchall()

day_pair = defaultdict(lambda: defaultdict(lambda: {
    "posts": 0, "total_up": 0, "subs": set(), "comments": [], "max_single": 0, "titles": [],
    "velocities": [], "comment_vels": []
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
            d["max_single"] = max(d["max_single"], max_up or 0)
            d["velocities"].append(vel or 0)
            d["comment_vels"].append(cvel or 0)
            if len(d["titles"]) < 2:
                d["titles"].append(title[:80])

days = sorted(day_pair.keys())
split = 8

print("Train: %s to %s" % (days[0], days[split - 1]))
print("Test:  %s to %s" % (days[split], days[-1]))


def build_records(day_range, min_posts=1, max_posts=3):
    recs = []
    for day in day_range:
        di = days.index(day)
        if di + 3 >= len(days):
            continue
        for pair, d in day_pair[day].items():
            if d["posts"] < min_posts or d["posts"] > max_posts:
                continue
            future = [day_pair[days[j]].get(pair, {}).get("posts", 0)
                      for j in range(di + 1, min(di + 4, len(days)))]
            peak = max(future) if future else 0
            if peak < 1:
                continue
            bc = max(d["comments"]) if d["comments"] else 0
            vels = d["velocities"]
            cvels = d["comment_vels"]
            recs.append({
                "pair": pair, "day": day,
                "posts": d["posts"], "total_up": d["total_up"],
                "subs": len(d["subs"]), "best_com": bc,
                "max_single": d["max_single"],
                "up_per_post": d["total_up"] / max(1, d["posts"]),
                "best_vel": max(vels) if vels else 0,
                "avg_vel": sum(vels) / max(1, len(vels)),
                "best_cvel": max(cvels) if cvels else 0,
                "avg_cvel": sum(cvels) / max(1, len(cvels)),
                "peak": peak,
                "title": d["titles"][0] if d["titles"] else "?",
            })
    return recs


def run_models(train, test, label):
    # --- Szabo-Huberman single signal ---
    print("\n  --- Szabo-Huberman single-signal ---")
    signals = [
        ("posts", lambda r: r["posts"]),
        ("total_upvotes", lambda r: r["total_up"]),
        ("best_comments", lambda r: r["best_com"]),
        ("up_per_post", lambda r: r["up_per_post"]),
    ]

    best_signal = None
    best_r2 = -999
    for name, fn in signals:
        x_tr = np.array([np.log1p(fn(r)) for r in train])
        y_tr = np.array([np.log1p(r["peak"]) for r in train])
        if np.std(x_tr) == 0:
            continue
        a, b = np.polyfit(x_tr, y_tr, 1)
        x_te = np.array([np.log1p(fn(r)) for r in test])
        y_te = np.array([np.log1p(r["peak"]) for r in test])
        pred_log = a * x_te + b
        ss_res = np.sum((y_te - pred_log) ** 2)
        ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        corr, _ = pearsonr(y_te, pred_log)
        print("    %-18s R2=%.3f  corr=%.3f  formula: log(peak) = %.3f*log(%s) + %.3f" % (
            name, r2, corr, a, name, b))
        if r2 > best_r2:
            best_r2 = r2
            best_signal = name
            best_a, best_b = a, b
            best_fn = fn

    print("    Best: %s (R2=%.3f)" % (best_signal, best_r2))

    # --- Multi-signal log-linear (without velocity) ---
    print("\n  --- Multi-signal log-linear (count only) ---")

    def log_features_base(r):
        return [np.log1p(r["posts"]), np.log1p(r["total_up"]),
                np.log1p(r["best_com"]), np.log1p(r["up_per_post"]),
                np.log1p(r["subs"]), 1.0]

    def log_features_vel(r):
        return [np.log1p(r["posts"]), np.log1p(r["total_up"]),
                np.log1p(r["best_com"]), np.log1p(r["up_per_post"]),
                np.log1p(r["subs"]),
                np.log1p(max(0, r["best_vel"])), np.log1p(max(0, r["avg_vel"])),
                np.log1p(max(0, r["best_cvel"])), np.log1p(max(0, r["avg_cvel"])),
                1.0]

    feat_names_base = ["log(posts)", "log(upvotes)", "log(comments)", "log(up/post)", "log(subs)", "intercept"]
    feat_names_vel = ["log(posts)", "log(upvotes)", "log(comments)", "log(up/post)", "log(subs)",
                      "log(best_vel)", "log(avg_vel)", "log(best_cvel)", "log(avg_cvel)", "intercept"]

    results = {}
    for version, feat_fn, feat_names_list in [
        ("count only", log_features_base, feat_names_base),
        ("+ velocity", log_features_vel, feat_names_vel),
    ]:
        X_tr = np.array([feat_fn(r) for r in train])
        y_tr = np.array([np.log1p(r["peak"]) for r in train])
        X_te = np.array([feat_fn(r) for r in test])
        y_te = np.array([np.log1p(r["peak"]) for r in test])

        coeffs, _, _, _ = lstsq(X_tr, y_tr, rcond=None)
        pred_log = X_te @ coeffs
        pred = np.expm1(pred_log)

        ss_res = np.sum((y_te - pred_log) ** 2)
        ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        corr, _ = pearsonr(y_te, pred_log)

        print("\n  --- Multi-signal log-linear (%s) ---" % version)
        print("    Coefficients:")
        for fn, c in zip(feat_names_list, coeffs):
            print("      %-18s %+.4f" % (fn, c))
        print("    R2=%.3f  corr=%.3f" % (r2, corr))
        results[version] = (r2, pred)

    r2_multi = results["+ velocity"][0]
    pred = results["+ velocity"][1]

    print("\n  --- Velocity impact ---")
    print("    Count only: R2=%.3f" % results["count only"][0])
    print("    + Velocity: R2=%.3f  (%+.3f)" % (results["+ velocity"][0],
          results["+ velocity"][0] - results["count only"][0]))

    # Show big topic predictions
    y_te_actual = np.array([r["peak"] for r in test])
    indexed = sorted(zip(test, pred), key=lambda x: -x[0]["peak"])

    print("\n  --- Big topic predictions ---")
    print("  %-25s %5s %5s %5s %6s %s" % ("Topic", "Posts", "Peak", "Pred", "Error", "Title"))
    print("  " + "-" * 90)
    for r, p in indexed[:20]:
        print("  %-25s %5d %5d %5.1f %+5.1f  %s" % (
            r["pair"], r["posts"], r["peak"], p, p - r["peak"], r["title"][:40]))

    big = [(r, p) for r, p in indexed if r["peak"] >= 8]
    if big:
        actual = [r["peak"] for r, _ in big]
        predicted = [p for _, p in big]
        from sklearn.metrics import mean_absolute_error
        print("\n  Topics 8+: count=%d, avg_actual=%.1f, avg_predicted=%.1f, MAE=%.1f" % (
            len(big), np.mean(actual), np.mean(predicted), mean_absolute_error(actual, predicted)))

    return best_r2, r2_multi


# === Run both windows ===
for min_p, max_p, label in [(1, 3, "EARLY (1-3 posts)"), (4, 8, "ESTABLISHED (4-8 posts)")]:
    print("\n" + "#" * 70)
    print("  %s" % label)
    print("#" * 70)

    train = build_records(days[:split], min_p, max_p)
    test = build_records(days[split:], min_p, max_p)
    print("  Train: %d records, Test: %d records" % (len(train), len(test)))

    if len(train) < 50 or len(test) < 20:
        print("  Not enough data, skipping")
        continue

    run_models(train, test, label)

conn.close()
print("\nDONE")
