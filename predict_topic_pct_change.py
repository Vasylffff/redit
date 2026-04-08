"""Predict percentage change in topic popularity per subreddit."""
import sqlite3, re, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from collections import defaultdict

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during'}

conn = sqlite3.connect(DB_PATH, timeout=30)

rows = conn.execute('''
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit
    FROM post_snapshots p
    JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
''').fetchall()

day_sub_kw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'posts': 0, 'total_up': 0, 'alive': 0})))
seen = set()
for pid, title, day, max_up, state, sub in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = re.findall(r'[a-z]+', title.lower())
    for w in set(words):
        if len(w) > 3 and w not in STOPWORDS:
            d = day_sub_kw[day][sub][w]
            d['posts'] += 1
            d['total_up'] += (max_up or 0)
            if state in ('surging', 'alive'):
                d['alive'] += 1

days = sorted(day_sub_kw.keys())


def build_feats(w):
    posts = [x['posts'] for x in w]
    total = [x['total_up'] for x in w]
    return [
        posts[-1], posts[-2], posts[-3],
        total[-1], total[-2], total[-3],
        (posts[-1] - posts[0]) / max(1, posts[0]),
        (total[-1] - total[0]) / max(1, total[0]),
        (posts[-1] - posts[-2]) / max(1, posts[-2]),
        (total[-1] - total[-2]) / max(1, total[-2]),
        sum(posts) / 3,
        sum(total) / 3,
    ]


print("=" * 70)
print("TOPIC PERCENTAGE CHANGE PREDICTION")
print("=" * 70)

for sub_name in ['news', 'politics', 'worldnews', 'technology', 'Games']:
    kw_series = defaultdict(list)
    for day in days:
        for kw, d in day_sub_kw[day].get(sub_name, {}).items():
            if d['posts'] < 2:
                continue
            kw_series[kw].append({'day': day, 'posts': d['posts'], 'total_up': d['total_up']})

    good = {kw: sorted(s, key=lambda x: x['day']) for kw, s in kw_series.items() if len(s) >= 5}
    if not good:
        continue

    # R2 scores for percentage change
    for days_ahead in [1, 2, 3]:
        X, y_pct_p, y_pct_u = [], [], []
        for kw, series in good.items():
            for i in range(3, len(series) - days_ahead + 1):
                w = series[i - 3:i]
                curr = series[i - 1]
                target = series[i + days_ahead - 1]
                X.append(build_feats(w))
                y_pct_p.append((target['posts'] - curr['posts']) / max(1, curr['posts']))
                y_pct_u.append((target['total_up'] - curr['total_up']) / max(1, curr['total_up']))

        if len(X) < 30:
            continue
        X = np.array(X)
        rf_p = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        rf_u = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        r2_p = cross_val_score(rf_p, X, np.array(y_pct_p), cv=min(5, len(X) // 5), scoring='r2')
        r2_u = cross_val_score(rf_u, X, np.array(y_pct_u), cv=min(5, len(X) // 5), scoring='r2')
        if days_ahead == 1:
            print(f"\n  r/{sub_name} ({len(good)} topics):")
        print(f"    {days_ahead}d ahead: Post% R2={r2_p.mean():.3f}  Upvote% R2={r2_u.mean():.3f}  n={len(X)}")

    # Actual predictions for last day
    X_tr, y_tr = [], []
    for kw, series in good.items():
        for i in range(3, len(series)):
            if series[i]['day'] >= days[-1]:
                continue
            w = series[max(0, i - 3):i]
            if len(w) < 3:
                continue
            curr = series[i - 1]
            X_tr.append(build_feats(w))
            y_tr.append((series[i]['posts'] - curr['posts']) / max(1, curr['posts']))

    if len(X_tr) < 10:
        continue

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(np.array(X_tr), np.array(y_tr))

    # Pick keywords per subreddit
    sub_kws = {
        'news': ['trump', 'iran', 'war', 'court', 'judge', 'military', 'oil', 'israel', 'ukraine', 'police'],
        'politics': ['trump', 'iran', 'war', 'bondi', 'hegseth', 'court', 'army', 'judge', 'ukraine', 'biden'],
        'worldnews': ['trump', 'iran', 'war', 'ukraine', 'russia', 'hormuz', 'nato', 'china', 'military', 'oil'],
        'technology': ['google', 'apple', 'microsoft', 'china', 'iran', 'starlink', 'oracle', 'data', 'tech', 'billion'],
        'Games': ['game', 'steam', 'trailer', 'launch', 'release', 'xbox', 'nintendo', 'online', 'free', 'update'],
    }

    print(f"\n    Predictions for {days[-1]} (r/{sub_name}):")
    print(f"    {'Keyword':<15} {'Yesterday':>10} {'Pred Chg':>10} {'Actual Chg':>10} {'Pred Tmrw':>10} {'Actual':>8} {'Hit?':>6}")
    print(f"    {'-' * 75}")

    for kw in sub_kws.get(sub_name, []):
        series = good.get(kw)
        if not series or len(series) < 4:
            continue
        w = series[-4:-1]
        if len(w) < 3:
            continue
        curr = series[-2]
        feat = np.array([build_feats(w)])
        pred_pct = rf.predict(feat)[0]

        actual_target = 0
        for s in series:
            if s['day'] == days[-1]:
                actual_target = s['posts']
        actual_pct = (actual_target - curr['posts']) / max(1, curr['posts']) if actual_target > 0 else 0

        pred_count = curr['posts'] * (1 + pred_pct)
        direction_match = (pred_pct > 0 and actual_pct > 0) or (pred_pct <= 0 and actual_pct <= 0)
        hit = "YES" if direction_match else "NO"

        print(f"    {kw:<15} {curr['posts']:>10} {pred_pct:>+9.0%} {actual_pct:>+9.0%} {pred_count:>10.0f} {actual_target:>8} {hit:>6}")

    print()

conn.close()
print("=" * 70)
print("DONE")
print("=" * 70)
