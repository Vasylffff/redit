"""Predict topic popularity - how many posts will a topic get tomorrow?"""
import sqlite3, re, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from collections import defaultdict

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")

STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during'}

conn = sqlite3.connect(DB_PATH)

rows = conn.execute('''
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit
    FROM post_snapshots p
    JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
''').fetchall()

day_topic = defaultdict(lambda: defaultdict(lambda: {'posts': 0, 'upvotes': [], 'alive': 0, 'subs': set()}))
seen = set()
for pid, title, day, max_up, state, sub in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = re.findall(r'[a-z]+', title.lower())
    for w in set(words):
        if len(w) > 3 and w not in STOPWORDS:
            d = day_topic[day][w]
            d['posts'] += 1
            d['upvotes'].append(max_up or 0)
            if state in ('surging', 'alive'):
                d['alive'] += 1
            d['subs'].add(sub)

days = sorted(day_topic.keys())
print(f"Days: {len(days)} ({days[0]} to {days[-1]})")

topic_series = defaultdict(list)
for day in days:
    for kw, d in day_topic[day].items():
        if d['posts'] < 2:
            continue
        topic_series[kw].append({
            'day': day, 'posts': d['posts'],
            'avg_up': sum(d['upvotes']) / d['posts'],
            'total_up': sum(d['upvotes']),
            'alive_rate': d['alive'] / d['posts'],
            'subs': len(d['subs']),
        })

good = {kw: sorted(s, key=lambda x: x['day']) for kw, s in topic_series.items() if len(s) >= 5}
print(f"Topics with 5+ days: {len(good)}")


def build_features(window):
    posts = [w['posts'] for w in window]
    ups = [w['avg_up'] for w in window]
    total = [w['total_up'] for w in window]
    alive = [w['alive_rate'] for w in window]
    subs = [w['subs'] for w in window]
    return [
        posts[-1], posts[-2], posts[-3],
        ups[-1], ups[-2],
        total[-1], total[-2],
        alive[-1], alive[-2],
        subs[-1], max(subs),
        (posts[-1] - posts[0]) / max(1, posts[0]),
        (total[-1] - total[0]) / max(1, total[0]),
        sum(posts) / 3,
    ]


# === REGRESSION MODELS ===
print("\n" + "=" * 70)
print("TOPIC POPULARITY REGRESSION")
print("=" * 70)

for target_name, target_key in [('Post Count', 'posts'), ('Total Upvotes', 'total_up'), ('Alive Rate', 'alive_rate')]:
    for days_ahead in [1, 2, 3]:
        X_data, y_data = [], []
        for kw, series in good.items():
            for i in range(3, len(series) - days_ahead + 1):
                window = series[i-3:i]
                target = series[i + days_ahead - 1]
                X_data.append(build_features(window))
                y_data.append(target[target_key])
        if len(X_data) < 50:
            continue
        X, y = np.array(X_data), np.array(y_data)
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        r2 = cross_val_score(rf, X, y, cv=5, scoring='r2')
        mae = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"  {target_name} {days_ahead}d ahead: R2={r2.mean():.3f}  MAE={-mae.mean():.1f}")
    print()


# === ACTUAL PREDICTIONS FOR TOP TOPICS ===
print("=" * 70)
print("TOPIC POPULARITY PREDICTIONS")
print("=" * 70)

# Train on all data except last day
test_day = days[-1]
X_train, y_train_posts, y_train_ups = [], [], []
for kw, series in good.items():
    for i in range(3, len(series)):
        if series[i]['day'] >= test_day:
            continue
        window = series[max(0, i-3):i]
        if len(window) < 3:
            continue
        X_train.append(build_features(window))
        y_train_posts.append(series[i]['posts'])
        y_train_ups.append(series[i]['total_up'])

rf_posts = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_posts.fit(np.array(X_train), np.array(y_train_posts))

rf_ups = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_ups.fit(np.array(X_train), np.array(y_train_ups))

print(f"\nPredicting for: {test_day}")
print(f"{'Topic':<20} {'Pred Posts':>10} {'Actual':>8} {'Error':>8} {'Pred Upvotes':>13} {'Trend':<20}")
print("-" * 85)

top_kws = ['trump', 'iran', 'war', 'ukraine', 'russia', 'israel', 'court', 'judge',
           'game', 'military', 'china', 'google', 'microsoft', 'oil', 'bondi',
           'hormuz', 'nato', 'energy', 'prices', 'apple']

for kw in top_kws:
    series = good.get(kw)
    if not series or len(series) < 4:
        continue

    recent = series[-4:-1]
    if len(recent) < 3:
        continue

    feat = np.array([build_features(recent)])
    pred_posts = rf_posts.predict(feat)[0]
    pred_ups = rf_ups.predict(feat)[0]

    actual = 0
    for s in series:
        if s['day'] == test_day:
            actual = s['posts']
            break

    error = abs(pred_posts - actual) if actual > 0 else 0
    posts = [w['posts'] for w in recent]
    trend = 'rising' if posts[-1] > posts[0] else ('falling' if posts[-1] < posts[0] else 'stable')

    print(f"{kw:<20} {pred_posts:>10.0f} {actual:>8} {error:>8.0f} {pred_ups:>13,.0f} {posts[0]}->{posts[-1]} ({trend})")


# === FORECAST NEXT 3 DAYS ===
print(f"\n{'=' * 70}")
print("3-DAY FORECAST FOR TOP TOPICS")
print(f"{'=' * 70}")
print(f"{'Topic':<15} {'Today':>8} {'Tomorrow':>10} {'Day 2':>10} {'Day 3':>10} {'Direction':<15}")
print("-" * 70)

for kw in ['trump', 'iran', 'war', 'ukraine', 'russia', 'israel', 'game', 'military', 'bondi']:
    series = good.get(kw)
    if not series or len(series) < 4:
        continue

    today = series[-1]['posts']
    recent = series[-3:]

    # Predict day 1
    if len(recent) >= 3:
        feat = np.array([build_features(recent)])
        day1 = rf_posts.predict(feat)[0]
    else:
        continue

    # Predict day 2 (use predicted day1 as input)
    window2 = [recent[-2].copy(), recent[-1].copy(), {'posts': day1, 'avg_up': recent[-1]['avg_up'], 'total_up': recent[-1]['total_up'], 'alive_rate': recent[-1]['alive_rate'], 'subs': recent[-1]['subs']}]
    feat2 = np.array([build_features(window2)])
    day2 = rf_posts.predict(feat2)[0]

    # Predict day 3
    window3 = [recent[-1].copy(), {'posts': day1, 'avg_up': recent[-1]['avg_up'], 'total_up': recent[-1]['total_up'], 'alive_rate': recent[-1]['alive_rate'], 'subs': recent[-1]['subs']}, {'posts': day2, 'avg_up': recent[-1]['avg_up'], 'total_up': recent[-1]['total_up'], 'alive_rate': recent[-1]['alive_rate'], 'subs': recent[-1]['subs']}]
    feat3 = np.array([build_features(window3)])
    day3 = rf_posts.predict(feat3)[0]

    if day3 > today * 1.1:
        direction = "RISING"
    elif day3 < today * 0.9:
        direction = "FALLING"
    else:
        direction = "STABLE"

    print(f"{kw:<15} {today:>8} {day1:>10.0f} {day2:>10.0f} {day3:>10.0f} {direction:<15}")

conn.close()
print(f"\n{'=' * 70}")
print("DONE")
print(f"{'=' * 70}")
