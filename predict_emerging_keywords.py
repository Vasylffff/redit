"""Predict emerging keywords - catch topics early before they explode."""
import sqlite3, re, os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www'}

conn = sqlite3.connect(DB_PATH, timeout=30)

rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state, l.subreddit
    FROM post_snapshots p
    JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

day_kw = defaultdict(lambda: defaultdict(lambda: {
    'posts': 0, 'total_up': 0, 'alive': 0, 'subs': set()
}))
seen = set()
for pid, title, day, max_up, state, sub in rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    words = re.findall(r'[a-z]+', title.lower())
    for w in set(words):
        if len(w) > 3 and w not in STOPWORDS:
            d = day_kw[day][w]
            d['posts'] += 1
            d['total_up'] += (max_up or 0)
            if state in ('surging', 'alive'):
                d['alive'] += 1
            d['subs'].add(sub)

days = sorted(day_kw.keys())
all_kws = set(w for d in day_kw.values() for w in d)

print("=" * 80)
print("EMERGING KEYWORD DETECTOR")
print("=" * 80)
print("Days: %d (%s to %s)" % (len(days), days[0], days[-1]))

# Find keywords that emerged and exploded
print("\nKEYWORDS THAT EMERGED FROM 1-5 POSTS AND EXPLODED:")
print("=" * 80)

emerging = []
for kw in all_kws:
    series = []
    for day in days:
        d = day_kw[day].get(kw)
        if d:
            series.append((day, d['posts'], d['total_up'], d['alive'], len(d['subs'])))
        else:
            series.append((day, 0, 0, 0, 0))

    # Find first meaningful appearance
    first_idx = None
    for i, (day, posts, _, _, _) in enumerate(series):
        if posts >= 1:
            first_idx = i
            break

    if first_idx is None:
        continue
    first_count = series[first_idx][1]
    if first_count > 5:
        continue  # already big

    # Find peak after first appearance
    peak_posts = max(s[1] for s in series[first_idx:])
    if peak_posts < 10:
        continue  # never got big

    peak_idx = first_idx
    for i in range(first_idx, len(series)):
        if series[i][1] == peak_posts:
            peak_idx = i
            break

    growth = peak_posts / max(1, first_count)
    days_to_peak = peak_idx - first_idx

    emerging.append({
        'kw': kw,
        'first_day': series[first_idx][0],
        'first_count': first_count,
        'first_upvotes': series[first_idx][2],
        'first_alive': series[first_idx][3],
        'first_subs': series[first_idx][4],
        'peak_day': series[peak_idx][0],
        'peak_count': peak_posts,
        'peak_upvotes': series[peak_idx][2],
        'growth': growth,
        'days_to_peak': days_to_peak,
        'timeline': [s[1] for s in series],
    })

emerging.sort(key=lambda x: -x['growth'])

print("%-15s %6s %6s %6s %7s %5s  Timeline (posts/day)" % (
    'Keyword', 'Start', 'Peak', 'Growth', 'UpvotK', 'Days'))
print("-" * 90)
for e in emerging[:30]:
    tl = ' '.join(["%3d" % p for p in e['timeline']])
    print("%-15s %6d %6d %5.0fx %7.0fK %5d  [%s]" % (
        e['kw'], e['first_count'], e['peak_count'], e['growth'],
        e['peak_upvotes']/1000, e['days_to_peak'], tl))

# BUILD PREDICTOR: on day 1 of a keyword (1-5 posts), will it explode?
print("\n" + "=" * 80)
print("EMERGENCE PREDICTION MODEL")
print("=" * 80)

X_data, y_data = [], []

for kw in all_kws:
    for i, day in enumerate(days[:-3]):
        d = day_kw[day].get(kw)
        if not d or d['posts'] < 1 or d['posts'] > 5:
            continue

        # How many days was this keyword seen before?
        prev_days = sum(1 for j in range(i) if day_kw[days[j]].get(kw, {}).get('posts', 0) > 0)

        # Day 1 features
        avg_up = d['total_up'] / max(1, d['posts'])
        alive_rate = d['alive'] / max(1, d['posts'])
        sub_count = len(d['subs'])

        # Did it explode to 10+ posts within next 3 days?
        max_future = max(
            day_kw[days[j]].get(kw, {}).get('posts', 0)
            for j in range(i + 1, min(i + 4, len(days)))
        )
        exploded = 1 if max_future >= 10 else 0

        X_data.append([
            d['posts'],
            avg_up,
            alive_rate,
            prev_days,
            d['total_up'],
            sub_count,
        ])
        y_data.append(exploded)

X = np.array(X_data)
y = np.array(y_data)
print("Observations: %d" % len(X))
print("Exploded: %d (%.1f%%)" % (sum(y), sum(y)/len(y)*100))
print("Didn't explode: %d (%.1f%%)" % (len(y)-sum(y), (len(y)-sum(y))/len(y)*100))

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
roc = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
acc = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print("\nEmergence prediction ROC AUC: %.3f" % roc.mean())
print("Accuracy: %.1f%%" % (acc.mean()*100))

rf.fit(X, y)
feat_names = ['initial_posts', 'avg_upvotes', 'alive_rate', 'prev_days_seen', 'total_upvotes', 'subreddit_count']
print("\nFeature importance (what predicts explosion?):")
for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
    bar = '#' * int(imp * 40)
    print("  %-20s %.3f  %s" % (name, imp, bar))

# What makes a keyword explode? Compare exploded vs not
print("\n" + "=" * 80)
print("WHAT MAKES A KEYWORD EXPLODE? (comparing day-1 signals)")
print("=" * 80)

exploded_idx = [i for i, v in enumerate(y_data) if v == 1]
not_idx = [i for i, v in enumerate(y_data) if v == 0]

print("%-25s %15s %15s" % ('Signal', 'Exploded', 'Didnt explode'))
print("-" * 60)
for j, name in enumerate(feat_names):
    exp_vals = [X_data[i][j] for i in exploded_idx]
    not_vals = [X_data[i][j] for i in not_idx]
    if exp_vals and not_vals:
        print("%-25s %15.1f %15.1f" % (
            name, np.mean(exp_vals), np.mean(not_vals)))

# Score current emerging keywords (last 2 days, 1-5 posts)
print("\n" + "=" * 80)
print("CURRENTLY EMERGING KEYWORDS (last 2 days, 1-5 posts)")
print("Ranked by explosion probability")
print("=" * 80)

current_emerging = []
for kw in all_kws:
    for day in days[-2:]:
        d = day_kw[day].get(kw)
        if not d or d['posts'] < 1 or d['posts'] > 5:
            continue
        prev_days = sum(1 for j, dd in enumerate(days) if dd < day and day_kw[dd].get(kw, {}).get('posts', 0) > 0)
        avg_up = d['total_up'] / max(1, d['posts'])
        alive_rate = d['alive'] / max(1, d['posts'])
        sub_count = len(d['subs'])
        feat = np.array([[d['posts'], avg_up, alive_rate, prev_days, d['total_up'], sub_count]])
        prob = rf.predict_proba(feat)[0][1]
        current_emerging.append({
            'kw': kw, 'day': day, 'posts': d['posts'],
            'avg_up': avg_up, 'alive_rate': alive_rate,
            'subs': sub_count, 'prob': prob, 'prev_days': prev_days,
        })

current_emerging.sort(key=lambda x: -x['prob'])

print("%-15s %6s %5s %8s %6s %5s %5s %8s" % (
    'Keyword', 'Day', 'Posts', 'Avg Up', 'Alive', 'Subs', 'Prev', 'Explode%'))
print("-" * 65)
for e in current_emerging[:30]:
    print("%-15s %6s %5d %8.0f %5.0f%% %5d %5d %7.0f%%" % (
        e['kw'], e['day'][-5:], e['posts'], e['avg_up'],
        e['alive_rate']*100, e['subs'], e['prev_days'], e['prob']*100))

conn.close()
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
