"""Use post RISING predictions to improve keyword growth prediction."""
import sqlite3, re, os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
STOPWORDS = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','not','with','from','by','as','it','its','has','have','had','that','this','be','been','will','would','could','should','may','might','can','do','does','did','than','then','so','if','just','about','up','out','no','all','more','some','into','over','after','before','new','says','said','say','get','gets','one','two','first','last','also','how','what','when','where','who','why','which','while','being','he','she','they','his','her','their','our','my','your','we','you','me','us','them','him','against','under','between','through','during','removed','moderator','meta','flair','subreddit','reddit','https','com','www'}

conn = sqlite3.connect(DB_PATH, timeout=30)
state_rank = {'dead': 0, 'dying': 1, 'cooling': 2, 'alive': 3, 'surging': 4}

# Build post rising model
post_rows = conn.execute("""
    SELECT post_id, age_minutes_at_snapshot, activity_state,
           upvotes_at_snapshot, comment_count_at_snapshot, upvote_velocity_per_hour
    FROM post_snapshots
    WHERE activity_state IN ('surging','alive','cooling','dying','dead')
      AND age_minutes_at_snapshot IS NOT NULL
    ORDER BY post_id, age_minutes_at_snapshot
""").fetchall()

post_timeline = defaultdict(list)
for pid, age, state, up, com, vel in post_rows:
    post_timeline[pid].append({
        'age_h': (age or 0) / 60, 'rank': state_rank.get(state, 0),
        'up': up or 0, 'com': com or 0, 'vel': vel or 0, 'state': state
    })

X_rise, y_rise = [], []
for pid, timeline in post_timeline.items():
    if len(timeline) < 2:
        continue
    st = sorted(timeline, key=lambda x: x['age_h'])
    for i in range(1, len(st)):
        rising = 1 if st[i]['rank'] > st[i - 1]['rank'] else 0
        X_rise.append([st[i]['rank'], st[i]['up'], st[i]['com'], st[i]['vel'], st[i]['age_h']])
        y_rise.append(rising)

rf_rise = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
rf_rise.fit(np.array(X_rise), np.array(y_rise))
print("Rising model: %d samples, %.1f%% rising" % (len(y_rise), sum(y_rise) / len(y_rise) * 100))

# Score every post
post_rise_score = {}
for pid, timeline in post_timeline.items():
    st = sorted(timeline, key=lambda x: x['age_h'])
    latest = st[-1]
    feat = np.array([[latest['rank'], latest['up'], latest['com'], latest['vel'], latest['age_h']]])
    post_rise_score[pid] = rf_rise.predict_proba(feat)[0][1]

print("Scored %d posts" % len(post_rise_score))

# Build keyword data
title_rows = conn.execute("""
    SELECT DISTINCT p.post_id, p.title, substr(p.snapshot_time_utc, 1, 10) as day,
           l.max_upvotes, l.latest_activity_state
    FROM post_snapshots p JOIN post_lifecycles l ON p.post_id = l.post_id
    WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
""").fetchall()

day_kw = defaultdict(lambda: defaultdict(list))
seen = set()
for pid, title, day, max_up, state in title_rows:
    key = (pid, day)
    if key in seen:
        continue
    seen.add(key)
    day_kw[day][pid] = {'title': title, 'up': max_up or 0, 'state': state,
                         'rise': post_rise_score.get(pid, 0),
                         'surging': 1 if state == 'surging' else 0,
                         'alive': 1 if state in ('surging', 'alive') else 0}
    for w in set(re.findall(r"[a-z]+", title.lower())):
        if len(w) > 3 and w not in STOPWORDS:
            day_kw[day][w].append(day_kw[day][pid])

# Remove non-keyword entries
for day in day_kw:
    to_remove = [k for k in day_kw[day] if not isinstance(day_kw[day][k], list)]
    for k in to_remove:
        del day_kw[day][k]

days = sorted(day_kw.keys())

# Test
for lo, hi, label in [(1, 5, '1-5'), (3, 15, '3-15'), (5, 15, '5-15')]:
    X_simple, X_enhanced, y = [], [], []

    for kw in set(w for d in day_kw.values() for w in d if isinstance(day_kw[list(day_kw.keys())[0]].get(w), list)):
        for i in range(len(days) - 3):
            posts = day_kw[days[i]].get(kw, [])
            if not isinstance(posts, list):
                continue
            n = len(posts)
            if n < lo or n > hi:
                continue

            max_future = max(
                len([p for p in day_kw[days[j]].get(kw, []) if isinstance(day_kw[days[j]].get(kw), list)])
                if isinstance(day_kw[days[j]].get(kw), list) else 0
                for j in range(i + 1, min(i + 4, len(days)))
            )
            grew = 1 if max_future > n * 1.5 else 0

            prev = sum(1 for j in range(i) if isinstance(day_kw[days[j]].get(kw), list) and len(day_kw[days[j]][kw]) > 0)
            total_up = sum(p['up'] for p in posts)

            X_simple.append([n, total_up, prev])

            rise_probs = [p['rise'] for p in posts]
            X_enhanced.append([n, total_up, prev,
                np.mean(rise_probs),
                max(rise_probs),
                sum(1 for r in rise_probs if r > 0.3) / n,
                sum(p['surging'] for p in posts) / n,
                sum(p['alive'] for p in posts) / n,
            ])
            y.append(grew)

    if len(y) < 100 or sum(y) < 20:
        print("%s: not enough data (%d samples, %d grew)" % (label, len(y), sum(y)))
        continue

    Xs = np.array(X_simple)
    Xe = np.array(X_enhanced)
    ya = np.array(y)

    roc_s = cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), Xs, ya, cv=5, scoring='roc_auc')
    roc_e = cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), Xe, ya, cv=5, scoring='roc_auc')
    diff = roc_e.mean() - roc_s.mean()
    star = '***' if diff > 0.03 else ('**' if diff > 0.02 else ('*' if diff > 0.01 else ''))

    print("\n%s posts (n=%d, grew=%.1f%%):" % (label, len(y), sum(y) / len(y) * 100))
    print("  Simple:   %.3f" % roc_s.mean())
    print("  +Rising:  %.3f" % roc_e.mean())
    print("  Diff:     %+.3f %s" % (diff, star))

    # Show profile
    gi = [i for i, v in enumerate(y) if v == 1]
    ni = [i for i, v in enumerate(y) if v == 0]
    fn = ['posts', 'total_up', 'prev', 'avg_rise', 'max_rise', 'high_rise_pct', 'surging_pct', 'alive_pct']
    for j, name in enumerate(fn):
        gv = np.mean([X_enhanced[i][j] for i in gi])
        nv = np.mean([X_enhanced[i][j] for i in ni])
        diff_val = gv - nv
        pct = diff_val / max(abs(nv), 0.001) * 100
        mark = " <--" if abs(pct) > 10 else ""
        print("    %-20s Grew:%.4f  Didnt:%.4f  %+.0f%%%s" % (name, gv, nv, pct, mark))

conn.close()
print("\nDONE")
