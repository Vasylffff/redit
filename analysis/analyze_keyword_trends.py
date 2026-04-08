"""
Keyword Trend Detection
========================
What topics are rising/falling across subreddits over time?
Extract keywords from titles, track frequency over snapshot windows.
"""

import csv
import os
import re
import sqlite3
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timezone

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "from", "by", "as", "it", "its",
    "has", "have", "had", "that", "this", "be", "been", "will", "would", "could",
    "should", "may", "might", "can", "do", "does", "did", "than", "then", "so",
    "if", "just", "about", "up", "out", "no", "all", "more", "some", "into",
    "over", "after", "before", "new", "says", "said", "say", "get", "gets",
    "one", "two", "first", "last", "also", "how", "what", "when", "where",
    "who", "why", "which", "while", "being", "been", "he", "she", "they",
    "his", "her", "their", "our", "my", "your", "we", "you", "i", "me",
    "us", "them", "him", "against", "under", "between", "through", "during",
    "each", "every", "both", "such", "only", "other", "any", "most", "very",
    "own", "same", "back", "now", "even", "still", "here", "there", "per",
    "off", "down", "well", "way", "going", "go", "come", "make", "take",
}


def extract_keywords(title):
    """Extract meaningful words from a title"""
    words = re.findall(r'[a-z]+', title.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def main():
    print("=" * 70)
    print("KEYWORD TREND DETECTION")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # Get all posts with snapshot times
    rows = conn.execute("""
        SELECT p.post_id, p.subreddit, p.title, p.snapshot_time_utc, p.activity_state
        FROM post_snapshots p
        WHERE p.title IS NOT NULL AND p.title != ''
          AND p.snapshot_time_utc IS NOT NULL
        ORDER BY p.snapshot_time_utc
    """).fetchall()

    print(f"  {len(rows)} snapshot observations with titles")

    # Parse dates and group into day buckets
    day_keywords = defaultdict(Counter)  # day_str -> Counter of keywords
    day_sub_keywords = defaultdict(lambda: defaultdict(Counter))  # day -> sub -> Counter
    keyword_states = defaultdict(lambda: defaultdict(int))  # keyword -> state -> count
    keyword_upvotes = defaultdict(list)

    # Also get upvotes for keyword scoring
    upvote_rows = conn.execute("""
        SELECT p.post_id, p.title, l.max_upvotes, l.latest_activity_state, l.subreddit
        FROM post_lifecycles l
        JOIN (SELECT post_id, title FROM post_snapshots GROUP BY post_id) p ON l.post_id = p.post_id
        WHERE p.title IS NOT NULL AND l.max_upvotes IS NOT NULL
    """).fetchall()

    for pid, title, max_up, state, sub in upvote_rows:
        keywords = extract_keywords(title)
        for kw in set(keywords):  # unique per post
            keyword_upvotes[kw].append(max_up)
            if state:
                keyword_states[kw][state] += 1

    seen_per_day = defaultdict(set)
    for pid, sub, title, snap_time, state in rows:
        try:
            day = snap_time[:10]  # YYYY-MM-DD
        except (TypeError, IndexError):
            continue

        # Only count each post once per day
        key = (pid, day)
        if key in seen_per_day[day]:
            continue
        seen_per_day[day].add(key)

        keywords = extract_keywords(title)
        for kw in set(keywords):
            day_keywords[day][kw] += 1
            day_sub_keywords[day][sub][kw] += 1

    days = sorted(day_keywords.keys())
    print(f"  {len(days)} days of data: {days[0]} to {days[-1]}")

    # Overall top keywords
    total_keywords = Counter()
    for day_counter in day_keywords.values():
        total_keywords += day_counter

    print(f"\n{'=' * 70}")
    print("OVERALL TOP KEYWORDS (across all days)")
    print(f"{'=' * 70}")
    for kw, count in total_keywords.most_common(30):
        avg_up = statistics.mean(keyword_upvotes.get(kw, [0]))
        states = keyword_states.get(kw, {})
        total_s = sum(states.values())
        alive_pct = (states.get("surging", 0) + states.get("alive", 0)) / total_s if total_s > 0 else 0
        print(f"  {kw:<20} mentions={count:>5}  avg_up={avg_up:>8,.0f}  alive={alive_pct:.0%}")

    # Trending keywords (rising from first half to second half)
    print(f"\n{'=' * 70}")
    print("TRENDING KEYWORDS (rising)")
    print(f"{'=' * 70}")

    mid = len(days) // 2
    first_half = Counter()
    second_half = Counter()
    for i, day in enumerate(days):
        if i < mid:
            first_half += day_keywords[day]
        else:
            second_half += day_keywords[day]

    trends = []
    for kw in set(list(first_half.keys()) + list(second_half.keys())):
        f_count = first_half.get(kw, 0)
        s_count = second_half.get(kw, 0)
        total = f_count + s_count
        if total < 5:
            continue
        # Normalize by number of days in each half
        f_rate = f_count / max(1, mid)
        s_rate = s_count / max(1, len(days) - mid)
        if f_rate > 0:
            change = (s_rate - f_rate) / f_rate
        elif s_count > 0:
            change = 10.0  # new keyword
        else:
            change = 0

        trends.append({
            "keyword": kw,
            "first_half_mentions": f_count,
            "second_half_mentions": s_count,
            "first_half_rate": round(f_rate, 2),
            "second_half_rate": round(s_rate, 2),
            "change_pct": round(change, 4),
            "total_mentions": total,
            "avg_upvotes": round(statistics.mean(keyword_upvotes.get(kw, [0]))),
        })

    # Rising keywords
    rising = sorted([t for t in trends if t["change_pct"] > 0.3], key=lambda x: -x["change_pct"])
    print(f"  {'Keyword':<20} {'1st half':>10} {'2nd half':>10} {'Change':>10} {'Avg Up':>8}")
    print(f"  {'-' * 65}")
    for t in rising[:20]:
        print(f"  {t['keyword']:<20} {t['first_half_mentions']:>10} {t['second_half_mentions']:>10} {t['change_pct']:>+9.0%} {t['avg_upvotes']:>8}")

    # Falling keywords
    print(f"\n{'=' * 70}")
    print("DECLINING KEYWORDS (falling)")
    print(f"{'=' * 70}")
    falling = sorted([t for t in trends if t["change_pct"] < -0.3], key=lambda x: x["change_pct"])
    print(f"  {'Keyword':<20} {'1st half':>10} {'2nd half':>10} {'Change':>10} {'Avg Up':>8}")
    print(f"  {'-' * 65}")
    for t in falling[:20]:
        print(f"  {t['keyword']:<20} {t['first_half_mentions']:>10} {t['second_half_mentions']:>10} {t['change_pct']:>+9.0%} {t['avg_upvotes']:>8}")

    # Keywords with highest upvote potential
    print(f"\n{'=' * 70}")
    print("HIGHEST UPVOTE KEYWORDS (min 10 posts)")
    print(f"{'=' * 70}")
    high_up = [t for t in trends if t["total_mentions"] >= 10]
    for t in sorted(high_up, key=lambda x: -x["avg_upvotes"])[:20]:
        print(f"  {t['keyword']:<20} avg_up={t['avg_upvotes']:>8,}  mentions={t['total_mentions']}")

    # Per-subreddit top keywords
    print(f"\n{'=' * 70}")
    print("TOP KEYWORDS BY SUBREDDIT (most recent day)")
    print(f"{'=' * 70}")
    latest_day = days[-1]
    for sub in sorted(day_sub_keywords[latest_day].keys()):
        top5 = day_sub_keywords[latest_day][sub].most_common(8)
        if top5:
            kws = ", ".join(f"{kw}({c})" for kw, c in top5)
            print(f"  {sub}: {kws}")

    # Save
    path = os.path.join(OUT_DIR, "keyword_trends.csv")
    if trends:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trends[0].keys()))
            w.writeheader()
            w.writerows(sorted(trends, key=lambda x: -abs(x["change_pct"])))
        print(f"\n  Saved: {path} ({len(trends)} keywords)")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
