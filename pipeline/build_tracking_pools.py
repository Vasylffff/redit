"""
build_tracking_pools.py  —  Smart tracking pool manager

Splits posts into three pools based on their velocity variance signature:

  active_pool   →  check every hour   (alive/surging or unpredictable variance)
  dormant_pool  →  check every 6h     (variance collapsed but <24h old — watch for revival)
  dropped_pool  →  stop tracking      (variance collapsed + 24h+ old + confirmed dead signals)

Dead detection uses variance collapse rather than just velocity threshold:
  - High variance before collapse  →  post was still alive/fighting
  - Low variance + declining after →  locked into decay, genuinely dead

Outputs:
  data/tracking/active_pool.csv
  data/tracking/dormant_pool.csv
  data/tracking/dropped_pool.csv
  data/tracking/pool_summary.csv
"""

import collections
import csv
import os
import statistics
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SNAPSHOTS_PATH = "data/history/reddit/post_snapshots.csv"
TRACKING_DIR   = "data/tracking"

# Variance collapse thresholds (tuned from data analysis)
STD_HIGH_THRESHOLD  = 15.0   # std before collapse must be above this
STD_LOW_THRESHOLD   = 8.0    # std after collapse must be below this
MEAN_VEL_THRESHOLD  = 40.0   # mean velocity after collapse must be below this
WINDOW_SIZE         = 4      # snapshots to look at before/after

# Confirmed dead — all of these must be true
DEAD_MIN_SNAPSHOTS  = 3      # need at least this many in dead-like state
DEAD_MAX_VELOCITY   = 5.0    # upvote velocity below this
DEAD_MAX_CV         = 0.5    # comment velocity below this

# Age thresholds
DORMANT_MAX_HOURS   = 24     # posts younger than this go to dormant, not dropped
MIN_SNAPSHOTS_TO_CLASSIFY = 5  # need at least this many snapshots to classify


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_snapshots():
    if not os.path.exists(SNAPSHOTS_PATH):
        raise FileNotFoundError(f"Not found: {SNAPSHOTS_PATH}")

    by_post = collections.defaultdict(list)
    with open(SNAPSHOTS_PATH, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            pid = row["post_id"]
            t   = row.get("snapshot_time_utc", "")
            try:
                v = float(row.get("upvote_velocity_per_hour") or 0)
            except ValueError:
                v = 0.0
            try:
                cv = float(row.get("comment_velocity_per_hour") or 0)
            except ValueError:
                cv = 0.0
            try:
                age_h = float(row.get("age_hours_at_snapshot") or 0)
            except ValueError:
                age_h = 0.0

            by_post[pid].append({
                "snapshot_id":  row["snapshot_id"],
                "time":         t,
                "subreddit":    row.get("subreddit", ""),
                "title":        row.get("title", ""),
                "url":          row.get("url", ""),
                "velocity":     v,
                "cv":           cv,
                "age_hours":    age_h,
                "state":        row.get("activity_state", "").strip(),
                "upvotes":      row.get("upvotes_at_snapshot", "0"),
                "listing_type": row.get("listing_type", ""),
                "seen_in_rising": row.get("listing_type", "") == "rising",
            })

    # Sort by time
    for pid in by_post:
        by_post[pid].sort(key=lambda x: x["time"])

    return by_post


# ---------------------------------------------------------------------------
# Variance collapse detection
# ---------------------------------------------------------------------------

def clean_velocities(velocities):
    """
    Remove collection gaps (zeros) from the velocity sequence.
    A zero is a gap if the value after it is significantly higher —
    meaning the post was simply not captured that hour, not actually dead.
    A zero is real decay if the values around it are also low.
    """
    if not velocities:
        return velocities

    cleaned = []
    for i, v in enumerate(velocities):
        if v == 0.0:
            # Look at neighbours to decide: gap or real zero?
            prev_val = velocities[i-1] if i > 0 else 0
            next_val = velocities[i+1] if i < len(velocities)-1 else 0
            # If surrounded by high values → collection gap, skip it
            if prev_val > 10 or next_val > 10:
                continue  # drop this zero — it's a gap
        cleaned.append(v)
    return cleaned


def find_variance_collapse(velocities):
    """
    Slide a window through velocity sequence looking for the point where
    variance drops and stays low — the dead trajectory lock-in.

    Cleans collection gaps first so zeros don't trigger false collapses.
    Returns collapse_index or None.
    """
    velocities = clean_velocities(velocities)
    n = len(velocities)
    if n < WINDOW_SIZE * 2 + 1:
        return None

    for i in range(WINDOW_SIZE, n - WINDOW_SIZE):
        before = velocities[max(0, i - WINDOW_SIZE):i]
        after  = velocities[i:i + WINDOW_SIZE]

        if len(before) < 2 or len(after) < 2:
            continue

        std_before = statistics.stdev(before)
        std_after  = statistics.stdev(after)
        mean_after = statistics.mean(after)

        # Extra check: after window must be strictly declining, not flat-high
        is_declining = after[-1] <= after[0] if len(after) >= 2 else True

        if (std_before > STD_HIGH_THRESHOLD
                and std_after  < STD_LOW_THRESHOLD
                and mean_after < MEAN_VEL_THRESHOLD
                and is_declining):
            return i

    return None


def is_confirmed_dead(snaps):
    """
    Check if the last N snapshots all show dead-like behaviour.
    Requires: low velocity, low comment velocity, consistent state.
    """
    if len(snaps) < DEAD_MIN_SNAPSHOTS:
        return False

    last = snaps[-DEAD_MIN_SNAPSHOTS:]
    velocities = [s["velocity"] for s in last]
    cvs        = [s["cv"]       for s in last]
    states     = [s["state"]    for s in last]

    low_vel  = all(v <= DEAD_MAX_VELOCITY for v in velocities)
    low_cv   = all(c <= DEAD_MAX_CV       for c in cvs)
    bad_state = sum(1 for s in states if s in ("dead", "dying")) >= 2

    return low_vel and low_cv and bad_state


def shows_revival_signal(snaps):
    """
    Check if a dormant post is showing signs of revival.
    Requires sustained signal across 2+ snapshots to avoid
    flagging collection gaps as revivals.
    """
    if len(snaps) < 4:
        return False

    last3 = snaps[-3:]
    vels  = [s["velocity"] for s in last3]
    cvs   = [s["cv"]       for s in last3]

    # Need at least 2 of last 3 snapshots above threshold
    # (not just 1, which could be a collection gap artifact)
    vel_spikes = sum(1 for v in vels if v > 20)
    cv_spikes  = sum(1 for c in cvs  if c > 5)

    if vel_spikes >= 2:
        return True

    # Appeared in rising in last 2 snapshots
    if sum(1 for s in last3 if s.get("seen_in_rising")) >= 1:
        return True

    # Strong comment velocity sustained
    if cv_spikes >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Classify each post
# ---------------------------------------------------------------------------

def classify_post(pid, snaps):
    """
    Returns (pool, reason, collapse_index)
    pool: 'active' | 'dormant' | 'dropped'
    """
    n = len(snaps)
    if n < MIN_SNAPSHOTS_TO_CLASSIFY:
        return "active", "insufficient data — keep watching", None

    latest     = snaps[-1]
    age_hours  = latest["age_hours"]
    velocities = [s["velocity"] for s in snaps]
    last_state = latest["state"]

    # Check for revival signal first — always moves back to active
    if shows_revival_signal(snaps):
        return "active", "revival signal detected", None

    # Find variance collapse point
    collapse_idx = find_variance_collapse(velocities)

    if collapse_idx is None:
        # No collapse detected — still in chaotic/alive phase
        return "active", "no variance collapse — still active", None

    # Variance has collapsed — decide dormant vs dropped
    confirmed_dead = is_confirmed_dead(snaps)

    if confirmed_dead and age_hours >= DORMANT_MAX_HOURS:
        return "dropped", f"variance collapsed at snap {collapse_idx}, confirmed dead, age {age_hours:.0f}h", collapse_idx

    if confirmed_dead and age_hours < DORMANT_MAX_HOURS:
        return "dormant", f"variance collapsed at snap {collapse_idx}, dead signals but young ({age_hours:.0f}h)", collapse_idx

    if collapse_idx is not None and age_hours >= DORMANT_MAX_HOURS:
        return "dormant", f"variance collapsed at snap {collapse_idx}, age {age_hours:.0f}h", collapse_idx

    # Collapsed but still young — dormant to watch
    return "dormant", f"variance collapsed at snap {collapse_idx}, monitoring for revival", collapse_idx


# ---------------------------------------------------------------------------
# Build pools
# ---------------------------------------------------------------------------

def build_pools(by_post):
    active  = []
    dormant = []
    dropped = []

    for pid, snaps in by_post.items():
        if not snaps:
            continue
        latest = snaps[-1]
        pool, reason, collapse_idx = classify_post(pid, snaps)

        velocities = [s["velocity"] for s in snaps]
        last_n_vel = velocities[-4:] if len(velocities) >= 4 else velocities
        try:
            recent_std = round(statistics.stdev(last_n_vel), 2) if len(last_n_vel) > 1 else 0
        except Exception:
            recent_std = 0

        entry = {
            "post_id":          pid,
            "subreddit":        latest["subreddit"],
            "title":            latest["title"][:80],
            "url":              latest["url"],
            "pool":             pool,
            "reason":           reason,
            "snapshots":        len(snaps),
            "age_hours":        round(latest["age_hours"], 1),
            "latest_state":     latest["state"],
            "latest_velocity":  round(latest["velocity"], 2),
            "latest_cv":        round(latest["cv"], 2),
            "recent_vel_std":   recent_std,
            "collapse_at_snap": collapse_idx if collapse_idx is not None else "",
            "last_seen":        latest["time"][:16],
        }

        if pool == "active":
            active.append(entry)
        elif pool == "dormant":
            dormant.append(entry)
        else:
            dropped.append(entry)

    return active, dormant, dropped


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

FIELDS = [
    "post_id", "subreddit", "title", "url", "pool", "reason",
    "snapshots", "age_hours", "latest_state", "latest_velocity",
    "latest_cv", "recent_vel_std", "collapse_at_snap", "last_seen"
]

def save_pool(rows, filename):
    os.makedirs(TRACKING_DIR, exist_ok=True)
    path = os.path.join(TRACKING_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def save_summary(active, dormant, dropped):
    path = os.path.join(TRACKING_DIR, "pool_summary.csv")
    # Count by subreddit and pool
    rows = []
    all_subs = sorted(set(
        r["subreddit"] for r in active + dormant + dropped
    ))
    for sub in all_subs:
        a = sum(1 for r in active  if r["subreddit"] == sub)
        d = sum(1 for r in dormant if r["subreddit"] == sub)
        x = sum(1 for r in dropped if r["subreddit"] == sub)
        rows.append({
            "subreddit": sub,
            "active":  a,
            "dormant": d,
            "dropped": x,
            "total":   a + d + x,
            "pct_active":  f"{100*a/(a+d+x):.0f}%" if (a+d+x) else "0%",
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subreddit","active","dormant","dropped","total","pct_active"])
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading snapshots...")
    by_post = load_snapshots()
    print(f"  {len(by_post):,} unique posts loaded")

    print("Classifying posts into pools...")
    active, dormant, dropped = build_pools(by_post)

    total = len(active) + len(dormant) + len(dropped)
    print(f"\n  {'active':<10} {len(active):>5}  ({100*len(active)/total:.0f}%)  — check every hour")
    print(f"  {'dormant':<10} {len(dormant):>5}  ({100*len(dormant)/total:.0f}%)  — check every 6h")
    print(f"  {'dropped':<10} {len(dropped):>5}  ({100*len(dropped)/total:.0f}%)  — stop tracking")

    # Breakdown: why were posts classified each way?
    import collections as col
    reasons = col.Counter(r["reason"].split(",")[0].strip() for r in dormant + dropped)
    print("\n  Top dormant/dropped reasons:")
    for reason, n in reasons.most_common(5):
        print(f"    {reason:<50} {n}")

    print("\nSaving pools...")
    p1 = save_pool(active,  "active_pool.csv")
    p2 = save_pool(dormant, "dormant_pool.csv")
    p3 = save_pool(dropped, "dropped_pool.csv")
    p4 = save_summary(active, dormant, dropped)

    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
    print(f"  {p4}")

    # Revival candidates — dormant posts showing early revival signal
    revivals = [r for r in active if "revival" in r["reason"]]
    if revivals:
        print(f"\n  Revival signals detected: {len(revivals)}")
        for r in revivals[:5]:
            print(f"    r/{r['subreddit']:<12} vel={r['latest_velocity']:>6.1f}/hr  {r['title'][:50]}")


if __name__ == "__main__":
    main()
