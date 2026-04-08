"""
Time-to-Death Predictor
========================
Given a post's current state and metrics, predict how many hours until it dies.
"""

import csv
import os
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "history", "reddit", "history.db")
OUT_DIR = os.path.join(PROJECT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("TIME-TO-DEATH PREDICTOR")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # Get posts that actually died — we know their full lifespan
    rows = conn.execute("""
        SELECT post_id, subreddit, latest_activity_state, observed_hours,
               max_upvotes, max_comments, total_upvote_growth, total_comment_growth,
               snapshot_count, first_upvotes, last_upvotes,
               first_comments, last_comments,
               last_upvote_velocity_per_hour, last_comment_velocity_per_hour,
               activity_states_seen, listing_types_seen, seen_in_hot, seen_in_rising
        FROM post_lifecycles
        WHERE snapshot_count >= 3
          AND observed_hours > 0
          AND latest_activity_state IN ('dead', 'dying', 'cooling')
    """).fetchall()

    print(f"  {len(rows)} dead/dying/cooling posts for training")

    # Also get state-at-each-snapshot for time-in-state calculations
    snap_rows = conn.execute("""
        SELECT post_id, age_minutes_at_snapshot, activity_state,
               upvotes_at_snapshot, comment_count_at_snapshot, upvote_velocity_per_hour
        FROM post_snapshots
        WHERE activity_state IS NOT NULL AND activity_state != ''
        ORDER BY post_id, age_minutes_at_snapshot
    """).fetchall()

    # Build per-post snapshot timeline
    post_snaps = defaultdict(list)
    for pid, age, state, up, com, vel in snap_rows:
        post_snaps[pid].append({
            "age_min": age or 0, "state": state,
            "upvotes": up or 0, "comments": com or 0, "velocity": vel or 0
        })

    # Empirical: time-to-death from each state
    print(f"\n{'=' * 70}")
    print("EMPIRICAL TIME-TO-DEATH FROM EACH STATE")
    print(f"{'=' * 70}")

    state_to_death = defaultdict(list)  # (state, sub) -> [hours_to_death]

    for pid, sub, final_state, obs_hours, *_ in rows:
        snaps = post_snaps.get(pid, [])
        if len(snaps) < 2:
            continue

        sorted_snaps = sorted(snaps, key=lambda x: x["age_min"])
        death_age = sorted_snaps[-1]["age_min"] / 60  # hours

        # For each snapshot, how long until death?
        for s in sorted_snaps:
            remaining = death_age - s["age_min"] / 60
            if remaining >= 0:
                state_to_death[(s["state"], sub)].append(remaining)
                state_to_death[(s["state"], "ALL")].append(remaining)

    # Print table
    states = ["surging", "alive", "emerging", "cooling", "dying", "dead"]
    print(f"\n  Overall (all subreddits):")
    print(f"  {'State':<12} {'Posts':>6} {'Med hrs':>10} {'Avg hrs':>10} {'P25':>8} {'P75':>8}")
    print(f"  {'-' * 50}")
    for state in states:
        hrs = state_to_death.get((state, "ALL"), [])
        if not hrs:
            continue
        s = sorted(hrs)
        print(f"  {state:<12} {len(hrs):>6} {statistics.median(hrs):>10.1f} {statistics.mean(hrs):>10.1f} {s[len(s)//4]:>8.1f} {s[3*len(s)//4]:>8.1f}")

    # Per subreddit
    print(f"\n  Per subreddit (median hours to death from alive state):")
    for sub in sorted(set(s for (st, s) in state_to_death.keys() if s != "ALL")):
        alive_hrs = state_to_death.get(("alive", sub), [])
        surging_hrs = state_to_death.get(("surging", sub), [])
        if alive_hrs:
            print(f"    {sub}: alive->{statistics.median(alive_hrs):.0f}h", end="")
        if surging_hrs:
            print(f"  surging->{statistics.median(surging_hrs):.0f}h", end="")
        print()

    # Build ML predictor
    if _HAS_SKLEARN:
        print(f"\n{'=' * 70}")
        print("TIME-TO-DEATH ML MODEL")
        print(f"{'=' * 70}")

        # Features from each snapshot point
        training = []
        for pid, sub, final_state, obs_hours, max_up, max_com, up_growth, com_growth, \
                snap_count, first_up, last_up, first_com, last_com, \
                last_up_vel, last_com_vel, states_seen, listings_seen, in_hot, in_rising in rows:

            snaps = post_snaps.get(pid, [])
            if len(snaps) < 2:
                continue

            sorted_snaps = sorted(snaps, key=lambda x: x["age_min"])
            death_age = sorted_snaps[-1]["age_min"] / 60

            # Use midpoint snapshot as "current" observation
            mid_idx = len(sorted_snaps) // 2
            mid = sorted_snaps[mid_idx]
            remaining = death_age - mid["age_min"] / 60

            if remaining < 0:
                continue

            state_num = {"surging": 5, "alive": 4, "emerging": 3, "cooling": 2, "dying": 1, "dead": 0}

            training.append({
                "remaining_hours": remaining,
                "current_state": state_num.get(mid["state"], 2),
                "current_upvotes": mid["upvotes"],
                "current_comments": mid["comments"],
                "current_velocity": mid["velocity"],
                "age_hours": mid["age_min"] / 60,
                "max_upvotes": max_up or 0,
                "snap_count_so_far": mid_idx + 1,
                "seen_in_hot": in_hot or 0,
                "seen_in_rising": in_rising or 0,
                "states_seen_count": len(set((states_seen or "").split("|"))),
                "up_growth": up_growth or 0,
            })

        print(f"  Training samples: {len(training)}")

        feature_names = [
            "current_state", "current_upvotes", "current_comments",
            "current_velocity", "age_hours", "max_upvotes",
            "snap_count_so_far", "seen_in_hot", "seen_in_rising",
            "states_seen_count", "up_growth"
        ]

        X = np.array([[t[f] for f in feature_names] for t in training])
        y = np.array([t["remaining_hours"] for t in training])

        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        r2 = cross_val_score(rf, X, y, cv=5, scoring="r2")
        mae = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error")

        print(f"  R2 score (5-fold): {r2.mean():.3f} (+/- {r2.std():.3f})")
        print(f"  Mean Absolute Error: {-mae.mean():.1f} hours")

        rf.fit(X, y)
        print(f"\n  Feature importance:")
        for name, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1]):
            if imp > 0.01:
                bar = "#" * int(imp * 40)
                print(f"    {name:<25} {imp:.3f}  {bar}")

        # Example predictions
        print(f"\n{'=' * 70}")
        print("EXAMPLE PREDICTIONS")
        print(f"{'=' * 70}")

        examples = [
            {"desc": "Surging post, 500 up, 2h old", "current_state": 5, "current_upvotes": 500,
             "current_comments": 50, "current_velocity": 200, "age_hours": 2,
             "max_upvotes": 500, "snap_count_so_far": 3, "seen_in_hot": 0,
             "seen_in_rising": 1, "states_seen_count": 2, "up_growth": 450},
            {"desc": "Alive post, 200 up, 8h old", "current_state": 4, "current_upvotes": 200,
             "current_comments": 30, "current_velocity": 20, "age_hours": 8,
             "max_upvotes": 250, "snap_count_so_far": 5, "seen_in_hot": 1,
             "seen_in_rising": 1, "states_seen_count": 3, "up_growth": 180},
            {"desc": "Cooling post, 100 up, 15h old", "current_state": 2, "current_upvotes": 100,
             "current_comments": 20, "current_velocity": 3, "age_hours": 15,
             "max_upvotes": 150, "snap_count_so_far": 8, "seen_in_hot": 0,
             "seen_in_rising": 0, "states_seen_count": 3, "up_growth": 80},
            {"desc": "Dying post, 50 up, 20h old", "current_state": 1, "current_upvotes": 50,
             "current_comments": 10, "current_velocity": 0.5, "age_hours": 20,
             "max_upvotes": 80, "snap_count_so_far": 10, "seen_in_hot": 0,
             "seen_in_rising": 0, "states_seen_count": 4, "up_growth": 30},
            {"desc": "Fresh post, 10 up, 30min old", "current_state": 3, "current_upvotes": 10,
             "current_comments": 2, "current_velocity": 15, "age_hours": 0.5,
             "max_upvotes": 10, "snap_count_so_far": 1, "seen_in_hot": 0,
             "seen_in_rising": 0, "states_seen_count": 1, "up_growth": 8},
        ]

        for ex in examples:
            feat = np.array([[ex[f] for f in feature_names]])
            pred = rf.predict(feat)[0]
            print(f"\n  {ex['desc']}")
            print(f"    Predicted time to death: ~{pred:.0f} hours")
            if pred < 3:
                print(f"    Status: IMMINENT DEATH")
            elif pred < 12:
                print(f"    Status: Declining, hours left")
            elif pred < 24:
                print(f"    Status: Still has a day")
            else:
                print(f"    Status: Healthy, day+ remaining")

    # Save empirical tables
    path = os.path.join(OUT_DIR, "time_to_death.csv")
    save_rows = []
    for (state, sub), hrs in state_to_death.items():
        if sub == "ALL" and hrs:
            save_rows.append({
                "state": state,
                "median_hours_to_death": round(statistics.median(hrs), 1),
                "avg_hours_to_death": round(statistics.mean(hrs), 1),
                "p25": round(sorted(hrs)[len(hrs)//4], 1),
                "p75": round(sorted(hrs)[3*len(hrs)//4], 1),
                "sample_count": len(hrs),
            })
    if save_rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(save_rows[0].keys()))
            w.writeheader()
            w.writerows(save_rows)
        print(f"\n  Saved: {path}")

    conn.close()
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
