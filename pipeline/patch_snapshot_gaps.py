"""
patch_snapshot_gaps.py  —  Fix collection gaps in post_snapshots.csv

For each post's snapshot sequence:
  gap < 3h  →  interpolate velocity from upvote diff / elapsed hours
  gap >= 3h →  flag as is_collection_gap = 1, leave velocity as-is

Run this AFTER build_reddit_history.py, BEFORE any analysis scripts.

Output:  overwrites post_snapshots.csv with patched version
         (original backed up to post_snapshots_pre_patch.csv)
"""

import collections
import csv
import os
import shutil

SNAPSHOTS_PATH  = "data/history/reddit/post_snapshots.csv"
BACKUP_PATH     = "data/history/reddit/post_snapshots_pre_patch.csv"

SMALL_GAP_MAX   = 3.0   # hours — interpolate below this
LARGE_GAP_MIN   = 3.0   # hours — flag above this


def parse_float(v, default=None):
    try:
        return float(v) if v not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def patch_gaps(rows_by_post):
    """
    For each post, walk through snapshots in time order.
    Patch velocity where gap is small, flag where gap is large.
    Returns flat list of all rows with new columns added.
    """
    patched = []

    for pid, snaps in rows_by_post.items():
        snaps.sort(key=lambda r: r["snapshot_time_utc"])

        for i, row in enumerate(snaps):
            gap = parse_float(row.get("hours_since_previous_snapshot"), default=None)

            # Default new columns
            row["is_collection_gap"]    = 0
            row["velocity_interpolated"] = 0
            row["is_reddit_fuzzing"]    = 0

            if gap is None or gap <= 0:
                patched.append(row)
                continue

            if gap < SMALL_GAP_MAX:
                vel     = parse_float(row.get("upvote_velocity_per_hour"), 0)
                upvotes = parse_float(row.get("upvotes_at_snapshot"), 0)
                prev_up = parse_float(row.get("previous_upvotes_at_snapshot"), None)
                delta   = parse_float(row.get("upvote_delta_from_previous_snapshot"), None)

                if vel == 0 and upvotes and upvotes > 20:
                    if delta is not None and delta > 0:
                        # Delta exists and is positive — recalculate velocity
                        interp_vel = delta / gap
                        row["upvote_velocity_per_hour"] = round(interp_vel, 4)
                        row["velocity_interpolated"]    = 1
                    elif delta == 0 and prev_up is not None and upvotes == prev_up:
                        # Reddit upvote fuzzing — same count shown twice
                        # This is a REAL observation showing zero growth
                        # Leave velocity as 0 — that's accurate
                        # Just note it's a fuzzing case not a missing window
                        row["is_reddit_fuzzing"] = 1

            else:
                # Large gap — flag it, don't trust the velocity
                row["is_collection_gap"] = 1
                # Recalculate velocity from upvote delta / actual gap time
                # so at least the magnitude is right even if timing is uncertain
                delta  = parse_float(row.get("upvote_delta_from_previous_snapshot"), None)
                if delta is not None and gap > 0:
                    interp_vel = delta / gap
                    row["upvote_velocity_per_hour"] = round(interp_vel, 4)
                    row["velocity_interpolated"]    = 1

            patched.append(row)

    return patched


def main():
    if not os.path.exists(SNAPSHOTS_PATH):
        print(f"ERROR: {SNAPSHOTS_PATH} not found.")
        return

    print("Loading post_snapshots.csv...")
    with open(SNAPSHOTS_PATH, encoding="utf-8", errors="replace") as f:
        reader   = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows     = list(reader)
    print(f"  {len(rows):,} rows loaded")

    # Group by post
    by_post = collections.defaultdict(list)
    for row in rows:
        by_post[row["post_id"]].append(row)
    print(f"  {len(by_post):,} unique posts")

    # Check if already patched
    if "is_collection_gap" in fieldnames:
        print("  Already patched — re-patching with latest logic")
        fieldnames = [f for f in fieldnames
                      if f not in ("is_collection_gap", "velocity_interpolated")]

    # Add new columns
    new_fields = fieldnames + ["is_collection_gap", "velocity_interpolated", "is_reddit_fuzzing"]

    # Patch
    print("Patching gaps...")
    patched = patch_gaps(by_post)

    # Count what we did
    flagged      = sum(1 for r in patched if int(r.get("is_collection_gap", 0)))
    interpolated = sum(1 for r in patched if int(r.get("velocity_interpolated", 0)))
    fuzzing      = sum(1 for r in patched if int(r.get("is_reddit_fuzzing", 0)))
    print(f"  Flagged as collection gap:    {flagged:,} rows ({100*flagged/len(patched):.1f}%)")
    print(f"  Velocity interpolated:        {interpolated:,} rows ({100*interpolated/len(patched):.1f}%)")
    print(f"  Reddit fuzzing (real zero):   {fuzzing:,} rows ({100*fuzzing/len(patched):.1f}%) — velocity=0 is correct")

    # Backup original
    print(f"Backing up original to {BACKUP_PATH}...")
    shutil.copy2(SNAPSHOTS_PATH, BACKUP_PATH)

    # Write patched version
    print(f"Writing patched {SNAPSHOTS_PATH}...")
    with open(SNAPSHOTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(patched)

    print(f"Done. {len(patched):,} rows written.")
    print()
    print("Next steps:")
    print("  - Re-run build_tracking_pools.py  (uses cleaned velocities)")
    print("  - Re-run detect_flow_deviation.py  (uses cleaned states)")
    print("  - Re-run predict_post_flow.py       (uses cleaned transitions)")


if __name__ == "__main__":
    main()
