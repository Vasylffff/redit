from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from normalize_reddit_json import clean_text, parse_datetime


DEFAULT_EXCLUDED_SUBREDDITS = ("pasta",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the merged Reddit history tables and write a machine-readable "
            "quality report."
        )
    )
    parser.add_argument(
        "--history-dir",
        default="data/history/reddit",
        help="Directory containing the merged history CSV files.",
    )
    parser.add_argument(
        "--output",
        default="data/history/reddit/validation_report.json",
        help="Path for the validation report JSON file.",
    )
    parser.add_argument(
        "--exclude-subreddits",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_SUBREDDITS),
        help="Subreddits that should not appear in the main history dataset.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_names(values: list[str]) -> set[str]:
    return {clean_text(value).lower().removeprefix("r/") for value in values if clean_text(value)}


def add_issue(issues: list[dict[str, Any]], *, severity: str, code: str, message: str, details: dict[str, Any] | None = None) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "details": details or {},
        }
    )


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir)
    excluded_subreddits = normalize_names(args.exclude_subreddits)

    post_snapshots = load_csv(history_dir / "post_snapshots.csv")
    post_lifecycles = load_csv(history_dir / "post_lifecycles.csv")
    analysis_focus = load_csv(history_dir / "analysis_focus_latest.csv")
    tracking_candidates = load_csv(history_dir / "tracking_candidates_latest.csv")
    activity_thresholds = load_csv(history_dir / "activity_thresholds.csv")

    issues: list[dict[str, Any]] = []

    duplicate_counter = Counter(
        (clean_text(row.get("snapshot_id")), clean_text(row.get("post_id")))
        for row in post_snapshots
        if clean_text(row.get("snapshot_id")) and clean_text(row.get("post_id"))
    )
    duplicates = [key for key, count in duplicate_counter.items() if count > 1]
    if duplicates:
        add_issue(
            issues,
            severity="error",
            code="duplicate_snapshot_post_keys",
            message="Duplicate (snapshot_id, post_id) combinations were found in post_snapshots.csv.",
            details={"count": len(duplicates), "examples": duplicates[:5]},
        )

    bad_time_order = 0
    negative_values = 0
    missing_core_fields = 0
    observed_subreddits: Counter[str] = Counter()
    excluded_hits: Counter[str] = Counter()
    for row in post_snapshots:
        subreddit = clean_text(row.get("subreddit")).lower()
        if subreddit:
            observed_subreddits[subreddit] += 1
        if subreddit in excluded_subreddits:
            excluded_hits[subreddit] += 1

        if not clean_text(row.get("snapshot_id")) or not clean_text(row.get("post_id")) or not subreddit:
            missing_core_fields += 1

        created_at = parse_datetime(row.get("created_at"))
        snapshot_time = parse_datetime(row.get("snapshot_time_utc"))
        if created_at and snapshot_time and created_at > snapshot_time:
            bad_time_order += 1

        for field in (
            "age_minutes_at_snapshot",
            "upvotes_at_snapshot",
            "comment_count_at_snapshot",
            "upvote_delta_from_previous_snapshot",
            "comment_delta_from_previous_snapshot",
            "upvote_delta_to_next_snapshot",
            "comment_delta_to_next_snapshot",
        ):
            value = parse_float(row.get(field))
            if value is not None and field in {"age_minutes_at_snapshot", "upvotes_at_snapshot", "comment_count_at_snapshot"} and value < 0:
                negative_values += 1

    if missing_core_fields:
        add_issue(
            issues,
            severity="error",
            code="missing_core_fields",
            message="Some post snapshot rows are missing snapshot_id, post_id, or subreddit.",
            details={"row_count": missing_core_fields},
        )
    if bad_time_order:
        add_issue(
            issues,
            severity="error",
            code="invalid_time_order",
            message="Some posts appear to have been created after their snapshot time.",
            details={"row_count": bad_time_order},
        )
    if negative_values:
        add_issue(
            issues,
            severity="error",
            code="negative_core_metrics",
            message="Negative ages or engagement counts were found in post_snapshots.csv.",
            details={"row_count": negative_values},
        )
    if excluded_hits:
        add_issue(
            issues,
            severity="warning",
            code="excluded_subreddits_present",
            message="Excluded/noise subreddits are still present in the history.",
            details={"counts": dict(excluded_hits)},
        )

    lifecycle_state_counts = Counter(clean_text(row.get("latest_activity_state")) for row in post_lifecycles)
    unknown_share = lifecycle_state_counts.get("unknown", 0) / len(post_lifecycles) if post_lifecycles else 0.0
    if unknown_share > 0.65:
        add_issue(
            issues,
            severity="warning",
            code="high_unknown_state_share",
            message="A high share of post lifecycles still have unknown latest state.",
            details={"unknown_share": unknown_share, "unknown_count": lifecycle_state_counts.get("unknown", 0)},
        )

    threshold_rows = {clean_text(row.get("subreddit")).lower(): row for row in activity_thresholds}
    for subreddit in sorted(observed_subreddits):
        if subreddit not in threshold_rows:
            add_issue(
                issues,
                severity="warning",
                code="missing_threshold_row",
                message=f"No threshold row was written for subreddit '{subreddit}'.",
            )
            continue
        source = clean_text(threshold_rows[subreddit].get("threshold_source"))
        if source == "global_fallback":
            add_issue(
                issues,
                severity="warning",
                code="fallback_thresholds_in_use",
                message=f"Subreddit '{subreddit}' is still using global fallback thresholds.",
                details={"subreddit": subreddit},
            )

    focus_post_ids = {clean_text(row.get("post_id")) for row in analysis_focus if clean_text(row.get("post_id"))}
    orphan_tracking = [
        clean_text(row.get("post_id"))
        for row in tracking_candidates
        if clean_text(row.get("post_id")) and clean_text(row.get("post_id")) not in focus_post_ids
    ]
    if orphan_tracking:
        add_issue(
            issues,
            severity="warning",
            code="tracking_candidates_outside_focus",
            message="Some tracking candidates do not appear in the latest analysis focus file.",
            details={"count": len(orphan_tracking), "examples": orphan_tracking[:5]},
        )

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "history_dir": str(history_dir),
        "summary": {
            "post_snapshot_rows": len(post_snapshots),
            "post_lifecycle_rows": len(post_lifecycles),
            "analysis_focus_rows": len(analysis_focus),
            "tracking_candidate_rows": len(tracking_candidates),
            "subreddits": dict(sorted(observed_subreddits.items())),
            "latest_state_counts": dict(sorted(lifecycle_state_counts.items())),
        },
        "issues": issues,
        "status": "ok" if not any(issue["severity"] == "error" for issue in issues) else "error",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation report written to {output_path}")
    print(f"Status: {report['status']}")
    print(f"Issues found: {len(issues)}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
