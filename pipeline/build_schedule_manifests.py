from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize_csv_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_rows.append(
            {
                (key.lstrip("\ufeff").strip() if key else key): value
                for key, value in row.items()
            }
        )
    return normalized_rows


SCHEDULE_DEFINITIONS = (
    {
        "schedule_name": "hourly_new",
        "listing": "new",
        "top_time": "",
        "run_hours": "*",
        "cadence_label": "hourly",
        "description": "Capture new posts entering each subreddit every hour.",
    },
    {
        "schedule_name": "two_hour_rising",
        "listing": "rising",
        "top_time": "",
        "run_hours": "0,2,4,6,8,10,12,14,16,18,20,22",
        "cadence_label": "every_2_hours",
        "description": "Capture posts that are starting to gain momentum every two hours.",
    },
    {
        "schedule_name": "four_hour_hot",
        "listing": "hot",
        "top_time": "",
        "run_hours": "0,4,8,12,16,20",
        "cadence_label": "every_4_hours",
        "description": "Capture actively popular posts every four hours.",
    },
    {
        "schedule_name": "twice_daily_top_day",
        "listing": "top",
        "top_time": "day",
        "run_hours": "0,12",
        "cadence_label": "twice_daily",
        "description": "Capture the strongest daily posts twice per day.",
    },
    {
        "schedule_name": "daily_top_week",
        "listing": "top",
        "top_time": "week",
        "run_hours": "0",
        "cadence_label": "daily",
        "description": "Capture the strongest weekly posts once per day.",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the broad discovery manifest into cadence-specific schedule manifests "
            "such as hourly new, two-hour rising, and daily top-week."
        )
    )
    parser.add_argument(
        "--source-manifest",
        default="configs/discovery_batch_manifest.csv",
        help="Source discovery manifest to split into scheduled subsets.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/schedules",
        help="Directory where scheduled manifest CSV files will be written.",
    )
    parser.add_argument(
        "--schedule-plan",
        default="configs/schedules/schedule_plan.csv",
        help="CSV file describing the generated schedule manifests and run hours.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Source manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = normalize_csv_rows(list(reader))
    if not rows:
        raise SystemExit("Source manifest is empty.")
    return rows


def select_rows(
    rows: list[dict[str, str]],
    *,
    listing: str,
    top_time: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("listing", "") == listing and row.get("top_time", "") == top_time
    ]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_manifest = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(source_manifest)

    schedule_rows: list[dict[str, str]] = []

    for definition in SCHEDULE_DEFINITIONS:
        matched_rows = select_rows(
            rows,
            listing=definition["listing"],
            top_time=definition["top_time"],
        )
        if not matched_rows:
            raise SystemExit(
                "No rows matched "
                f"{definition['schedule_name']} in {source_manifest}"
            )

        manifest_path = output_dir / f"{definition['schedule_name']}.csv"
        write_manifest(manifest_path, matched_rows)

        schedule_rows.append(
            {
                "schedule_name": definition["schedule_name"],
                "cadence_label": definition["cadence_label"],
                "run_hours": definition["run_hours"],
                "manifest_path": str(manifest_path),
                "job_count": str(len(matched_rows)),
                "subreddit_count": str(
                    len({row.get("subreddit", "") for row in matched_rows})
                ),
                "description": definition["description"],
            }
        )

    schedule_plan_path = Path(args.schedule_plan)
    schedule_plan_path.parent.mkdir(parents=True, exist_ok=True)
    with schedule_plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(schedule_rows[0].keys()))
        writer.writeheader()
        writer.writerows(schedule_rows)

    print(f"Wrote {len(schedule_rows)} schedule manifests to {output_dir}")
    print(f"Schedule plan written to {schedule_plan_path}")


if __name__ == "__main__":
    main()
