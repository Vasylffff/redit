from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_TABLE_FILES = (
    "post_snapshots.csv",
    "comment_snapshots.csv",
    "subreddit_snapshots.csv",
    "subreddit_health_trend.csv",
    "subreddit_health_latest.csv",
    "snapshot_catalog.csv",
    "activity_thresholds.csv",
    "post_lifecycles.csv",
    "top_posts.csv",
    "latest_post_status.csv",
    "analysis_focus_latest.csv",
    "current_attention_leaderboard.csv",
    "general_popularity_leaderboard.csv",
    "naive_next_hour_forecast_latest.csv",
    "naive_forecast_leaderboard.csv",
    "naive_forecast_watchlist_by_subreddit.csv",
    "naive_forecast_evaluation_overall.csv",
    "naive_forecast_evaluation_by_subreddit.csv",
    "post_case_studies_latest.csv",
    "subreddit_attention_latest.csv",
    "cooling_posts_latest.csv",
    "dying_posts_latest.csv",
    "dead_posts_latest.csv",
    "top_risers_latest.csv",
    "tracking_candidates_latest.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the merged Reddit history CSVs into a SQLite database for easier "
            "querying and downstream analysis."
        )
    )
    parser.add_argument(
        "--history-dir",
        default="data/history/reddit",
        help="Directory containing the history CSV files.",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models/reddit",
        help="Directory containing model-ready CSV files to include when available.",
    )
    parser.add_argument(
        "--output",
        default="data/history/reddit/history.db",
        help="Path for the SQLite database file.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def infer_sqlite_type(values: list[str]) -> str:
    cleaned = [value for value in values if value not in ("", None)]
    if not cleaned:
        return "TEXT"

    int_like = True
    float_like = True
    for value in cleaned:
        text = str(value)
        try:
            int(text)
        except ValueError:
            int_like = False
        try:
            float(text)
        except ValueError:
            float_like = False
        if not int_like and not float_like:
            return "TEXT"

    if int_like:
        return "INTEGER"
    if float_like:
        return "REAL"
    return "TEXT"


def sqlite_table_name(csv_name: str) -> str:
    return csv_name.replace(".csv", "")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_csv_to_table(connection: sqlite3.Connection, table_name: str, csv_path: Path) -> int:
    fieldnames, rows = load_rows(csv_path)
    connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    if not fieldnames:
        connection.execute(f'CREATE TABLE "{table_name}" (_empty TEXT)')
        return 0

    column_types = {
        field: infer_sqlite_type([row.get(field, "") for row in rows])
        for field in fieldnames
    }
    column_sql = ", ".join(f'"{field}" {column_types[field]}' for field in fieldnames)
    connection.execute(f'CREATE TABLE "{table_name}" ({column_sql})')

    if not rows:
        return 0

    placeholders = ", ".join("?" for _ in fieldnames)
    columns = ", ".join(f'"{field}"' for field in fieldnames)
    insert_sql = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'
    payload = [[row.get(field, "") for field in fieldnames] for row in rows]
    connection.executemany(insert_sql, payload)
    return len(rows)


def create_indexes(connection: sqlite3.Connection) -> None:
    statements = [
        'CREATE INDEX IF NOT EXISTS idx_post_snapshots_post_id ON post_snapshots (post_id)',
        'CREATE INDEX IF NOT EXISTS idx_post_snapshots_snapshot_time ON post_snapshots (snapshot_time_utc)',
        'CREATE INDEX IF NOT EXISTS idx_post_snapshots_story_domain ON post_snapshots (subreddit, listing_type, snapshot_time_utc)',
        'CREATE INDEX IF NOT EXISTS idx_comment_snapshots_post_time ON comment_snapshots (subreddit, post_id, snapshot_time_utc)',
        'CREATE INDEX IF NOT EXISTS idx_comment_snapshots_comment_id ON comment_snapshots (comment_id)',
        'CREATE INDEX IF NOT EXISTS idx_post_lifecycles_post_id ON post_lifecycles (post_id)',
        'CREATE INDEX IF NOT EXISTS idx_post_lifecycles_subreddit_state ON post_lifecycles (subreddit, latest_activity_state)',
        'CREATE INDEX IF NOT EXISTS idx_analysis_focus_post_id ON analysis_focus_latest (post_id)',
        'CREATE INDEX IF NOT EXISTS idx_current_attention_rank ON current_attention_leaderboard (current_attention_rank_overall)',
        'CREATE INDEX IF NOT EXISTS idx_general_popularity_rank ON general_popularity_leaderboard (general_popularity_rank_overall)',
        'CREATE INDEX IF NOT EXISTS idx_naive_forecast_rank ON naive_forecast_leaderboard (forecast_rank_overall)',
        'CREATE INDEX IF NOT EXISTS idx_naive_watchlist_sub ON naive_forecast_watchlist_by_subreddit (subreddit, watchlist_rank_in_subreddit)',
        'CREATE INDEX IF NOT EXISTS idx_naive_eval_sub ON naive_forecast_evaluation_by_subreddit (segment)',
        'CREATE INDEX IF NOT EXISTS idx_case_studies_rank ON post_case_studies_latest (case_rank)',
        'CREATE INDEX IF NOT EXISTS idx_subreddit_attention_latest_sub ON subreddit_attention_latest (subreddit)',
        'CREATE INDEX IF NOT EXISTS idx_tracking_candidates_post_id ON tracking_candidates_latest (post_id)',
        'CREATE INDEX IF NOT EXISTS idx_post_timeline_points_post_time ON post_timeline_points (subreddit, post_id, snapshot_time_utc)',
        'CREATE INDEX IF NOT EXISTS idx_prediction_next_hour_post_id ON prediction_next_hour (post_id)',
        'CREATE INDEX IF NOT EXISTS idx_prediction_sequences_post_id ON prediction_sequences (post_id)',
        'CREATE INDEX IF NOT EXISTS idx_prediction_sequences_first_seen ON prediction_sequences (subreddit, first_seen_snapshot_time_utc)',
        'CREATE INDEX IF NOT EXISTS idx_subreddit_health_trend_sub ON subreddit_health_trend (subreddit, snapshot_time_utc)',
        'CREATE INDEX IF NOT EXISTS idx_subreddit_health_latest_sub ON subreddit_health_latest (subreddit)',
        'CREATE INDEX IF NOT EXISTS idx_subreddit_health_latest_score ON subreddit_health_latest (health_score)',
    ]
    for statement in statements:
        try:
            connection.execute(statement)
        except sqlite3.OperationalError:
            continue


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    ensure_parent(output_path)

    csv_paths: list[Path] = []
    for name in DEFAULT_TABLE_FILES:
        path = history_dir / name
        if path.is_file():
            csv_paths.append(path)
    for name in (
        "prediction_all_snapshots.csv",
        "prediction_next_hour.csv",
        "prediction_sequences.csv",
        "post_timeline_points.csv",
    ):
        path = model_dir / name
        if path.is_file():
            csv_paths.append(path)

    if not csv_paths:
        raise SystemExit("No CSV files were found to export.")

    connection = sqlite3.connect(output_path)
    try:
        exported_counts: dict[str, int] = {}
        for csv_path in csv_paths:
            table_name = sqlite_table_name(csv_path.name)
            exported_counts[table_name] = export_csv_to_table(connection, table_name, csv_path)
        create_indexes(connection)
        connection.commit()
    finally:
        connection.close()

    print(f"SQLite database written to {output_path}")
    for table_name, row_count in exported_counts.items():
        print(f"- {table_name}: {row_count} row(s)")


if __name__ == "__main__":
    main()
