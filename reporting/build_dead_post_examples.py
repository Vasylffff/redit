from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_visual_report import (
    FONT_LABEL,
    FONT_SUBTITLE,
    FONT_TITLE,
    Image,
    ImageDraw,
    BG,
    ascii_label,
    draw_line_panel,
    draw_text,
    draw_wrapped_text,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build example timeline charts for dead posts in selected subreddits."
    )
    parser.add_argument("--history-dir", default="data/history/reddit")
    parser.add_argument("--models-dir", default="data/models/reddit")
    parser.add_argument("--output-dir", default="data/analysis/reddit/visuals/dead_examples")
    parser.add_argument("--subreddits", nargs="*", default=["politics", "worldnews"])
    parser.add_argument("--min-snapshots", type=int, default=12)
    return parser.parse_args()


def choose_dead_examples(
    latest_status: pd.DataFrame,
    *,
    subreddits: list[str],
    min_snapshots: int,
) -> pd.DataFrame:
    frame = latest_status.copy()
    frame["snapshot_count"] = pd.to_numeric(frame["snapshot_count"], errors="coerce")
    frame["observed_hours"] = pd.to_numeric(frame["observed_hours"], errors="coerce")
    frame["current_attention_score"] = pd.to_numeric(frame["current_attention_score"], errors="coerce")
    frame["latest_activity_state"] = frame["latest_activity_state"].astype(str)
    frame = frame[frame["subreddit"].isin(subreddits)].copy()
    frame = frame[frame["latest_activity_state"].str.lower() == "dead"].copy()
    frame = frame[frame["snapshot_count"].fillna(0) >= min_snapshots].copy()
    frame = frame.sort_values(
        by=["subreddit", "snapshot_count", "observed_hours", "current_attention_score"],
        ascending=[True, False, False, False],
    )
    return frame.groupby("subreddit", as_index=False).head(1).copy()


def build_dead_timeline_chart(post: pd.Series, timeline: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1400, 940), BG)
    draw = ImageDraw.Draw(image)

    draw_text(draw, (30, 20), "Dead Post Example", FONT_TITLE)
    draw_text(draw, (30, 58), ascii_label(str(post["subreddit"]), max_length=20), FONT_SUBTITLE)
    draw_wrapped_text(draw, (30, 88), ascii_label(str(post["title"]), max_length=140), FONT_LABEL, max_width=1320, line_height=22)

    latest_state = str(post.get("latest_activity_state", "")).lower()
    snapshot_count = int(float(post.get("snapshot_count") or 0))
    observed_hours = float(post.get("observed_hours") or 0.0)
    subtitle = (
        f"Latest state: {latest_state or 'unknown'} | "
        f"Snapshots: {snapshot_count} | "
        f"Observed hours: {observed_hours:.1f}"
    )
    draw_text(draw, (30, 138), subtitle, FONT_LABEL)

    boxes = [
        (30, 190, 685, 540),
        (715, 190, 1370, 540),
        (30, 560, 685, 910),
        (715, 560, 1370, 910),
    ]
    labels = [str(i + 1) for i in range(len(timeline))]

    draw_line_panel(draw, boxes[0], timeline["upvotes_at_snapshot"].fillna(0.0).tolist(), "#1f77b4", "Upvotes", labels)
    draw_line_panel(draw, boxes[1], timeline["comment_count_at_snapshot"].fillna(0.0).tolist(), "#ff7f0e", "Comments", labels)
    draw_line_panel(draw, boxes[2], timeline["upvote_velocity_per_hour"].fillna(0.0).tolist(), "#2ca02c", "Upvote Velocity / Hour", labels)
    draw_line_panel(draw, boxes[3], timeline["comment_velocity_per_hour"].fillna(0.0).tolist(), "#d62728", "Comment Velocity / Hour", labels)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_status = pd.read_csv(history_dir / "latest_post_status.csv")
    timeline = pd.read_csv(
        models_dir / "post_timeline_points.csv",
        usecols=[
            "subreddit",
            "post_id",
            "title",
            "snapshot_time_utc",
            "sequence_position",
            "upvotes_at_snapshot",
            "comment_count_at_snapshot",
            "upvote_velocity_per_hour",
            "comment_velocity_per_hour",
            "activity_state",
            "listing_type",
        ],
    )

    selected = choose_dead_examples(
        latest_status,
        subreddits=args.subreddits,
        min_snapshots=args.min_snapshots,
    )
    if selected.empty:
        raise SystemExit("No dead-post examples matched the current filters.")

    summary_lines = ["# Dead Post Examples", ""]
    for _, post in selected.iterrows():
        post_timeline = timeline[
            (timeline["subreddit"] == post["subreddit"]) & (timeline["post_id"] == post["post_id"])
        ].sort_values("sequence_position").copy()
        if post_timeline.empty:
            continue
        chart_path = output_dir / f"dead_post_timeline_{slugify(str(post['subreddit']))}_{slugify(str(post['post_id']))}.png"
        build_dead_timeline_chart(post, post_timeline, chart_path)
        summary_lines.extend(
            [
                f"## r/{post['subreddit']}",
                "",
                f"- Post ID: `{post['post_id']}`",
                f"- Title: {ascii_label(str(post['title']), max_length=180)}",
                f"- Latest state: `{post['latest_activity_state']}`",
                f"- Snapshot count: `{int(float(post['snapshot_count']) if pd.notna(post['snapshot_count']) else 0)}`",
                f"- Observed hours: `{float(post['observed_hours']) if pd.notna(post['observed_hours']) else 0.0:.1f}`",
                f"- Chart: `{chart_path.name}`",
                "",
            ]
        )
        print(f"Saved dead post chart -> {chart_path}")

    summary_path = output_dir / "dead_post_examples_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
