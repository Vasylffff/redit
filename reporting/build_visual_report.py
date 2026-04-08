from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a visual report with one post timeline and subreddit overview charts."
    )
    parser.add_argument("--history-dir", default="data/history/reddit")
    parser.add_argument("--models-dir", default="data/models/reddit")
    parser.add_argument("--output-dir", default="data/analysis/reddit/visuals")
    parser.add_argument("--min-snapshots", type=int, default=8)
    return parser.parse_args()


def slugify(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-") or "chart"


def ascii_label(value: str, max_length: int = 110) -> str:
    text = (value or "").encode("ascii", "ignore").decode("ascii").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def choose_example_post(leaderboard: pd.DataFrame, min_snapshots: int) -> pd.Series:
    candidates = leaderboard[leaderboard["snapshot_count"].fillna(0) >= min_snapshots].copy()
    if candidates.empty:
        candidates = leaderboard.copy()
    candidates = candidates.sort_values(
        by=["current_attention_rank_overall", "snapshot_count", "current_attention_score"],
        ascending=[True, False, False],
    )
    return candidates.iloc[0]


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT_TITLE = get_font(28)
FONT_SUBTITLE = get_font(20)
FONT_LABEL = get_font(16)
FONT_SMALL = get_font(14)

BG = "#ffffff"
TEXT = "#1a1a1a"
GRID = "#d9d9d9"
AXIS = "#333333"
SERIES_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")
STATE_ORDER = ["surging", "alive", "cooling", "dying", "dead"]
STATE_COLORS = {
    "surging": "#1f77b4",
    "alive": "#17becf",
    "cooling": "#ffbf00",
    "dying": "#ff7f0e",
    "dead": "#d62728",
}
AGE_BUCKETS = [
    ("under_30m", "<30m"),
    ("30m_to_1h", "30m-1h"),
    ("1h_to_3h", "1-3h"),
    ("3h_to_6h", "3-6h"),
    ("6h_to_12h", "6-12h"),
    ("12h_to_24h", "12-24h"),
    ("over_24h", ">24h"),
]


def topic_label(value: str) -> str:
    return ascii_label(value.replace("_", " "), max_length=28)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str = TEXT) -> None:
    draw.text(xy, text, fill=fill, font=font)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    line_height: int,
    fill: str = TEXT,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for index, line in enumerate(lines):
        draw.text((x, y + index * line_height), line, fill=fill, font=font)
    return len(lines) * line_height


def chart_bounds(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return left + 55, top + 30, right - 20, bottom - 38


def draw_panel_title(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    left, top, _, _ = box
    draw.text((left + 8, top + 6), title, fill=TEXT, font=FONT_SUBTITLE)


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = chart_bounds(box)
    draw.rectangle(box, outline="#bfbfbf", width=1)
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)
    return left, top, right, bottom


def normalize(value: float, low: float, high: float) -> float:
    if math.isclose(high, low):
        return 0.5
    return (value - low) / (high - low)


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    color: str,
    title: str,
    x_labels: list[str] | None = None,
) -> None:
    draw_panel_title(draw, box, title)
    left, top, right, bottom = draw_axes(draw, box)
    if not values:
        draw.text((left + 10, top + 10), "No data", fill=TEXT, font=FONT_LABEL)
        return

    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        y_min = min(0.0, y_min)
        y_max = y_max + 1.0

    for fraction in (0.25, 0.5, 0.75):
        y = bottom - int((bottom - top) * fraction)
        draw.line((left, y, right, y), fill=GRID, width=1)

    n = len(values)
    usable_width = max(1, right - left - 10)
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        x = left + 5 if n == 1 else left + 5 + int(index * usable_width / (n - 1))
        y = bottom - int(normalize(value, y_min, y_max) * (bottom - top - 6)) - 3
        points.append((x, y))

    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=color)

    draw.text((left - 2, top - 2), f"{y_max:.0f}", fill=TEXT, font=FONT_SMALL)
    draw.text((left - 2, bottom - 18), f"{y_min:.0f}", fill=TEXT, font=FONT_SMALL)

    if x_labels:
        step = max(1, len(x_labels) // 4)
        for index in range(0, len(x_labels), step):
            if index >= len(points):
                continue
            x, _ = points[index]
            label = x_labels[index]
            draw.text((x - 10, bottom + 6), label, fill=TEXT, font=FONT_SMALL)


def draw_multi_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    frame: pd.DataFrame,
    title: str,
    value_column: str,
    x_column: str,
    series_column: str,
    series_order: list[str] | None = None,
    series_color_map: dict[str, str] | None = None,
) -> None:
    draw_panel_title(draw, box, title)
    left, top, right, bottom = draw_axes(draw, box)
    if frame.empty:
        draw.text((left + 10, top + 10), "No data", fill=TEXT, font=FONT_LABEL)
        return

    ordered_times = list(dict.fromkeys(frame[x_column].tolist()))
    if not ordered_times:
        draw.text((left + 10, top + 10), "No data", fill=TEXT, font=FONT_LABEL)
        return

    values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    if values.empty:
        draw.text((left + 10, top + 10), "No data", fill=TEXT, font=FONT_LABEL)
        return

    plot_right = right - 120
    y_min = float(values.min())
    y_max = float(values.max())
    if math.isclose(y_min, y_max):
        y_min = min(0.0, y_min)
        y_max = y_max + 1.0

    for fraction in (0.25, 0.5, 0.75):
        y = bottom - int((bottom - top) * fraction)
        draw.line((left, y, plot_right, y), fill=GRID, width=1)

    usable_width = max(1, plot_right - left - 20)
    x_positions = {
        value: (left + 10 if len(ordered_times) == 1 else left + 10 + int(index * usable_width / (len(ordered_times) - 1)))
        for index, value in enumerate(ordered_times)
    }

    unique_series = list(dict.fromkeys(frame[series_column].tolist()))
    if series_order:
        present = set(unique_series)
        ordered_series = [series_name for series_name in series_order if series_name in present]
        ordered_series.extend(series_name for series_name in unique_series if series_name not in ordered_series)
    else:
        ordered_series = unique_series

    legend_left = plot_right + 10
    legend_top = top + 8
    draw.rectangle(
        (legend_left - 6, legend_top - 4, right - 4, legend_top + 14 * len(ordered_series) + 8),
        fill="#ffffff",
    )

    for series_index, series_name in enumerate(ordered_series):
        series_rows = frame[frame[series_column] == series_name]
        color = (
            series_color_map.get(str(series_name), SERIES_COLORS[series_index % len(SERIES_COLORS)])
            if series_color_map
            else SERIES_COLORS[series_index % len(SERIES_COLORS)]
        )
        points: list[tuple[int, int]] = []
        for _, row in series_rows.iterrows():
            raw_value = row.get(value_column)
            if pd.isna(raw_value):
                continue
            x_value = row.get(x_column)
            if x_value not in x_positions:
                continue
            x = x_positions[x_value]
            y = bottom - int(normalize(float(raw_value), y_min, y_max) * (bottom - top - 6)) - 3
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        elif len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=color)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=color)

        legend_x = legend_left
        legend_y = legend_top + (series_index * 16)
        draw.text((legend_x, legend_y), ascii_label(str(series_name), max_length=12), fill=color, font=FONT_SMALL)

    draw.text((left - 2, top - 2), f"{y_max:.0f}", fill=TEXT, font=FONT_SMALL)
    draw.text((left - 2, bottom - 18), f"{y_min:.0f}", fill=TEXT, font=FONT_SMALL)

    label_step = max(1, len(ordered_times) // 4)
    for index in range(0, len(ordered_times), label_step):
        x = x_positions[ordered_times[index]]
        label = str(ordered_times[index])
        draw.text((x - 10, bottom + 6), label, fill=TEXT, font=FONT_SMALL)


def draw_stacked_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    frame: pd.DataFrame,
) -> None:
    draw_panel_title(draw, box, "Subreddit State Mix")
    left, top, right, bottom = draw_axes(draw, box)
    ordered = frame.sort_values("total_current_attention_score", ascending=False).copy()
    ordered["alive_share"] = ordered["alive_or_surging_count"] / ordered["post_count"]
    ordered["cooling_share"] = ordered["cooling_count"] / ordered["post_count"]
    ordered["dead_share"] = ordered["dead_count"] / ordered["post_count"]

    colors = {
        "alive_share": "#2ca02c",
        "cooling_share": "#ffbf00",
        "dead_share": "#d62728",
    }
    count = len(ordered)
    bar_width = max(30, int((right - left) / max(1, count * 1.8)))
    spacing = max(12, int((right - left - count * bar_width) / max(1, count + 1)))
    current_x = left + spacing

    for _, row in ordered.iterrows():
        y_cursor = bottom
        for field in ("alive_share", "cooling_share", "dead_share"):
            bar_height = int((bottom - top) * float(row[field]))
            draw.rectangle(
                (current_x, y_cursor - bar_height, current_x + bar_width, y_cursor),
                fill=colors[field],
                outline=colors[field],
            )
            y_cursor -= bar_height
        label = ascii_label(str(row["subreddit"]), max_length=12)
        draw.text((current_x, bottom + 8), label, fill=TEXT, font=FONT_SMALL)
        current_x += bar_width + spacing

    draw.text((right - 130, top + 5), "green = alive", fill="#2ca02c", font=FONT_SMALL)
    draw.text((right - 130, top + 22), "yellow = cooling", fill="#b8860b", font=FONT_SMALL)
    draw.text((right - 130, top + 39), "red = dead", fill="#d62728", font=FONT_SMALL)


def draw_scatter_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    frame: pd.DataFrame,
) -> None:
    draw_panel_title(draw, box, "Subreddit Attention vs Popularity")
    left, top, right, bottom = draw_axes(draw, box)
    x_min = float(frame["avg_general_popularity_score"].min())
    x_max = float(frame["avg_general_popularity_score"].max())
    y_min = float(frame["avg_current_attention_score"].min())
    y_max = float(frame["avg_current_attention_score"].max())

    for _, row in frame.iterrows():
        x_norm = normalize(float(row["avg_general_popularity_score"]), x_min, x_max)
        y_norm = normalize(float(row["avg_current_attention_score"]), y_min, y_max)
        x = left + 10 + int(x_norm * (right - left - 20))
        y = bottom - 10 - int(y_norm * (bottom - top - 20))
        radius = max(8, int(math.sqrt(max(1.0, float(row["post_count"]))) * 1.2))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#7db7ff", outline="#1f4e79", width=2)
        draw.text((x + radius + 4, y - 6), ascii_label(str(row["subreddit"]), max_length=12), fill=TEXT, font=FONT_SMALL)

    draw.text((left, top - 18), f"Popularity {x_min:.1f} to {x_max:.1f}", fill=TEXT, font=FONT_SMALL)
    draw.text((left, bottom + 8), f"Attention {y_min:.2f} to {y_max:.2f}", fill=TEXT, font=FONT_SMALL)


def build_timeline_chart(post: pd.Series, timeline: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1400, 900), BG)
    draw = ImageDraw.Draw(image)

    draw_text(draw, (30, 20), "Example Post Timeline", FONT_TITLE)
    draw_text(draw, (30, 58), ascii_label(str(post["subreddit"]), max_length=20), FONT_SUBTITLE)
    draw_wrapped_text(draw, (30, 88), ascii_label(str(post["title"]), max_length=140), FONT_LABEL, max_width=1320, line_height=22)

    boxes = [
        (30, 170, 685, 520),
        (715, 170, 1370, 520),
        (30, 540, 685, 870),
        (715, 540, 1370, 870),
    ]
    labels = [str(i + 1) for i in range(len(timeline))]

    draw_line_panel(draw, boxes[0], timeline["upvotes_at_snapshot"].fillna(0.0).tolist(), "#1f77b4", "Upvotes", labels)
    draw_line_panel(draw, boxes[1], timeline["comment_count_at_snapshot"].fillna(0.0).tolist(), "#ff7f0e", "Comments", labels)
    draw_line_panel(draw, boxes[2], timeline["upvote_velocity_per_hour"].fillna(0.0).tolist(), "#2ca02c", "Upvote Velocity / Hour", labels)
    draw_line_panel(draw, boxes[3], timeline["comment_velocity_per_hour"].fillna(0.0).tolist(), "#d62728", "Comment Velocity / Hour", labels)

    image.save(output_path)


def build_subreddit_state_mix_chart(frame: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1200, 720), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (30, 20), "Subreddit Overview", FONT_TITLE)
    draw_text(draw, (30, 58), "How much of each subreddit is alive, cooling, or dead right now", FONT_SUBTITLE)
    draw_stacked_bar_chart(draw, (30, 110, 1170, 680), frame)
    image.save(output_path)


def build_subreddit_scatter_chart(frame: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1200, 720), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (30, 20), "Subreddit Attention vs Popularity", FONT_TITLE)
    draw_text(draw, (30, 58), "Bubble size reflects how many posts we have for that subreddit", FONT_SUBTITLE)
    draw_scatter_chart(draw, (30, 110, 1170, 680), frame)
    image.save(output_path)


def build_subreddit_trend_chart(frame: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1400, 900), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (30, 20), "Subreddit Trend Overview", FONT_TITLE)
    draw_text(draw, (30, 58), "Recent hourly-new 100-post samples across all tracked subreddits", FONT_SUBTITLE)

    boxes = [
        (30, 110, 685, 460),
        (715, 110, 1370, 460),
        (30, 500, 685, 850),
        (715, 500, 1370, 850),
    ]

    draw_multi_line_panel(draw, boxes[0], frame, "Total Upvotes In Snapshot", "total_upvotes_in_snapshot", "time_label", "subreddit")
    draw_multi_line_panel(draw, boxes[1], frame, "Total Comments In Snapshot", "total_comments_in_snapshot", "time_label", "subreddit")
    draw_multi_line_panel(draw, boxes[2], frame, "Persisting Posts Share %", "persisting_posts_share_pct", "time_label", "subreddit")
    draw_multi_line_panel(draw, boxes[3], frame, "New Posts Since Previous", "new_post_count_since_previous_snapshot", "time_label", "subreddit")
    image.save(output_path)


def normalize_distribution(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {state: 1.0 / len(STATE_ORDER) for state in STATE_ORDER}
    return {state: counts.get(state, 0.0) / total for state in STATE_ORDER}


def classify_deviation_kind(surge_ratio: float, alive_ratio: float, dead_ratio: float, threshold: float = 1.4) -> tuple[str, float]:
    if surge_ratio >= threshold * 1.5:
        return "SURGE SPIKE", surge_ratio
    if surge_ratio >= threshold:
        return "elevated activity", surge_ratio
    if alive_ratio <= 1 / threshold:
        return "unusually quiet", 1 / max(alive_ratio, 0.01)
    if dead_ratio >= threshold * 1.5:
        return "mass die-off", dead_ratio
    return "normal", 1.0


def prepare_state_rows(post_snapshots: pd.DataFrame, prediction_rows: pd.DataFrame) -> pd.DataFrame:
    topic_map = (
        prediction_rows[["post_id", "subreddit", "content_topic_primary"]]
        .dropna(subset=["post_id", "content_topic_primary"])
        .drop_duplicates(subset=["post_id"])
        .rename(columns={"content_topic_primary": "topic"})
    )
    frame = post_snapshots[
        [
            "snapshot_id",
            "snapshot_time_utc",
            "post_id",
            "subreddit",
            "age_bucket",
            "activity_state",
            "next_snapshot_id",
        ]
    ].copy()
    frame = frame[frame["activity_state"].isin(STATE_ORDER)].copy()
    frame = frame.merge(topic_map[["post_id", "topic"]], on="post_id", how="left")
    frame = frame.dropna(subset=["topic"]).copy()
    frame["hour_key"] = frame["snapshot_time_utc"].astype(str).str.slice(0, 13)
    return frame


def prepare_flow_trajectory_frame(state_rows: pd.DataFrame) -> pd.DataFrame:
    lookup = (
        state_rows[["snapshot_id", "post_id", "activity_state"]]
        .drop_duplicates(subset=["snapshot_id", "post_id"])
        .rename(columns={"snapshot_id": "next_snapshot_id", "activity_state": "next_state"})
    )
    transitions = state_rows.merge(lookup, on=["next_snapshot_id", "post_id"], how="left")
    transitions = transitions[transitions["next_state"].isin(STATE_ORDER)].copy()

    combo_counts = (
        transitions.groupby(["subreddit", "topic"]).size().reset_index(name="transition_count")
        .sort_values(["subreddit", "transition_count"], ascending=[True, False])
    )
    top_combos = combo_counts.groupby("subreddit", group_keys=False).head(1).reset_index(drop=True)
    if top_combos.empty:
        return pd.DataFrame()

    combo_transition_counts: dict[tuple[str, str, str, str], dict[str, float]] = {}
    global_transition_counts: dict[tuple[str, str], dict[str, float]] = {}

    combo_grouped = transitions.groupby(["subreddit", "topic", "age_bucket", "activity_state", "next_state"]).size()
    for (subreddit, topic, age_bucket, from_state, next_state), count in combo_grouped.items():
        key = (subreddit, topic, age_bucket, from_state)
        combo_transition_counts.setdefault(key, {state: 0.0 for state in STATE_ORDER})
        combo_transition_counts[key][next_state] += float(count)

    global_grouped = transitions.groupby(["age_bucket", "activity_state", "next_state"]).size()
    for (age_bucket, from_state, next_state), count in global_grouped.items():
        key = (age_bucket, from_state)
        global_transition_counts.setdefault(key, {state: 0.0 for state in STATE_ORDER})
        global_transition_counts[key][next_state] += float(count)

    rows: list[dict[str, object]] = []
    early_buckets = {"under_30m", "30m_to_1h"}

    for panel_index, combo in top_combos.iterrows():
        subreddit = str(combo["subreddit"])
        topic = str(combo["topic"])
        combo_slice = state_rows[(state_rows["subreddit"] == subreddit) & (state_rows["topic"] == topic)].copy()
        init_counts = (
            combo_slice[combo_slice["age_bucket"].isin(early_buckets)]["activity_state"].value_counts().to_dict()
        )
        if not init_counts:
            init_counts = combo_slice["activity_state"].value_counts().to_dict()
        dist = normalize_distribution({state: float(init_counts.get(state, 0.0)) for state in STATE_ORDER})

        for age_bucket, age_label in AGE_BUCKETS:
            new_dist = {state: 0.0 for state in STATE_ORDER}
            for from_state, probability in dist.items():
                combo_counts_for_row = combo_transition_counts.get((subreddit, topic, age_bucket, from_state), {})
                if sum(combo_counts_for_row.values()) >= 10:
                    row_dist = normalize_distribution(combo_counts_for_row)
                else:
                    row_dist = normalize_distribution(global_transition_counts.get((age_bucket, from_state), {}))
                for state in STATE_ORDER:
                    new_dist[state] += probability * row_dist.get(state, 0.0)
            dist = normalize_distribution(new_dist)
            for state in STATE_ORDER:
                rows.append(
                    {
                        "panel_index": panel_index,
                        "panel_label": f"r/{subreddit} | {topic_label(topic)}",
                        "subreddit": subreddit,
                        "topic": topic,
                        "age_label": age_label,
                        "state": state,
                        "probability_pct": dist[state] * 100.0,
                    }
                )

    return pd.DataFrame(rows)


def prepare_live_pulse_frame(state_rows: pd.DataFrame) -> pd.DataFrame:
    if state_rows.empty:
        return pd.DataFrame()
    hours = sorted(hour for hour in state_rows["hour_key"].dropna().unique().tolist() if hour)
    if not hours:
        return pd.DataFrame()

    latest_hour = hours[-1]
    baseline_cutoff = hours[-3] if len(hours) >= 4 else latest_hour

    baseline_rows = state_rows[state_rows["hour_key"] < baseline_cutoff].copy()
    current_rows = state_rows[state_rows["hour_key"] == latest_hour].copy()
    if baseline_rows.empty or current_rows.empty:
        return pd.DataFrame()

    def summarise(frame: pd.DataFrame) -> pd.DataFrame:
        grouped = frame.groupby(["topic", "subreddit", "activity_state"]).size().unstack(fill_value=0)
        for state in STATE_ORDER:
            if state not in grouped.columns:
                grouped[state] = 0
        grouped = grouped[STATE_ORDER].copy()
        grouped["total"] = grouped.sum(axis=1)
        grouped["active_rate"] = (grouped["surging"] + grouped["alive"]) / grouped["total"]
        grouped["surge_rate"] = grouped["surging"] / grouped["total"]
        grouped["dead_rate"] = grouped["dead"] / grouped["total"]
        return grouped.reset_index()

    baseline = summarise(baseline_rows)
    current = summarise(current_rows)
    baseline = baseline[baseline["total"] >= 10].copy()
    current = current[current["total"] >= 3].copy()
    if baseline.empty or current.empty:
        return pd.DataFrame()

    merged = current.merge(
        baseline,
        on=["topic", "subreddit"],
        suffixes=("_current", "_baseline"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["surge_ratio"] = merged["surge_rate_current"] / merged["surge_rate_baseline"].clip(lower=0.01)
    merged["alive_ratio"] = merged["active_rate_current"] / merged["active_rate_baseline"].clip(lower=0.01)
    merged["dead_ratio"] = merged["dead_rate_current"] / merged["dead_rate_baseline"].clip(lower=0.01)
    deviation_info = merged.apply(
        lambda row: classify_deviation_kind(float(row["surge_ratio"]), float(row["alive_ratio"]), float(row["dead_ratio"])),
        axis=1,
    )
    merged["kind"] = [value[0] for value in deviation_info]
    merged["magnitude"] = [value[1] for value in deviation_info]
    merged["label"] = merged.apply(
        lambda row: f"{topic_label(str(row['topic']))} | r/{row['subreddit']}",
        axis=1,
    )
    return merged.sort_values(["magnitude", "active_rate_current"], ascending=[False, False]).head(10).copy()


def prepare_deviation_timeline_frame(state_rows: pd.DataFrame) -> pd.DataFrame:
    if state_rows.empty:
        return pd.DataFrame()
    counts = state_rows.groupby(["hour_key", "topic", "subreddit", "activity_state"]).size().unstack(fill_value=0).reset_index()
    for state in STATE_ORDER:
        if state not in counts.columns:
            counts[state] = 0
    counts["total"] = counts[STATE_ORDER].sum(axis=1)
    counts["active_rate"] = (counts["surging"] + counts["alive"]) / counts["total"]
    counts["surge_rate"] = counts["surging"] / counts["total"]
    counts["dead_rate"] = counts["dead"] / counts["total"]

    hours = sorted(hour for hour in counts["hour_key"].dropna().unique().tolist() if hour)
    history_rows: list[dict[str, object]] = []

    for hour_index, hour_key in enumerate(hours):
        current = counts[(counts["hour_key"] == hour_key) & (counts["total"] >= 3)].copy()
        label = hour_key.replace("T", " ")
        if hour_index < 3 or current.empty:
            history_rows.append(
                {
                    "hour_key": hour_key,
                    "time_label": label[5:16],
                    "max_magnitude": 1.0,
                    "top_label": "warming up",
                    "kind": "normal",
                }
            )
            continue

        cutoff_hour = hours[hour_index - 2]
        baseline_pool = counts[counts["hour_key"] < cutoff_hour].copy()
        baseline = baseline_pool.groupby(["topic", "subreddit"])[STATE_ORDER].sum().reset_index()
        if baseline.empty:
            history_rows.append(
                {
                    "hour_key": hour_key,
                    "time_label": label[5:16],
                    "max_magnitude": 1.0,
                    "top_label": "warming up",
                    "kind": "normal",
                }
            )
            continue

        baseline["total"] = baseline[STATE_ORDER].sum(axis=1)
        baseline = baseline[baseline["total"] >= 10].copy()
        baseline["active_rate"] = (baseline["surging"] + baseline["alive"]) / baseline["total"]
        baseline["surge_rate"] = baseline["surging"] / baseline["total"]
        baseline["dead_rate"] = baseline["dead"] / baseline["total"]

        merged = current.merge(
            baseline[["topic", "subreddit", "active_rate", "surge_rate", "dead_rate"]],
            on=["topic", "subreddit"],
            suffixes=("_current", "_baseline"),
            how="inner",
        )
        if merged.empty:
            history_rows.append(
                {
                    "hour_key": hour_key,
                    "time_label": label[5:16],
                    "max_magnitude": 1.0,
                    "top_label": "normal",
                    "kind": "normal",
                }
            )
            continue

        merged["surge_ratio"] = merged["surge_rate_current"] / merged["surge_rate_baseline"].clip(lower=0.01)
        merged["alive_ratio"] = merged["active_rate_current"] / merged["active_rate_baseline"].clip(lower=0.01)
        merged["dead_ratio"] = merged["dead_rate_current"] / merged["dead_rate_baseline"].clip(lower=0.01)
        deviation_info = merged.apply(
            lambda row: classify_deviation_kind(float(row["surge_ratio"]), float(row["alive_ratio"]), float(row["dead_ratio"])),
            axis=1,
        )
        merged["kind"] = [value[0] for value in deviation_info]
        merged["magnitude"] = [value[1] for value in deviation_info]
        best = merged.sort_values(["magnitude", "active_rate_current"], ascending=[False, False]).iloc[0]
        history_rows.append(
            {
                "hour_key": hour_key,
                "time_label": label[5:16],
                "max_magnitude": min(float(best["magnitude"]), 6.0),
                "top_label": f"{topic_label(str(best['topic']))} | r/{best['subreddit']}",
                "kind": str(best["kind"]),
            }
        )

    return pd.DataFrame(history_rows)


def build_flow_trajectory_chart(frame: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1500, 1040), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (30, 20), "Flow Trajectory Chart", FONT_TITLE)
    draw_text(draw, (30, 58), "Historical state-transition curves by age bucket for one representative topic in each subreddit", FONT_SUBTITLE)

    if frame.empty:
        draw_text(draw, (30, 110), "Not enough transition data for a flow trajectory chart.", FONT_LABEL)
        image.save(output_path)
        return

    boxes = [
        (30, 110, 490, 480),
        (520, 110, 980, 480),
        (1010, 110, 1470, 480),
        (30, 530, 490, 900),
        (520, 530, 980, 900),
    ]
    panel_order = frame[["panel_index", "panel_label"]].drop_duplicates().sort_values("panel_index")
    for box, (_, panel_row) in zip(boxes, panel_order.iterrows()):
        panel_frame = frame[frame["panel_label"] == panel_row["panel_label"]].copy()
        draw_multi_line_panel(
            draw,
            box,
            panel_frame,
            str(panel_row["panel_label"]),
            "probability_pct",
            "age_label",
            "state",
            series_order=STATE_ORDER,
            series_color_map=STATE_COLORS,
        )

    legend_box = (1010, 530, 1470, 900)
    draw.rectangle(legend_box, outline="#bfbfbf", width=1)
    draw_text(draw, (legend_box[0] + 20, legend_box[1] + 18), "How To Read", FONT_SUBTITLE)
    notes = [
        "Each panel follows a typical post path for one topic/subreddit pair.",
        "The lines show the estimated chance of being in each state as a post ages.",
        "High surging or alive early means the topic often catches momentum quickly.",
        "A sharp rise in dying or dead later means that topic tends to lose steam fast.",
    ]
    y = legend_box[1] + 56
    for note in notes:
        y += draw_wrapped_text(draw, (legend_box[0] + 20, y), note, FONT_LABEL, max_width=400, line_height=22)
        y += 10
    legend_y = y + 10
    for state in STATE_ORDER:
        color = STATE_COLORS[state]
        draw.line((legend_box[0] + 24, legend_y + 8, legend_box[0] + 60, legend_y + 8), fill=color, width=4)
        draw.text((legend_box[0] + 72, legend_y), state, fill=color, font=FONT_LABEL)
        legend_y += 28

    image.save(output_path)


def build_live_pulse_dashboard(frame: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1500, 980), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (30, 20), "Live Pulse Dashboard", FONT_TITLE)
    draw_text(draw, (30, 58), "Current active-rate versus baseline for the strongest topic and subreddit deviations right now", FONT_SUBTITLE)

    plot_box = (30, 120, 1470, 930)
    draw.rectangle(plot_box, outline="#bfbfbf", width=1)
    left, top, right, bottom = plot_box
    chart_left = left + 320
    chart_right = right - 180
    row_height = 68
    max_width = chart_right - chart_left

    if frame.empty:
        draw_text(draw, (left + 20, top + 20), "No current-vs-baseline pulse rows were available.", FONT_LABEL)
        image.save(output_path)
        return

    draw.line((chart_left, top + 30, chart_left, bottom - 30), fill=AXIS, width=2)
    draw.line((chart_left, bottom - 30, chart_right, bottom - 30), fill=AXIS, width=2)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        x = chart_left + int(max_width * fraction)
        draw.line((x, top + 30, x, bottom - 30), fill=GRID, width=1)
        draw.text((x - 12, bottom - 18), f"{int(fraction * 100)}%", fill=TEXT, font=FONT_SMALL)
    draw.text((chart_left, top + 6), "0%", fill=TEXT, font=FONT_SMALL)

    legend_y = top + 12
    draw.rectangle((right - 155, legend_y, right - 135, legend_y + 12), fill="#a6a6a6", outline="#a6a6a6")
    draw.text((right - 125, legend_y - 4), "baseline active", fill=TEXT, font=FONT_SMALL)
    draw.rectangle((right - 155, legend_y + 24, right - 135, legend_y + 36), fill="#1f77b4", outline="#1f77b4")
    draw.text((right - 125, legend_y + 20), "current active", fill=TEXT, font=FONT_SMALL)

    start_y = top + 70
    for index, (_, row) in enumerate(frame.iterrows()):
        y = start_y + index * row_height
        label = ascii_label(str(row["label"]), max_length=38)
        kind = str(row["kind"])
        baseline_pct = float(row["active_rate_baseline"]) * 100.0
        current_pct = float(row["active_rate_current"]) * 100.0
        magnitude = float(row["magnitude"])
        current_color = "#1f77b4"
        if kind in {"SURGE SPIKE", "elevated activity"}:
            current_color = "#2ca02c"
        elif kind in {"unusually quiet", "mass die-off"}:
            current_color = "#d62728"

        draw.text((left + 16, y - 6), label, fill=TEXT, font=FONT_LABEL)
        draw.text((left + 16, y + 16), ascii_label(kind, max_length=20), fill=current_color, font=FONT_SMALL)

        baseline_width = int((baseline_pct / 100.0) * max_width)
        current_width = int((current_pct / 100.0) * max_width)
        draw.rectangle((chart_left, y, chart_left + baseline_width, y + 14), fill="#b5b5b5", outline="#b5b5b5")
        draw.rectangle((chart_left, y + 20, chart_left + current_width, y + 34), fill=current_color, outline=current_color)

        draw.text((chart_right + 12, y - 2), f"{baseline_pct:.0f}%", fill="#666666", font=FONT_SMALL)
        draw.text((chart_right + 12, y + 18), f"{current_pct:.0f}%  x{magnitude:.2f}", fill=current_color, font=FONT_SMALL)

    image.save(output_path)


def build_deviation_history_timeline(frame: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGB", (1500, 900), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (30, 20), "Deviation History Timeline", FONT_TITLE)
    draw_text(draw, (30, 58), "Reconstructed hourly peak deviation magnitude from historical topic and subreddit state shifts", FONT_SUBTITLE)

    box = (30, 110, 1470, 860)
    left, top, right, bottom = draw_axes(draw, box)
    if frame.empty:
        draw.text((left + 10, top + 10), "No deviation history available", fill=TEXT, font=FONT_LABEL)
        image.save(output_path)
        return

    values = frame["max_magnitude"].fillna(1.0).astype(float).tolist()
    x_labels = frame["time_label"].astype(str).tolist()
    y_min = min(1.0, min(values))
    y_max = max(values)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    for fraction in (0.25, 0.5, 0.75):
        y = bottom - int((bottom - top) * fraction)
        draw.line((left, y, right, y), fill=GRID, width=1)
    threshold_y = bottom - int(normalize(1.4, y_min, y_max) * (bottom - top - 6)) - 3
    draw.line((left, threshold_y, right, threshold_y), fill="#ffbf00", width=2)
    draw.text((right - 140, threshold_y - 18), "alert threshold 1.4", fill="#b8860b", font=FONT_SMALL)

    usable_width = max(1, right - left - 20)
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        x = left + 10 if len(values) == 1 else left + 10 + int(index * usable_width / (len(values) - 1))
        y = bottom - int(normalize(value, y_min, y_max) * (bottom - top - 6)) - 3
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill="#1f4e79", width=3)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#1f77b4", outline="#1f77b4")

    draw.text((left - 2, top - 2), f"{y_max:.1f}", fill=TEXT, font=FONT_SMALL)
    draw.text((left - 2, bottom - 18), f"{y_min:.1f}", fill=TEXT, font=FONT_SMALL)

    label_step = max(1, len(x_labels) // 6)
    for index in range(0, len(x_labels), label_step):
        x, _ = points[index]
        draw.text((x - 18, bottom + 6), x_labels[index], fill=TEXT, font=FONT_SMALL)

    spike_rows = frame.sort_values("max_magnitude", ascending=False).head(4).reset_index(drop=True)
    for _, spike in spike_rows.iterrows():
        match_index = frame.index[frame["hour_key"] == spike["hour_key"]]
        if len(match_index) == 0:
            continue
        point_index = int(match_index[0])
        x, y = points[point_index]
        annotation = ascii_label(f"{spike['top_label']} ({spike['kind']})", max_length=42)
        draw.line((x, y - 2, x, max(top + 20, y - 40)), fill="#666666", width=1)
        draw.text((min(right - 320, x + 6), max(top + 8, y - 56)), annotation, fill=TEXT, font=FONT_SMALL)

    image.save(output_path)


def write_summary(
    *,
    output_path: Path,
    post: pd.Series,
    timeline: pd.DataFrame,
    subreddit_df: pd.DataFrame,
    chart_paths: dict[str, Path],
    pulse_frame: pd.DataFrame,
    deviation_timeline: pd.DataFrame,
) -> None:
    top_attention = subreddit_df.sort_values("total_current_attention_score", ascending=False).iloc[0]
    top_alive = subreddit_df.sort_values("alive_or_surging_count", ascending=False).iloc[0]
    pulse_leader = pulse_frame.iloc[0] if not pulse_frame.empty else None
    top_spike = deviation_timeline.sort_values("max_magnitude", ascending=False).iloc[0] if not deviation_timeline.empty else None
    lines = [
        "# Visual Report",
        "",
        "## Example Post",
        "",
        f"- Subreddit: `{post['subreddit']}`",
        f"- Post ID: `{post['post_id']}`",
        f"- Title: {ascii_label(str(post['title']), max_length=180)}",
        f"- Snapshots shown: `{int(post['snapshot_count'])}`",
        f"- Latest activity state: `{post['latest_activity_state']}`",
        f"- Current attention score: `{float(post['current_attention_score']):.2f}`",
        f"- Timeline chart: `{chart_paths['timeline'].name}`",
        "",
        "## Subreddit Overview",
        "",
        f"- Highest total current attention: `{top_attention['subreddit']}`",
        f"- Most alive or surging posts: `{top_alive['subreddit']}`",
        f"- State mix chart: `{chart_paths['state_mix'].name}`",
        f"- Attention vs popularity chart: `{chart_paths['scatter'].name}`",
        f"- Hourly subreddit trends chart: `{chart_paths['trend'].name}`",
        f"- Flow trajectory chart: `{chart_paths['flow_trajectory'].name}`",
        f"- Live pulse dashboard: `{chart_paths['pulse_dashboard'].name}`",
        f"- Deviation history timeline: `{chart_paths['deviation_timeline'].name}`",
        "",
    ]
    if pulse_leader is not None:
        lines.extend(
            [
                "## Pulse Leader",
                "",
                f"- Strongest live deviation: `{pulse_leader['label']}`",
                f"- Kind: `{pulse_leader['kind']}`",
                f"- Current active rate: `{float(pulse_leader['active_rate_current']) * 100:.1f}%`",
                f"- Baseline active rate: `{float(pulse_leader['active_rate_baseline']) * 100:.1f}%`",
                "",
            ]
        )
    if top_spike is not None:
        lines.extend(
            [
                "## Biggest Historical Spike",
                "",
                f"- Peak timestamp: `{top_spike['hour_key']}`",
                f"- Peak magnitude: `{float(top_spike['max_magnitude']):.2f}`",
                f"- Top driver: `{top_spike['top_label']}`",
                f"- Kind: `{top_spike['kind']}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = pd.read_csv(history_dir / "current_attention_leaderboard.csv")
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
    prediction_rows = pd.read_csv(
        models_dir / "prediction_next_hour.csv",
        usecols=["post_id", "subreddit", "content_topic_primary"],
    )
    subreddit_df = pd.read_csv(history_dir / "subreddit_attention_latest.csv")
    subreddit_snapshots = pd.read_csv(history_dir / "subreddit_snapshots.csv")
    post_snapshots = pd.read_csv(
        history_dir / "post_snapshots.csv",
        usecols=[
            "snapshot_id",
            "snapshot_time_utc",
            "post_id",
            "subreddit",
            "age_bucket",
            "activity_state",
            "next_snapshot_id",
        ],
    )

    post = choose_example_post(leaderboard, args.min_snapshots)
    post_timeline = timeline[timeline["post_id"] == post["post_id"]].sort_values("sequence_position").copy()

    timeline_chart = output_dir / f"example_post_timeline_{slugify(str(post['post_id']))}.png"
    state_mix_chart = output_dir / "subreddit_state_mix.png"
    scatter_chart = output_dir / "subreddit_attention_vs_popularity.png"
    trend_chart = output_dir / "subreddit_hourly_trends.png"
    flow_trajectory_chart = output_dir / "flow_trajectory_by_subreddit.png"
    pulse_dashboard_chart = output_dir / "live_pulse_dashboard.png"
    deviation_timeline_chart = output_dir / "deviation_history_timeline.png"
    summary_path = output_dir / "visual_report_summary.md"

    hourly_subreddit = subreddit_snapshots[
        (subreddit_snapshots["schedule_name"] == "hourly_new")
        | (subreddit_snapshots["listing_type"] == "new")
    ].copy()
    hourly_subreddit["snapshot_dt"] = pd.to_datetime(hourly_subreddit["snapshot_time_utc"], utc=True, errors="coerce")
    hourly_subreddit = hourly_subreddit.dropna(subset=["snapshot_dt"]).sort_values(["subreddit", "snapshot_dt"])
    hourly_subreddit = hourly_subreddit.groupby("subreddit", group_keys=False).tail(24).copy()
    hourly_subreddit["time_label"] = hourly_subreddit["snapshot_dt"].dt.strftime("%d %H:%M")
    hourly_subreddit["average_upvotes"] = pd.to_numeric(hourly_subreddit["average_upvotes"], errors="coerce")
    hourly_subreddit["average_comment_count"] = pd.to_numeric(hourly_subreddit["average_comment_count"], errors="coerce")
    hourly_subreddit["post_count_in_snapshot"] = pd.to_numeric(hourly_subreddit["post_count_in_snapshot"], errors="coerce")
    hourly_subreddit["persisting_post_count_from_previous_snapshot"] = pd.to_numeric(
        hourly_subreddit["persisting_post_count_from_previous_snapshot"], errors="coerce"
    )
    hourly_subreddit["new_post_count_since_previous_snapshot"] = pd.to_numeric(
        hourly_subreddit["new_post_count_since_previous_snapshot"], errors="coerce"
    ).fillna(0.0)
    hourly_subreddit["total_upvotes_in_snapshot"] = (
        hourly_subreddit["average_upvotes"] * hourly_subreddit["post_count_in_snapshot"]
    )
    hourly_subreddit["total_comments_in_snapshot"] = (
        hourly_subreddit["average_comment_count"] * hourly_subreddit["post_count_in_snapshot"]
    )
    hourly_subreddit["persisting_posts_share_pct"] = (
        hourly_subreddit["persisting_post_count_from_previous_snapshot"] / hourly_subreddit["post_count_in_snapshot"]
    ).fillna(0.0) * 100.0

    state_rows = prepare_state_rows(post_snapshots, prediction_rows)
    flow_trajectory_frame = prepare_flow_trajectory_frame(state_rows)
    pulse_frame = prepare_live_pulse_frame(state_rows)
    deviation_timeline = prepare_deviation_timeline_frame(state_rows)

    build_timeline_chart(post, post_timeline, timeline_chart)
    build_subreddit_state_mix_chart(subreddit_df, state_mix_chart)
    build_subreddit_scatter_chart(subreddit_df, scatter_chart)
    build_subreddit_trend_chart(hourly_subreddit, trend_chart)
    build_flow_trajectory_chart(flow_trajectory_frame, flow_trajectory_chart)
    build_live_pulse_dashboard(pulse_frame, pulse_dashboard_chart)
    build_deviation_history_timeline(deviation_timeline, deviation_timeline_chart)
    write_summary(
        output_path=summary_path,
        post=post,
        timeline=post_timeline,
        subreddit_df=subreddit_df,
        pulse_frame=pulse_frame,
        deviation_timeline=deviation_timeline,
        chart_paths={
            "timeline": timeline_chart,
            "state_mix": state_mix_chart,
            "scatter": scatter_chart,
            "trend": trend_chart,
            "flow_trajectory": flow_trajectory_chart,
            "pulse_dashboard": pulse_dashboard_chart,
            "deviation_timeline": deviation_timeline_chart,
        },
    )

    print(f"Saved timeline chart -> {timeline_chart}")
    print(f"Saved subreddit state mix chart -> {state_mix_chart}")
    print(f"Saved subreddit attention chart -> {scatter_chart}")
    print(f"Saved subreddit trend chart -> {trend_chart}")
    print(f"Saved flow trajectory chart -> {flow_trajectory_chart}")
    print(f"Saved live pulse dashboard -> {pulse_dashboard_chart}")
    print(f"Saved deviation history timeline -> {deviation_timeline_chart}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
