from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual sentiment analysis: VADER scoring + K-means clustering on Reddit comments."
    )
    parser.add_argument(
        "--db",
        default="data/history/reddit/history.db",
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis",
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of K-means clusters (default 5).",
    )
    return parser.parse_args()


def load_comments(db_path: str) -> list[dict[str, str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT subreddit, post_id, body FROM comment_snapshots "
        "WHERE body IS NOT NULL AND body != '' AND body != '[deleted]' AND body != '[removed]'"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_titles(db_path: str) -> list[dict[str, str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT post_id, subreddit, title FROM post_snapshots "
        "WHERE title IS NOT NULL AND title != ''"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_vader(texts: list[str]) -> list[float]:
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def classify_vader(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def run_kmeans(texts: list[str], n_clusters: int) -> tuple[list[int], list[list[str]]]:
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X).tolist()

    feature_names = vectorizer.get_feature_names_out()
    cluster_words: list[list[str]] = []
    for i in range(n_clusters):
        center = km.cluster_centers_[i]
        top_indices = center.argsort()[-10:][::-1]
        cluster_words.append([feature_names[j] for j in top_indices])

    return labels, cluster_words


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # --- Comment sentiment ---
    print("Loading comments...")
    comment_rows = load_comments(args.db)
    comment_texts = [r["body"] for r in comment_rows]
    print("  %d comments loaded" % len(comment_texts))

    print("Running VADER on comments...")
    comment_scores = run_vader(comment_texts)

    print("Running K-means on comments (k=%d)..." % args.clusters)
    comment_labels, cluster_words = run_kmeans(comment_texts, args.clusters)

    comment_output = []
    for i, row in enumerate(comment_rows):
        comment_output.append({
            "subreddit": row["subreddit"],
            "post_id": row["post_id"],
            "vader_score": round(comment_scores[i], 4),
            "vader_label": classify_vader(comment_scores[i]),
            "cluster": comment_labels[i],
            "body_preview": row["body"][:200],
        })

    write_csv(comment_output, output_dir / "comment_sentiment.csv")
    print("  Saved -> %s" % (output_dir / "comment_sentiment.csv"))

    # --- Title sentiment ---
    print("Loading titles...")
    title_rows = load_titles(args.db)
    title_texts = [r["title"] for r in title_rows]
    print("  %d unique titles loaded" % len(title_texts))

    print("Running VADER on titles...")
    title_scores = run_vader(title_texts)

    title_output = []
    for i, row in enumerate(title_rows):
        title_output.append({
            "post_id": row["post_id"],
            "subreddit": row["subreddit"],
            "title": row["title"],
            "vader_score": round(title_scores[i], 4),
            "vader_label": classify_vader(title_scores[i]),
        })

    write_csv(title_output, output_dir / "title_sentiment.csv")
    print("  Saved -> %s" % (output_dir / "title_sentiment.csv"))

    # --- Subreddit summary ---
    sub_comment_scores: dict[str, list[float]] = {}
    sub_title_scores: dict[str, list[float]] = {}
    for row in comment_output:
        sub_comment_scores.setdefault(row["subreddit"], []).append(row["vader_score"])
    for row in title_output:
        sub_title_scores.setdefault(row["subreddit"], []).append(row["vader_score"])

    summary_rows = []
    for sub in sorted(set(list(sub_comment_scores.keys()) + list(sub_title_scores.keys()))):
        c_scores = sub_comment_scores.get(sub, [])
        t_scores = sub_title_scores.get(sub, [])
        c_neg = sum(1 for s in c_scores if s <= -0.05) / len(c_scores) * 100 if c_scores else 0
        t_neg = sum(1 for s in t_scores if s <= -0.05) / len(t_scores) * 100 if t_scores else 0
        summary_rows.append({
            "subreddit": sub,
            "comment_count": len(c_scores),
            "comment_avg_sentiment": round(np.mean(c_scores), 4) if c_scores else "",
            "comment_negative_pct": round(c_neg, 1),
            "title_count": len(t_scores),
            "title_avg_sentiment": round(np.mean(t_scores), 4) if t_scores else "",
            "title_negative_pct": round(t_neg, 1),
        })

    write_csv(summary_rows, output_dir / "sentiment_by_subreddit.csv")
    print("  Saved -> %s" % (output_dir / "sentiment_by_subreddit.csv"))

    # --- Cluster summary ---
    cluster_summary = []
    for i in range(args.clusters):
        c_scores = [comment_scores[j] for j in range(len(comment_labels)) if comment_labels[j] == i]
        count = len(c_scores)
        pos = sum(1 for s in c_scores if s >= 0.05)
        neg = sum(1 for s in c_scores if s <= -0.05)
        neu = count - pos - neg
        cluster_summary.append({
            "cluster": i,
            "count": count,
            "avg_sentiment": round(np.mean(c_scores), 4) if c_scores else "",
            "positive_pct": round(100 * pos / count, 1) if count else 0,
            "neutral_pct": round(100 * neu / count, 1) if count else 0,
            "negative_pct": round(100 * neg / count, 1) if count else 0,
            "top_words": ", ".join(cluster_words[i]),
        })

    write_csv(cluster_summary, output_dir / "cluster_summary.csv")
    print("  Saved -> %s" % (output_dir / "cluster_summary.csv"))

    # --- Print summary ---
    print("\n=== VADER OVERALL ===")
    positive = sum(1 for s in comment_scores if s >= 0.05)
    negative = sum(1 for s in comment_scores if s <= -0.05)
    neutral = len(comment_scores) - positive - negative
    print("  Positive: %d (%.1f%%)" % (positive, 100 * positive / len(comment_scores)))
    print("  Neutral:  %d (%.1f%%)" % (neutral, 100 * neutral / len(comment_scores)))
    print("  Negative: %d (%.1f%%)" % (negative, 100 * negative / len(comment_scores)))

    print("\n=== BY SUBREDDIT ===")
    for row in summary_rows:
        print("  r/%s: comments avg=%.3f (%s%% neg), titles avg=%s (%s%% neg)" % (
            row["subreddit"], float(row["comment_avg_sentiment"] or 0),
            row["comment_negative_pct"], row["title_avg_sentiment"], row["title_negative_pct"],
        ))

    print("\n=== CLUSTERS ===")
    for row in cluster_summary:
        print("  Cluster %d (%d comments, avg=%.3f): %s" % (
            row["cluster"], row["count"], float(row["avg_sentiment"] or 0), row["top_words"],
        ))

    print("\nDone.")


if __name__ == "__main__":
    main()
