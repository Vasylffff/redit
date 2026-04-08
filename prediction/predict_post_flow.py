"""
predict_post_flow.py  —  Multi-layer Reddit post flow predictor

Layers:
  1. Baseline      Markov chain: historical P(next_state | state, topic, subreddit, age)
  2. Live heat     How hot is this topic RIGHT NOW vs historical norm
  3. Scenario      User-injected event assumption (e.g. "major political event")
  4. Anchor        After 2h of real data, anchor to observed state + project forward
  5. Discussion    Comment signals: will the discussion be good quality?
  6. Sentiment     Upvote-weighted comment sentiment shifts surging/alive probabilities

Usage:
    python predict_post_flow.py --topic war_geopolitics --subreddit worldnews
    python predict_post_flow.py --topic politics_government --subreddit politics --scenario major
    python predict_post_flow.py --topic ai_software --subreddit technology --anchor-state alive --anchor-upvotes 340 --anchor-cv 8
    python predict_post_flow.py --list-topics
    python predict_post_flow.py --all
"""

import argparse
import collections
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SNAPSHOTS_PATH  = "data/history/reddit/post_snapshots.csv"
PREDICTION_PATH = "data/models/reddit/prediction_next_hour.csv"

STATES       = ["surging", "alive", "cooling", "dying", "dead"]
VALID_STATES = set(STATES)

AGE_BUCKETS = [
    ("under_30m",  "<30m"),
    ("30m_to_1h",  "30m-1h"),
    ("1h_to_3h",   "1-3h"),
    ("3h_to_6h",   "3-6h"),
    ("6h_to_12h",  "6-12h"),
    ("12h_to_24h", "12-24h"),
    ("over_24h",   ">24h"),
]

# Named scenarios: multiplier applied to surge/alive probability
# > 1.0 = more likely to surge,  < 1.0 = less likely
SCENARIOS = {
    "normal":    1.0,
    "moderate":  2.0,   # press conference, notable tweet, minor scandal
    "major":     4.0,   # election result, resignation, war development
    "breaking":  7.0,   # historic / once-in-a-decade event
    "quiet":     0.4,   # holiday, slow news day
}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

C = {
    "surging": "\033[92m",
    "alive":   "\033[96m",
    "cooling": "\033[93m",
    "dying":   "\033[33m",
    "dead":    "\033[91m",
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "heat_hi": "\033[91m",
    "heat_lo": "\033[94m",
    "heat_ok": "\033[92m",
}

def col(state, text=None):
    return C.get(state, "") + (text or state) + C["reset"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    for p in (SNAPSHOTS_PATH, PREDICTION_PATH):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found.")

    print("Loading data...")

    post_meta = {}
    with open(PREDICTION_PATH, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            pid = row["post_id"]
            if pid not in post_meta and row.get("content_topic_primary"):
                post_meta[pid] = {
                    "topic":    row["content_topic_primary"],
                    "subreddit": row["subreddit"],
                }

    snap_state = {}
    snap_rows  = []
    with open(SNAPSHOTS_PATH, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            state = row.get("activity_state", "").strip()
            if state in VALID_STATES:
                snap_state[(row["snapshot_id"], row["post_id"])] = state
            snap_rows.append(row)

    print(f"  {len(post_meta):,} posts  |  {len(snap_rows):,} snapshots  |  {len(snap_state):,} with valid state")
    return snap_rows, snap_state, post_meta


# ---------------------------------------------------------------------------
# Layer 1 — Markov chain
# ---------------------------------------------------------------------------

def velocity_bucket(row):
    """Bucket upvote velocity into low/med/high using per-subreddit thresholds."""
    try:
        vel     = float(row.get("upvote_velocity_per_hour") or 0)
        alive_t = float(row.get("alive_upvote_velocity_threshold") or 4)
        surge_t = float(row.get("surging_upvote_velocity_threshold") or 60)
    except ValueError:
        return "med"
    if vel >= surge_t:
        return "high"
    if vel >= alive_t:
        return "med"
    return "low"


def build_transitions(snap_rows, snap_state, post_meta):
    sub_trans    = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    global_trans = collections.defaultdict(collections.Counter)

    for row in snap_rows:
        pid  = row["post_id"]
        nsid = row.get("next_snapshot_id", "").strip()
        if not nsid or pid not in post_meta:
            continue
        fs = snap_state.get((row["snapshot_id"], pid))
        ts = snap_state.get((nsid, pid))
        if not fs or not ts:
            continue
        vbucket = velocity_bucket(row)
        # Key now includes velocity bucket for finer-grained transitions
        key        = (fs, row.get("age_bucket", ""), vbucket)
        key_global = (fs, row.get("age_bucket", ""))   # global fallback without velocity
        sub_trans[row["subreddit"]][key][ts] += 1
        global_trans[key_global][ts] += 1

    total = sum(sum(c.values()) for m in sub_trans.values() for c in m.values())
    print(f"  {total:,} transitions built  (with velocity bucketing)")
    return sub_trans, global_trans


def normalise(counts):
    total = sum(counts.values())
    if not total:
        return {s: 1.0 / len(STATES) for s in STATES}
    return {s: counts.get(s, 0) / total for s in STATES}


def get_matrix_row(sub_trans, global_trans, subreddit, from_state, bucket, vbucket="med"):
    # Try specific (subreddit, state, age, velocity)
    key = (from_state, bucket, vbucket)
    counts = sub_trans[subreddit].get(key, {})
    if sum(counts.values()) >= 10:
        return normalise(counts)
    # Fallback: any velocity for this subreddit+state+age
    combined = collections.Counter()
    for vb in ("low", "med", "high"):
        combined.update(sub_trans[subreddit].get((from_state, bucket, vb), {}))
    if sum(combined.values()) >= 10:
        return normalise(combined)
    # Global fallback
    return normalise(global_trans.get((from_state, bucket), {}))


def run_chain(init_dist, sub_trans, global_trans, subreddit, init_vbucket="med"):
    dist    = dict(init_dist)
    vbucket = init_vbucket
    results = []
    for bucket, label in AGE_BUCKETS:
        new_dist = {s: 0.0 for s in STATES}
        for fs, p in dist.items():
            if fs not in VALID_STATES:
                continue
            row_p = get_matrix_row(sub_trans, global_trans, subreddit, fs, bucket, vbucket)
            for ts in STATES:
                new_dist[ts] += p * row_p.get(ts, 0.0)
        total = sum(new_dist.values())
        if total:
            new_dist = {s: v / total for s, v in new_dist.items()}
        results.append((label, dict(new_dist)))
        dist = new_dist
        # Velocity naturally decays as posts age — shift toward med after early buckets
        if bucket in ("3h_to_6h", "6h_to_12h") and vbucket == "high":
            vbucket = "med"
    return results


# ---------------------------------------------------------------------------
# Initial state distribution (historical)
# ---------------------------------------------------------------------------

def initial_state_dist(snap_rows, snap_state, post_meta, topic, subreddit):
    early = {"under_30m", "30m_to_1h"}
    base  = collections.Counter()
    cv_buckets = {"low": collections.Counter(),
                  "med": collections.Counter(),
                  "high": collections.Counter()}

    for row in snap_rows:
        pid = row["post_id"]
        if pid not in post_meta:
            continue
        m = post_meta[pid]
        if m["topic"] != topic or m["subreddit"] != subreddit:
            continue
        if row.get("age_bucket") not in early:
            continue
        state = snap_state.get((row["snapshot_id"], pid))
        if not state:
            continue
        base[state] += 1
        try:
            cv = float(row.get("comment_velocity_per_hour") or 0)
        except ValueError:
            cv = 0.0
        tier = "high" if cv >= 5 else ("med" if cv > 0 else "low")
        cv_buckets[tier][state] += 1

    fallback = {"surging": 0.15, "alive": 0.60, "cooling": 0.15,
                "dying": 0.07, "dead": 0.03}
    base_dist = normalise(base) if sum(base.values()) >= 5 else fallback
    cv_dists  = {k: (normalise(v) if sum(v.values()) >= 3 else base_dist)
                 for k, v in cv_buckets.items()}
    return base_dist, cv_dists


# ---------------------------------------------------------------------------
# Layer 2 — Live heat
# ---------------------------------------------------------------------------

def compute_live_heat(snap_rows, snap_state, post_meta, topic, subreddit,
                      historical_surge_alive_rate):
    """
    Compare current surge+alive rate to historical norm.
    Uses snapshots from the last 3 hours as 'right now'.
    Returns (heat_multiplier, current_rate, historical_rate, label)
    """
    if not snap_rows:
        return 1.0, 0.0, historical_surge_alive_rate, "unknown"

    # Get the most recent snapshot time
    times = [r["snapshot_time_utc"] for r in snap_rows if r.get("snapshot_time_utc")]
    if not times:
        return 1.0, 0.0, historical_surge_alive_rate, "unknown"
    latest_time = max(times)

    # Filter to same topic/subreddit in recent snapshots
    recent_states = []
    for row in snap_rows:
        pid = row["post_id"]
        if pid not in post_meta:
            continue
        m = post_meta[pid]
        if m["topic"] != topic or m["subreddit"] != subreddit:
            continue
        t = row.get("snapshot_time_utc", "")
        if not t or t < latest_time[:13]:  # within ~same few hours
            continue
        state = snap_state.get((row["snapshot_id"], pid))
        if state:
            recent_states.append(state)

    if len(recent_states) < 3:
        return 1.0, 0.0, historical_surge_alive_rate, "insufficient data"

    current_rate = sum(1 for s in recent_states if s in ("surging", "alive")) / len(recent_states)

    if historical_surge_alive_rate > 0:
        heat = current_rate / historical_surge_alive_rate
    else:
        heat = 1.0

    if heat >= 2.0:
        label = "VERY HOT"
    elif heat >= 1.3:
        label = "hot"
    elif heat >= 0.7:
        label = "normal"
    elif heat >= 0.4:
        label = "cool"
    else:
        label = "cold"

    return heat, current_rate, historical_surge_alive_rate, label


def historical_surge_alive_rate(snap_rows, snap_state, post_meta, topic, subreddit):
    counts = collections.Counter()
    for row in snap_rows:
        pid = row["post_id"]
        if pid not in post_meta:
            continue
        m = post_meta[pid]
        if m["topic"] != topic or m["subreddit"] != subreddit:
            continue
        state = snap_state.get((row["snapshot_id"], pid))
        if state:
            counts[state] += 1
    total = sum(counts.values())
    if not total:
        return 0.6  # fallback
    return (counts.get("surging", 0) + counts.get("alive", 0)) / total


# ---------------------------------------------------------------------------
# Layer 3 — Scenario parameter
# ---------------------------------------------------------------------------

def apply_scenario(init_dist, multiplier):
    """
    Shift the initial state distribution based on scenario multiplier.
    multiplier > 1  →  more likely to start surging/alive
    multiplier < 1  →  more likely to start dying/dead
    """
    if multiplier == 1.0:
        return init_dist

    adjusted = {}
    for state in STATES:
        if state in ("surging", "alive"):
            adjusted[state] = init_dist.get(state, 0) * multiplier
        else:
            adjusted[state] = init_dist.get(state, 0) / max(multiplier, 0.1)

    total = sum(adjusted.values())
    return {s: v / total for s, v in adjusted.items()}


# ---------------------------------------------------------------------------
# Layer 4 — Anchor (project from real 2h observation)
# ---------------------------------------------------------------------------

def anchor_dist(anchor_state, anchor_upvotes, anchor_cv,
                topic_historical_avg_upvotes, topic_historical_avg_cv):
    """
    Build an initial distribution anchored to a real observation.
    The certainty shifts based on how above/below average the post is.
    """
    if anchor_state not in VALID_STATES:
        anchor_state = "alive"

    # Start with most probability on the anchor state
    base = {s: 0.02 for s in STATES}
    base[anchor_state] = 0.80

    # Adjust based on upvote performance vs topic avg
    if topic_historical_avg_upvotes > 0:
        upvote_ratio = anchor_upvotes / topic_historical_avg_upvotes
    else:
        upvote_ratio = 1.0

    # Adjust based on comment velocity vs topic avg
    if topic_historical_avg_cv > 0:
        cv_ratio = anchor_cv / topic_historical_avg_cv
    else:
        cv_ratio = 1.0

    # Combined performance signal
    performance = (upvote_ratio + cv_ratio) / 2.0

    if performance >= 2.0:
        # Well above average — shift toward surging
        base["surging"] += 0.15
        base["alive"]   += 0.05
        base["dying"]   -= 0.01
        base["dead"]    -= 0.01
    elif performance <= 0.5:
        # Well below average — shift toward dying
        base["dying"]   += 0.10
        base["dead"]    += 0.05
        base["surging"] -= 0.01
        base["alive"]   -= 0.01

    # Renormalise
    total = sum(base.values())
    return {s: max(v, 0) / total for s, v in base.items()}, performance


def topic_averages(snap_rows, post_meta, topic, subreddit):
    upvotes = []
    cvs     = []
    for row in snap_rows:
        pid = row["post_id"]
        if pid not in post_meta:
            continue
        m = post_meta[pid]
        if m["topic"] != topic or m["subreddit"] != subreddit:
            continue
        try:
            upvotes.append(float(row.get("upvotes_at_snapshot") or 0))
        except ValueError:
            pass
        try:
            cvs.append(float(row.get("comment_velocity_per_hour") or 0))
        except ValueError:
            pass
    avg_u = sum(upvotes) / len(upvotes) if upvotes else 100
    avg_c = sum(cvs)    / len(cvs)     if cvs     else 2
    return avg_u, avg_c


# ---------------------------------------------------------------------------
# Layer 5 — Discussion quality
# ---------------------------------------------------------------------------

def discussion_quality(snap_rows, snap_state, post_meta, topic, subreddit,
                       anchor_cv=None, anchor_upvotes=None):
    """
    Estimate whether this post will generate quality discussion.
    Returns a score 0-100 and a label.
    """
    # Gather comment signal stats for this topic/subreddit
    q_shares   = []
    avg_ups    = []
    unique_com = []
    body_lens  = []

    pred_path = PREDICTION_PATH
    with open(pred_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("content_topic_primary") != topic:
                continue
            if row.get("subreddit") != subreddit:
                continue
            if row.get("has_comment_sample") != "1":
                continue
            try:
                q_shares.append(float(row["question_comment_share_sample"]))
                avg_ups.append(float(row["avg_comment_upvotes_sample"]))
                unique_com.append(float(row["unique_commenter_count_sample"]))
                body_lens.append(float(row["avg_comment_body_word_count_sample"]))
            except (ValueError, KeyError):
                pass

    if not q_shares:
        return None, "insufficient comment data"

    avg_q   = sum(q_shares)   / len(q_shares)
    avg_u   = sum(avg_ups)    / len(avg_ups)
    avg_uc  = sum(unique_com) / len(unique_com)
    avg_bl  = sum(body_lens)  / len(body_lens)

    # Score components (each 0-25)
    q_score  = min(avg_q  * 100,  25)           # question share → debate
    u_score  = min(avg_u  / 2,    25)            # comment upvotes → quality
    uc_score = min(avg_uc / 4,    25)            # unique commenters → breadth
    bl_score = min(avg_bl / 6,    25)            # word count → depth

    score = q_score + u_score + uc_score + bl_score

    # Anchor adjustment — if real cv and upvotes are provided, shift the
    # score relative to how this post is actually performing vs topic average
    if anchor_cv is not None or anchor_upvotes is not None:
        avg_cv_topic     = sum(
            float(r.get("comment_velocity_per_hour") or 0)
            for r in snap_rows
            if post_meta.get(r["post_id"], {}).get("topic") == topic
               and post_meta.get(r["post_id"], {}).get("subreddit") == subreddit
        )
        n_cv = sum(
            1 for r in snap_rows
            if post_meta.get(r["post_id"], {}).get("topic") == topic
               and post_meta.get(r["post_id"], {}).get("subreddit") == subreddit
               and r.get("comment_velocity_per_hour")
        )
        avg_cv_topic = avg_cv_topic / n_cv if n_cv else 2.0

        cv_ratio = (anchor_cv / avg_cv_topic) if anchor_cv and avg_cv_topic > 0 else 1.0

        # cv well above average → discussion will be more engaged than baseline
        # cv well below average → discussion will be shallower than baseline
        if cv_ratio >= 3.0:
            score = min(score * 1.4, 100)
        elif cv_ratio >= 1.5:
            score = min(score * 1.15, 100)
        elif cv_ratio <= 0.3:
            score = score * 0.6
        elif cv_ratio <= 0.7:
            score = score * 0.85

    if score >= 75:
        label = "deep & engaged"
    elif score >= 55:
        label = "good discussion"
    elif score >= 35:
        label = "moderate"
    else:
        label = "shallow / low engagement"

    return round(score), label


# ---------------------------------------------------------------------------
# Layer 6 — Sentiment signal
# ---------------------------------------------------------------------------

def compute_sentiment_signal(topic, subreddit):
    """
    Read upvote-weighted comment sentiment from the prediction dataset.
    Returns (multiplier, avg_sentiment, label).
    """
    sentiment_values = []

    try:
        with open(PREDICTION_PATH, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if row.get("content_topic_primary") != topic:
                    continue
                if row.get("subreddit") != subreddit:
                    continue
                if row.get("has_comment_sample") != "1":
                    continue
                raw = row.get("sentiment_weighted_mean_sample")
                if raw is None or raw == "":
                    raw = row.get("sentiment_mean_sample")
                if raw is None or raw == "":
                    continue
                try:
                    sentiment_values.append(float(raw))
                except (ValueError, TypeError):
                    pass
    except FileNotFoundError:
        return 1.0, None, "no prediction data"

    if not sentiment_values:
        return 1.0, None, "insufficient sentiment data"

    avg_sentiment = sum(sentiment_values) / len(sentiment_values)

    if avg_sentiment > 0.15:
        mult, label = 1.3, "strongly positive"
    elif avg_sentiment > 0.05:
        mult, label = 1.1, "mildly positive"
    elif avg_sentiment < -0.15:
        mult, label = 0.7, "strongly negative"
    elif avg_sentiment < -0.05:
        mult, label = 0.9, "mildly negative"
    else:
        mult, label = 1.0, "neutral"

    return mult, round(avg_sentiment, 4), label


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_flow(results, topic, subreddit, scenario_name, heat_info,
               anchor_info, discussion_info, sentiment_info, layers_used):
    print()
    print(C["bold"] + "=" * 72 + C["reset"])
    print(C["bold"] + "  POST FLOW PREDICTOR" + C["reset"])
    print("=" * 72)
    print(f"  Topic:     {col('surging', topic)}   Subreddit: {col('alive', subreddit)}")

    # Show which layers are active
    print(f"  Layers:    ", end="")
    layer_labels = {
        "baseline":   "Baseline",
        "heat":       "Live Heat",
        "scenario":   f"Scenario({scenario_name})",
        "anchor":     "Anchor",
        "discussion": "Discussion",
        "sentiment":  "Sentiment",
    }
    print("  |  ".join(
        C["bold"] + layer_labels[l] + C["reset"]
        for l in layers_used
    ))

    # Heat
    heat_mult, current_rate, hist_rate, heat_label = heat_info
    heat_color = "heat_hi" if heat_mult >= 1.3 else ("heat_lo" if heat_mult < 0.7 else "heat_ok")
    print(f"\n  Live heat: {C[heat_color]}{heat_label}{C['reset']}"
          f"  (current={current_rate:.0%}  hist={hist_rate:.0%}"
          f"  x{heat_mult:.1f})")

    # Scenario
    if scenario_name != "normal":
        smult = SCENARIOS.get(scenario_name, 1.0)
        print(f"  Scenario:  {C['bold']}{scenario_name}{C['reset']}"
              f"  (surge multiplier x{smult})")

    # Anchor
    if anchor_info:
        state, upvotes, cv, perf = anchor_info
        print(f"  Anchor:    state={col(state)}  upvotes={upvotes}"
              f"  cv={cv}/hr  performance={perf:.1f}x avg")

    # Discussion
    disc_score, disc_label = discussion_info
    if disc_score is not None:
        bar_n = round(disc_score / 4)
        bar   = "[" + "#" * bar_n + "." * (25 - bar_n) + "]"
        print(f"  Discussion quality:  {disc_score}/100  {bar}  {C['bold']}{disc_label}{C['reset']}")

    # Sentiment
    sent_mult, sent_avg, sent_label = sentiment_info
    if sent_avg is not None:
        direction = "+" if sent_avg >= 0 else ""
        print(f"  Sentiment signal:   {direction}{sent_avg}  mult={sent_mult}x  {C['bold']}{sent_label}{C['reset']}")

    print()

    # Flow table
    print(f"  {'Age':<9}", end="")
    for s in STATES:
        print(f"  {col(s, f'{s:<12}')}", end="")
    print("  dominant")
    print("  " + "-" * 74)

    dominant_states = []
    for label, probs in results:
        dominant = max(probs, key=probs.get)
        dominant_states.append(dominant)
        print(f"  {label:<9}", end="")
        for s in STATES:
            p = probs.get(s, 0.0)
            bar_n = round(p * 8)
            bar   = "[" + "#" * bar_n + "." * (8 - bar_n) + "]"
            print(f"  {col(s, f'{p:>5.0%} {bar}')}", end="")
        print(f"  {col(dominant)}")

    print()
    print("  TRAJECTORY: " + " -> ".join(col(s) for s in dominant_states))
    print("=" * 72)
    print()


def print_all_summary(topics, subreddits, snap_rows, snap_state, post_meta,
                      sub_trans, global_trans):
    bucket_labels = [label for _, label in AGE_BUCKETS]
    print()
    print(C["bold"] + "=" * 95 + C["reset"])
    print(C["bold"] + "  FLOW COMPARISON — all topics (baseline, normal scenario)" + C["reset"])
    print()
    print(f"  {'Topic':<28} {'Sub':<12}", end="")
    for lbl in bucket_labels:
        print(f" {lbl:<9}", end="")
    print()
    print("  " + "-" * 95)

    for topic in topics:
        for sub in subreddits:
            base_dist, _ = initial_state_dist(
                snap_rows, snap_state, post_meta, topic, sub)
            results = run_chain(base_dist, sub_trans, global_trans, sub)
            print(f"  {topic:<28} {sub:<12}", end="")
            for _, probs in results:
                dominant = max(probs, key=probs.get)
                print(f" {col(dominant, f'{dominant[:8]:<9}')}", end="")
            print()
        print()
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-layer Reddit post flow predictor")
    parser.add_argument("--topic",     default=None)
    parser.add_argument("--subreddit", default=None)
    parser.add_argument("--scenario",  default="normal",
                        choices=list(SCENARIOS.keys()),
                        help="Event scenario (normal/moderate/major/breaking/quiet)")
    parser.add_argument("--cv-tier",   default="base",
                        choices=["base", "low", "med", "high"],
                        help="Comment velocity tier for initial state")
    # Layer 4 anchor args
    parser.add_argument("--anchor-state",   default=None,
                        choices=STATES,
                        help="Observed state at ~2h (enables anchor layer)")
    parser.add_argument("--anchor-upvotes", type=float, default=None,
                        help="Observed upvotes at anchor point")
    parser.add_argument("--anchor-cv",      type=float, default=None,
                        help="Observed comment velocity at anchor point")
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--list-topics", action="store_true")
    args = parser.parse_args()

    snap_rows, snap_state, post_meta = load_data()
    print("Building transitions...")
    sub_trans, global_trans = build_transitions(snap_rows, snap_state, post_meta)

    topics     = sorted(set(v["topic"]     for v in post_meta.values()))
    subreddits = sorted(set(v["subreddit"] for v in post_meta.values()))

    if args.list_topics:
        tc = collections.Counter(v["topic"]     for v in post_meta.values())
        sc = collections.Counter(v["subreddit"] for v in post_meta.values())
        print("\nTopics:")
        for t, n in tc.most_common():
            print(f"  {t:<35} ({n:,} posts)")
        print("\nSubreddits:")
        for s, n in sc.most_common():
            print(f"  {s:<35} ({n:,} posts)")
        return

    if args.all:
        print_all_summary(topics, subreddits, snap_rows, snap_state,
                          post_meta, sub_trans, global_trans)
        return

    topic     = args.topic     or "war_geopolitics"
    subreddit = args.subreddit or "worldnews"

    layers_used = ["baseline"]

    # --- Layer 1+2: baseline + live heat ---
    base_dist, cv_dists = initial_state_dist(
        snap_rows, snap_state, post_meta, topic, subreddit)
    init_dist = cv_dists.get(args.cv_tier, base_dist) \
                if args.cv_tier != "base" else base_dist

    hist_rate = historical_surge_alive_rate(
        snap_rows, snap_state, post_meta, topic, subreddit)
    heat_info = compute_live_heat(
        snap_rows, snap_state, post_meta, topic, subreddit, hist_rate)
    layers_used.append("heat")

    heat_mult = heat_info[0]
    if heat_mult != 1.0:
        init_dist = apply_scenario(init_dist, heat_mult)

    # --- Layer 3: scenario ---
    scenario_mult = SCENARIOS.get(args.scenario, 1.0)
    if scenario_mult != 1.0:
        init_dist = apply_scenario(init_dist, scenario_mult)
        layers_used.append("scenario")

    # --- Layer 4: anchor ---
    anchor_info = None
    init_vbucket = "med"
    if args.anchor_state:
        avg_u, avg_c = topic_averages(snap_rows, post_meta, topic, subreddit)
        upvotes = args.anchor_upvotes or avg_u
        cv      = args.anchor_cv      or avg_c
        init_dist, performance = anchor_dist(
            args.anchor_state, upvotes, cv, avg_u, avg_c)
        anchor_info = (args.anchor_state, upvotes, cv, performance)
        layers_used.append("anchor")
        # Derive velocity bucket from anchor upvotes vs topic average
        if performance >= 2.0:
            init_vbucket = "high"
        elif performance <= 0.5:
            init_vbucket = "low"

    # --- Layer 5: discussion quality ---
    disc_score, disc_label = discussion_quality(
        snap_rows, snap_state, post_meta, topic, subreddit,
        anchor_cv=args.anchor_cv, anchor_upvotes=args.anchor_upvotes)
    if disc_score is not None:
        layers_used.append("discussion")
    discussion_info = (disc_score, disc_label)

    # --- Layer 6: sentiment signal ---
    sent_mult, sent_avg, sent_label = compute_sentiment_signal(topic, subreddit)
    if sent_avg is not None and sent_mult != 1.0:
        init_dist = apply_scenario(init_dist, sent_mult)
        layers_used.append("sentiment")
    sentiment_info = (sent_mult, sent_avg, sent_label)

    # --- Run chain ---
    results = run_chain(init_dist, sub_trans, global_trans, subreddit, init_vbucket)

    print_flow(results, topic, subreddit, args.scenario,
               heat_info, anchor_info, discussion_info, sentiment_info, layers_used)


if __name__ == "__main__":
    main()
