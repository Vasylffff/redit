# REdit — Reddit Flow Intelligence

A data collection, modelling, and prediction system built on top of the Reddit API. Tracks how posts live and die across subreddits, builds Markov chain transition models of post flow, detects news events through statistical deviation, and manages smart tracking pools that stop wasting requests on dead posts.

---

## What this project does

Reddit posts follow predictable trajectories: they surge, cool, die. But *how* they do it depends on the topic, the subreddit, the time of day, and what is happening in the real world. This project:

1. **Collects** hourly snapshots of Reddit posts across multiple subreddits and listing types
2. **Builds** a merged history of every post's velocity, state, and engagement over time
3. **Patches** collection gaps so that a missed hourly window doesn't corrupt velocity calculations
4. **Models** post-state transitions as Markov chains, conditioned on topic, subreddit, age, and current velocity
5. **Predicts** how a post will evolve over the next 24 hours — with optional real-observation anchoring
6. **Detects** when a topic is deviating from its historical baseline (= something is happening in the news)
7. **Manages** smart tracking pools that prioritise live posts and drop confirmed-dead ones

---

## Architecture overview

```
Reddit API
    |
    v
collect_reddit_data.py          -- pulls raw posts + comments per subreddit
run_free_collection_schedule.py -- runs only the manifests due this hour
    |
    v
build_reddit_history.py         -- merges all raw runs into post_snapshots.csv
    |
    v
patch_snapshot_gaps.py          -- fixes velocity corruption from missed windows
    |
    v
build_prediction_dataset.py     -- labels each snapshot with next-hour outcome
    |
    v
predict_post_flow.py            -- 5-layer Markov predictor
detect_flow_deviation.py        -- news-pulse / deviation detector
build_tracking_pools.py         -- variance-collapse-based dead-post manager
```

---

## Setup

### 1. Create a virtual environment

```powershell
C:\Users\Basyl\miniforge3\python.exe -m venv .venv
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Create your `.env` file

```powershell
Copy-Item .env.example .env
```

Fill in:
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`

Get these from [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — create a "script" app.

---

## Data collection

### Run a one-off collection

```powershell
.\.venv\Scripts\python.exe collect_reddit_data.py wallstreetbets stocks investing --post-limit 100 --comment-limit-per-post 20
```

Writes timestamped raw data to `data/raw/20260324_141500/`.

### Run the scheduled queue (recommended)

```powershell
.\.venv\Scripts\python.exe run_free_collection_schedule.py
```

Only runs the manifests due for the current hour:

| Schedule | Cadence | Listing |
|---|---|---|
| `hourly_new.csv` | every hour | new |
| `two_hour_rising.csv` | every 2 hours | rising |
| `four_hour_hot.csv` | every 4 hours | hot |
| `twice_daily_top_day.csv` | twice a day | top/day |
| `daily_top_week.csv` | once a day | top/week |

Dry-run preview (no network):

```powershell
.\.venv\Scripts\python.exe run_free_collection_schedule.py --hour 14 --dry-run
```

### Install Windows Task Scheduler job

```powershell
powershell -ExecutionPolicy Bypass -File .\install_collection_schedule_task.ps1
```

This creates `REdit Hourly Collection` — runs the collection, rebuilds history, rebuilds predictions, validates data, and exports to SQLite. Shows a popup summary each run.

---

## Building the history

```powershell
.\.venv\Scripts\python.exe build_reddit_history.py
```

Merges all raw collection runs into:
- `data/history/reddit/post_snapshots.csv` — one row per post per snapshot
- `data/history/reddit/subreddit_snapshots.csv` — subreddit-level aggregates
- `data/history/reddit/activity_thresholds.csv` — empirical alive/surging/dead velocity cutoffs per subreddit
- `data/history/reddit/analysis_focus_latest.csv` — currently surging/alive posts
- `data/history/reddit/tracking_candidates_latest.csv` — per-subreddit priority watchlist

Each snapshot row carries:
- `upvotes_at_snapshot`, `comments_at_snapshot`
- `upvote_velocity_per_hour`, `comment_velocity_per_hour`
- `upvote_delta_from_previous_snapshot`
- `hours_since_previous_snapshot`
- `activity_state` — one of `surging / alive / cooling / dying / dead`
- `age_hours_at_snapshot`
- `listing_type` — which schedule captured this snapshot

---

## Patching collection gaps

**Run this every time after `build_reddit_history.py`, before any modelling scripts.**

```powershell
.\.venv\Scripts\python.exe patch_snapshot_gaps.py
```

### Why gaps corrupt velocity

When the collector misses a window (laptop offline, subway, sleep), the next snapshot shows `velocity = 0` even though the post was actively gaining upvotes during the gap. This fakes a dead-post signal, corrupts Markov transitions, and triggers false variance collapse.

### What the patcher does

| Gap size | Action |
|---|---|
| `< 3 hours` | If velocity = 0 but upvotes > 20 and delta > 0: recalculate velocity from `delta / gap_hours`. Flag `velocity_interpolated = 1`. |
| `>= 3 hours` | Flag `is_collection_gap = 1`. Recalculate velocity from delta / gap anyway (magnitude is better than zero). |
| Zero delta, same upvotes | This is Reddit upvote fuzzing — real zero activity. Flag `is_reddit_fuzzing = 1`, keep velocity = 0 (it's correct). |

Adds three new columns to `post_snapshots.csv`:
- `is_collection_gap` — 1 if the gap before this snapshot was 3h+
- `velocity_interpolated` — 1 if velocity was recalculated
- `is_reddit_fuzzing` — 1 if zero velocity is a genuine zero (Reddit showing same upvote count twice)

Backs up the original to `post_snapshots_pre_patch.csv` before overwriting.

---

## Build prediction tables

```powershell
.\.venv\Scripts\python.exe build_prediction_dataset.py
```

Writes:
- `data/models/reddit/prediction_next_hour.csv` — rows with next-snapshot state labels
- `data/models/reddit/prediction_all_snapshots.csv` — full snapshot-to-snapshot table
- `data/models/reddit/prediction_sequences.csv` — ordered trajectories per post

---

## Predicting post flow

```powershell
.\.venv\Scripts\python.exe predict_post_flow.py --topic war_geopolitics --subreddit worldnews
```

### How it works — 5 layers

#### Layer 1: Markov chain baseline

Builds transition matrices `P(next_state | current_state, topic, subreddit, age_bucket, velocity_bucket)` from historical data.

Velocity is bucketed:
- `low` — below the subreddit's alive threshold
- `med` — alive threshold up to surging threshold
- `high` — above surging threshold

Three-level fallback for sparse data:
1. Full key: `(state, age_bucket, velocity_bucket)` — needs 10+ observations
2. Drop velocity bucket: `(state, age_bucket)` — needs 10+ observations
3. Global fallback: `(state, age_bucket)` across all subreddits

#### Layer 2: Live heat

Compares current surge+alive rate for this topic/subreddit against its 7-day historical baseline (excluding the last 3 hours, so the baseline is never contaminated by the current window).

- Heat ratio > 1.5 → shift initial distribution toward surging
- Heat ratio < 0.7 → shift toward cooling/dead

#### Layer 3: Scenario

Optional user-defined event assumption. Applied as a multiplier to the surge/alive probability in the initial distribution.

| Scenario | Multiplier | Use case |
|---|---|---|
| `quiet` | x0.4 | Holiday, slow news day |
| `normal` | x1.0 | Default — no event assumption |
| `moderate` | x2.0 | Press conference, notable tweet |
| `major` | x4.0 | Election result, resignation, war development |
| `breaking` | x7.0 | Historic / once-in-a-decade event |

#### Layer 4: Anchor (2-hour real observation)

After 2 hours of real observation, replace the historical initial distribution with what you've actually seen. Provide:
- `--anchor-state` — the state right now
- `--anchor-upvotes` — upvote count at 2h
- `--anchor-cv` — comment velocity right now

The anchor's velocity performance (actual vs expected) shifts the velocity bucket used for projections.

#### Layer 5: Discussion quality

Scores the discussion 0–100 from:
- `question_share` — fraction of comments that are questions (engagement signal)
- `avg_comment_upvotes` — community is upvoting the discussion
- `unique_commenters` — breadth of participation
- `post_body_length` — substantial OP encourages better replies

Adjusted by anchor comment velocity ratio when anchor data is present.

### Usage examples

```powershell
# Basic prediction
.\.venv\Scripts\python.exe predict_post_flow.py --topic war_geopolitics --subreddit worldnews

# With scenario assumption
.\.venv\Scripts\python.exe predict_post_flow.py --topic politics_government --subreddit politics --scenario major

# With 2-hour real observation
.\.venv\Scripts\python.exe predict_post_flow.py --topic ai_software --subreddit technology --anchor-state alive --anchor-upvotes 340 --anchor-cv 8

# List all available topics
.\.venv\Scripts\python.exe predict_post_flow.py --list-topics

# Run all topic/subreddit combinations
.\.venv\Scripts\python.exe predict_post_flow.py --all
```

---

## News-pulse deviation detector

```powershell
.\.venv\Scripts\python.exe detect_flow_deviation.py
```

Compares the *current* surge/alive/dead rate for each topic+subreddit against the 7-day historical baseline. Flags when something unusual is happening.

### Deviation types

| Signal | Meaning |
|---|---|
| `SURGE SPIKE` | Surge rate is 2x+ the baseline — unusual volume of activity |
| `elevated activity` | Surge rate 1.4x+ baseline |
| `unusually quiet` | Active rate well below baseline — topic has gone cold |
| `mass die-off` | Dead rate 2x+ baseline — posts dying unusually fast |
| `normal` | Within expected range |

### Cross-subreddit signals

The same topic deviating in the same direction across 2+ subreddits simultaneously is a much stronger signal than a single subreddit. Shown separately as `MULTI-SUB SURGE` or `MULTI-SUB QUIET`.

### History log

Every run appends to `data/history/reddit/deviation_log.csv` — build up a record of when topics were deviating and by how much. Useful for correlating with actual news events.

### Options

```powershell
# Filter to one topic
.\.venv\Scripts\python.exe detect_flow_deviation.py --topic war_geopolitics

# Adjust detection threshold (default 1.4)
.\.venv\Scripts\python.exe detect_flow_deviation.py --threshold 1.6

# Adjust current window (default 3 hours)
.\.venv\Scripts\python.exe detect_flow_deviation.py --hours 6
```

---

## Smart tracking pools

```powershell
.\.venv\Scripts\python.exe build_tracking_pools.py
```

Splits all tracked posts into three pools so the collector stops wasting requests on dead posts.

### The pools

| Pool | Check frequency | Condition |
|---|---|---|
| `active_pool.csv` | every hour | Variance has not collapsed, or revival signal detected |
| `dormant_pool.csv` | every 6 hours | Variance collapsed but post is young (<24h) — watch for revival |
| `dropped_pool.csv` | stop tracking | Variance collapsed + confirmed dead + age 24h+ |

### Variance collapse detection

A post is *dead* when its velocity variance collapses — the chaotic high-variance alive phase ends and velocity locks into a smooth low decline. This is more reliable than a velocity threshold alone, which gets triggered by collection gaps.

Detection criteria:
- `std_before > 15` — post was genuinely volatile before this point
- `std_after < 8` — variance has locked down after this point
- `mean_after < 40` — the after-window isn't flat-high
- `is_declining` — the after-window is actually declining, not plateauing

Collection gap zeros are stripped from the velocity sequence before analysis (a zero surrounded by high values is a gap artefact, not real zero activity).

### Confirmed dead check

Requires *all* of the last 3 snapshots to show:
- upvote velocity <= 5/hr
- comment velocity <= 0.5/hr
- at least 2 of 3 snapshots in `dead` or `dying` state

### Revival signal

Requires 2 of the last 3 snapshots (not just 1, to avoid gap artefacts) to show velocity > 20/hr, or sustained comment velocity, or appearance in the `rising` listing. Posts showing revival signal are moved back to `active_pool`.

---

## Recommended pipeline order

Run these in sequence after each collection cycle:

```powershell
.\.venv\Scripts\python.exe build_reddit_history.py
.\.venv\Scripts\python.exe patch_snapshot_gaps.py
.\.venv\Scripts\python.exe build_prediction_dataset.py
.\.venv\Scripts\python.exe build_tracking_pools.py
.\.venv\Scripts\python.exe detect_flow_deviation.py
```

The Windows Task Scheduler popup wrapper (`run_collection_schedule_window.ps1`) runs the first three automatically after each collection. Run the last two manually when you want analysis.

---

## Training models

Several training scripts are included but are secondary to the Markov approach:

```powershell
.\.venv\Scripts\python.exe train_next_hour_classification.py   # Random Forest classifier
.\.venv\Scripts\python.exe train_next_hour_regression.py       # Linear regression on velocity
.\.venv\Scripts\python.exe train_next_hour_gradient_descent.py # SGD
.\.venv\Scripts\python.exe train_next_hour_trees.py            # Gradient boosted trees
.\.venv\Scripts\python.exe evaluate_naive_forecast.py          # Naive (same-as-now) baseline
```

These write model outputs to `data/models/reddit/classification/`, `regression/`, etc.

---

## Validate and export

```powershell
# Run data quality checks
.\.venv\Scripts\python.exe validate_history_data.py

# Export to SQLite for easier querying
.\.venv\Scripts\python.exe export_history_to_sqlite.py
```

---

## Data files (what is committed vs excluded)

Large runtime files are excluded from git (see `.gitignore`). The repository contains only code, configs, and small summary CSVs.

| File | Size | In git |
|---|---|---|
| `data/history/reddit/post_snapshots.csv` | ~110 MB | No — too large |
| `data/history/reddit/history.db` | ~600 MB | No — regenerable |
| `data/models/reddit/prediction_next_hour.csv` | ~110 MB | No — regenerable |
| `configs/schedules/*.csv` | KB | Yes |
| `data/history/reddit/activity_thresholds.csv` | KB | Yes |
| `data/history/reddit/deviation_log.csv` | KB | Yes |

To rebuild everything from scratch after cloning:
1. Run the collection schedule for a few hours
2. Run `build_reddit_history.py`
3. Run `patch_snapshot_gaps.py`
4. Run `build_prediction_dataset.py`

---

## Key concepts

**Activity states** — assigned per-snapshot based on upvote velocity relative to subreddit thresholds:
- `surging` — velocity above surging threshold
- `alive` — velocity above alive threshold
- `cooling` — velocity positive but declining
- `dying` — velocity near zero, comments slowing
- `dead` — effectively no new engagement

**Velocity buckets** — three tiers used for Markov conditioning:
- `low` — below alive threshold
- `med` — alive to surging
- `high` — above surging threshold

**Collection gap** — a snapshot where the previous window was missed. `is_collection_gap = 1`. Velocity is recalculated from the upvote delta divided by the actual gap length.

**Reddit fuzzing** — Reddit occasionally shows the same upvote count in consecutive snapshots even when a post is getting upvotes (anti-scraping obfuscation). `is_reddit_fuzzing = 1` means velocity = 0 is correct, not a gap.

**Variance collapse** — the point in a post's lifecycle where velocity variance drops sharply and stays low. Indicates the post has locked into its decay trajectory and is effectively dead for new engagement.

---

## Notes

- Keep your `.env` file private. It is git-ignored by default.
- The `--dry-run` flag works on all collection scripts.
- All scripts use the project `.venv` interpreter: `.\.venv\Scripts\python.exe <script>.py`
- The Markov predictor needs at least a few weeks of history data to make good predictions. Accuracy improves with volume.
- Subreddits with low post volume (e.g. `games` ~289 posts) will fall back to global transition matrices more often than high-volume ones (e.g. `politics` ~1,400 posts).
