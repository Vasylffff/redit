# Scraping System Evolution

This document describes how the data collection system evolved through four major versions, each addressing limitations discovered during operation.

---

## Version 1: PRAW (March 24 — never deployed)

**What it was:** `collect_reddit_data.py` using the PRAW library (Python Reddit API Wrapper).

**How it worked:** Official Reddit API with authenticated requests. Would have provided full access to posts, comments, user data, and historical content.

**Why it failed:** Required Reddit API credentials. An application was submitted on 24 March with a formal research proposal. Reddit did not approve the application. Without credentials, the script could not authenticate and returned errors.

**Limitations:**
- Completely dependent on Reddit's approval process
- No fallback mechanism
- Application included: student email (QMUL), supervisor name, full research proposal
- Reddit never responded — no rejection, no approval, just silence

**Files:** `collect_reddit_data.py` (still in repo but unused)

---

## Version 2: Apify (March 26 — briefly tested, abandoned)

**What it was:** `fetch_apify_actor_data.py` using the Apify platform with the `trudax/reddit-scraper` actor.

**How it worked:** Sent configuration JSON to Apify's cloud service, which ran a headless browser to scrape Reddit pages. Returned posts and some comments.

**What it produced:** Successfully scraped data — first 10 items, then batches of 25-100 per subreddit. Multiple batch runs were completed across r/worldnews, r/news, r/politics, r/technology.

**Why it was abandoned:**
- Cost money (Apify credits per run)
- Data format was inconsistent with the free JSON format
- Could not easily re-observe the same posts (no tracking mechanism)
- The same data was available for free through Reddit's public JSON endpoints
- Key realisation from session: *"i do not understand the usefulness of apify because for our prediction we tend to use same thing that we can find freely no?"*

**Limitations:**
- Commercial dependency
- Rate limited by Apify's concurrent run limits
- No hourly automation — each run was manual or semi-manual
- Comment data was partial
- Mixed data format with the free collection made merging difficult

**Files:** `fetch_apify_actor_data.py` (removed during Codex cleanup), configs in `configs/discovery_batch/`

---

## Version 3: Free JSON — No Tracking (March 26 — early operation)

**What it was:** `collect_reddit_free.py` fetching Reddit's public JSON endpoints (e.g. `reddit.com/r/technology/new.json`).

**How it worked:** Appended `.json` to Reddit listing URLs. Each hourly run fetched whatever posts were currently in the listing. Saved raw JSON to `data/raw/reddit_json/` with timestamps.

**Discovery:** The developer found that Reddit serves structured JSON for any listing page without authentication. Claude Code verified the approach was viable for hourly collection and wrote the scraper.

**What it collected:**
- Post metadata: title, author, upvotes, upvote ratio, number of comments, creation time, subreddit, flair, URL
- 1.1-second delay between requests (rate limiting)
- 5 subreddits: technology, news, worldnews, politics, Games
- Only "new" listing initially

**Critical limitation — no tracking:**
- Each run fetched whatever was in "new" at that moment
- No mechanism to re-observe the same post over time
- If a post appeared in two consecutive hourly scrapes, it would have two snapshots — but this was by luck
- Many posts were observed only once, making velocity calculation impossible
- The "history" was really just a pile of one-time snapshots with some lucky duplicates
- Could not compute reliable velocity (upvotes per hour) because most posts had only one data point

**Other limitations:**
- Single listing type ("new" only) — missed rising, hot, and top posts
- No comment collection — only post metadata
- No gap detection — when the laptop was asleep, the next snapshot showed velocity = 0 for every post, creating false dead-post signals
- No validation of data integrity

**Files:** `collect_reddit_free.py`, `build_reddit_history.py` (merged all JSON into timeline)

---

## Version 4: Free JSON — Multi-Cadence with Tracking (March 27 onwards, refined through April)

**What changed:**

### 4a: Multiple listing types (March 27)

Added five collection cadences to capture posts at different lifecycle stages:

| Schedule | Cadence | Listing | What it captures |
|----------|---------|---------|-----------------|
| `hourly_new.csv` | Every hour | new | Freshly submitted posts |
| `two_hour_rising.csv` | Every 2 hours | rising | Posts gaining momentum |
| `four_hour_hot.csv` | Every 4 hours | hot | Currently popular posts |
| `twice_daily_top_day.csv` | Twice daily | top/day | Best posts of the day |
| `daily_top_week.csv` | Daily | top/week | Best posts of the week |

This meant the same post could appear in "new" at hour 1, "rising" at hour 3, and "hot" at hour 6 — giving multiple snapshots automatically.

### 4b: Comment collection (March 28-29)

Added comment scraping to `collect_reddit_free.py`:
- Fetches individual post JSON (e.g. `reddit.com/r/technology/comments/{id}.json`)
- Extracts comment text, author, upvotes, creation time
- Enabled VADER sentiment analysis and Gini coefficient features
- Limited to top-level comments (Reddit JSON nests replies but free endpoint doesn't paginate deeply)

### 4c: Tracking pool (March 31 — Codex refactoring)

Split observation into two lanes:

1. **Prediction observation pool** (`prediction_observation_pool_latest.csv`) — Fixed cohort. Posts enter when first discovered and stay for a defined observation window. This provides clean trajectory data because every post is guaranteed multiple snapshots.

2. **Live watch pool** (`live_watch_pool_latest.csv`) — Rolling shortlist of currently interesting posts. Updated each cycle. Used for monitoring, not for training data.

3. **Combined pool** (`free_observation_pool_latest.csv`) — Union of both, fed to the exact-post re-observation system.

**Why the split mattered:** Before the split, the tracking pool was rolling — it only watched posts that looked interesting right now. This biased the dataset toward already-strong posts and meant weak posts were dropped before their full trajectory could be observed. The fixed cohort ensures every post gets a complete observation window regardless of how it performs.

### 4d: Gap patching (April 1)

`patch_snapshot_gaps.py` addressed velocity corruption from missed collection windows:

| Gap size | Problem | Fix |
|----------|---------|-----|
| < 3 hours | velocity = 0 even for active posts | Recalculate from upvote delta / gap hours. Flag `velocity_interpolated = 1` |
| >= 3 hours | Magnitude is unreliable | Flag `is_collection_gap = 1`. Still recalculate (better than zero) |
| Zero delta, same upvotes | Reddit upvote fuzzing | Flag `is_reddit_fuzzing = 1`. Keep velocity = 0 (it's correct) |

### 4e: Two-machine collection (April 7)

A second machine was set up running the same Task Scheduler job. Data from both machines was merged:
- Machine 1: 3,278 raw JSON files
- Machine 2: 4,427 raw JSON files  
- Merged: 5,140 unique files (1,862 were new from machine 2)

This filled gaps caused by laptop sleep, subway commutes, and different active hours. Post snapshot count increased from 149,323 to 216,944 (+45%). Collection gap rate dropped from 17.9% to 12.1%.

**Files:** `run_free_collection_schedule.py`, `build_free_tracking_pool.py`, `patch_snapshot_gaps.py`

---

## Current System (as of April 8, 2026)

**Architecture:**
```
Windows Task Scheduler (hourly)
    |
    v
run_free_collection_schedule.py
    |--- checks which manifests are due this hour
    |--- runs collect_reddit_free.py for each
    |
    v
collect_reddit_free.py
    |--- fetches listing JSON (new/rising/hot/top)
    |--- fetches individual post JSON for tracked posts
    |--- fetches comments for tracked posts
    |--- 1.1s rate limit between requests
    |
    v
data/raw/reddit_json/YYYYMMDD_HHMMSS_*.json
    |
    v
build_reddit_history.py (merges all snapshots)
    v
patch_snapshot_gaps.py (fixes velocity corruption)
    v
build_prediction_dataset.py (adds ML labels)
```

**What it collects per post:**
- title, author, subreddit, flair, URL
- upvotes, upvote_ratio, num_comments
- creation time, snapshot time
- upvote velocity (upvotes per hour since last snapshot)
- comment velocity
- activity state (surging/alive/cooling/dying/dead)
- age at snapshot (hours since creation)
- listing type that captured this snapshot

**What it collects per comment:**
- text, author, upvotes, creation time
- parent post ID
- (No reply threading — Reddit JSON nests replies but free endpoint returns flat structure)

**Current dataset:**
- 216,944 post snapshots
- 972,353 comment snapshots
- 6,826 unique posts
- 5 subreddits
- 13 days of collection (March 26 — April 7)
- 5,140 raw JSON files

**Known remaining limitations:**
- No reply threading (99% of reply counts are zero — Reddit JSON limitation)
- Collection gaps during subway commute (~1 hour daily)
- Limited to 5 subreddits due to rate limiting constraints
- Reddit occasionally serves cached/stale JSON (upvote fuzzing)
- No authentication means no access to removed/deleted post content
- Free JSON endpoint could be restricted by Reddit at any time
