# Reddit Community Analysis — Project Context

## Project Overview
Queen Mary University of London — SPC4001 Assessment 2
Reddit data collection and prediction project. Collect Reddit post/comment data, analyse community dynamics, predict post engagement and topic lifecycle.

## Project Location
`C:\Users\Basyl\OneDrive - Queen Mary, University of London\REdit`
GitHub: https://github.com/Vasylffff/redit

## Python Executable
`C:\Users\Basyl\miniconda3\python.exe` (NOT `python` or `python3`)

## Tracked Subreddits
technology, news, worldnews, politics, Games

## Data Collection (3 methods)
1. `collect_reddit_data.py` — PRAW (official Reddit API) — requires credentials
2. `fetch_apify_actor_data.py` — Apify actor `trudax/reddit-scraper` — costs money
3. `collect_reddit_free.py` — Reddit public `.json` endpoints — **preferred, free, no credentials**

## Automation
- `run_free_collection_window.ps1` — hourly runner with popup summary
- `install_free_collection_task.ps1` — installs Windows Task Scheduler job
- Known gap: no internet on subway (~1hr daily), causes collection gaps

## Data Pipeline
1. `collect_reddit_free.py` → raw JSON snapshots in data/snapshots/
2. `build_reddit_history.py` → merges snapshots into data/history/ CSVs
3. `build_prediction_dataset.py` → features for ML models
4. Various `predict_*.py` and `train_*.py` scripts → models

## Key Data Files
- `data/snapshots/` — raw per-collection JSON files
- `data/history/` — merged timeline CSVs (posts + comments over time)
- `data/predictions/` — model outputs

## Analysis Scripts (~20)
- `analyze_sentiment.py` — NLP sentiment scoring
- `analyze_post_timing.py` — optimal posting times
- `analyze_keyword_trends.py` — trending terms
- `analyze_velocity_curves.py` — post engagement acceleration
- `analyze_cross_subreddit.py` — topic spread between subreddits
- `analyze_domains.py` — link domain analysis
- `analyze_comment_engagement.py` — comment depth/quality
- `analyze_title_style.py` — title characteristics vs engagement
- `analyze_authors.py` — author posting patterns

## Prediction Models (~15)
- `predict_post_flow.py` — general post engagement trajectory
- `predict_post_outcome.py` — will post succeed or die
- `predict_topic_popularity.py` — topic-level growth prediction
- `predict_emerging_keywords.py` — next trending keywords
- `predict_mood.py` — subreddit sentiment direction
- `predict_time_to_death.py` — when a post stops getting engagement
- `predict_subreddit_direction.py` — subreddit activity trend
- `predict_crosspost_success.py` — cross-posting success probability
- `train_next_hour_*.py` — next-hour engagement prediction (4 model types: classification, regression, gradient descent, trees)

## Test Scripts (~20)
- `test_virality_models.py`, `test_topic_lifecycle.py`
- `test_momentum_escalation.py`, `test_organic_spread.py`
- `test_temporal_validation.py`, `test_hyperparameter_search.py`
- etc.

## Key Findings (from Session 1, March 31)
- ~100k posts collected across 5 subreddits
- Prediction accuracy varies by subreddit: politics best at 72%, average ~64%
- More data = better predictions (politics has most posts)
- General flow prediction harder than individual post prediction
- Dead post detection: most die within 2-4 hours if no initial engagement
- Comment velocity is the strongest predictor of post success

## Known Issues
- Activity state labels (surging/alive/cooling/dying/dead) are USELESS for topic-level prediction — they're post-level labels that add noise, not signal. Use raw metrics instead.
- Collection gaps from subway commute (no internet ~1hr daily)
- Reddit API credentials impossible to get — use free JSON endpoint only
- Limited to 5 subreddits due to rate limiting on free endpoint

## Reports
- `Assessment_2_Final.docx` — main submission document
- `FINAL_REPORT.md` — markdown version
- Multiple drafts in root folder

## Previous Sessions
### Session 1 (March 31 - April 1): Main analysis
- Built prediction models for post flow
- Explored general topic flow prediction
- Created tracking pools and dead post detection
- Pushed to GitHub
- Data sharing setup (OneDrive)

### Session 2 (April 7-8): Report building
- Background agents running for assessment report
- Figure generation
- Word document building

## Important Feedback
1. Don't use activity state labels as features for topic prediction
2. Update report incrementally as work progresses, not at the end
3. Stick to raw engagement metrics (velocity, comments, upvotes, author count) for topic-level models
