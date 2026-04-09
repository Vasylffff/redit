# REdit — Reddit Topic Lifecycle Prediction

Predicts how topics emerge, spread, and die on Reddit using only engagement signals. Tracks 6,826 posts across 5 subreddits over 13 days, applies classification at every lifecycle stage, and achieves 0.86–0.99 ROC AUC depending on the task.

**Module:** Exploring AI: Understanding and Applications (SPC4004), Queen Mary University of London  
**Assessment 2:** Code Generation Project

---

## What it does

The system operates at two levels:

**Post-level** — predicts individual post trajectories:
- Will this post surge, survive, or die? (0.987 ROC for surging detection)
- How long will it last? (0.843 ROC at 1h, decaying to 0.57 at 7 days)
- What drives survival? (Gini coefficient of comment upvotes = strongest feature at 46%)

**Topic-level** — tracks word pairs ("russian+tanker", "crimson+desert") through their complete lifecycle:

| Stage | Question | ROC AUC |
|-------|----------|---------|
| Filter | Is this noise? (discards 87% of pairs) | 0.850 |
| Birth | Will it grow from 1–3 to 5+ posts? | 0.860 |
| Growth | Has it peaked or still growing? | 0.958 |
| Spread | Will it reach r/politics tomorrow? | 0.756 |
| Decline | Is it dying? | 0.992 |
| Death | Will it die tomorrow? | 0.890 |
| Death speed | Quick death or slow death? | 0.996 |
| Revival | Ongoing story or one-shot event? | 0.970 |

The system is content-agnostic — the same algorithm catches political scandals, game launches, and viral memes using identical engagement features.

---

## Project structure

```
REdit/
├── collection/          Data scrapers and scheduling
│   ├── collect_reddit_free.py      Free Reddit JSON endpoint scraper
│   ├── collect_reddit_data.py      PRAW-based scraper (unused, API not approved)
│   ├── normalize_reddit_json.py    Normalise raw JSON into tables
│   └── run_free_collection_schedule.py  Hourly scheduler
│
├── pipeline/            Data processing and feature engineering
│   ├── build_reddit_history.py     Merge snapshots into unified timeline
│   ├── patch_snapshot_gaps.py      Fix velocity corruption from missed hours
│   ├── build_prediction_dataset.py ML feature labels
│   ├── build_tracking_pools.py     Smart post tracking (active/dormant/dropped)
│   ├── build_free_tracking_pool.py Tracking pool for free collection
│   ├── build_naive_forecast.py     Baseline Markov forecaster
│   ├── evaluate_naive_forecast.py  Forecast evaluation
│   ├── build_subreddit_health.py   Subreddit health scoring
│   ├── export_history_to_sqlite.py SQLite database export
│   ├── validate_history_data.py    Data integrity checks
│   └── ...
│
├── prediction/          ML models
│   ├── predict_post_flow.py        5-layer Markov chain predictor
│   ├── predict_post_outcome.py     Pop or flop prediction
│   ├── predict_time_to_death.py    Hours until post dies
│   ├── predict_mood.py             Sentiment trajectory prediction
│   ├── predict_emerging_keywords.py Keyword emergence detection
│   ├── predict_topic_popularity.py  Topic growth regression
│   ├── predict_subreddit_direction.py Subreddit trend prediction
│   ├── predict_crosspost_success.py Cross-post success probability
│   ├── train_next_hour_*.py        4 model variants (RF, SGD, regression, trees)
│   └── ...
│
├── analysis/            Data analysis scripts
│   ├── analyze_sentiment.py        VADER + K-means on comments
│   ├── analyze_comment_engagement.py Gini coefficient analysis
│   ├── analyze_cross_subreddit.py  Topic spread patterns
│   ├── analyze_keyword_signal.py   Keyword emergence signals
│   ├── analyze_velocity_curves.py  Engagement acceleration
│   ├── analyze_post_timing.py      Optimal posting hours
│   └── ...
│
├── tests/               Validation and experiments
│   ├── test_temporal_pairs.py      Co-occurrence pair emergence detection
│   ├── test_topic_lifecycle.py     Topic state prediction
│   ├── test_subreddit_spread.py    Cross-subreddit spread prediction
│   ├── test_topic_dying.py         Topic death detection
│   ├── test_revival_type.py        Ongoing vs one-shot classification
│   ├── test_false_death.py         False death analysis
│   ├── test_model_comparison.py    5 classifiers x 7 tasks
│   ├── test_hyperparameter_search.py 36 config tuning
│   ├── test_virality_models.py     Szabo-Huberman and log-linear models
│   └── ...
│
├── reporting/           Report and figure generation
│   ├── build_report_html.py        Live HTML report preview
│   ├── build_final_report_docx.py  Word document builder
│   ├── build_assessment_figures.py Figure generation
│   ├── build_visual_report.py      Dashboard visuals
│   └── ...
│
├── reports/             All report versions (markdown + docx)
├── docs/                Documentation
│   ├── scraping_evolution.md       How collection evolved (4 versions)
│   ├── codex_handover_summary.md   Pipeline refactoring handover
│   └── session_history_reconstruction.md  Full development timeline
├── configs/             Schedule and batch configurations
├── data/                Collected data, models, figures
│
├── run_full_analysis.py            One-click: generate all 16 figures
├── report_preview.html             Live report preview (open in browser)
├── requirements.txt
└── *.ps1                           Windows Task Scheduler automation
```

---

## Dataset

Self-collected over 13 days (26 March – 7 April 2026) using Reddit's free public JSON endpoints. No API key required.

| Metric | Value |
|--------|-------|
| Post snapshots | 216,944 |
| Comment snapshots | 972,353 |
| Unique posts | 6,826 |
| Subreddits | r/technology, r/news, r/worldnews, r/politics, r/Games |
| Collection | Hourly via Windows Task Scheduler (2 machines) |
| Raw files | 5,140 JSON files |

The full `data/` folder is not in git (files are 100MB–600MB). Download from OneDrive:

**[Download data folder](https://qmulprod-my.sharepoint.com/:f:/g/personal/ap25150_qmul_ac_uk/IgCrPEbEU-ShRL7wNRfmOlZcAfQOlu9h5u4LOnrz3c2r_mA?email=v.shcherbatykh%40se25.qmul.ac_uk&e=zONS93)**

---

## Setup

### 1. Create virtual environment

```powershell
python -m venv .venv
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run collection (optional — data is available via OneDrive)

```powershell
.\.venv\Scripts\python.exe collection/run_free_collection_schedule.py
```

### 4. Run full analysis pipeline

```powershell
.\.venv\Scripts\python.exe pipeline/build_reddit_history.py
.\.venv\Scripts\python.exe pipeline/patch_snapshot_gaps.py
.\.venv\Scripts\python.exe pipeline/build_prediction_dataset.py
.\.venv\Scripts\python.exe pipeline/export_history_to_sqlite.py
```

### 5. Generate all figures

```powershell
.\.venv\Scripts\python.exe run_full_analysis.py
```

Produces 16 figures in `data/analysis/reddit/figures/`.

---

## Key findings

1. **Content-agnostic detection works.** The same algorithm catches political crises, game launches, and viral memes — it tracks word pairs spreading, not content meaning.

2. **Posts and topics follow opposite predictability patterns.** Post prediction decays from 0.843 ROC (1h) to 0.57 (7d). Topic prediction improves at longer horizons because topics follow momentum.

3. **Controversy drives survival.** Negative comment sentiment correlates with longer post survival. r/politics (most negative) is growing; r/Games (most positive) is declining.

4. **Gini coefficient is the strongest post predictor.** Comment upvote concentration (46% feature importance) outperforms all velocity and upvote features.

5. **Topic death needs 2+ days to confirm.** One-day definition has 13.1% false-death rate. Bigger topics revive more often (44.8% for 8–11 peak posts).

6. **Logistic Regression beats Random Forest** on the core emergence detection task (0.860 vs 0.829), indicating the signal is linear.

7. **Magnitude is unpredictable.** Detection works (0.86 ROC) but predicting HOW viral a topic will become fails (R²=0.22). Tested Szabo-Huberman log-linear models — confirmed by established research.

8. **Revival is driven by external events.** Predicting WHEN a topic revives: 0.578 ROC (random). Predicting WHICH topics are the type that revives: 0.970 ROC. Multiple previous peaks is the strongest signal (15.5x ratio).

---

## Collection evolution

The scraping system went through 4 major versions — see [docs/scraping_evolution.md](docs/scraping_evolution.md):

1. **PRAW** — Reddit API application not approved
2. **Apify** — tested, abandoned (cost, inconsistent data)
3. **Free JSON, no tracking** — posts observed by luck, not design
4. **Free JSON, multi-cadence + tracking** — 5 listing types, fixed observation cohorts, gap patching, two-machine merge

---

## AI tools used

- **Claude Code** (Anthropic, Claude Opus 4.6) — primary tool for code generation, analysis, modelling, and topic lifecycle pipeline
- **OpenAI Codex** — initial setup, infrastructure, pipeline refactoring

All development was conversational. See [docs/session_history_reconstruction.md](docs/session_history_reconstruction.md) for the complete development timeline.

---

## Model comparison

Five classifiers tested across seven tasks with 36 hyperparameter configurations:

| Task | Random Forest | Extra Trees | Grad. Boost | Logistic Reg. | Decision Tree |
|------|--------------|-------------|-------------|---------------|---------------|
| Emergence | 0.829 | 0.808 | 0.680 | **0.851** | 0.560 |
| Escalation | 0.708 | **0.736** | 0.565 | 0.712 | 0.510 |
| Failure filter | 0.829 | 0.808 | 0.760 | **0.850** | 0.560 |
| Death tomorrow | **0.886** | 0.869 | 0.874 | 0.787 | 0.864 |
| Peaked/growing | 0.731 | 0.657 | **0.734** | 0.581 | 0.658 |
| Quick/slow death | 0.995 | 0.996 | 0.996 | **0.999** | 0.973 |
| Subreddit spread | **0.659** | 0.644 | 0.619 | 0.636 | 0.606 |

No single model dominates. Hyperparameter tuning improved Decision Tree by +0.265 and Gradient Boosting by +0.122.

---

## References

- Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proc. ICWSM*.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *JMLR*, 12, pp. 2825-2830.
- Szabo, G. and Huberman, B.A. (2010) 'Predicting the Popularity of Online Content', *Communications of the ACM*, 53(8), pp. 80-88.
