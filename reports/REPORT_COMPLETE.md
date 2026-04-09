# Predicting Reddit Post Survival and Emerging Topic Detection Using Engagement Analysis

**Module:** Exploring AI: Understanding and Applications (SPC4004)
**Assessment:** Code Generation Project
**GitHub:** https://github.com/Vasylffff/redit

---

## 1. Problem Definition & Dataset Justification

### The Problem

This project addresses two interconnected prediction tasks on Reddit:

1. **Post-level prediction**: Given a post's current engagement metrics, predict whether it will survive (remain active) or die, across time horizons from 1 hour to 7 days.

2. **Topic-level prediction**: Given a keyword or topic that has just appeared in 1-3 posts, predict whether it will grow into a significant topic.

The project applies multiple machine learning approaches: Markov chain transition modelling for state-to-state prediction, binary and multi-class classification (Random Forest, Extra Trees, Decision Trees) for survival prediction, regression for upvote and time-to-death estimation, unsupervised clustering (K-means) for comment analysis, and rule-based NLP (VADER) for sentiment scoring.

### The Dataset

Self-collected using Reddit's public JSON endpoints over 12 days (26 March to 6 April 2026) across five subreddits: r/technology, r/news, r/worldnews, r/politics, and r/Games.

| Metric | Value |
|--------|-------|
| Post snapshots | 185,048 |
| Comment snapshots | 677,569 |
| Unique post lifecycles | 6,292 |
| Collection frequency | Hourly (automated via Windows Task Scheduler) |

Each post is tracked through lifecycle states (surging, alive, cooling, dying, dead) computed from upvote and comment velocity thresholds per subreddit. This provides naturally labelled training data without manual annotation.

### Why This Dataset

Pre-existing datasets provide static snapshots. Our time-series collection captures the same post at multiple points, enabling trajectory prediction. Comment text enables sentiment and engagement analysis. Different subreddits provide natural experimental variation (r/politics churns in 11 hours, r/Games lives 49 hours).

---

## 2. Initial Code & Explanation of AI Use

The initial codebase was generated using Claude Code (Anthropic's CLI, Claude Opus 4.6). The primary prompt:

> "This is my project and I want you to create a folder and set this up so I could scrape Reddit from here"

This produced a data collection pipeline, schedule runner, and a five-layer Markov chain predictor. The initial system could scrape data and predict state transitions using baseline probabilities but had no sentiment analysis, no comment engagement features, no multi-horizon prediction, and no topic-level analysis.

All subsequent development used Claude Code with the developer directing analysis priorities, questioning results, and pushing exploration beyond what the AI initially suggested was possible.

---

## 3. Critique of Initial Code

### Compatibility Issues
- `datetime.UTC` (Python 3.11+) used across 12 files; development machine ran Python 3.10
- f-string backslash syntax incompatible with Python 3.10
- scikit-learn version requirement incompatible with Python 3.10

### Algorithmic Limitations
- **No sentiment analysis**: Blind to emotional content of discussions
- **No comment engagement analysis**: Ignored upvote distribution patterns (Gini coefficient later proved the strongest predictor)
- **Single-step prediction**: Markov chain converges to equilibrium; cannot discriminate at longer horizons
- **No per-subreddit models**: r/politics (11h median alive) treated same as r/Games (49h)
- **No topic-level analysis**: Operated only at individual post level

### Infrastructure Issues
Task Scheduler script crashed non-interactively due to WPF popup dependency and aggressive error handling treating Python stderr as fatal errors. Unicode em dashes corrupted under Windows cp1252 encoding.

---

## 4. Iterative Development & Justification

### Iteration 1: Compatibility and Infrastructure Fixes
Replaced `datetime.UTC` in 12 files, fixed f-string syntax, pinned scikit-learn 1.7, removed WPF popups, fixed encoding. Verified: all scripts run, scheduler collects hourly.

### Iteration 2: VADER Sentiment Integration
Scored 677,569 comments with VADER. Added 5 sentiment columns to prediction dataset.

**Finding**: Posts with negative sentiment survive longer (alive avg -0.006 vs dead +0.045). Controversy drives engagement; apathy kills posts. Decision Tree achieved 74.5% accuracy but feature importance revealed comment count dominated at 74% — the model was counting comments, not analysing sentiment.

### Iteration 3: Comment Upvote Gini Coefficient
Implemented Gini coefficient on comment upvote distributions. High Gini = few dominant comments (community consensus); low Gini = diffuse attention (unfocused discussion).

**Finding**: Gini became the strongest predictor (46% feature importance). Surviving posts: Gini 0.63-0.72. Dying posts: 0.36. Classifier improved to 77.6%.

**Insight**: What predicts survival is not whether people are happy or angry, but whether the discussion has focus.

### Iteration 4: Per-Subreddit Classification
Separate Random Forest classifiers per subreddit. r/politics: 81% accuracy. r/Games: 64%. Politically focused communities follow more systematic patterns than entertainment ones.

### Iteration 5: Multi-Horizon Prediction
17 separate classifiers for 1-72 hour horizons, each trained on 100K-160K samples.

**Finding**: ROC AUC decays systematically: 0.843 (1h) → 0.834 (4h) → 0.809 (12h) → 0.771 (24h) → 0.726 (48h) → ~0.57 (7 days).

Post half-lives: surging ~48h, alive ~24h, cooling ~3h, dying ~1h.

**Insight**: Beyond 48 hours, accuracy paradoxically rises to 85% while ROC drops to 0.57. This occurs because class imbalance increases (only 14% alive at 7 days), so predicting "everything dies" gets high accuracy. ROC AUC, insensitive to class imbalance, reveals negligible discriminative power beyond 3 days.

### Iteration 6: State Rise/Fall Prediction
Predicted whether posts will improve in state (not just survive).

**Finding**: Rise prediction achieves 0.947 ROC at 1h and barely decays to 0.926 at 24h. Rise is more predictable than fall because rising requires strong signals (velocity, engagement) while falling is absence of signal.

### Iteration 7: Per-State Predictions
Separate models for each state transition.

| Target | 1h ROC | 12h ROC | Notes |
|--------|--------|---------|-------|
| Surging | 0.971 | 0.958 | Nearly perfect, stays strong |
| Dead | 0.945 | 0.904 | Death is very predictable |
| Cooling | 0.846 | 0.663 | Hardest — transitional state |
| Dying/Dead | 0.892 | 0.768 | Strong early signal |

### Iteration 8: Topic Growth Prediction
Shifted from individual posts to keyword-level prediction using daily frequency tracking across 1,793 keywords over 12 days.

**Finding**: Topic growth prediction achieves 0.839 ROC at 3 days ahead — and **improves** with longer horizons (opposite of posts). Topics follow momentum; posts follow chaos.

**Per-subreddit topic formulas** (linear regression):
- r/news: R2=0.808 for topic upvote prediction
- r/Games: R2=0.727
- r/politics: R2=0.551 (most chaotic)

**Inverse relationship discovered**: r/politics is best for post prediction (81%) but worst for topic prediction (R2=0.551). r/news is worst for posts (68%) but best for topics (R2=0.808). Post-level and topic-level dynamics operate on different mechanisms.

### Iteration 9: Emerging Keyword Detection
Detecting which keywords appearing with 1-5 posts will reach 10+ posts within 3 days.

**Result**: 0.866 ROC AUC. Primary signals: post count, total upvotes, cross-subreddit spread, days previously seen. However, when controlling for simple counts, quality signals add only +0.013 improvement.

### Iteration 10: Co-occurrence Pair Analysis
Instead of single keywords (generic), used word pairs appearing in the same title to represent specific topics.

**Examples detected**: "birthright+citizenship" (1→17 posts), "russian+tanker" (2→16), "kash+patel" (1→14), "attorney+general" (1→16).

**Result**: 0.845 ROC for pair emergence, 0.838 on temporal validation (trained on days 1-8, tested on days 9-12).

### Iteration 11: Connecting Post Quality to Topic Prediction
Multiple attempts to use post-level quality signals (Gini, sentiment, velocity, rising predictions) to improve keyword/topic prediction:

| Approach | Improvement over simple counts |
|----------|-------------------------------|
| Average post quality features | -0.003 (worse) |
| Best post features | -0.002 (worse) |
| Ratio features (surging%, alive%, dead%) | -0.020 (worse) |
| Post survival predictions as features | -0.001 (same) |
| Post rising predictions as features | -0.001 (same) |
| Quality at 3+ days observation | +0.013 (small) |
| 5x rocketing keywords with rising | +0.060 (but n=15) |
| **First post's comment count** | **+0.034** (best) |

**The breakthrough**: First post's comment count (384 avg for growing keywords vs 179 for non-growing) provides the strongest cross-level signal at +0.034 improvement, achieving 0.859 ROC. This connects post quality to topic prediction: if people discuss the first post extensively, more posts about the same topic follow.

**Honest assessment**: This improvement is real but modest. The fundamental challenge remains: topic growth depends on real-world events (court rulings, military actions, political scandals) that engagement metrics cannot predict. A ceasefire proposal post with 1,400 comments and 40K upvotes received only 1% explosion probability from our model — the signal was screaming but the model missed it because it lacked sufficient training examples of such extreme cases.

### Iteration 12: Engagement Speed and Acceleration
Tested whether the RATE of early engagement (upvotes per hour, comments per hour) and ACCELERATION (is it getting faster?) predicts topic growth better than total counts.

**Finding**: Growing topics show 2.8x higher upvote acceleration and 2.2x higher comment speed on their first posts. However, the classifier ROC actually decreased (0.767 → 0.740) because acceleration features add noise with only 40 test events. The signal exists in the profile but the model cannot exploit it reliably.

### Iteration 13: Organic Spread Signals
Tested novel features: unique authors (organic vs single-person posting), title diversity (different angles on same topic), time concentration (posts clustered vs spread), and cross-subreddit patterns.

**Finding**: Title diversity emerged as the strongest organic signal at 3.3x ratio between growing and non-growing topics. Topics with posts approaching the subject from different angles (diversity 0.76) are far more likely to grow than single-angle stories (0.23). Unique author count was the 2nd most important feature (0.267 importance). However, ROC improvement was only +0.002 on temporal validation.

### Iteration 14: Practical Detection Evaluation
Evaluated practical usefulness rather than just ROC scores. At a 10% probability threshold:

| Metric | Value |
|--------|-------|
| Keywords flagged | 565 |
| Actually grew | 180 (32% precision) |
| False alarms | 385 |
| Topics caught | 72% of all growers |
| Practical summary | 1 in 3 flagged keywords actually grows |

At 50% threshold: 64% precision but only catches 9% of growing topics. The model is most confident about keywords that already look obviously big — the genuinely early, subtle detections remain the unsolved challenge.

### Summary of Topic-Level Exploration

All attempts to use post-level quality signals for topic prediction:

| Approach | ROC Improvement | Why |
|----------|----------------|-----|
| Average post quality | -0.003 | Averages wash out signal |
| Best post features | -0.002 | Not enough contrast |
| Post survival predictions | -0.001 | Different mechanism |
| Post rising predictions | -0.001 | Same problem |
| Ratio features (surge%, dead%) | -0.020 | Added noise |
| Quality at 3+ days observation | +0.013 | Small but real |
| First post comment count | **+0.034** | Best cross-level signal |
| Engagement speed/acceleration | Profile clear (2.8x) but model -0.027 | Needs more data |
| Title diversity | Profile clear (3.3x) but model +0.002 | Needs more data |
| 5x rocketing events | **+0.060** | But only 15 events |

**Core conclusion**: Post-level quality signals show clear statistical separation between growing and non-growing topics (2-3x ratios in profiles). However, translating this into improved classification requires more training examples of topic explosion events than 12 days provides. The signals are real; the data volume is insufficient to exploit them.

---

## 5. Final Code Evaluation and Reflection

### Complete Model Performance

**Post-Level Models (genuinely strong):**
| Model | ROC AUC | Notes |
|-------|---------|-------|
| Surging detection (1h) | 0.987 | Near-perfect |
| Post state rise (1h) | 0.947 | Barely decays to 0.926 at 24h |
| Dead detection (1h) | 0.945 | Death is very predictable |
| Post survival (1h) | 0.843 | Good |
| Post survival (4h) | 0.834 | Sweet spot |
| Post survival (24h) | 0.771 | Declining |
| Post survival (48h) | 0.726 | Approaching random |

**Topic-Level Models (decent, different mechanism):**
| Model | Score | Notes |
|-------|-------|-------|
| Emerging keyword detection | 0.866 ROC | Mostly counting |
| Co-occurrence pair emergence | 0.845 ROC | Real topics |
| Temporal validation (pairs) | 0.838 ROC | Trained on past, tested on future |
| Topic growth 3-day ahead | 0.839 ROC | Improves with longer horizons |
| First-post-to-topic prediction | 0.859 ROC | Comment count is the bridge |
| r/news topic upvotes | R2=0.808 | Linear regression with interpretable formula |

**Regression Models (weakest):**
| Model | R2 | Notes |
|-------|-----|-------|
| Topic post count (1d) | 0.549 | Decent |
| Time-to-death | 0.459 | Ballpark |
| Peak upvotes | 0.42 | Rough |
| Comment volume | 0.34 | Weakest |

### Key Findings

1. **Controversy drives engagement, apathy kills posts.** Negative comment sentiment correlates with longer survival. r/politics (sentiment -0.10) shows strongest growth; r/Games (+0.38) is declining.

2. **Comment upvote concentration (Gini) is the strongest post-level predictor** (46% importance). Posts with focused discussion survive; diffuse discussion kills.

3. **Predictability decays for posts but improves for topics.** Posts become unpredictable beyond 24h (ROC 0.843→0.726). Topics become MORE predictable at 3 days (ROC 0.839) because they follow momentum.

4. **Post prediction and topic prediction operate on fundamentally different mechanisms.** Post survival depends on engagement quality (Gini, velocity, comments). Topic growth depends on volume momentum (post count, upvotes, subreddit spread). Cross-level connection is weak (+0.013-0.034) with 12 days of data.

5. **First post's comment count is the bridge between levels.** A keyword whose first post generates extensive discussion (384 vs 179 comments) is more likely to become a topic (0.859 ROC with +0.034 improvement). Discussion drives topic spread.

6. **r/politics and r/news show inverse predictability.** Politics: best post prediction (81%), worst topic prediction (R2=0.551). News: worst post prediction (68%), best topic prediction (R2=0.808). Predictable mechanics, unpredictable events vs unpredictable posts, predictable momentum.

7. **Accuracy can be misleading.** At 7 days, 85% accuracy with 0.57 ROC — predicting "everything dies" scores high accuracy but has no discriminative power.

### Limitations

- **12 days of data** limits rare event detection (only 39 topic explosions in test set, 15 rocketing events)
- **Reddit public JSON** provides no reply threading (99% zeros), preventing argument depth analysis
- **Topic prediction mostly relies on counting** — quality signals exist in feature importance (10-15%) but don't significantly improve ROC
- **Cannot predict real-world events** — ceasefire proposals, court rulings, political scandals drive topic emergence beyond what engagement metrics capture
- **All findings correlational** — cannot establish that negative sentiment causes longer survival

---

## 6. Reflection on AI-Assisted Coding

### Where AI Was Effective
Claude Code excelled at rapid prototyping, algorithm selection (Random Forest, VADER, K-means), and boilerplate generation. The conversational workflow enabled rapid iteration. AI correctly suggested the Gini coefficient for upvote concentration, multi-horizon prediction, and co-occurrence pair analysis.

### Where AI Was Misleading
- **Task Scheduler crash**: WPF + unicode em dashes worked interactively, crashed in deployment
- **Timeout miscalculation**: 5 subreddits x 100 posts x 10 comments x 1.1s = 92 min, exceeding 30 min timeout
- **Misleading accuracy**: Initial 74.5% appeared good until feature importance showed 74% was comment counting
- **Premature ceiling claims**: AI declared "we've hit the ceiling" multiple times but developer pushed past each one, finding co-occurrence pairs, first-post comment signal, and the post-topic bridge

### Validation Approach
Execution testing, 5-fold cross-validation, **temporal validation** (trained on past, tested on future), feature importance analysis, confusion matrices, and cross-subreddit consistency checks. The temporal validation (0.838 ROC) is the most rigorous test — it proves the model works on unseen future data, not just shuffled historical data.

### Ethical Considerations
Responsible scraping (1.1s delays), aggregate analysis rather than individual profiling, transparent AI documentation.

---

## References

Bird, S., Klein, E. and Loper, E. (2009) *Natural Language Processing with Python*. O'Reilly Media.

Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proc. ICWSM*.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *JMLR*, 12, pp. 2825-2830.

---

## List of Figures

**Figure 1**: ROC curves for post survival prediction across 5 time horizons (1h-48h), showing predictability decay. File: `roc_prediction_decay.png`

**Figure 2**: Survival probability curves for 5 post archetypes with half-life annotations. File: `survival_probability_curves.png`

**Figure 3**: ROC curves + confusion matrix for 4-hour classifier. File: `roc_curves_all_horizons.png`

**Figure 4**: Subreddit lifecycle state distribution. File: `subreddit_state_mix.png`

**Figure 5**: Flow trajectories by subreddit. File: `flow_trajectory_by_subreddit.png`

**Figure 6**: Activity dashboard. File: `live_pulse_dashboard.png`

---

## Appendix: Scripts Created

| Script | Purpose |
|--------|---------|
| `analyze_sentiment.py` | VADER + K-means comment analysis |
| `predict_mood.py` | Sentiment-to-lifecycle correlation |
| `analyze_sentiment_trajectory.py` | Sentiment change over time |
| `analyze_comment_engagement.py` | Gini coefficient analysis |
| `analyze_post_timing.py` | Best posting hours |
| `analyze_velocity_curves.py` | Upvote velocity by subreddit |
| `analyze_cross_subreddit.py` | Story propagation detection |
| `predict_subreddit_direction.py` | Subreddit trend forecasting |
| `predict_post_outcome.py` | Single post predictor |
| `predict_time_to_death.py` | Time-to-death regression |
| `predict_crosspost_success.py` | Cross-posting success |
| `analyze_domains.py` | Link domain performance |
| `analyze_authors.py` | Author success rates |
| `analyze_keyword_trends.py` | Keyword frequency tracking |
| `analyze_title_style.py` | Title formatting impact |
| `predict_topic_popularity.py` | Topic volume prediction |
| `predict_topic_with_quality.py` | Quality-enhanced topic prediction |
| `predict_emerging_keywords.py` | Emerging keyword detection |
| `predict_topic_pct_change.py` | Topic percentage change prediction |
| `predict_emerging_keywords.py` | Emerging keyword detection |
| `test_temporal_pairs.py` | Temporal validation with topic pairs |
| `test_temporal_validation.py` | Temporal validation with keywords |
| `test_best_post_keyword.py` | Best-post vs average features for keywords |
| `test_sweet_spot.py` | 5-15 post keyword ratio features |
| `test_rising_to_keyword.py` | Post rising predictions as keyword features |
| `test_hourly_emergence.py` | Quality signals at different observation windows |
| `test_engagement_speed.py` | Engagement speed and acceleration |
| `test_organic_spread.py` | Organic spread: authors, title diversity, time |
