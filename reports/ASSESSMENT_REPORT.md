# Exploring AI: Understanding and Applications (SPC4004)
## Assessment 2 -- Code Generation Project

**GitHub Repository:** https://github.com/Vasylffff/redit

---

## 1. Problem Definition & Dataset Justification

### Machine Learning Task

This project addresses a **binary classification problem**: predicting whether a Reddit post will remain active (surging or alive) or decay (cooling, dying, or dead) over different time horizons, from 1 hour to 72 hours ahead. The system additionally performs sentiment analysis, engagement prediction, and subreddit-level trend forecasting.

Reddit posts follow a lifecycle: they emerge, potentially surge in popularity, maintain activity, then gradually cool and die. The ability to predict this trajectory has practical applications in content strategy, news monitoring, and understanding online engagement dynamics.

### Dataset

The dataset is **self-collected** using Reddit's public JSON endpoints (e.g., `https://www.reddit.com/r/{subreddit}/new.json`), requiring no API key. Data was collected over **12 days** (26 March -- 6 April 2026) across five subreddits: r/technology, r/news, r/worldnews, r/politics, and r/Games.

**Dataset statistics:**
- 185,048 post snapshots (hourly observations of post state)
- 677,569 comment snapshots with full text
- 6,292 unique post lifecycles tracked
- 2,174 subreddit-level health trend observations

Each post snapshot includes: upvote count, comment count, upvote velocity, age, rank position, and activity state. Comment snapshots include: body text, upvote count, author, and reply count. This time-series structure allows tracking how posts and their discussions evolve over time.

The dataset was chosen because it provides a real-world, naturally labelled classification problem: post lifecycle states (surging, alive, cooling, dying, dead) are computed from velocity thresholds, providing ground truth labels without manual annotation.

---

## 2. Initial Code & Explanation of AI Use

### Starting Point

The initial codebase was generated using **Claude Code** (Anthropic's CLI for Claude, powered by Claude Opus 4.6). The starting prompt was:

> *"This is my project and I want you to create a folder and set this up so I could scrape Reddit from here"*

The AI generated the data collection pipeline (`collect_reddit_free.py`), schedule runner, and a 5-layer Markov chain predictor (`predict_post_flow.py`). The initial system could collect Reddit data and predict post state transitions using baseline transition probabilities, but had no sentiment analysis, no comment engagement features, and no multi-horizon prediction capability.

### AI Tools Used

- **Claude Code (Claude Opus 4.6)** -- primary tool for code generation, debugging, and iterative development throughout the project
- **VADER Sentiment Analyzer** -- pre-trained NLP model for comment sentiment scoring (Bird et al., 2009)
- **scikit-learn 1.7** -- machine learning library for Random Forest classifiers, Decision Trees, and K-means clustering

All code generation was performed through conversational prompts, with the AI suggesting approaches and the developer directing which analyses to pursue and evaluating results.

---

## 3. Critique of Initial Code

### Correctness Issues

The initial code contained several compatibility bugs:
- **Python 3.10 incompatibility**: The code used `datetime.UTC` (introduced in Python 3.11) across 12 files. This caused `ImportError` on the development machine running Python 3.10.6.
- **f-string syntax error**: `export_history_to_sqlite.py` used backslashes inside f-string expressions, which Python 3.10 does not permit.
- **Dependency version mismatch**: `requirements.txt` specified `scikit-learn>=1.8.0`, which requires Python 3.11+.

### Algorithm Limitations

The initial predictor used a **5-layer Markov chain** that only considered upvote and comment velocity to predict state transitions. This approach had fundamental limitations:

1. **No sentiment analysis** -- the model was blind to whether comments were positive, negative, or controversial. A post with 100 angry comments was treated identically to one with 100 supportive comments.
2. **No comment engagement features** -- comment upvote patterns (concentration, dominance) were ignored entirely.
3. **Single-step prediction only** -- could only predict the next state transition, not the trajectory over hours or days.
4. **No per-subreddit models** -- used one global model despite vastly different dynamics across subreddits (e.g., r/politics posts churn in 11 hours vs r/Games posts lasting 49 hours).

### Code Structure

The initial code was well-structured with clear separation of concerns (collection, history building, prediction). However, it lacked modularity for adding new analysis layers, and the pipeline runner script (`run_free_collection_window.ps1`) used WPF MessageBox popups that crashed when run non-interactively by Windows Task Scheduler.

---

## 4. Iterative Development & Justification

### Iteration 1: Compatibility Fixes

**Problem:** Code could not run on Python 3.10.
**Change:** Replaced `datetime.UTC` with `timezone.utc` in 12 files, extracted f-string backslash expressions to variables, pinned scikit-learn to version 1.7.x.
**Verification:** All scripts executed without import errors.

### Iteration 2: VADER Sentiment Analysis

**Problem:** The model had no understanding of comment sentiment.
**Change:** Created `analyze_sentiment.py` integrating the VADER sentiment analyzer (Hutto & Gilbert, 2014) to score 677,569 comments. Added 5 sentiment columns to the prediction dataset: mean, weighted mean, positive share, negative share, and variance.
**Result:** Discovered that **negative sentiment correlates with longer post survival** (alive posts avg sentiment -0.006 vs dead posts +0.045). This counterintuitive finding -- that controversy drives engagement -- became a central theme.
**Verification:** Cross-referenced sentiment scores with lifecycle states across 3,520 posts.

### Iteration 3: K-means Comment Clustering

**Problem:** VADER provides a single sentiment score but misses thematic patterns.
**Change:** Applied K-means clustering (k=5) on TF-IDF vectorized comment text to discover natural comment archetypes.
**Result:** Identified 5 distinct clusters: casual discussion, political debate, war/geopolitics commentary, meta/moderation, and tech/AI discussion. Each cluster showed different sentiment distributions.
**Verification:** Cluster centroids showed coherent topic groupings with distinct average sentiment scores.

### Iteration 4: Comment Upvote Gini Coefficient

**Problem:** Sentiment alone only added ~2-3% accuracy over comment volume. Needed a stronger engagement signal.
**Change:** Implemented Gini coefficient analysis on comment upvote distributions in `analyze_comment_engagement.py`. The Gini coefficient measures how concentrated upvotes are -- do a few comments dominate (high Gini) or are upvotes spread evenly (low Gini)?
**Result:** **Gini coefficient became the #1 predictor** (46% feature importance). Surging/alive posts have high Gini (0.63-0.72) -- a few winning comments capture community attention. Dying posts have low Gini (0.36) -- no comment stands out.
**Verification:** Classifier accuracy improved from 74.5% to 77.6% with Gini features (5-fold cross-validation). This finding -- that upvote concentration predicts survival better than raw sentiment -- was the project's most significant discovery.

### Iteration 5: Multi-Horizon Prediction

**Problem:** The model could only predict one hour ahead. Real-world utility requires longer-range forecasts.
**Change:** Built separate Random Forest classifiers for 17 prediction horizons (1h to 72h), training each on 100K-160K samples. Created survival probability curves showing the decay of predictability over time.
**Result:** ROC AUC degradation curve: 0.843 (1h) -> 0.834 (4h) -> 0.809 (12h) -> 0.771 (24h) -> 0.726 (48h). Computed **post half-lives**: surging posts survive ~48h, alive ~24h, cooling ~3h, dying ~1h.
**Verification:** 5-fold cross-validation at each horizon with consistent results. The decay curve itself is a finding: Reddit post trajectories become inherently unpredictable beyond ~24 hours.

### Iteration 6: Per-Subreddit Models

**Problem:** A global model treats all subreddits equally despite different engagement dynamics.
**Change:** Trained separate classifiers for each subreddit.
**Result:** r/politics achieved 81% accuracy (vs 71% global), while r/Games only reached 64% -- confirming that some communities are more predictable than others.
**Verification:** Per-subreddit confusion matrices showed distinct error patterns reflecting each community's dynamics.

---

## 5. Final Code Evaluation and Reflection

### Model Performance Summary

The final system includes multiple prediction models, each evaluated with 5-fold cross-validation:

| Model | Metric | Score |
|-------|--------|-------|
| Surging prediction (1h) | ROC AUC | 0.987 |
| Post survival (1h) | ROC AUC | 0.843 |
| Post survival (4h) | ROC AUC | 0.834 |
| Post survival (12h) | ROC AUC | 0.809 |
| Post survival (24h) | ROC AUC | 0.771 |
| Post survival (48h) | ROC AUC | 0.726 |
| Rise/fall (r/politics) | Accuracy | 81% |
| Time-to-death | R2 | 0.459 |

Figure 1 shows ROC curves for all prediction horizons, demonstrating the systematic decay of prediction accuracy over time. Figure 2 shows survival probability curves for different post types, illustrating how initial engagement predicts long-term trajectory.

### Key Findings

1. **Controversy drives engagement**: Posts with negative comment sentiment survive longer than positive ones. The most active subreddit (r/politics, sentiment -0.10) is growing, while the most positive (r/Games, +0.38) is declining.

2. **Comment upvote concentration is the strongest predictor**: The Gini coefficient of comment upvotes outperforms raw sentiment, upvote count, and comment volume in predicting post survival.

3. **Predictability decays over time**: Post trajectories are highly predictable within 4 hours (AUC 0.834) but approach randomness by 48 hours (AUC 0.726), suggesting an inherent chaotic component to Reddit engagement.

4. **Post half-life varies by state**: Surging posts have a half-life of ~48 hours; dying posts ~1 hour. This provides actionable estimates for content monitoring.

### Confusion Matrix Analysis

The 4-hour binary classifier (Figure 3) correctly identifies 80% of surviving posts and 67% of dying posts. The main failure mode is **false optimism** -- predicting posts will survive when they actually die (3,900 false positives out of 32,381 test samples). This bias reflects the dataset's class imbalance: most snapshots occur while posts are still alive.

### Limitations

- **Reply depth unavailable**: Reddit's public JSON returns limited reply count data (99% zeros), preventing analysis of argument thread structure.
- **12 days of data**: Longer collection periods would improve temporal patterns and reduce model variance.
- **Regression models underperform**: Predicting exact upvote counts (R2=0.42) is substantially harder than state classification, suggesting Reddit virality has a significant random component.
- **No causal claims**: The correlations found (e.g., negative sentiment predicting survival) may reflect confounding factors rather than causal relationships.

---

## 6. Reflection on AI-Assisted Coding

### Where AI Was Useful

Claude Code was highly effective for **rapid prototyping**: generating analysis scripts, suggesting appropriate ML algorithms (Random Forest for tabular data, VADER for sentiment), and debugging compatibility issues. The conversational workflow allowed iterating quickly -- describing a finding verbally and having working code within minutes.

The AI correctly suggested using Gini coefficient for upvote concentration, which became the project's strongest finding. It also efficiently handled boilerplate tasks: SQLite exports, CSV I/O, cross-validation setup, and matplotlib chart generation.

### Where AI Was Misleading or Required Correction

Several AI-generated components required human intervention:

1. **Task Scheduler crash**: The generated PowerShell script used `$ErrorActionPreference = "Stop"` which treated Python stderr warnings as fatal errors, and included WPF popups that crash in non-interactive sessions. Required manual debugging to identify.
2. **Unicode encoding**: AI-generated em dashes in string literals corrupted under Windows cp1252 encoding, causing parser errors. A subtle bug that only appeared at runtime.
3. **Timeout miscalculation**: AI suggested collecting 10 comments per post on the hourly schedule without calculating that 5 subreddits x 100 posts x 10 comments x 1.1s delay = 90+ minutes, exceeding the 30-minute timeout.
4. **Over-optimistic initial sentiment model**: The first mood predictor reported 74.5% accuracy, but feature importance analysis revealed comment count (74% importance) dominated, meaning the model was essentially counting comments rather than analysing sentiment. This required critical evaluation to identify.

### Validation Approach

All AI-generated code was validated through: (a) running scripts and checking output against expected values, (b) cross-validation with held-out test sets, (c) feature importance analysis to ensure models captured meaningful patterns rather than artifacts, and (d) confusion matrices to understand specific failure modes.

### Ethical Considerations

The project scrapes publicly available Reddit data without authentication. While this is permitted by Reddit's public JSON endpoints, ethical considerations include: user privacy (usernames are collected but not the focus of analysis), rate limiting compliance (1.1s delay between requests), and responsible data handling (no personal data is shared or published). The project analyses aggregate patterns rather than individual behaviour.

---

## References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
- Hutto, C.J. & Gilbert, E.E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *Proceedings of ICWSM*.
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
- Reddit Public JSON API. Available at: https://www.reddit.com/dev/api/

---

## Figures

**Figure 1:** ROC curves showing prediction accuracy decay across 5 time horizons (1h to 48h). Located at `data/analysis/reddit/visuals/roc_prediction_decay.png`

**Figure 2:** Survival probability curves for 5 post types, showing estimated half-life for each. Located at `data/analysis/reddit/visuals/survival_probability_curves.png`

**Figure 3:** Confusion matrix for the 4-hour binary classifier. Located at `data/analysis/reddit/visuals/roc_curves_all_horizons.png`

**Figure 4:** Subreddit state distribution comparison. Located at `data/analysis/reddit/visuals/subreddit_state_mix.png`

**Figure 5:** Post lifecycle flow trajectories by subreddit. Located at `data/analysis/reddit/visuals/flow_trajectory_by_subreddit.png`

**Figure 6:** Live pulse dashboard showing current activity levels. Located at `data/analysis/reddit/visuals/live_pulse_dashboard.png`
