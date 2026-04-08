# Predicting Reddit Topic Lifecycles: Emergence, Spread, and Death

**Module:** Exploring AI: Understanding and Applications (SPC4004)
**Assessment 2 -- Code Generation Project**
**GitHub:** https://github.com/Vasylffff/redit

---

## 1. Problem Definition & Dataset Justification

This project builds a complete topic lifecycle prediction system for Reddit. Rather than predicting individual posts, it tracks how *topics* -- represented as co-occurring word pairs in post titles -- emerge, spread across communities, and die. The system is entirely content-agnostic: the same algorithm detects a political scandal, a game launch, or a viral meme using only engagement signals, never reading the actual content.

The machine learning task combines multiple problems: binary classification (will a topic grow?), multi-class classification (what state will it be in tomorrow?), and regression (how many days until death?). Different algorithms proved appropriate for different lifecycle stages, and a key finding was that the simplest model -- Logistic Regression -- outperforms complex ensembles on the core detection task.

### Dataset

Entirely self-collected over 13 days (26 March -- 7 April 2026) using Reddit's public JSON endpoints across five subreddits: r/technology, r/news, r/worldnews, r/politics, and r/Games. Data was collected hourly via Windows Task Scheduler from two machines, then merged.

| Metric | Value |
|--------|-------|
| Post snapshots | 216,944 |
| Comment snapshots | 972,353 |
| Unique posts tracked | 6,826 |
| Collection frequency | Hourly |
| Raw JSON files | 5,140 |
| Co-occurrence word pairs analysed | 200,000+ per day |

Posts are tracked through lifecycle states -- surging, alive, cooling, dying, dead -- computed from upvote and comment velocity thresholds per subreddit. These serve as natural labels without manual annotation. Topics are represented as two-word pairs extracted from titles (e.g. "russian+tanker", "crimson+desert"), which capture specific stories rather than generic vocabulary (Szabo and Huberman, 2010).

*(Figure 11: Data Collection Coverage by Subreddit)*

---

## 2. Initial Code & Explanation of AI Use

The codebase was generated using Claude Code (Anthropic, Claude Opus 4.6). Initial prompt:

> "This is my project and I want you to create a folder and set this up so I could scrape Reddit from here"

This produced a data collection pipeline and a five-layer Markov chain post predictor using only upvote velocity. The initial system had no topic-level prediction, no comment engagement analysis, no cross-subreddit tracking, and no sentiment features. All subsequent development used Claude Code conversationally -- the developer directed priorities, questioned results, and pushed exploration beyond what the AI initially suggested was achievable.

---

## 3. Critique of Initial Code

### Bugs

- `datetime.UTC` incompatible with Python 3.10 across 12 files
- f-string backslash syntax errors
- scikit-learn version incompatibility
- Windows Task Scheduler crashed due to WPF popups in non-interactive mode
- Unicode em dashes corrupted under Windows cp1252 encoding

### Algorithmic Limitations

**No topic-level analysis.** The initial code predicted individual posts but had no concept of "topics" -- it could not detect that multiple posts about the same story were related.

**Single-keyword approach.** When topic analysis was first attempted, it used single keywords ("trump", "iran"), which are too generic. The word "trump" appears in 2,192 posts -- it is not a topic, it is noise. The shift to co-occurrence pairs ("russian+tanker", "birthright+citizenship") was a critical improvement that the AI did not suggest initially.

**Random Forest everywhere.** The initial code defaulted to Random Forest for every task. Testing five classifiers across seven tasks revealed that Logistic Regression outperforms Random Forest on the core emergence detection task (0.860 vs 0.829 ROC AUC), indicating the signal is fundamentally linear.

**No death definition analysis.** The initial code defined topic death as "<2 posts for 1 day". Analysis showed this produces a 13.1% false-death rate -- topics declared dead that actually revive. A 2-consecutive-day definition reduces this to 6.8%.

**Default hyperparameters.** All models used default settings. Tuning revealed that Gradient Boosting was severely misconfigured (depth=6 caused overfitting, ROC improved from 0.713 to 0.835 at depth=2), and Decision Tree improved from 0.577 to 0.842 by reducing depth from 8 to 5.

*(Figure 4: Overfitting -- Deeper Trees Perform Worse)*

---

## 4. Iterative Development & Justification

### Phase 1: Making It Work

**Iteration 1 -- Compatibility fixes.** Replaced `datetime.UTC` with `timezone.utc` in 12 files. Fixed f-string syntax. Pinned scikit-learn. Fixed Task Scheduler encoding issues. Verified: system collects data hourly without crashing.

### Phase 2: Post-Level Engagement

**Iteration 2 -- VADER sentiment.** Scored 972K comments. Found negative sentiment correlates with longer post survival (alive avg -0.006 vs dead +0.045). Controversy drives engagement.

**Iteration 3 -- Gini coefficient.** Comment upvote distribution (how concentrated vs diffuse) became the strongest post-level predictor at 46% feature importance. High Gini = community consensus = survival.

**Iteration 4 -- Per-subreddit models.** r/politics: 81% accuracy. r/Games: 64%. Each community has distinct engagement patterns.

### Phase 3: Post Prediction at Multiple Horizons

**Iteration 5 -- Multi-horizon classifiers.** Built 17 classifiers for 1h to 72h horizons. ROC AUC decays from 0.843 (1h) to 0.57 (7 days). Beyond 48 hours, accuracy paradoxically rises to 85% while ROC drops -- predicting "everything dies" gets high accuracy but zero discriminative power.

**Iteration 6 -- Surging and dead detection.** Surging detection: 0.987 ROC AUC. Dead detection: 0.945. These are the strongest post-level models.

### Phase 4: From Posts to Topics

**Iteration 7 -- Co-occurrence pairs.** Single keywords are too generic. Two-word pairs from the same title represent specific stories. "birthright+citizenship" is a Supreme Court case. "russian+tanker" is a naval incident. Temporal validation: 0.813 ROC AUC for detecting topics at 1-3 posts that will grow to 5+.

*(Figure 1: Topic Lifecycle Trajectories)*

**Iteration 8 -- Content-agnostic detection.** The same algorithm catches completely different types of content:
- `official+trailer` -- 122 posts, game announcements
- `hormuz+strait` -- 252 posts, geopolitical crisis
- `crimson+desert` -- 65 posts, game launch
- `media+social` -- 127 posts, tech policy debate

The model does not read content. It tracks two words appearing together, people engaging, posts multiplying. A viral meme and a war follow identical spread patterns from the model's perspective.

### Phase 5: Complete Topic Lifecycle

**Iteration 9 -- "Has it peaked?"** Given a topic's first two days of data, predict whether it has already peaked or will keep growing. ROC AUC: 0.958. The signal is simple: post_growth_d1_d2 (52.7% importance) and upvote_growth_d1_d2 (38.1%). If posts are increasing day-over-day, it has not peaked.

**Iteration 10 -- Topic death prediction.** "Will this topic die tomorrow?" ROC AUC: 0.890. Top features: current post count (46.1%), subreddit coverage (17.1%). Fewer posts and losing subreddit coverage predict death.

**Iteration 11 -- Quick vs slow death.** After a topic peaks, will it die in 0-1 days or survive 2+ days? ROC AUC: 0.996. The single feature that drives this: decline_rate_d1 (67.8% importance) -- how fast posts dropped the day after the peak.

*(Figure 5: Complete Topic Lifecycle Prediction Pipeline)*

**Iteration 12 -- Subreddit spread prediction.** If a topic exists in subreddit A, will it appear in subreddit B tomorrow? Per-target ROC AUC: r/politics 0.756, r/Games 0.679, r/news 0.661, r/worldnews 0.625.

Key finding: r/politics breaks stories first (47,580 times), not r/news. r/worldnews spreads fastest (avg 1.1 days from first appearance). The top cross-subreddit route is news -> worldnews (2,354 times).

*(Figure 8: Subreddit Spread Patterns)*

**Iteration 13 -- Ongoing vs one-shot classification.** When a topic drops, will it come back? The answer depends on whether it is an ongoing story or a one-shot event. ROC AUC: 0.970. The strongest signal: multiple_peaks (21.5% importance). An ongoing story has multiple surges and dips; a one-shot event has one peak and dies.

Profile comparison:
| Signal | Ongoing story | One-shot event | Ratio |
|--------|--------------|----------------|-------|
| Multiple peaks | 0.9 | 0.1 | 15.5x |
| Consistency | 0.4 | 0.1 | 4.6x |
| Active days | 3.2 | 0.8 | 4.0x |

*(Figure 12: Ongoing vs One-Shot Feature Ratios)*

**Iteration 14 -- Topic death definition.** Analysing 8,880 drop events revealed the original 1-day death definition has a 13.1% false-death rate. Bigger topics revive more often: topics peaking at 8-11 posts have a 44.8% revival rate vs 22.8% for small (3-4 peak) topics. Switching to 2 consecutive days reduces false deaths to 6.8%.

*(Figure 6: Revival Rates by Topic Size)*

**Iteration 15 -- Noise filter.** 98.6% of word pairs never grow. At 99% confidence, the model filters out 87% of all pairs as noise with 99.5% precision, leaving only ~5,000 candidates to monitor. ROC AUC: 0.824.

**Iteration 16 -- Model comparison and hyperparameter tuning.** Tested five classifiers (Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, Decision Tree) across all tasks with 36 hyperparameter configurations.

*(Figure 7: Model Performance Heatmap)*

---

## 5. Final Code Evaluation and Reflection

### The Complete Lifecycle Pipeline

*(Figure 5: Pipeline Summary)*

| Lifecycle Stage | Task | Best ROC AUC | Best Model |
|----------------|------|-------------|------------|
| Filter | Discard noise (87% filtered) | 0.850 | Logistic Regression |
| Birth | Detect emerging topic (1-3 -> 5+ posts) | 0.860 | Logistic Regression |
| Growth | Has it peaked or still growing? | 0.958 | Gradient Boosting |
| Spread | Will it reach r/politics? | 0.756 | Random Forest |
| Decline | Topic dying state detection | 0.992 | Random Forest |
| Death | Will it die tomorrow? | 0.890 | Random Forest |
| Death speed | Quick death (0-1d) or slow (2+d)? | 0.999 | Logistic Regression |
| Revival | Ongoing story or one-shot? | 0.970 | Random Forest |

### Model Comparison Results

*(Figure 2: ROC Curves -- Multi-Model Comparison)*

No single model dominates. Logistic Regression wins on emergence detection (0.860 vs RF 0.829), which means the topic emergence signal is fundamentally linear. Random Forest wins on tasks requiring complex feature interactions (death prediction, subreddit spread). Gradient Boosting is sensitive to hyperparameters -- depth=6 (default) gives 0.713, depth=2 gives 0.835.

*(Figure 10: Gradient Boosting Learning Rate Tuning)*

### Hyperparameter Tuning Impact

| Model | Default ROC | Tuned ROC | Improvement |
|-------|-----------|----------|-------------|
| Decision Tree | 0.577 | 0.842 | +0.265 |
| Gradient Boosting | 0.713 | 0.835 | +0.122 |
| Random Forest | 0.820 | 0.846 | +0.027 |
| Logistic Regression | 0.859 | 0.860 | +0.001 |

Key finding: overfitting is the dominant failure mode. Across all model families, shallower trees outperform deeper ones. Random Forest peaks at depth=4, Decision Tree at depth=5. Unlimited depth (RF depth=None) drops to 0.698.

*(Figure 4: Depth vs ROC -- Overfitting Analysis)*

### Topic State Transition Matrix

*(Figure 13: Topic State Transition Matrix)*

From observed transitions:
- Surging topics: 48% go stable next day, 30% die, only 4% stay surging
- Dead topics: 94% stay dead, but 6% revive
- Dying topics: 44% revive to stable (topic death is not permanent)

### What Doesn't Work

**Magnitude prediction.** We can detect whether a topic will grow (0.86 ROC) but cannot predict how much. Tested Szabo-Huberman log-linear model (R2=0.22), Random Forest regression (R2=0.16), power-law percentile ranges, and growth multipliers. All fail because at 1-3 posts, 99.5% of word pairs stay small -- the model cannot distinguish a future 10-post topic from a future 22-post topic.

**Exact timing.** "How many days until death?" gives R2=-0.5 (worse than guessing the mean). "How many days until peak?" gives R2=0.046. The binary question works (will it die? 0.89 ROC); the regression question does not.

**Revival timing.** "When will a dead topic come back?" gives 0.578 ROC -- barely above random. Revival depends on external real-world events (new developments in an ongoing story), not on the topic's own engagement metrics. However, predicting which topics *are the type* that revives works at 0.970 ROC.

### Limitations

- 13 days of data limits rare event learning (only ~62 topic growth events in test sets)
- Daily granularity -- hourly topic tracking would provide more signal
- No natural language understanding -- cannot distinguish ongoing stories from one-shot events by content, only by trajectory shape
- Magnitude prediction remains unsolved from early signals
- All findings are correlational, not causal

---

## 6. Reflection on AI-Assisted Coding

### Where AI Was Effective

Rapid prototyping of data collection, algorithm selection, and boilerplate generation. The conversational workflow enabled moving from idea to working analysis in minutes. Claude Code generated the initial pipeline, all analysis scripts, and figure generation code.

### Where AI Was Wrong or Misleading

**Default model assumptions.** AI defaulted to Random Forest for every task. Testing revealed Logistic Regression outperforms it on the core task -- the AI never suggested trying simpler models first.

**Activity state labels for topics.** AI suggested using post-level activity states (surging/alive/cooling/dying/dead) as features for topic prediction. These are discretised velocity values and carry zero signal at topic level (alive_ratio showed 0.9x ratio between growing and non-growing topics). The developer identified this as redundant information.

**Premature ceiling claims.** AI declared "we've hit the ceiling" multiple times. The developer pushed past each one, discovering the topic lifecycle pipeline, subreddit spread prediction, death definition analysis, and the ongoing-vs-one-shot classification.

**Hyperparameter negligence.** AI used near-default parameters throughout. Tuning revealed Decision Tree improved by +0.265 and Gradient Boosting by +0.122 -- significant improvements that were not attempted until the developer asked.

### How AI-Generated Code Was Validated

- Temporal validation (trained on past, tested on future) rather than cross-validation
- Feature importance analysis to verify models learn real patterns
- Confusion matrices to understand failure modes
- Multiple model comparison to avoid algorithm bias
- Testing established research models (Szabo-Huberman) against our data
- Honest reporting of what does not work alongside what does

### Ethical Considerations

- 1.1-second rate limiting between API requests
- Aggregate analysis only, no individual user profiling
- Public data only (Reddit JSON endpoints, no authentication required)
- Transparent documentation of AI tool usage throughout

---

## References

Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proceedings of the International AAAI Conference on Web and Social Media*.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.

Szabo, G. and Huberman, B.A. (2010) 'Predicting the Popularity of Online Content', *Communications of the ACM*, 53(8), pp. 80-88.

---

## Figures

**Figure 1:** Topic lifecycle trajectories showing ongoing stories vs one-shot events. `fig1_topic_trajectories.png`

**Figure 2:** ROC curves comparing five classifiers on topic emergence detection. `fig2_roc_emergence_models.png`

**Figure 3:** Feature importance for topic emergence detection (Random Forest). `fig3_feature_importance.png`

**Figure 4:** Overfitting analysis -- tree depth vs ROC AUC for Random Forest and Decision Tree. `fig4_depth_vs_roc.png`

**Figure 5:** Complete topic lifecycle prediction pipeline with ROC AUC scores. `fig5_pipeline_summary.png`

**Figure 6:** Topic death: false death rates by definition and revival rates by topic size. `fig6_revival_rates.png`

**Figure 7:** Model performance heatmap across all tasks and classifiers. `fig7_model_heatmap.png`

**Figure 8:** Subreddit spread patterns: who breaks stories first and top spread routes. `fig8_subreddit_spread.png`

**Figure 9:** Confusion matrix for emergence detection. `fig9_confusion_matrix.png`

**Figure 10:** Gradient Boosting learning rate tuning curve. `fig10_gbm_learning_rate.png`

**Figure 11:** Daily data collection coverage by subreddit. `fig11_data_coverage.png`

**Figure 12:** Ongoing vs one-shot topic feature ratios. `fig12_ongoing_vs_oneshot.png`

**Figure 13:** Topic state transition matrix. `fig13_transition_matrix.png`

**Figure 14:** Emergence detection ROC AUC at different growth targets. `fig14_target_vs_roc.png`
