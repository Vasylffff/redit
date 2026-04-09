# Predicting Reddit Post Survival Using Comment Engagement and Sentiment Analysis

**Module:** Exploring AI: Understanding and Applications (SPC4004)
**Assessment:** Code Generation Project
**GitHub:** https://github.com/Vasylffff/redit

---

## 1. Problem Definition & Dataset Justification

### The Problem

Every day, thousands of posts are submitted to Reddit. Some surge to the front page with tens of thousands of upvotes. Most die within hours with barely any engagement. The question this project seeks to answer is: **can we predict, based on a post's early signals, whether it will survive or die -- and how far into the future can that prediction remain reliable?**

This is framed as a **binary classification problem**: given a post's current state (upvotes, comments, velocity, age, comment sentiment, and comment engagement patterns), classify it as either **surviving** (will be in surging or alive state) or **dying** (will be in cooling, dying, or dead state) at a future time horizon.

The problem extends across multiple time horizons: 1 hour ahead, 4 hours, 12 hours, 24 hours, and 48 hours. This multi-horizon approach allows us to study how predictability itself decays over time -- a finding that proved to be one of the project's most significant contributions.

### Why This Problem Matters

Understanding post engagement dynamics has practical applications. Content creators want to know which posts to invest effort in promoting. News monitoring systems need to identify which stories are gaining traction. Researchers studying online discourse benefit from understanding what drives engagement versus apathy. From a machine learning perspective, it presents an interesting challenge: a time-series classification problem with naturally labelled data, high dimensionality, and an inherent stochastic component.

### The Dataset

The dataset was **entirely self-collected** using Reddit's public JSON endpoints (for example, `https://www.reddit.com/r/technology/new.json`). No API key or authentication was required. A custom scraping system was built that runs on Windows Task Scheduler, collecting data hourly across five subreddits.

**Subreddits tracked:** r/technology, r/news, r/worldnews, r/politics, r/Games

**Collection period:** 12 days (26 March to 6 April 2026)

**Final dataset size:**
- 185,048 post snapshots (hourly observations of the same posts over time)
- 677,569 comment snapshots with full text body
- 6,292 unique post lifecycles tracked from first appearance to death
- 2,174 subreddit-level health trend observations

**What each post snapshot contains:**
- Upvote count and upvote ratio at time of observation
- Comment count
- Upvote velocity (upvotes per hour)
- Comment velocity (comments per hour)
- Age of post in minutes
- Rank position within the subreddit listing
- Computed activity state (surging, alive, cooling, dying, dead)

**What each comment snapshot contains:**
- Full comment text body
- Comment upvote count
- Author username
- Whether it is a top-level comment or a reply
- Age of the comment at time of observation

### Why This Dataset?

This dataset was chosen over pre-existing datasets (such as those on Kaggle) for several reasons. First, it provides **naturally labelled training data**: lifecycle states are computed from empirical velocity thresholds per subreddit, providing ground truth labels without manual annotation. Second, the **time-series structure** enables trajectory prediction -- we observe the same post at multiple points in time, allowing models to learn from how posts evolve rather than just their static features at a single moment. Third, the **comment text** enables sentiment and engagement analysis that static post metadata alone cannot provide. Fourth, collecting our own data demonstrates understanding of the full machine learning pipeline from data acquisition through to prediction.

### Activity States Explained

Posts are classified into lifecycle states based on their upvote and comment velocity relative to subreddit-specific thresholds:

- **Surging**: Velocity significantly above the subreddit median. The post is gaining upvotes rapidly.
- **Alive**: Velocity above the minimum activity threshold. The post is actively being engaged with.
- **Cooling**: Velocity declining but still above zero. The post is losing momentum.
- **Dying**: Velocity near zero. Engagement has nearly stopped.
- **Dead**: No meaningful engagement detected across multiple snapshots.

These thresholds are computed empirically for each subreddit because engagement norms differ dramatically: a post gaining 5 upvotes per hour is healthy on r/Games but essentially dead on r/news.

---

## 2. Initial Code & Explanation of AI Use

### How the Initial Code Was Generated

The entire initial codebase was generated using **Claude Code**, Anthropic's command-line interface powered by the Claude Opus 4.6 language model. The development process was conversational: the developer described goals in natural language, and the AI generated working Python code.

The first prompt that initiated the project was:

> "This is my project and I want you to create a folder and set this up so I could scrape Reddit from here"

This produced:
- `collect_reddit_free.py` -- a Reddit scraper using public JSON endpoints with 1.1-second delay between requests for rate limiting
- `run_free_collection_schedule.py` -- a schedule runner that reads CSV configuration files to determine which subreddits and listing types to scrape at each hour
- `run_free_collection_window.ps1` -- a PowerShell wrapper for Windows Task Scheduler with a mutex to prevent concurrent runs
- `build_reddit_history.py` -- merges raw JSON scrapes into structured CSV history files
- `predict_post_flow.py` -- a five-layer Markov chain predictor

### The Initial Prediction System

The AI-generated predictor used five multiplicative layers:
1. **Baseline**: Markov chain transition probabilities computed from observed state changes
2. **Live Heat**: Adjusts predictions based on whether the subreddit is currently more or less active than its historical average
3. **Scenario**: Topic-specific adjustments
4. **Anchor**: Stabilises predictions against extreme values
5. **Discussion Quality**: A basic engagement score based on comment count

Each layer produces a multiplier that adjusts the probability distribution of next states. This is conceptually sound but, as the critique will discuss, limited in practice.

### AI Tools Used Throughout

- **Claude Code (Claude Opus 4.6)**: Primary tool for all code generation, debugging, analysis design, and iterative improvement
- **VADER Sentiment Analyser** (Hutto and Gilbert, 2014): Pre-trained rule-based model for scoring comment sentiment on a scale from -1 (most negative) to +1 (most positive)
- **scikit-learn 1.7** (Pedregosa et al., 2011): Random Forest classifiers, Decision Trees, Extra Trees, K-means clustering, and cross-validation utilities
- **matplotlib**: Chart and figure generation
- **pandas**: Data manipulation for visual report generation

All AI interactions were conversational, with the developer directing which analyses to perform and critically evaluating all outputs before incorporating them.

---

## 3. Critique of Initial Code

### Correctness and Compatibility Issues

The initial code contained three categories of bugs that prevented it from running on the development machine:

**1. Python version incompatibility.** The code used `from datetime import UTC`, which was introduced in Python 3.11. The development machine ran Python 3.10.6. This import error appeared in 12 different files, making the entire codebase non-functional without modification. This illustrates a common risk of AI-generated code: the model may generate code targeting a different Python version than the user's environment.

**2. f-string syntax error.** In `export_history_to_sqlite.py`, the code contained backslashes inside f-string expressions:
```python
insert_sql = f'INSERT INTO "{table_name}" ({", ".join(f"""\"{field}\"""" for field in fieldnames)}) VALUES ({placeholders})'
```
Python 3.10 does not allow backslashes inside f-string expressions. This had to be refactored to extract the column list to an intermediate variable.

**3. Dependency version mismatch.** The `requirements.txt` specified `scikit-learn>=1.8.0`, which requires Python 3.11+. This had to be pinned to `scikit-learn>=1.7.0,<1.8.0`.

### Algorithmic Limitations

**No sentiment analysis.** The initial predictor was entirely blind to the emotional content of discussions. A post with 100 supportive comments was treated identically to one with 100 angry comments. Given that comment sentiment proved to correlate significantly with post survival (negative sentiment posts live longer), this was a major missing feature.

**No comment engagement analysis.** The system counted comments but did not analyse their upvote patterns. As later iterations discovered, the distribution of upvotes across comments (measured by Gini coefficient) is the single strongest predictor of post survival -- stronger than sentiment, velocity, or comment volume.

**Single-step prediction only.** The Markov chain could predict the next state transition but could not forecast trajectories over multiple hours or days. Furthermore, the Markov chain's probability distributions converge to a fixed equilibrium after approximately 10 transitions regardless of starting state, making it fundamentally unable to discriminate between posts at longer horizons.

**No subreddit-specific models.** A single global model was used despite dramatically different engagement dynamics across subreddits. For example, r/politics posts have a median alive duration of just 11 hours with rapid churn, while r/Games posts live a median of 49 hours. Treating these identically introduces systematic prediction errors.

### Infrastructure Issues

The Task Scheduler automation had two critical bugs:

**1. WPF popup dependency.** The PowerShell script loaded `PresentationFramework` at startup to display Windows MessageBox popups showing collection results. When Windows Task Scheduler runs scripts non-interactively (no desktop session), this causes the entire script to crash before any data collection occurs.

**2. Aggressive error handling.** The script set `$ErrorActionPreference = "Stop"`, which treats any output to stderr as a terminating error. Python commonly writes warnings and progress information to stderr, causing the PowerShell script to abort on benign messages. Combined with the popup issue, this meant the automated collection system was completely non-functional -- the scheduler appeared to run every hour but produced no data.

### Code Structure Assessment

Positively, the initial code was well-organised with clear separation of concerns: collection, history building, health computation, prediction dataset construction, and prediction were each handled by separate scripts with a defined pipeline order. The configuration system using CSV schedule files was flexible and well-designed. However, the lack of modularity for adding new analysis layers meant that integrating sentiment analysis required modifying existing scripts rather than adding new pipeline stages cleanly.

---

## 4. Iterative Development & Justification

### Iteration 1: Compatibility and Infrastructure Fixes

**Problem addressed:** The entire codebase was non-functional on Python 3.10, and the Task Scheduler automation crashed silently.

**Changes made:**
- Replaced `datetime.UTC` with `timezone.utc` in 12 Python files
- Extracted f-string backslash expressions to intermediate variables in `export_history_to_sqlite.py`
- Pinned scikit-learn to version 1.7.x in requirements
- Removed WPF `PresentationFramework` dependency from `run_free_collection_window.ps1`, wrapping popups in try/catch so they display when interactive but silently skip in Task Scheduler
- Changed `$ErrorActionPreference` from "Stop" to "Continue"
- Replaced unicode em dash characters (`\u2014`) with ASCII dashes in string literals (Windows cp1252 encoding corrupted them)

**Why this improves the system:** Without these fixes, no code could execute and no data could be collected. The Task Scheduler fix was particularly critical because it enabled the 12-day automated data collection that underpins all subsequent analysis.

**Verification:** All scripts executed without import or syntax errors. The Task Scheduler completed an hourly collection cycle with exit code 0 and produced new JSON data files.

### Iteration 2: VADER Sentiment Integration

**Problem addressed:** The model had no understanding of comment sentiment. Whether discussions were supportive, hostile, or controversial was invisible to the predictor.

**Changes made:**
- Created `analyze_sentiment.py` integrating the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyser
- Scored all 677,569 comments with VADER's compound sentiment score (-1 to +1)
- Added five new columns to the prediction dataset in `build_prediction_dataset.py`:
  - `sentiment_mean_sample`: Average VADER score across comments
  - `sentiment_weighted_mean_sample`: Upvote-weighted average (comments with more upvotes contribute more)
  - `sentiment_positive_share_sample`: Proportion of comments with positive sentiment (>0.05)
  - `sentiment_negative_share_sample`: Proportion of comments with negative sentiment (<-0.05)
  - `sentiment_variance_sample`: Variance of sentiment scores (measures polarisation)
- Added Layer 6 (Sentiment Signal) to `predict_post_flow.py` using the weighted sentiment as a multiplier

**Key finding:** This iteration revealed a **counterintuitive result**: posts with negative comment sentiment survive longer than posts with positive sentiment. Across 3,520 posts:
- Alive/surging posts average sentiment: **-0.006** (slightly negative)
- Dying/dead posts average sentiment: **+0.045** (slightly positive)
- Difference: **-0.072**

This means controversial, angry discussions keep posts alive. Apathy -- not negativity -- kills posts. This finding was consistent across all five subreddits and became a central theme of the project.

**Model performance:** A Decision Tree classifier using sentiment features achieved **74.5% accuracy** (5-fold cross-validation). However, feature importance analysis revealed that `comment_count` dominated at 74% importance, meaning the model was essentially counting comments rather than analysing sentiment. This prompted the search for stronger engagement signals in Iteration 3.

**Verification:** Cross-referenced sentiment scores with lifecycle states across 3,520 posts. The negative correlation was statistically consistent across subreddits. The 74.5% accuracy was validated with 5-fold cross-validation to prevent overfitting.

### Iteration 3: Comment Upvote Gini Coefficient

**Problem addressed:** Sentiment features added only 2-3% accuracy above comment volume alone. A stronger engagement signal was needed.

**What is the Gini coefficient?** Originally an economics measure of income inequality, the Gini coefficient measures how concentrated a distribution is. Applied to comment upvotes:
- **High Gini (close to 1)**: A few comments receive most of the upvotes. The community has coalesced around specific viewpoints.
- **Low Gini (close to 0)**: Upvotes are spread evenly across all comments. No comment stands out; the discussion lacks focus.

**Changes made:**
- Created `analyze_comment_engagement.py` implementing Gini coefficient calculation on comment upvote distributions
- Computed per-post engagement features: Gini coefficient, top-comment share, top-3 endorsed sentiment, sentiment gap (top-upvoted vs rest), early vs late sentiment shift
- Built a new classifier incorporating these features

**Key finding:** The Gini coefficient became the **single strongest predictor** of post survival, with 46% feature importance -- exceeding raw sentiment, upvote count, comment volume, and all other features.

The data showed clear separation:
- **Surging/alive posts**: Gini 0.63-0.72 (concentrated upvotes, community consensus)
- **Dying posts**: Gini 0.36 (diffuse upvotes, no standout comments)
- **Dead posts**: Gini 0.58 (moderate -- some structure remains)

Additionally, dying posts showed the largest sentiment gap (-0.135): the most-upvoted comments were negative while the overall sentiment was neutral. The community was endorsing angry takes even when the majority of comments were balanced.

**Model performance:** Classifier accuracy improved from 74.5% to **77.6%** (5-fold cross-validation).

**Why this matters:** This finding -- that comment upvote concentration predicts post survival better than raw sentiment -- is an original contribution. It suggests that what matters for engagement is not whether people are happy or angry, but whether the discussion has **focus**: a few clear talking points that the community rallies around.

**Verification:** 5-fold cross-validation with consistent accuracy across folds. Feature importance analysis confirmed Gini as the dominant feature. The pattern held across all five subreddits.

### Iteration 4: Per-Subreddit Classification

**Problem addressed:** A single global model treats all subreddits equally despite dramatically different engagement norms and dynamics.

**Changes made:** Trained separate Random Forest classifiers for each subreddit using the same feature set.

**Results:**
| Subreddit | Accuracy | Character |
|-----------|----------|-----------|
| r/politics | 81% | Fast churn (11h median alive), predictable patterns |
| r/worldnews | 76% | News-driven, moderate predictability |
| r/technology | 70% | Mixed content, moderate predictability |
| r/news | 68% | Breaking news spikes make prediction harder |
| r/Games | 64% | Entertainment content, most unpredictable |

**Why this improves the system:** The 10-percentage-point improvement for r/politics (81% vs 71% global) demonstrates that community-specific models capture patterns that global models miss. The variation in accuracy itself is informative: politically focused communities follow more systematic engagement patterns (predictable controversy cycles) than entertainment communities (where a surprise game announcement can create unpredictable spikes).

**Verification:** Per-subreddit confusion matrices showed distinct error patterns reflecting each community's dynamics. For example, r/politics had fewer false positives (less false optimism) while r/Games had more (frequently predicted survival for posts that actually died).

### Iteration 5: Multi-Horizon Prediction

**Problem addressed:** All previous iterations could only predict one step ahead. Real-world utility requires knowing not just "will this post be alive in 1 hour" but "what's its trajectory over the next 2 days?"

**Changes made:**
- Built 17 separate Random Forest classifiers for prediction horizons from 1 hour to 72 hours
- Each model trained on 100,000-160,000 labelled samples (a snapshot at time T paired with the observed state at time T+horizon)
- Generated ROC curves for each horizon to visualise accuracy decay
- Computed survival probability curves by combining predictions across all horizons
- Calculated post "half-lives" (time until survival probability drops below 50%)

**Results -- ROC AUC by horizon:**
| Horizon | ROC AUC | Interpretation |
|---------|---------|----------------|
| 1 hour | 0.843 | Strong discrimination |
| 4 hours | 0.834 | Sweet spot -- best balance of accuracy and usefulness |
| 12 hours | 0.809 | Still solid |
| 24 hours | 0.771 | Declining |
| 48 hours | 0.726 | Approaching limits of predictability |
| 72 hours | Approaches baseline | Near-random |

**Post half-life estimates:**
| Post Type | Half-Life |
|-----------|-----------|
| Surging (500 up, 50 comments) | ~48 hours |
| Alive (100 up, 20 comments) | ~24 hours |
| Fresh (5 up, no comments) | ~10 hours |
| Cooling (50 up, 10 comments) | ~3 hours |
| Dying (20 up, 3 comments) | ~1 hour |

**Why this is significant:** The systematic decay of prediction accuracy is itself a scientific finding. It demonstrates that Reddit post trajectories have a **deterministic component** (captured by our features, dominant in the first 4-12 hours) and a **stochastic component** (which grows over time, making long-range prediction inherently unreliable). The boundary between predictable and chaotic falls at approximately 24 hours, beyond which the model's advantage over random guessing diminishes rapidly.

**Verification:** 5-fold cross-validation at each horizon produced consistent results. The decay curve is smooth and monotonic, suggesting it reflects genuine signal degradation rather than model artifacts.

### Additional Analysis Scripts Built

Beyond the core prediction pipeline, several supporting analyses were developed to provide context for the findings:

- **`analyze_post_timing.py`**: Best posting hours analysis. Found 08:00 UTC optimal (30.4% alive rate) and 03:00 UTC worst (57.9% dead rate).
- **`analyze_velocity_curves.py`**: Upvote velocity curves by subreddit and state. Posts gaining 500+ upvotes in the first hour have a 49% survival rate versus 17% for posts gaining 0-10.
- **`analyze_cross_subreddit.py`**: Detected 1,573 cross-posted stories. Worldnews breaks stories first (516 times). Median cross-subreddit propagation time: 11.1 hours.
- **`predict_subreddit_direction.py`**: Subreddit-level trend forecasting. Currently r/politics (+65, strong uptrend) and r/worldnews (+65), while r/Games (-50, mild decline).
- **`analyze_domains.py`**: Link domain performance analysis across 5,935 posts and 142 domains.
- **`analyze_authors.py`**: Author success rate analysis across 1,899 unique authors. Found that being prolific does not improve success rate (26% average for prolific authors, same as overall).
- **`analyze_keyword_trends.py`**: Keyword frequency tracking over 12 days with acceleration detection.
- **`analyze_title_style.py`**: Title formatting impact analysis. Shock/sensational words achieve 59.1% alive rate versus 23.8% baseline.
- **`predict_crosspost_success.py`**: Cross-posting success predictor. Posts alive in r/politics have 72% success rate when cross-posted to r/news.
- **`predict_time_to_death.py`**: Time-to-death regression model (R2=0.459, MAE 9.1 hours).

---

## 5. Final Code Evaluation and Reflection

### Performance Summary

The final system includes multiple prediction models evaluated with rigorous cross-validation:

**Classification Models (ROC AUC):**
| Model | 1h | 4h | 12h | 24h | 48h |
|-------|-----|-----|------|------|------|
| Post survival | 0.843 | 0.834 | 0.809 | 0.771 | 0.726 |
| Surging detection | 0.987 | -- | -- | -- | -- |

**Per-subreddit accuracy (rise/fall classification):**
| Subreddit | Accuracy |
|-----------|----------|
| r/politics | 81% |
| r/worldnews | 76% |
| r/technology | 70% |
| r/news | 68% |
| r/Games | 64% |

**Regression Models:**
| Target | R2 | MAE |
|--------|-----|-----|
| Time to death | 0.459 | 9.1 hours |
| Peak upvotes | 0.42 | varies by subreddit |
| Comment volume | 0.34 | 117 comments |

### Confusion Matrix Analysis (4-Hour Binary Classifier)

The confusion matrix for the best-performing binary model (4-hour horizon) reveals:

|  | Predicted Alive | Predicted Dead |
|--|----------------|----------------|
| **Actually Alive** | 16,570 (true positive) | 4,189 (false negative) |
| **Actually Dead** | 3,900 (false positive) | 8,057 (true negative) |

- **True positive rate (recall):** 80% -- the model catches 4 out of 5 surviving posts
- **True negative rate:** 67% -- the model catches 2 out of 3 dying posts
- **Primary failure mode:** False optimism (3,900 cases where the model predicted survival but the post died)

The false optimism bias reflects the training data's class imbalance: most snapshots are collected while posts are still active, so the model has seen more "alive" examples than "dead" ones.

### Key Findings

**1. Controversy drives engagement; apathy kills posts.**
Across 3,520 posts with comment data, negative comment sentiment correlates with longer post survival. The most active subreddit (r/politics, average sentiment -0.10) shows the strongest growth trajectory, while the most positive (r/Games, +0.38) is declining. This suggests that emotional intensity -- even negative intensity -- sustains audience attention.

**2. Comment upvote concentration is the strongest predictor.**
The Gini coefficient of comment upvote distributions outperforms raw sentiment, upvote count, comment volume, and velocity in predicting post survival. Posts where the community rallies around a few standout comments survive; posts with diffuse, unfocused discussion die.

**3. Predictability decays systematically over time.**
The ROC AUC curve from 0.843 (1h) to 0.726 (48h) quantifies the transition from deterministic to stochastic behaviour in Reddit engagement. Posts are highly predictable in the first 4-12 hours, but beyond 24 hours, inherent randomness dominates.

**4. Post half-life varies dramatically by starting state.**
Surging posts (high velocity, concentrated engagement) have a half-life of approximately 48 hours. Dying posts have approximately 1 hour. This provides actionable, quantitative estimates rather than vague qualitative assessments.

**5. Subreddit-specific models outperform global models.**
The 10-percentage-point improvement for r/politics demonstrates that community dynamics are sufficiently distinct to warrant separate models. A one-size-fits-all approach systematically misclassifies posts in communities with unusual engagement patterns.

### Remaining Limitations

**Data limitations:**
- Reddit's public JSON returns limited reply threading data (99% of reply counts are zero), preventing analysis of argument depth and conversation structure
- The 12-day collection window, while substantial, limits detection of weekly or seasonal patterns
- Only five subreddits were tracked; results may not generalise to smaller or differently structured communities

**Model limitations:**
- Regression models (predicting exact upvote counts) performed poorly (R2 = 0.42), confirming that the magnitude of viral success has a large random component even when survival/death is predictable
- The false optimism bias in the classifier could be addressed with class-weight balancing or oversampling, which was not implemented
- All findings are correlational; the project cannot establish that negative sentiment causes longer survival (the causal direction could be reversed -- popular posts attract more negative comments)

**Technical limitations:**
- The system requires a continuously running machine with internet access for data collection
- No real-time prediction interface was built; analyses are run as batch scripts
- Comment sentiment scoring with VADER treats all comments equally regardless of their position in the conversation thread

---

## 6. Reflection on AI-Assisted Coding

### Where AI Was Effective

Claude Code excelled at several aspects of the development process:

**Rapid prototyping.** The conversational workflow allowed moving from a verbal description of an analysis to working, executable code within minutes. For example, describing "I want to know if comment upvote concentration predicts post survival" produced a complete analysis script with Gini coefficient calculation, classifier training, and cross-validation in a single generation.

**Algorithm selection.** The AI correctly recommended Random Forest classifiers for tabular data with mixed feature types, VADER for pre-trained sentiment analysis without training data, and K-means for unsupervised comment clustering. These choices were appropriate for the data characteristics and problem structure.

**Boilerplate handling.** Data pipeline code (CSV I/O, SQLite exports, matplotlib configuration, cross-validation setup) was generated reliably, freeing the developer to focus on analysis design and interpretation.

**Novel suggestions.** The AI suggested using the Gini coefficient for upvote concentration analysis, which became the project's most significant finding. It also suggested multi-horizon prediction as a natural extension, leading to the predictability decay analysis.

### Where AI Was Misleading or Required Correction

**1. The Task Scheduler crash.** The AI generated a PowerShell script with WPF dependencies and unicode em dashes that worked perfectly in interactive testing but crashed silently when run by Task Scheduler. This is a class of bug that AI is particularly prone to: the code is syntactically correct and works in the developer's environment but fails in the deployment environment due to assumptions about runtime context.

**2. Timeout miscalculation.** When asked to add comment collection to the hourly schedule, the AI set `max_comments=10` without calculating the total request time: 5 subreddits x 100 posts x 10 comments x 1.1s delay = 92 minutes, far exceeding the 30-minute timeout. This required two rounds of correction (reducing to 5, then to 0) before the schedule completed within time limits.

**3. Misleading accuracy.** The initial sentiment model reported 74.5% accuracy, which appeared to validate the approach. However, critical examination of feature importances revealed that `comment_count` contributed 74% of predictive power -- the model was essentially counting comments rather than performing sentiment analysis. Without human scrutiny of the feature importances, this misleading result would have been accepted as evidence that sentiment analysis works for prediction.

**4. Python version assumptions.** The AI generated code targeting Python 3.11 features despite the development machine running Python 3.10. This affected 12 files and required systematic replacement.

### How AI-Generated Code Was Validated

A multi-layered validation approach was employed:

1. **Execution testing:** Every generated script was run immediately to catch syntax and runtime errors
2. **Cross-validation:** All classifiers were evaluated with 5-fold cross-validation to prevent overfitting to training data
3. **Feature importance analysis:** After every model was trained, feature importances were examined to ensure the model was learning from meaningful patterns rather than spurious correlations or data leakage
4. **Confusion matrix inspection:** Understanding not just accuracy but the specific types of errors the model makes
5. **Sanity checking findings:** Counterintuitive results (like negative sentiment correlating with survival) were cross-referenced across subreddits and time periods to confirm they were genuine patterns rather than artifacts

### Ethical and Professional Considerations

**Data collection ethics.** The scraper uses Reddit's public JSON endpoints with a 1.1-second delay between requests, complying with rate-limiting norms. No authentication or API abuse is involved. Data collection focuses on publicly visible post and comment metadata.

**Privacy.** While author usernames are collected as part of post metadata, the analysis focuses on aggregate patterns (average behaviour of post types, subreddit-level trends) rather than individual user profiling. No personally identifiable information beyond public Reddit usernames is used or shared.

**AI transparency.** All AI tool usage is documented in this report. The GitHub repository contains both the initially generated code and the final refined version, allowing full traceability of AI contributions versus human modifications.

**Limitations of automated analysis.** The sentiment analysis assigns scores to text without understanding context, sarcasm, or cultural nuance. Results should be interpreted as approximate measures of emotional valence rather than precise assessments of human sentiment.

---

## References

Bird, S., Klein, E. and Loper, E. (2009) *Natural Language Processing with Python*. Sebastopol: O'Reilly Media.

Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media*. Ann Arbor, MI, June 2014.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.

---

## List of Figures

**Figure 1: ROC Prediction Decay Curves.** ROC curves for post survival prediction across five time horizons (1h, 4h, 12h, 24h, 48h). The systematic curve degradation from green (1h, AUC=0.843) to red (48h, AUC=0.726) demonstrates the transition from deterministic to stochastic post behaviour. The dashed diagonal represents random guessing (AUC=0.500). File: `roc_prediction_decay.png`

**Figure 2: Post Survival Probability Curves.** Survival probability over 72 hours for five post archetypes: surging, alive, cooling, dying, and fresh. The horizontal dashed line marks the 50% half-life threshold. Surging posts maintain above 50% for approximately 48 hours while dying posts drop below 50% within 1 hour. File: `survival_probability_curves.png`

**Figure 3: ROC Curves and Confusion Matrix.** Left panel: ROC curves for all prediction horizons overlaid for comparison. Right panel: Confusion matrix for the 4-hour binary classifier, showing 16,570 true positives, 8,057 true negatives, 3,900 false positives (false optimism), and 4,189 false negatives. File: `roc_curves_all_horizons.png`

**Figure 4: Subreddit State Distribution.** Stacked bar chart comparing the distribution of post lifecycle states (surging, alive, cooling, dying, dead) across all five tracked subreddits, revealing distinct community engagement profiles. File: `subreddit_state_mix.png`

**Figure 5: Lifecycle Flow Trajectories.** Sankey-style flow diagrams showing how posts transition between lifecycle states in each subreddit, with arrow width proportional to transition frequency. File: `flow_trajectory_by_subreddit.png`

**Figure 6: Live Activity Dashboard.** Dashboard showing current engagement metrics, post volumes, and computed health scores for all monitored subreddits at time of generation. File: `live_pulse_dashboard.png`
