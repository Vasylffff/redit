Assessment 2 - Report Preview

# Predicting Reddit Topic Lifecycles:Emergence, Spread, and Death
Exploring AI: Understanding and Applications (SPC4004)
Assessment 2 — Code Generation Project
Vasyl Shcherbatykh
GitHub: https://github.com/Vasylffff/redit

# 1. Problem Definition & Dataset Justification

## What This Project Does
Every day, thousands of stories appear on Reddit. Some disappear within hours. Others spread across multiple communities, generate thousands of comments, and dominate the platform for days. This project asks: can we predict which stories will grow, how they will spread, and when they will die, using only early engagement signals?

The system tracks topics through their complete lifecycle: emergence (a new story appears in 1–3 posts), growth (it gains momentum), spread (it crosses into new subreddits), decline (engagement drops), and death (the story stops generating new posts). At each stage, a separate machine learning model makes predictions about what will happen next.

## The Machine Learning Tasks

The project applies classification and regression at two scales:

- **Post survival prediction** — will an individual post still be alive at 1h, 4h, 12h, 24h, 48h, or 7 days?
- **Post state detection** — is this post surging, dying, or about to change state?
- **Post flow modelling** — Markov chain transition probabilities between lifecycle states
- **Time-to-death regression** — how many hours until a post dies?
- **Sentiment and engagement analysis** — VADER comment scoring and Gini coefficient of comment upvote distribution
- **Topic emergence detection** — will a new word pair grow from 1–3 posts to 5+?
- **Topic lifecycle state prediction** — what state will a topic be in tomorrow?
- **Topic death and revival** — will a declining topic die permanently or come back?
- **Cross-subreddit spread** — will a topic appear in a new subreddit tomorrow?
- **Magnitude prediction (attempted)** — how large will a topic get? Tested with Szabo-Huberman log-linear models; did not achieve useful accuracy (R²=0.22)

Five classifiers were compared: Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, and Decision Tree, with 36 hyperparameter configurations tested.

## Why This Problem Matters
Understanding how topics emerge and die online has practical applications in several areas. News monitoring systems need to identify which stories are gaining traction before they peak. Content moderators need to anticipate which topics will spread across communities. Researchers studying misinformation need to understand how stories propagate between subreddits and whether spread patterns can be detected early.

From a machine learning perspective, the problem is interesting because it requires prediction at different timescales and granularities. Post-level prediction (will this individual post survive?) works well at short horizons but degrades beyond 24 hours. Topic-level prediction (will this story grow?) follows the opposite pattern: it improves at longer horizons because topics follow momentum while individual posts are chaotic. This inverse relationship between post and topic predictability was one of the project's key discoveries.

## Why This Dataset
Five subreddits were chosen to represent different community types: r/politics and r/news (fast-moving political discussion), r/worldnews (international affairs), r/technology (industry news), and r/Games (entertainment). This diversity tests whether the same models generalise across communities with different engagement patterns. For example, r/politics posts have an average lifecycle of 11 hours while r/Games posts survive 49 hours.

The dataset was self-collected rather than using an existing corpus because the project requires repeated observations of the same posts over time. Existing Reddit datasets typically provide single snapshots. Hourly collection over 13 days produced 216,944 post snapshots capturing how each post's engagement changed hour by hour.

| Metric | Value 
| Post snapshots | 216,944 
| Comment snapshots | 972,353 
| Unique posts tracked | 6,826 
| Collection frequency | Hourly 
| Raw JSON files | 5,140 
| Co-occurrence pairs analysed | 200,000+ per day 

[embedded image]
*Figure 1: Daily data collection coverage by subreddit across the 13-day collection period.*

# 2. Initial Code & Explanation of AI Use

## The Starting Point

The project began on 24 March 2026 with OpenAI Codex. The first task was getting Reddit data. Three approaches were attempted before finding one that worked:

**Attempt 1: PRAW (Reddit API).** Codex generated collect_reddit_data.py using the PRAW library, which requires Reddit API credentials. I submitted an API application on 24 March, providing a formal research proposal describing the project's aims. The application was not approved. Without credentials, the PRAW-based collector could not run.

**Attempt 2: Apify.** On 26 March, Codex suggested using Apify, a commercial web scraping service with a Reddit actor (trudax/reddit-scraper). This worked technically — it returned post and comment data — but the data format was inconsistent and did not provide the repeated hourly snapshots needed for trajectory tracking. I tested it across several subreddits but concluded it was not suitable for the project's requirements.

**Attempt 3: Free JSON endpoints.** I discovered that appending .json to any Reddit listing URL (e.g. reddit.com/r/technology/new.json) returns structured post metadata without authentication or rate limiting beyond basic politeness. I asked Claude Code whether this approach was viable for hourly collection and to write the collector. This produced collect_reddit_free.py, which became the sole data collection method for the rest of the project.

## Initial Generated Code

The first working version was minimal:

- **collect_reddit_free.py** — scrapes Reddit public JSON endpoints for new posts across five subreddits, with 1.1-second rate limiting
- **build_reddit_history.py** — merges all raw JSON snapshots into a unified timeline CSV
- A basic Windows Task Scheduler job to run collection hourly

At this stage there was no tracking pool — each hourly collection simply fetched whatever posts were currently listed as "new." There was no mechanism to follow the same post over time. If a post happened to appear in consecutive hourly scrapes, it would have multiple snapshots, but this was by luck rather than design. Many posts were observed only once. This created significant uncertainty about what kind of dataset was actually being built and whether it would be sufficient for trajectory prediction.

## AI Tools Used

Two AI tools were used throughout the project. Claude Code (Anthropic, Claude Opus 4.6) was the primary development tool, used for the majority of code generation, data collection infrastructure, comment scraping, analysis, modelling, and the complete topic lifecycle pipeline. OpenAI Codex assisted with initial setup, specific infrastructure tasks, and a major pipeline refactoring on 31 March that separated prediction tracking from live monitoring and redesigned the lifecycle state model. All development was conversational: I directed priorities, questioned results, and pushed exploration when AI suggested premature ceilings.

# 3. Critique of Initial Code, Iterative Development & Justification

With the initial code in place, the first priority was making it actually run. From there, each limitation was discovered and addressed in sequence — from basic bugs through to the complete topic lifecycle system.

## Phase 1: Getting It Running (March 24–26)

**Iteration 1: Fixing AI-generated bugs.** The initial code generated by AI could not run. datetime.UTC was used across 12 files but requires Python 3.11+ (the machine ran 3.10). f-string backslash syntax caused runtime failures. The Task Scheduler job crashed silently because AI generated WPF popup windows that fail in non-interactive mode. Unicode em dashes in the code corrupted under Windows cp1252 encoding. All of these required manual identification and correction before any data collection could begin.

**Iteration 2: Automated hourly collection.** The initial collector only fetched "new" posts, missing rising, hot, and top listings which capture posts at different lifecycle stages. A schedule system was built with five cadences: hourly (new), every 2 hours (rising), every 4 hours (hot), twice daily (top/day), and daily (top/week). Task Scheduler was installed to run collection automatically. Later in the project, a second machine was added and results were merged to fill gaps caused by laptop sleep and commuting.

## Phase 2: Post-Level Prediction (March 27 – April 1)

**Iteration 3: Tracking pool (Codex session 2).** A pool was added to follow the same posts over time. It was split into a fixed prediction cohort and a rolling live shortlist — without this, most posts were observed only once, making trajectory prediction unreliable. Dead posts were dropped via a three-tier system: active (hourly), dormant (6-hourly), dropped (archived).

**Iteration 4: History building and dead definition.** Raw snapshots were merged into a unified timeline. Gap patching fixed velocity corruption from missed collections — gaps showed zero velocity even for active posts, faking dead-post signals. The deeper problem was definitional: the initial "dead" label was too blunt and triggered during collection gaps. This led to the variance collapse method — a post is dead when its velocity variance drops sharply and stays low, more reliable than a simple threshold. The dying/dead distinction motivated adding "dying" as a separate early-warning state, so posts are flagged before they fully stop.

![](data/analysis/reddit/figures/fig_gap_problem.png)
*Figure: Collection gap problem. Left: velocity drops during a 4-hour gap, creating a false dead signal (348/hr to near-zero). Right: the post gained +1,344 upvotes during the same period.*

**Iteration 5: Markov chain predictor.** Given a post's current state (surging, alive, cooling, dying, dead), what is the probability of transitioning to each other state next hour? Transition matrices were computed from 116,000 observed transitions, conditioned on subreddit, age, and velocity. A three-level fallback handles sparse data: full key first, then dropping velocity, then global fallback. This was the first working prediction model — it could project 24 hours forward by chaining hourly transitions. However, Markov chains converge to equilibrium, so predictions beyond ~10 steps became indistinguishable. This motivated adding Random Forest classifiers for specific-horizon prediction, while the Markov predictor was retained for trajectory projection.

**Iteration 6: Sentiment analysis.** VADER sentiment scoring (Hutto and Gilbert, 2014) was applied to 972,353 comments. The result was counterintuitive: negative comment sentiment correlates with longer post survival (alive posts average −0.006 vs dead posts +0.045), suggesting controversy sustains engagement while apathy accelerates decline. However, the initial sentiment classifier achieved only 74.5% accuracy, and feature importance analysis revealed comment count alone accounted for 74% of predictions — the model had learned to count comments rather than analyse their sentiment, exposing a fundamental feature leakage problem.

**Iteration 7: Comment engagement features (Gini coefficient).** The Gini coefficient measures how comment upvotes are distributed within a post's discussion. High Gini (0.63–0.72) means a few comments dominate — community consensus. Low Gini (0.36) means diffuse, unfocused discussion. This became the strongest single post-level predictor at 46% feature importance, surpassing all velocity and upvote features. Classifier accuracy improved from 74.5% to 77.6% with this single feature.

**Iteration 8: Per-subreddit models.** The single model trained across all five subreddits achieved 64% accuracy. Breaking down performance by subreddit revealed that r/politics scored 72% while r/Games was much lower — accuracy correlated with both data volume and how structured the community's engagement patterns are. Training separate classifiers per subreddit improved r/politics to 81%. r/politics posts churn in 11 hours on average; r/Games posts survive 49 hours.

**Iteration 9: Survival prediction and multi-horizon extension.** A binary Random Forest classifier was trained to predict whether a post would still be alive one hour later, using velocity, comment count, Gini coefficient, sentiment, and age as features. This achieved 0.843 ROC AUC at 1 hour. The model was then extended to seventeen classifiers covering horizons from 1 hour to 7 days:

| Horizon | ROC AUC | Accuracy 
| 1 hour | 0.843 | 72% 
| 4 hours | 0.834 | 74% 
| 12 hours | 0.809 | 77% 
| 24 hours | 0.771 | 80% 
| 48 hours | 0.726 | 83% 
| 7 days | ~0.57 | 85% 

![](data/analysis/reddit/figures/fig14_multihorizon_roc_decay.png)

*Figure: ROC AUC decay across prediction horizons. Performance drops from 0.843 at 1 hour to ~0.57 at 7 days, approaching random chance.*

At 7 days, accuracy paradoxically rises to 85% while ROC drops to 0.57 — predicting "everything dies" achieves high accuracy but zero discriminative power. This demonstrates why ROC AUC is more appropriate than accuracy for imbalanced classification.

**Iteration 10: Surging and dead detection.** Specialised binary classifiers for specific states achieved the project's strongest post-level results: surging detection at 0.987 ROC AUC and dead detection at 0.945. State rise prediction achieved 0.947 ROC and barely decayed to 0.926 at 24 hours. Rise is more predictable than fall because it requires detectable signals; falling is the absence of signal.

**Iteration 11: Post outcome and time-to-death.** predict_post_outcome.py combined empirical growth multipliers per subreddit with state transition matrices from 116,000 observed transitions to produce pop/flop probabilities with estimated peak upvote range. predict_time_to_death.py achieved R²=0.459 with MAE of 9.1 hours. The top feature was states_seen_count (30%) — posts that had been through more state transitions were more predictable.

**Iteration 12: Cross-subreddit success prediction.** 1,305 detected cross-posts were analysed using title similarity matching. The politics→news route had 72% success rate. worldnews broke stories first 516 times. Median propagation time: 11.1 hours.

## Phase 3: Topic Lifecycle Prediction (April 7–8)

Post-level prediction worked well at short horizons but degraded beyond 24 hours. I wanted to predict *general* flow — not individual posts but the overall story. This motivated the shift to topic-level prediction, tracking how stories emerge and spread regardless of which individual posts carry them.

**Iteration 13: Co-occurrence pair detection.** Two-word pairs from the same title represent specific stories. "birthright+citizenship" is a Supreme Court case. "russian+tanker" is a naval incident. Temporal validation: 0.813 ROC AUC.

[embedded image]
*Figure 3: Topic lifecycle trajectories. Ongoing stories (Hormuz Strait) show repeated surges. One-shot events (Easter+Trump) spike once and die.*

**Iteration 14: Content-agnostic detection.** The same algorithm detected: "official+trailer" (122 posts, game announcements), "hormuz+strait" (252 posts, geopolitical crisis), "crimson+desert" (65 posts, game launch), "media+social" (127 posts, tech policy). The model operates on engagement patterns alone.

**Iteration 15: Growth peak detection.** Predict whether a topic has peaked or will keep growing. ROC AUC: 0.958. Dominant features: post growth day-over-day (52.7%) and upvote growth (38.1%).

**Iteration 16: Topic death prediction.** "Will this topic die tomorrow?" ROC AUC: 0.890. Top features: post count (46.1%), subreddit coverage (17.1%).

**Iteration 17: Death speed classification.** Quick death (0–1 days) vs slow death (2+ days). ROC AUC: 0.996. Decisive feature: decline rate on day after peak (67.8%).

[embedded image]
*Figure 4: Complete topic lifecycle prediction pipeline. Green = excellent (&gt;0.9), orange = good (&gt;0.8).*

**Iteration 18: Subreddit spread prediction.** Will a topic appear in a new subreddit tomorrow? r/politics: 0.756 ROC AUC. r/politics breaks stories first (47,580 times), not r/news. Most common route: news → worldnews (2,354 times).

[embedded image]
*Figure 5: Left: which subreddit breaks stories first. Right: most common cross-subreddit spread routes.*

**Iteration 19: Ongoing story vs one-shot event.** When a topic drops, will it come back? ROC AUC: 0.970. Strongest feature: multiple peaks (21.5% importance, 15.5x ratio).

| Signal | Ongoing story | One-shot event | Ratio 
| Multiple peaks | 0.9 | 0.1 | 15.5x 
| Consistency | 0.4 | 0.1 | 4.6x 
| Active days | 3.2 | 0.8 | 4.0x 
| Peak posts | 3.8 | 1.6 | 2.3x 

[embedded image]
*Figure 6: Feature ratios distinguishing ongoing stories from one-shot events.*

**Iteration 20: Topic death definition.** 1-day definition has 13.1% false-death rate. Topics peaking at 8–11 posts revive 44.8% of the time. Two-consecutive-day definition reduces false deaths to 6.8%.

[embedded image]
*Figure 7: Left: false death rates by definition. Right: revival rates by topic size.*

**Iteration 21: Noise filter.** 98.6% of word pairs never grow. At 99% confidence, filters out 87% of pairs with 99.5% precision. ROC AUC: 0.824.

**Iteration 22: Model comparison and hyperparameter tuning.**

[embedded image]
*Figure 8: Model performance across all tasks. Bold = best per task. No single model dominates.*

| Model | Default ROC | Tuned ROC | Improvement 
| Decision Tree | 0.577 | 0.842 | +0.265 
| Gradient Boosting | 0.713 | 0.835 | +0.122 
| Random Forest | 0.820 | 0.846 | +0.027 
| Logistic Regression | 0.859 | 0.860 | +0.001 

[embedded image]
*Figure 9: Gradient Boosting learning rate tuning. Peaks at 0.02, degrades at higher rates.*

# 4. Final Code Evaluation and Reflection

## Post-Level Prediction Results

Post-level models predict the trajectory of individual posts. These were developed first and provided the foundation for topic-level work.

| Task | ROC AUC | Key Feature 
| Surging detection | 0.987 | Upvote velocity 
| State rise prediction (1h) | 0.947 | Current velocity + comment rate 
| Dead detection | 0.945 | Velocity collapse 
| Survival (1h) | 0.843 | Gini coefficient (46%) 
| Survival (4h) | 0.834 | Gini + velocity 
| Survival (24h) | 0.771 | Per-subreddit models 
| Survival (7d) | ~0.57 | Base rate dominates 

Post prediction accuracy decays with time horizon: from 0.843 ROC at 1 hour to approximately 0.57 at 7 days. The Gini coefficient of comment upvote distribution was the strongest single feature at 46% importance. VADER sentiment analysis revealed that negative sentiment correlates with longer survival (alive average -0.006 vs dead +0.045), confirming that controversy drives engagement.

## Topic Lifecycle Pipeline Results

| Lifecycle Stage | Task | Best ROC AUC | Best Model 
| Filter | Discard noise (87% filtered) | 0.850 | Logistic Regression 
| Birth | Detect emerging topic (1-3 to 5+) | 0.860 | Logistic Regression 
| Growth | Has it peaked or still growing? | 0.958 | Gradient Boosting 
| Spread | Will it reach r/politics? | 0.756 | Random Forest 
| Decline | Topic dying state detection | 0.992 | Random Forest 
| Death | Will it die tomorrow? | 0.890 | Random Forest 
| Death speed | Quick death or slow death? | 0.999 | Logistic Regression 
| Revival | Ongoing story or one-shot? | 0.970 | Random Forest 

A key discovery was the inverse relationship between post and topic predictability. Individual posts become chaotic beyond 24 hours (ROC decays from 0.843 to 0.57). Topics become more predictable at longer horizons because they follow momentum. r/politics has the best post prediction (81%) but the worst topic prediction (R²=0.551). r/news is the reverse: worst posts (68%) but best topics (R²=0.808).

[embedded image]
*Figure 10: ROC curves comparing five classifiers on topic emergence detection.*

## Topic State Transitions

[embedded image]
*Figure 11: Topic state transition matrix. Surging topics have only 4% chance of remaining surging. Dead topics stay dead 94% of the time, but 6% revive.*

## What Does Not Work

**Magnitude prediction.** The system detects whether a topic will grow (0.86 ROC) but cannot predict how large. Szabo-Huberman log-linear model achieved R²=0.22. All regression approaches failed because at 1–3 posts, 99.5% of word pairs remain small.

**Exact timing.** "Days until death" gives R²=-0.5 (worse than guessing the mean). Binary classification works; regression does not.

**Revival timing.** "When will a dead topic come back?" achieves 0.578 ROC — barely above random. Revival depends on external events. However, predicting which topics are the type that revives works at 0.970 ROC.

## Limitations

- 13 days of data limits rare event learning (~62 topic growth events in test sets)
- Daily topic granularity; hourly resolution would provide more signal
- No NLP at topic level; ongoing stories distinguished by trajectory shape only
- Magnitude prediction unsolved from early signals
- All findings correlational, not causal

# 5. Reflection on AI-Assisted Coding

## Where AI Was Effective
AI tools enabled rapid prototyping across the entire pipeline. The initial collection system, history builder, and prediction framework were functional within the first session. Algorithm selection and boilerplate code generation were particularly effective.

## Where AI Was Misleading
**Default model assumptions.** Both Codex and Claude Code defaulted to Random Forest for every task. Logistic Regression outperforms it on the core task — not discovered until I explicitly requested comparison.
**Redundant feature suggestions.** Claude Code recommended post-level activity states for topic prediction. These showed no separation (alive ratio: 0.9x). The developer identified this as redundant.
**Premature ceiling claims.** AI declared ceilings multiple times. The developer pushed past each, discovering the lifecycle pipeline, spread prediction, and ongoing-vs-one-shot classification.
**Hyperparameter negligence.** Default parameters used throughout. Decision Tree improved +0.265 and Gradient Boosting +0.122 with basic tuning.

## Validation Approach

- Temporal validation: trained on days 1–8, tested on days 9–13
- Feature importance analysis to verify real patterns
- Multiple classifier comparison to avoid algorithm bias
- Testing established research models (Szabo-Huberman) against data
- Honest reporting of negative results

## Ethical Considerations

- 1.1-second rate limiting between API requests
- Aggregate analysis only; no individual user profiling
- Public data only via documented JSON endpoints
- Transparent documentation of all AI tools used

# References
Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proceedings of the International AAAI Conference on Web and Social Media*.
Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.
Szabo, G. and Huberman, B.A. (2010) 'Predicting the Popularity of Online Content', *Communications of the ACM*, 53(8), pp. 80-88.