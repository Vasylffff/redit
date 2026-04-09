# Predicting Reddit Post Survival and Emerging Topic Detection

**Module:** Exploring AI: Understanding and Applications (SPC4004)
**Assessment 2 -- Code Generation Project**
**GitHub:** https://github.com/Vasylffff/redit

---

## 1. Problem Definition & Dataset Justification

This project builds a system that does two things:

1. **Detects new topics** as they emerge on Reddit from their first 1-3 posts
2. **Predicts the trajectory** of every post within those topics -- will it surge, survive, or die?

Think of it as a radar: it spots incoming stories (topic detection), then tracks each one in detail (post prediction).

### Dataset

Self-collected over 13 days (26 March -- 7 April 2026) using Reddit's public JSON endpoints across five subreddits (r/technology, r/news, r/worldnews, r/politics, r/Games). No API key required.

| Metric | Value |
|--------|-------|
| Post snapshots | 348,505 |
| Comment snapshots | 1,576,702 |
| Post lifecycles | 6,700 |
| Collection | Hourly via Windows Task Scheduler |

Posts are tracked through lifecycle states -- surging, alive, cooling, dying, dead -- computed from upvote and comment velocity thresholds per subreddit. These serve as natural labels without manual annotation.

### Machine Learning Approaches Used

The project applies multiple ML methods: Markov chain transition modelling, binary and multi-class classification (Random Forest, Extra Trees, Decision Trees), regression (Random Forest, Ridge), unsupervised clustering (K-means on TF-IDF comment vectors), and rule-based NLP (VADER sentiment scoring).

---

## 2. Initial Code & Explanation of AI Use

The codebase was generated using Claude Code (Anthropic, Claude Opus 4.6). Initial prompt:

> "This is my project and I want you to create a folder and set this up so I could scrape Reddit from here"

This produced a data collection pipeline and a five-layer Markov chain predictor using only upvote velocity. No sentiment analysis, no comment engagement features, no topic-level prediction.

All development used Claude Code conversationally. The developer directed priorities, questioned results, and repeatedly pushed exploration beyond the AI's initial assessments of what was achievable.

---

## 3. Critique of Initial Code

**Bugs:** `datetime.UTC` (Python 3.11) across 12 files on a Python 3.10 machine. f-string backslash syntax error. scikit-learn version incompatibility. Task Scheduler crashed due to WPF popups in non-interactive mode and unicode em dashes corrupted under Windows cp1252 encoding.

**Algorithmic limitations:** No sentiment analysis -- treated 100 angry comments the same as 100 supportive ones. No comment engagement analysis -- the Gini coefficient, later found to be the strongest predictor, was completely ignored. Single-step prediction only -- Markov chain converges to equilibrium, cannot discriminate beyond ~10 transitions. No per-subreddit models despite r/politics churning in 11 hours vs r/Games living 49 hours. No topic-level analysis at all.

---

## 4. Iterative Development

### Phase 1: Making It Work

**Iteration 1 -- Compatibility fixes.** Replaced `datetime.UTC` with `timezone.utc` in 12 files. Fixed f-string syntax. Pinned scikit-learn 1.7. Fixed Task Scheduler (removed WPF, fixed encoding, changed error policy). Result: system runs and collects data hourly.

### Phase 2: Understanding Post Engagement

**Iteration 2 -- VADER sentiment.** Scored 1.58 million comments. Found that negative sentiment correlates with longer survival (alive avg -0.006 vs dead +0.045). Controversy drives engagement; apathy kills posts. Initial classifier: 74.5% accuracy -- but feature importance showed comment count at 74%, meaning the model was counting, not analysing sentiment.

**Iteration 3 -- Gini coefficient.** Measured how comment upvotes are distributed. High Gini (0.63-0.72) means a few comments dominate -- community consensus. Low Gini (0.36) means diffuse attention -- unfocused discussion. Became the strongest predictor at 46% feature importance. Classifier improved to 77.6%.

**Iteration 4 -- Per-subreddit models.** r/politics: 81% accuracy. r/Games: 64%. Each community has distinct patterns. Politically focused communities are more systematic; entertainment communities are more random.

### Phase 3: Predicting the Future

**Iteration 5 -- Multi-horizon prediction.** Built 17 classifiers for 1h to 72h horizons:

| Horizon | ROC AUC |
|---------|---------|
| 1 hour | 0.843 |
| 4 hours | 0.834 |
| 12 hours | 0.809 |
| 24 hours | 0.771 |
| 48 hours | 0.726 |
| 7 days | ~0.57 |

Beyond 48 hours, accuracy paradoxically rises to 85% while ROC drops to 0.57. This occurs because only 14% of posts survive at 7 days -- predicting "everything dies" gets 85% accuracy but ROC reveals no actual discriminative power. This demonstrates why accuracy alone is a misleading metric.

**Iteration 6 -- State rise prediction.** Predicting whether a post will improve in state: 0.947 ROC at 1h, barely decaying to 0.926 at 24h. Rise is more predictable than fall because it requires detectable signals (velocity, engagement); falling is absence of signal.

**Iteration 7 -- Surging detection.** 0.987 ROC AUC -- near-perfect detection of posts about to surge. Dead detection: 0.945. These are the project's strongest models.

**Iteration 8 -- Survival probability curves.** Combined all horizon models into post half-life estimates: surging ~48h, alive ~24h, cooling ~3h, dying ~1h. (Figure 2)

### Phase 4: From Posts to Topics

**Iteration 9 -- Topic growth prediction.** Tracked 1,793 keywords daily. Topic growth achieves 0.839 ROC at 3 days -- and IMPROVES with longer horizons (opposite of posts). Individual posts are chaotic long-term; topics follow momentum.

**Key discovery -- inverse predictability.** r/politics: best post prediction (81%), worst topic prediction (R2=0.551). r/news: worst posts (68%), best topics (R2=0.808). Predictable engagement mechanics vs predictable aggregate momentum.

**Iteration 10 -- Co-occurrence pairs.** Two-word pairs from the same title represent specific stories rather than generic vocabulary. "birthright+citizenship", "russian+tanker", "kash+patel" -- actual news events detected from 1-2 posts.

Temporal validation (trained on days 1-9, tested on days 10-13): **0.838 ROC AUC.**

Real examples detected:
- "russian+tanker" -- 2 posts on day 1, grew to 16 (naval drone strikes oil tanker)
- "birthright+citizenship" -- 1 post, grew to 17 (Supreme Court case)
- "kash+patel" -- 1 post, grew to 14 (FBI director story)
- "easter+trump" -- 1 post, grew to 22 (political Easter controversy)

### Phase 5: Bridging Posts and Topics

Multiple attempts to use post-level quality signals to improve topic prediction:

| Approach | ROC Change |
|----------|-----------|
| Average post quality | -0.003 |
| Best post features | -0.002 |
| Post survival predictions | -0.001 |
| Post rising predictions | -0.001 |
| Engagement speed/acceleration | -0.043 |
| Quality at 3+ days observation | +0.013 |
| **First post's comment count** | **+0.034** |
| Title diversity (3.3x ratio) | +0.002 |
| 5x rocketing events | +0.060 (n=15) |

**The bridge:** First post's comment count -- keywords whose initial post generates extensive discussion (384 avg comments vs 179 for non-growers) are more likely to become topics. Achievement: 0.859 ROC. Discussion drives topic spread.

**Profile signals that exist but don't improve ROC:**
- Title diversity: 3.3x ratio (growing topics have posts from different angles)
- Upvote acceleration: 2.8x ratio (growing topics have faster-accelerating posts)
- Unique authors: 2.2x ratio (organic spread, not one person spamming)

These signals clearly separate growing from non-growing topics in profiles but cannot improve classification with current data volume. The patterns are real; the sample size is insufficient.

---

## 5. Final Evaluation

### The Complete System

**Post-level (strong -- genuinely useful):**

| Model | ROC AUC |
|-------|---------|
| Surging detection | 0.987 |
| Post state rise | 0.947 |
| Dead detection | 0.945 |
| Survival (1h) | 0.843 |
| Survival (4h) | 0.834 |
| Survival (24h) | 0.771 |

**Topic-level (decent -- real but limited):**

| Model | ROC AUC |
|-------|---------|
| Emerging keyword detection | 0.868 |
| Co-occurrence pair emergence | 0.838 (temporal) |
| Topic growth 3-day | 0.839 |
| First-post-to-topic | 0.859 |

**Practical detection (at 10% threshold):**
- Flags 565 keywords, 180 correct (32% precision)
- Catches 72% of topics that actually grow
- 1 in 3 flags is real; 2 in 3 are false alarms

### How It Works Together

1. New topic appears with 1-3 posts → detected at 0.838 ROC
2. Each post within topic → survival predicted at 0.987 ROC
3. Discussion tracked → sentiment, Gini concentration, engagement quality
4. Post lifespan estimated → half-life curves per state
5. Topic detection is the weakest link; everything after detection is strong

### Key Findings

1. **Controversy drives engagement.** Negative comment sentiment correlates with longer post survival. r/politics (most negative, -0.10) is growing; r/Games (most positive, +0.38) is declining.

2. **Comment upvote concentration (Gini) is the strongest post predictor.** Focused discussion (high Gini) = survival. Diffuse discussion = death.

3. **Posts and topics follow opposite predictability patterns.** Posts become chaotic beyond 24h. Topics become MORE predictable at 3+ days. Different mechanisms at different scales.

4. **Post quality signals exist at topic level but need more data.** Title diversity (3.3x), acceleration (2.8x), unique authors (2.2x) clearly separate growing from non-growing topics in profiles, but 13 days is insufficient for the model to exploit these patterns.

5. **Comment count on the first post bridges post and topic levels.** The strongest cross-level signal (+0.034 ROC improvement). Discussion on the first post drives topic spread.

### Per-Subreddit Formulas

Linear regression produces interpretable formulas for topic upvote prediction:

**r/news (R2=0.808):**
`predicted_upvotes = 57 * posts_yesterday - 321 * post_momentum + 0.9 * upvotes_yesterday + 1614`

**r/Games (R2=0.727):**
`predicted_upvotes = 29 * posts_yesterday - 232 * post_momentum + 0.8 * upvotes_yesterday + 215`

The negative `post_momentum` coefficient reveals **oversaturation**: topics accelerating too fast (too many posts too quickly) get fewer upvotes per post. The audience fatigues. This effect is strongest in r/worldnews (-1168 coefficient) and absent in entertainment subreddits.

r/technology shows unique behaviour: MORE posts correlate with FEWER total upvotes (-90 coefficient). Tech audiences ignore repetitive coverage. News audiences engage more when coverage increases (+57).

### Confusion Matrix Analysis

The 4-hour binary classifier (Figure 3):

|  | Predicted Alive | Predicted Dead |
|--|----------------|----------------|
| **Actually Alive** | 16,570 (TP, 80%) | 4,189 (FN) |
| **Actually Dead** | 3,900 (FP) | 8,057 (TN, 67%) |

Primary failure mode: false optimism (3,900 posts predicted to survive that actually died). This reflects training data class imbalance — most snapshots capture posts while still active.

### Brand New Topics

802 brand new co-occurrence pairs emerged over the collection period. Of these, 30 became significant (5+ posts). Detection characteristics:
- 64% detected on their first day of appearance
- 50% peak within 24 hours of first appearance
- Day-1 signals: growing topics average 5.6 posts and 17K upvotes vs non-growing 2.9 posts and 11K upvotes

The distinction between growing and non-growing new topics is partly about **story structure**: ongoing stories (court cases, diplomatic negotiations, political scandals) generate follow-up posts. One-shot events (data breach revealed, single announcement) get massive engagement on one post but no continuation. Our model cannot reliably distinguish these from engagement metrics alone.

### State Transition Matrix

From 186,815 observed transitions:

| From \ To | surging | alive | cooling | dying | dead |
|-----------|---------|-------|---------|-------|------|
| surging | 75% | 13% | 8% | 2% | 1% |
| alive | 3% | 63% | 11% | 11% | 13% |
| cooling | 12% | 39% | 34% | 6% | 7% |
| dying | 2% | 44% | 8% | 44% | 2% |
| dead | 1% | 42% | 9% | 0% | 48% |

Dying posts have a 44% chance of reviving to alive. Dead posts revive 42% of the time. Reddit posts are surprisingly resilient -- a finding the Markov chain captures well.

### Additional Analyses Performed

- **Best posting hours**: 08:00 UTC optimal (30.4% alive rate), 03:00 UTC worst (57.9% dead)
- **Upvote velocity curves**: Posts gaining 500+ upvotes in first hour have 49% survival vs 17% for 0-10 upvotes
- **Cross-subreddit propagation**: 1,573 cross-posted stories detected. Worldnews breaks stories first (516 times). Median propagation time: 11.1 hours. Politics→news cross-posting has 72% success rate
- **Domain analysis**: Reuters most posted (405 posts). euromaidanpress.com highest upvotes (median 5,584). hindustantimes.com best survival (71%)
- **Author analysis**: 1,899 unique authors. Prolific authors (10+ posts) average 26% alive rate -- same as everyone else. Being prolific does not improve success
- **Title style**: Shock/sensational words achieve 59.1% alive rate vs 23.8% baseline. Positive titles underperform (17.2% vs 20.5%). Very negative titles get the most upvotes
- **Keyword trends**: "trump" dominates (2,192 mentions). "bondi" fastest rising (+1620%). "xbox" declining
- **Flow deviation**: 12 active anomalies detected including business_economy 5.2x surge in r/politics
- **Time-to-death**: R2=0.459, MAE 9.1 hours. Top feature: states_seen_count (30%)

### Subreddit Direction

| Subreddit | Score | Direction | Sentiment |
|-----------|-------|-----------|-----------|
| politics | +65 | STRONG UPTREND | -0.10 (negative) |
| worldnews | +65 | STRONG UPTREND | -0.11 |
| technology | +15 | STABLE | +0.01 |
| news | -30 | MILD DECLINE | -0.04 |
| Games | -50 | MILD DECLINE | +0.38 (positive) |

The most negative subreddits are growing. The most positive is declining. Controversy fuels engagement across the entire platform.

### Limitations

- 13 days of data limits rare event learning (only ~250 topic growth events in test sets)
- Reddit public JSON provides no reply threading (99% zeros), preventing argument depth analysis
- Topic detection mostly relies on counting; quality signals (title diversity 3.3x, acceleration 2.8x, unique authors 2.2x) show clear profile separation but need months of data to improve classifier ROC
- Cannot predict real-world events that create topics -- a ceasefire proposal post with 1,400 comments and 40K upvotes received only 1% explosion probability because the model lacked sufficient examples of such extreme events
- All findings correlational, not causal -- popular posts may attract negative comments rather than negative comments causing survival

---

## 6. Reflection on AI-Assisted Coding

### Where AI Was Effective
Rapid prototyping, algorithm selection (Random Forest, VADER, K-means, Gini coefficient), and boilerplate generation. The conversational workflow enabled moving from idea to working analysis in minutes.

### Where AI Was Wrong
- **Task Scheduler crash:** Unicode em dashes + WPF popups worked interactively, crashed in deployment
- **Timeout miscalculation:** 5 subreddits x 100 posts x 10 comments x 1.1s = 92 minutes, exceeding 30-minute timeout
- **Misleading accuracy:** Initial 74.5% was essentially comment counting, not sentiment analysis
- **Premature ceiling claims:** AI declared "we've hit the ceiling" multiple times; developer pushed past each one, discovering co-occurrence pairs, title diversity, first-post comment bridge, and the post-topic inverse predictability relationship

### Validation
Temporal validation (trained on past, tested on future) rather than just cross-validation. Feature importance analysis to ensure models learn real patterns. Confusion matrices to understand failure modes. Honest assessment of what works and what doesn't.

### Ethics
1.1-second rate limiting between requests. Aggregate analysis, not individual profiling. Transparent AI tool documentation.

---

## References

Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proc. ICWSM*.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *JMLR*, 12, pp. 2825-2830.

---

## Figures

**Figure 1:** ROC curves for post survival across 5 horizons showing predictability decay. `roc_prediction_decay.png`

**Figure 2:** Survival probability curves with half-life annotations. `survival_probability_curves.png`

**Figure 3:** ROC curves + confusion matrix for 4-hour classifier. `roc_curves_all_horizons.png`

**Figure 4:** Subreddit state distribution. `subreddit_state_mix.png`

**Figure 5:** Flow trajectories by subreddit. `flow_trajectory_by_subreddit.png`

**Figure 6:** Activity dashboard. `live_pulse_dashboard.png`
