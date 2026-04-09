# Technical Analysis — Reddit Topic Lifecycle Prediction

Complete technical documentation of all models, failures, and findings. This document contains everything that could not fit in the 2,000-word assessment report.

---

## 1. Data Pipeline Issues

### Collection Gap Problem

Hourly collection ran on two machines via Windows Task Scheduler, but gaps occurred from:
- Laptop sleep during subway commute (~1hr daily)
- Task Scheduler failing silently after Windows updates
- OneDrive sync conflicts when both machines wrote simultaneously

**Impact:** 12.1% of snapshot rows were flagged as collection gaps. When a gap occurred, the next snapshot showed velocity = 0 even for actively growing posts — this faked a dead-post signal.

**Fix:** `patch_snapshot_gaps.py` recalculates velocity from the upvote delta divided by actual gap length. For gaps under 3 hours, velocity is interpolated. Longer gaps are flagged but still corrected. 8.6% of velocities were interpolated.

**Example:** A politics post ("Trump says government should stop funding Medicare") was gaining 348 upvotes/hour before a 4-hour gap. During the gap, raw velocity showed near-zero. But the post gained +1,344 upvotes during that "dead" period. Gap patching corrected the velocity to ~336/hr (1344/4).

### Reddit Vote Fuzzing

Reddit deliberately fuzzes vote counts on popular posts (anti-bot measure). A post might show 15,234 upvotes one minute and 15,189 the next, despite receiving upvotes. This creates negative velocity values that are physically impossible (upvotes cannot decrease on Reddit).

**Impact:** 3.2% of snapshots showed negative upvote velocity. This corrupted the velocity-based state classification — a surging post could briefly appear "dying" due to fuzzing.

**Fix:** `is_reddit_fuzzing` flag added. When a snapshot shows negative delta but the post is clearly active (high comment velocity, recent creation), the upvote velocity is clamped to zero rather than going negative.

### Tracking Pool Design Flaw

Initially there was no tracking pool. Each hourly scrape fetched whatever was currently listed as "new" on Reddit. If a post happened to appear in consecutive hourly scrapes, it would have multiple snapshots, but this was by luck rather than design.

**Impact:** 68% of posts in the first week had only 1-2 snapshots. Trajectory prediction requires 5+ observations minimum.

**Fix:** Codex session 2 (March 31) split tracking into:
- **Prediction cohort** — fixed set of posts tracked for their full observation window
- **Live watch pool** — rolling shortlist of currently interesting posts
- **Three-tier monitoring** — active (hourly), dormant (6-hourly), dropped (archived)

This increased median observations per post from 2 to 14.

---

## 2. Dead Post Definition — The Hardest Problem

### Attempt 1: Velocity Threshold

Initial approach: a post is "dead" when upvote velocity drops below 5/hour.

**Problem:** Triggered during collection gaps, Reddit fuzzing, and natural lulls in otherwise healthy posts. False death rate: ~25%.

### Attempt 2: Time-Based

A post is "dead" after 24 hours with velocity below threshold.

**Problem:** Many posts die within 4-6 hours. Waiting 24 hours wastes tracking resources and delays classification.

### Attempt 3: Variance Collapse (Final)

A post is dead when its velocity variance drops sharply and stays low. Specifically:
- Calculate rolling variance over a 5-snapshot window
- `std_before > 15` (was volatile) AND `std_after < 8` (now stable) AND declining trend
- More reliable than single-snapshot threshold because it requires sustained low activity

**Result:** False death rate reduced from ~25% to ~8%.

### Topic-Level Death Definition

At topic level, "death" is even harder. Analysis of 8,880 drop events revealed:

| Definition | False Death Rate |
|---|---|
| 1 day below 2 posts | 13.1% |
| 2 consecutive days below 2 posts | 6.8% |
| 3 consecutive days | 3.2% |

Larger topics revive more frequently:
- Topics peaking at 2-3 posts: 22.8% revival rate
- Topics peaking at 4-7 posts: 33.5% revival rate
- Topics peaking at 8-11 posts: 44.8% revival rate

The two-consecutive-day definition was adopted as the best trade-off between false deaths and detection delay.

---

## 3. Post-Level Models — What Works

### Markov Chain Predictor

`predict_post_flow.py` — 5-layer system:

1. **Baseline Markov** — transition matrix P(next_state | current_state, topic, subreddit, age_bucket, velocity_bucket) from 116,000 observed transitions
2. **Live Heat** — compare current surge+alive rate vs 7-day baseline. Heat ratio > 1.5 shifts toward surging; < 0.7 shifts toward dying
3. **Scenario** — user-specified event multiplier (quiet 0.4x to breaking 7x)
4. **Anchor** — after 2 hours, replace initial distribution with observed reality
5. **Discussion Quality** — 0-100 score from question_share, avg_comment_upvotes, unique_commenters, post_body_length

**Limitation:** Markov chains converge to equilibrium. After ~10 transitions, predictions become indistinguishable regardless of starting state. This is a fundamental mathematical property, not a data issue.

### Random Forest Survival Classifier

**Features (in order of importance):**

| Feature | Importance | Description |
|---|---|---|
| gini_coefficient | 46% | Comment upvote concentration |
| upvote_velocity_per_hour | 18% | Speed of upvote growth |
| comment_velocity_per_hour | 12% | Speed of comment growth |
| age_minutes_at_snapshot | 8% | Post age |
| comment_count | 7% | Total comments |
| upvote_ratio | 5% | Reddit's upvote/downvote ratio |
| sentiment_compound | 4% | VADER compound sentiment |

**Key insight:** Gini coefficient at 46% importance means comment distribution structure matters more than raw velocity. A post where 3 comments dominate (Gini 0.7) has focused discussion and survives. A post where engagement is diffuse (Gini 0.3) has unfocused discussion and dies.

### Multi-Horizon Decay

| Horizon | ROC AUC | Accuracy | Notes |
|---|---|---|---|
| 1 hour | 0.843 | 72% | Strong — recent features still predictive |
| 4 hours | 0.834 | 74% | Slight decay |
| 12 hours | 0.809 | 77% | Moderate decay |
| 24 hours | 0.771 | 80% | Accuracy rises but ROC drops |
| 48 hours | 0.726 | 83% | Class imbalance dominates |
| 7 days | ~0.57 | 85% | Predicting "everything dies" gets 85% accuracy |

**The accuracy trap at 7 days:** 86% of posts are dead at 7 days. A model that predicts "dead" for everything achieves 85% accuracy — but ROC AUC of 0.5 (random chance). Our model's 0.57 ROC means it has almost no discriminative power at this horizon. This is why ROC AUC is essential for imbalanced classification — accuracy alone is dangerously misleading.

### State Detection

| State | ROC AUC | Why it works |
|---|---|---|
| Surging | 0.987 | Strong velocity signal — surging posts are unmistakable |
| Dead | 0.945 | Velocity collapse is detectable |
| Rising | 0.947 | Requires detectable engagement acceleration |
| Cooling | 0.53 (48h) | Hardest — cooling is absence of signal, looks like noise |

**Why cooling is hardest:** Surging and dead have strong signals (high velocity, zero velocity). Cooling is the absence of signal — it looks like noise, random fluctuation, or a temporary lull. The model cannot distinguish "naturally slowing down" from "about to surge again" without external context.

### Per-Subreddit Accuracy

| Subreddit | Accuracy | Avg Lifecycle | Data Volume |
|---|---|---|---|
| r/politics | 81% | 11 hours | Most data |
| r/worldnews | 72% | 18 hours | High |
| r/news | 68% | 15 hours | Medium |
| r/technology | 66% | 29 hours | Medium |
| r/Games | 64% | 49 hours | Least data |

**Inverse relationship with lifecycle length:** Short-lived communities (politics, 11h) are more predictable because posts follow tighter patterns. Long-lived communities (Games, 49h) have more variance and external dependencies (game release timing, announcements).

**Accuracy correlates with data volume:** More posts = more training examples = better model. This is visible in the politics→Games gradient.

---

## 4. Sentiment Analysis — The Feature Leakage Lesson

### VADER on 972,353 Comments

- Alive posts: average sentiment -0.006 (slightly negative)
- Dead posts: average sentiment +0.045 (slightly positive)
- Surging posts: most negative sentiment on average

**Finding:** Negative sentiment = longer survival. Controversy drives engagement; apathy kills posts.

### The 74.5% Accuracy Trap

Initial sentiment classifier achieved 74.5% accuracy. Looked good. Feature importance analysis revealed:

| Feature | Importance |
|---|---|
| comment_count | 74% |
| sentiment_compound | 8% |
| sentiment_positive | 6% |
| sentiment_negative | 5% |
| Other | 7% |

The model learned to count comments, not analyse sentiment. More comments = more likely alive. This is **feature leakage** — comment count is a proxy for the label (alive posts have more comments by definition). The sentiment features contributed almost nothing.

**Fix:** Gini coefficient replaced raw comment count as the primary comment feature. It measures the distribution of engagement, not the volume, and achieved 46% feature importance — genuinely learning from comment structure.

---

## 5. Topic-Level Models — What Works

### Co-occurrence Pair Detection

Single keywords ("trump", 2,192 posts) are too generic. Two-word pairs from the same title represent specific stories:

| Pair | Posts | What it represents |
|---|---|---|
| russian+tanker | 16 | Naval drone strikes oil tanker |
| birthright+citizenship | 17 | Supreme Court case |
| kash+patel | 14 | FBI director story |
| easter+trump | 22 | Political Easter controversy |
| hormuz+strait | 252 | Geopolitical crisis |
| official+trailer | 122 | Game announcements |
| crimson+desert | 65 | Game launch |

**Temporal validation** (trained days 1-8, tested days 9-13): 0.813 ROC AUC.

**Content-agnostic:** The same algorithm detects politics, games, and tech topics. It uses engagement patterns only, not NLP.

### The Complete Lifecycle Pipeline

| Stage | Task | Best ROC | Best Model |
|---|---|---|---|
| Filter | Discard noise (87% filtered) | 0.850 | Logistic Regression |
| Birth | Detect emerging topic (1-3 to 5+) | 0.860 | Logistic Regression |
| Growth | Peaked or still growing? | 0.958 | Gradient Boosting |
| Spread | Will it reach r/politics? | 0.756 | Random Forest |
| Decline | Topic dying state detection | 0.992 | Random Forest |
| Death | Will it die tomorrow? | 0.890 | Random Forest |
| Death speed | Quick death or slow death? | 0.999 | Logistic Regression |
| Revival | Ongoing story or one-shot? | 0.970 | Random Forest |

### Inverse Predictability Discovery

| Subreddit | Post Accuracy | Topic R² | Interpretation |
|---|---|---|---|
| r/politics | 81% (best) | 0.551 (worst) | Structured posts, chaotic topics |
| r/news | 68% (worst) | 0.808 (best) | Chaotic posts, predictable topics |

**Why:** r/politics posts follow rigid engagement mechanics (partisan reactions, predictable comment patterns). But topics in r/politics depend on unpredictable real-world events (political decisions, scandals). r/news is the opposite — individual posts are noisy but topics follow aggregate momentum.

---

## 6. What Failed — Honest Negative Results

### Magnitude Prediction (R² = 0.22)

**Goal:** Given a topic at 1-3 posts, predict how many posts it will reach at peak.

**Attempts:**
1. Szabo-Huberman log-linear model (2010) — R² = 0.22
2. Random Forest regression — R² = 0.19
3. Power-law fitting — R² = 0.15
4. Ridge regression — R² = 0.21

**Why it fails:** At 1-3 posts, 99.5% of word pairs remain small. The signal-to-noise ratio is too low. Classification (will it grow? yes/no) works because binary decisions tolerate noise. Regression (by how much?) does not.

### Exact Timing (R² = -0.5)

**Goal:** Predict "days until topic death."

**Result:** R² = -0.5. The model performs worse than predicting the mean. Binary classification of the same events works well (0.890 ROC); regression on the exact timing does not.

**Why:** Topic death depends on external events (new developments in a story). The model can detect that a topic is dying but cannot predict when an external event will interrupt the decline.

### Revival Timing (ROC = 0.578)

**Goal:** Predict when a dead topic will revive.

**Result:** 0.578 ROC — barely above random.

**Why:** Revival is driven entirely by external real-world events. A topic about a court case revives when the court issues a ruling — this is unpredictable from engagement metrics.

**However:** Predicting which topics are the *type* that revives (ongoing stories vs one-shot events) achieves 0.970 ROC. The model can distinguish ongoing stories from one-shot events; it just cannot predict the timing of specific revivals.

### Post Quality → Topic Prediction (+0.034 max)

**Goal:** Use post-level quality signals to improve topic-level prediction.

**Systematic exploration:**

| Approach | ROC Change vs Baseline |
|---|---|
| Average post quality features | -0.003 |
| Best post features | -0.002 |
| Post survival predictions as features | -0.001 |
| Post rising predictions as features | -0.001 |
| Engagement speed/acceleration | -0.043 |
| Ratio features (surge%, dead%) | -0.020 |
| Quality at 3+ days observation | +0.013 |
| **First post's comment count** | **+0.034** (best) |
| Title diversity (3.3x ratio) | +0.002 |
| Upvote acceleration (2.8x ratio) | -0.043 |
| 5x rocketing events | +0.060 (but only n=15) |

**The bridge:** First post comment count is the strongest cross-level signal. Keywords whose initial post generates 384 average comments (vs 179 for non-growers) are more likely to become significant topics. Discussion on the first post drives topic spread.

**Profile vs classifier paradox:** Title diversity (3.3x), upvote acceleration (2.8x), and unique authors (2.2x) show clear separation between growing and non-growing topics in aggregate profiles. But they do not improve classifier ROC. The patterns are real — visible in the data — but 13 days is insufficient for the model to exploit them. More data would likely close this gap.

---

## 7. Model Comparison and Hyperparameter Tuning

### 5 Classifiers x 7 Tasks

No single model dominated all tasks:
- **Logistic Regression** — best on emergence detection (0.860), death speed (0.999), noise filtering (0.850). The signal is linear.
- **Random Forest** — best on death prediction (0.890), spread (0.756), revival (0.970). Handles feature interactions.
- **Gradient Boosting** — best on growth peak detection (0.958). Captures sequential patterns.
- **Extra Trees** — competitive but never the best on any task.
- **Decision Tree** — worst by default but improved most with tuning.

### Hyperparameter Impact

| Model | Default ROC | Tuned ROC | Improvement |
|---|---|---|---|
| Decision Tree | 0.577 | 0.842 | +0.265 |
| Gradient Boosting | 0.713 | 0.835 | +0.122 |
| Random Forest | 0.820 | 0.846 | +0.027 |
| Logistic Regression | 0.859 | 0.860 | +0.001 |

**Key finding:** Decision Tree improved by +0.265 simply by limiting depth to 3-5 (default was 8+). It was massively overfitting. Both AI tools (Codex and Claude Code) defaulted to Random Forest with near-default parameters for every task, never suggesting that simpler models or basic tuning might work better.

### Overfitting Analysis

Both Random Forest and Decision Tree perform worse with deeper trees. Optimal depth is 3-5, not the default 8-12. Gradient Boosting peaks at learning rate 0.02 and degrades substantially at higher rates.

---

## 8. Feature Engineering Notes

### Features That Work

| Feature | Type | Importance | Why |
|---|---|---|---|
| Gini coefficient | Post | 46% | Comment structure predicts survival |
| Upvote velocity | Post | 18% | Direct momentum signal |
| Comment velocity | Post | 12% | Discussion activity |
| Post count growth | Topic | 52% | Direct growth signal |
| Subreddit count | Topic | 17% | Spread = importance |
| Previous peaks | Revival | 21.5% | Ongoing story indicator |
| Decline rate day 1 | Death speed | 67.8% | Fast decline = fast death |

### Features That Don't Work

| Feature | Why Not |
|---|---|
| Activity state labels at topic level | Discretised velocity — redundant with raw velocity |
| Raw comment count | Proxy for label (alive = more comments). Feature leakage. |
| Sentiment at topic level | Too noisy when aggregated across posts |
| Author count | Correlated with post count, adds no independent signal |
| Post body length | Inconsistent across subreddits |

---

## 9. Ethical Considerations

### Scraping Legality

Reddit's Terms of Service place automated scraping in a grey area. The public JSON endpoints are documented and accessible without authentication, but Reddit does not explicitly authorize bulk collection. Mitigation: 1.1-second rate limiting between all requests, well below any practical impact on Reddit's infrastructure.

### Misuse Potential

The system's ability to detect emerging topics at 0.86 ROC from just 1-3 posts could theoretically be used to amplify coordinated content or manipulate narratives. The same signals that predict organic virality could identify targets for artificial amplification.

### Privacy

All data is publicly visible on Reddit. Sentiment analysis was performed on public comments without individual consent, consistent with standard academic practice. No individual user profiling was performed — all analysis is aggregate.

### AI Transparency

All AI tools documented with exact session dates, prompt counts, and key prompts. Developer decisions clearly separated from AI suggestions. Failed AI advice (state labels, Random Forest everywhere, premature ceilings) documented alongside successful AI contributions.

---

## 10. Limitations and Future Work

### Data Limitations

- **13 days** — insufficient for rare event learning (~62 topic growth events in test sets)
- **5 subreddits** — limited by rate limiting on free endpoints
- **Hourly granularity** — minute-level data would catch early surges faster
- **No reply threading** — Reddit's public JSON provides 99% zeros for reply depth
- **No NLP** — content-agnostic. Cannot distinguish ongoing stories from one-shot events by reading text, only by trajectory shape

### Model Limitations

- **Magnitude prediction unsolved** — can detect growth direction but not size
- **External events unpredictable** — revival timing, topic creation driven by real-world events
- **Cooling state unpredictable** — absence of signal looks like noise
- **All findings correlational** — cannot establish causal mechanisms from observational data
- **Class imbalance at long horizons** — 86% dead at 7 days means accuracy is meaningless

### Future Work

- Collect 3+ months of data to enable the profile signals (title diversity 3.3x, acceleration 2.8x) to improve classifier ROC
- Add NLP features at topic level to distinguish ongoing vs one-shot stories from text
- Minute-level collection for the first 6 hours of a topic's life
- Cross-platform tracking (Reddit + Twitter/X) for story propagation
- Real-time deployment with streaming predictions
