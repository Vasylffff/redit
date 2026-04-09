# Predicting Reddit Post Survival Using Comment Engagement and Sentiment Analysis

**Module:** Exploring AI: Understanding and Applications (SPC4004)
**Assessment:** Code Generation Project
**GitHub:** https://github.com/Vasylffff/redit

---

## 1. Problem Definition & Dataset Justification

This project addresses a binary classification problem: given a Reddit post's current metrics, can we predict whether it will remain active or decay within a specified time horizon?

Reddit posts follow a measurable lifecycle, transitioning through states: emerging, surging, alive, cooling, dying, and dead. These states are determined by upvote and comment velocity thresholds computed empirically per subreddit. Predicting this trajectory has applications in content strategy, news monitoring, and understanding online engagement dynamics.

The dataset was self-collected using Reddit's public JSON endpoints, requiring no API authentication. Over 12 days (26 March to 6 April 2026), the system collected hourly snapshots from five subreddits: r/technology, r/news, r/worldnews, r/politics, and r/Games. The final dataset comprises 185,048 post snapshots, 677,569 comment snapshots, and 6,292 unique post lifecycles. Each snapshot records upvote count, comment count, velocity metrics, rank position, and computed activity state. Comment snapshots include full text, enabling natural language processing.

This dataset was chosen because it provides naturally labelled training data: lifecycle states serve as ground truth without requiring manual annotation. The time-series structure enables both single-point classification and trajectory prediction across multiple horizons.

---

## 2. Initial Code & Explanation of AI Use

The initial codebase was generated using Claude Code (Anthropic's CLI, powered by Claude Opus 4.6). The primary prompt was:

> "This is my project and I want you to create a folder and set this up so I could scrape Reddit from here"

This produced a data collection pipeline (`collect_reddit_free.py`), an automated schedule runner, and a five-layer Markov chain predictor (`predict_post_flow.py`). The initial system could scrape Reddit data and predict state transitions using baseline transition probabilities computed from observed state changes.

Subsequent development used Claude Code throughout, with the developer directing analysis priorities and evaluating outputs. Additional tools included the VADER sentiment analyser (Hutto and Gilbert, 2014) for comment scoring and scikit-learn (Pedregosa et al., 2011) for classification and clustering algorithms.

---

## 3. Critique of Initial Code

The initial code contained several technical issues. First, it used `datetime.UTC`, introduced in Python 3.11, across twelve files, causing ImportError on the development machine running Python 3.10. Second, `export_history_to_sqlite.py` contained backslashes inside f-string expressions, which Python 3.10 does not permit. Third, the dependency specification required scikit-learn 1.8+, incompatible with Python 3.10.

Beyond compatibility, the initial predictor had fundamental algorithmic limitations. It relied exclusively on upvote and comment velocity, treating a post with 100 angry comments identically to one with 100 supportive comments. It could only predict one step ahead rather than forecasting trajectories over hours or days. It used a single global model despite vastly different dynamics across subreddits; for instance, r/politics posts have a median alive duration of 11 hours compared to 49 hours for r/Games. The Markov chain approach also produced probability distributions that converged to a fixed equilibrium regardless of starting conditions, limiting its discriminative power for long-range prediction.

The pipeline automation script used WPF MessageBox popups and aggressive error handling (`$ErrorActionPreference = "Stop"`) that caused crashes when executed non-interactively by Windows Task Scheduler, rendering the automated collection system non-functional.

---

## 4. Iterative Development & Justification

**Iteration 1: Compatibility and Infrastructure Fixes.** Replaced `datetime.UTC` with `timezone.utc` across twelve files, extracted problematic f-string expressions to intermediate variables, and pinned scikit-learn to version 1.7. Fixed the Task Scheduler script by replacing WPF popups with silent logging and changing the error policy to Continue. Verified by confirming all scripts executed without errors and the scheduler completed hourly collections successfully.

**Iteration 2: VADER Sentiment Integration.** Integrated the VADER sentiment analyser to score all 677,569 comments, adding five features to the prediction dataset: mean sentiment, upvote-weighted sentiment, positive comment share, negative comment share, and sentiment variance. This revealed a counterintuitive finding: posts with negative comment sentiment survive longer (alive posts average sentiment -0.006 versus dead posts +0.045, difference -0.072). Controversial discussions drive engagement; apathy kills posts. A Decision Tree classifier using these features achieved 74.5% accuracy (5-fold cross-validation). However, feature importance analysis revealed comment count dominated at 74%, meaning the model was essentially counting comments rather than analysing sentiment.

**Iteration 3: Comment Upvote Gini Coefficient.** To find a stronger engagement signal, implemented Gini coefficient analysis on comment upvote distributions. The Gini coefficient measures concentration: high values indicate a few dominant comments capturing most upvotes, while low values indicate evenly distributed attention. This became the strongest predictor found in the project (46% feature importance). Surviving posts exhibit high Gini (0.63-0.72), indicating clear community consensus around specific comments. Dying posts show low Gini (0.36), suggesting diffuse, unfocused discussion. Adding Gini features improved classifier accuracy to 77.6%.

**Iteration 4: Per-Subreddit Classification.** Trained separate Random Forest classifiers per subreddit, recognising that each community has distinct engagement patterns. This improved r/politics accuracy to 81% while revealing that r/Games (64%) is inherently less predictable. The variation itself is informative: politically focused communities follow more systematic engagement patterns than entertainment-focused ones.

**Iteration 5: Multi-Horizon Prediction.** Extended the system from single-step to multi-horizon prediction by training seventeen separate classifiers for horizons from 1 to 72 hours. Each model was trained on 100,000 to 160,000 labelled samples. This produced two key outputs: ROC curves showing systematic accuracy decay across horizons (Figure 1), and survival probability curves estimating post half-lives (Figure 2). The 4-hour model achieved 0.834 ROC AUC while the 48-hour model degraded to 0.726, quantifying the inherent unpredictability of long-range Reddit engagement.

---

## 5. Final Code Evaluation and Reflection

### Performance Metrics

| Prediction Target | Horizon | ROC AUC |
|---|---|---|
| Post survival | 1 hour | 0.843 |
| Post survival | 4 hours | 0.834 |
| Post survival | 12 hours | 0.809 |
| Post survival | 24 hours | 0.771 |
| Post survival | 48 hours | 0.726 |
| Surging detection | 1 hour | 0.987 |

The surging detection model achieves near-perfect discrimination (0.987 ROC AUC), correctly identifying posts about to surge in popularity.

### Survival Probability and Post Half-Life

By combining predictions across all horizons, the system computes survival probability curves for any post (Figure 2). Estimated half-lives: surging posts approximately 48 hours, alive posts approximately 24 hours, cooling posts approximately 3 hours, dying posts approximately 1 hour. These provide actionable estimates for content monitoring applications.

### Confusion Matrix Analysis

The 4-hour binary classifier (Figure 3) correctly identifies 80% of surviving posts and 67% of dying posts. The primary failure mode is false optimism: predicting survival when posts actually die, reflecting class imbalance in the training data where most snapshots capture posts while still active.

### Key Findings

First, controversy drives engagement. Negative comment sentiment correlates with longer post survival. The most active subreddit (r/politics, average sentiment -0.10) shows strong growth while the most positive (r/Games, +0.38) is declining. Second, comment upvote concentration outperforms all other features in predicting survival. Third, predictability decays systematically over time, suggesting an inherent chaotic component to Reddit engagement beyond approximately 24 hours.

### Limitations

Reddit's public JSON provides limited reply threading data, preventing analysis of argument depth. The 12-day collection period, while substantial, limits seasonal pattern detection. Regression models for predicting exact upvote counts performed poorly (R2 = 0.42), confirming that Reddit virality has a significant random component. All findings are correlational rather than causal.

---

## 6. Reflection on AI-Assisted Coding

Claude Code proved highly effective for rapid prototyping: generating analysis scripts, suggesting appropriate algorithms, and handling data pipeline boilerplate. The conversational workflow enabled rapid iteration, moving from verbal description to working code within minutes. The AI correctly suggested using the Gini coefficient for upvote concentration analysis, which produced the project's strongest finding.

However, several AI-generated components required significant human correction. The Task Scheduler script crashed due to unicode em dashes corrupting under Windows cp1252 encoding, a runtime-only bug invisible in the source code. The AI suggested collecting 10 comments per post hourly without calculating that this would exceed the 30-minute timeout (5 subreddits multiplied by 100 posts multiplied by 10 comments at 1.1-second delay equals 92 minutes). The initial sentiment model's 74.5% accuracy appeared promising until feature importance analysis revealed it was essentially counting comments rather than analysing sentiment, requiring critical human evaluation to identify.

Validation relied on cross-validation with held-out test sets, feature importance analysis to ensure models captured meaningful patterns, and confusion matrices to understand specific failure modes. This multi-layered validation approach was essential because AI-generated code can produce plausible but misleading results.

Ethical considerations include responsible scraping practices (1.1-second delays between requests, compliance with rate limits), analysis of aggregate patterns rather than individual user behaviour, and transparent documentation of AI tool usage throughout the development process.

---

## References

Bird, S., Klein, E. and Loper, E. (2009) *Natural Language Processing with Python*. Sebastopol: O'Reilly Media.

Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media*. Ann Arbor, MI, June 2014.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.

---

## List of Figures

**Figure 1:** ROC curves for post survival prediction across five time horizons (1h, 4h, 12h, 24h, 48h), demonstrating systematic decay in predictive accuracy. See `data/analysis/reddit/visuals/roc_prediction_decay.png`.

**Figure 2:** Survival probability curves for five post archetypes (surging, alive, cooling, dying, fresh), with estimated half-life annotations. See `data/analysis/reddit/visuals/survival_probability_curves.png`.

**Figure 3:** Confusion matrix for the 4-hour binary survival classifier showing 16,570 true positives, 8,057 true negatives, 3,900 false positives, and 4,189 false negatives. See `data/analysis/reddit/visuals/roc_curves_all_horizons.png`.

**Figure 4:** Subreddit lifecycle state distribution comparison across all five tracked subreddits. See `data/analysis/reddit/visuals/subreddit_state_mix.png`.

**Figure 5:** Post lifecycle flow trajectories by subreddit, showing state transition patterns. See `data/analysis/reddit/visuals/flow_trajectory_by_subreddit.png`.

**Figure 6:** Live activity pulse dashboard showing current engagement levels across all subreddits. See `data/analysis/reddit/visuals/live_pulse_dashboard.png`.
