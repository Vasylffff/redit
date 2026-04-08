"""Build HTML preview of the assessment report. Open in browser, refresh to see changes."""
import os, base64

FIGURES_DIR = "data/analysis/reddit/figures"
OUTPUT_PATH = "report_preview.html"


def img_base64(name):
    fpath = os.path.join(FIGURES_DIR, name)
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return '<img src="data:image/png;base64,%s" style="max-width:100%%;margin:20px auto;display:block;">' % data
    return '<p style="color:red;">[Missing: %s]</p>' % name


html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Assessment 2 - Report Preview</title>
<style>
body { font-family: Calibri, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #222; }
h1 { color: #000; border-bottom: 2px solid #333; padding-bottom: 8px; margin-top: 40px; }
h2 { color: #333; margin-top: 30px; }
h3 { color: #555; }
table { border-collapse: collapse; width: 100%%; margin: 15px 0; }
th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; font-size: 14px; }
th { background: #4472C4; color: white; }
tr:nth-child(even) { background: #f5f5f5; }
blockquote { border-left: 4px solid #4472C4; margin: 15px 0; padding: 10px 20px; background: #f9f9f9; font-style: italic; }
.caption { text-align: center; font-style: italic; font-size: 13px; color: #666; margin-bottom: 25px; }
.title { text-align: center; margin: 60px 0 40px 0; }
.title h1 { border: none; font-size: 28px; }
.title p { font-size: 14px; color: #555; }
ul { margin: 10px 0; }
li { margin: 4px 0; }
.highlight { background: #fff3cd; padding: 2px 4px; }
img { border: 1px solid #ddd; border-radius: 4px; }
</style>
</head>
<body>

<div class="title">
<h1>Predicting Reddit Topic Lifecycles:<br>Emergence, Spread, and Death</h1>
<p>Exploring AI: Understanding and Applications (SPC4004)<br>
Assessment 2 &mdash; Code Generation Project<br><br>
Vasyl Shcherbatykh<br><br>
<a href="https://github.com/Vasylffff/redit">GitHub: https://github.com/Vasylffff/redit</a></p>
</div>

<h1>1. Problem Definition &amp; Dataset Justification</h1>

<h2>What This Project Does</h2>
<p>Every day, thousands of stories appear on Reddit. Some disappear within hours. Others spread across multiple communities, generate thousands of comments, and dominate the platform for days. This project asks: can we predict which stories will grow, how they will spread, and when they will die, using only early engagement signals?</p>

<p>The system tracks topics through their complete lifecycle: emergence (a new story appears in 1&ndash;3 posts), growth (it gains momentum), spread (it crosses into new subreddits), decline (engagement drops), and death (the story stops generating new posts). At each stage, a separate machine learning model makes predictions about what will happen next.</p>

<h2>The Machine Learning Tasks</h2>

<p>The project applies classification and regression at two scales:</p>

<ul>
<li><strong>Post survival prediction</strong> &mdash; will an individual post still be alive at 1h, 4h, 12h, 24h, 48h, or 7 days?</li>
<li><strong>Post state detection</strong> &mdash; is this post surging, dying, or about to change state?</li>
<li><strong>Post flow modelling</strong> &mdash; Markov chain transition probabilities between lifecycle states</li>
<li><strong>Time-to-death regression</strong> &mdash; how many hours until a post dies?</li>
<li><strong>Sentiment and engagement analysis</strong> &mdash; VADER comment scoring and Gini coefficient of comment upvote distribution</li>
<li><strong>Topic emergence detection</strong> &mdash; will a new word pair grow from 1&ndash;3 posts to 5+?</li>
<li><strong>Topic lifecycle state prediction</strong> &mdash; what state will a topic be in tomorrow?</li>
<li><strong>Topic death and revival</strong> &mdash; will a declining topic die permanently or come back?</li>
<li><strong>Cross-subreddit spread</strong> &mdash; will a topic appear in a new subreddit tomorrow?</li>
<li><strong>Magnitude prediction (attempted)</strong> &mdash; how large will a topic get? Tested with Szabo-Huberman log-linear models; did not achieve useful accuracy (R&sup2;=0.22)</li>
</ul>

<p>Five classifiers were compared: Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, and Decision Tree, with 36 hyperparameter configurations tested.</p>

<h2>Why This Problem Matters</h2>
<p>Understanding how topics emerge and die online has practical applications in several areas. News monitoring systems need to identify which stories are gaining traction before they peak. Content moderators need to anticipate which topics will spread across communities. Researchers studying misinformation need to understand how stories propagate between subreddits and whether spread patterns can be detected early.</p>

<p>From a machine learning perspective, the problem is interesting because it requires prediction at different timescales and granularities. Post-level prediction (will this individual post survive?) works well at short horizons but degrades beyond 24 hours. Topic-level prediction (will this story grow?) follows the opposite pattern: it improves at longer horizons because topics follow momentum while individual posts are chaotic. This inverse relationship between post and topic predictability was one of the project's key discoveries.</p>

<h2>Why This Dataset</h2>
<p>Five subreddits were chosen to represent different community types: r/politics and r/news (fast-moving political discussion), r/worldnews (international affairs), r/technology (industry news), and r/Games (entertainment). This diversity tests whether the same models generalise across communities with different engagement patterns. For example, r/politics posts have an average lifecycle of 11 hours while r/Games posts survive 49 hours.</p>

<p>The dataset was self-collected rather than using an existing corpus because the project requires repeated observations of the same posts over time. Existing Reddit datasets typically provide single snapshots. Hourly collection over 13 days produced 216,944 post snapshots capturing how each post's engagement changed hour by hour.</p>

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Post snapshots</td><td>216,944</td></tr>
<tr><td>Comment snapshots</td><td>972,353</td></tr>
<tr><td>Unique posts tracked</td><td>6,826</td></tr>
<tr><td>Collection frequency</td><td>Hourly</td></tr>
<tr><td>Raw JSON files</td><td>5,140</td></tr>
<tr><td>Co-occurrence pairs analysed</td><td>200,000+ per day</td></tr>
</table>

""" + img_base64("fig11_data_coverage.png") + """
<p class="caption">Figure 1: Daily data collection coverage by subreddit across the 13-day collection period.</p>

<h1>2. Initial Code &amp; Explanation of AI Use</h1>

<h2>The Starting Point</h2>

<p>The project began on 24 March 2026 with OpenAI Codex. The first task was getting Reddit data. Three approaches were attempted before finding one that worked:</p>

<p><strong>Attempt 1: PRAW (Reddit API).</strong> Codex generated collect_reddit_data.py using the PRAW library, which requires Reddit API credentials. I submitted an API application on 24 March, providing a formal research proposal describing the project's aims. The application was not approved. Without credentials, the PRAW-based collector could not run.</p>

<p><strong>Attempt 2: Apify.</strong> On 26 March, Codex suggested using Apify, a commercial web scraping service with a Reddit actor (trudax/reddit-scraper). This worked technically &mdash; it returned post and comment data &mdash; but the data format was inconsistent and did not provide the repeated hourly snapshots needed for trajectory tracking. I tested it across several subreddits but concluded it was not suitable for the project's requirements.</p>

<p><strong>Attempt 3: Free JSON endpoints.</strong> I discovered that appending .json to any Reddit listing URL (e.g. reddit.com/r/technology/new.json) returns structured post metadata without authentication or rate limiting beyond basic politeness. I asked Claude Code whether this approach was viable for hourly collection and to write the collector. This produced collect_reddit_free.py, which became the sole data collection method for the rest of the project.</p>

<h2>Initial Generated Code</h2>

<p>The first working version was minimal:</p>
<ul>
<li><strong>collect_reddit_free.py</strong> &mdash; scrapes Reddit public JSON endpoints for new posts across five subreddits, with 1.1-second rate limiting</li>
<li><strong>build_reddit_history.py</strong> &mdash; merges all raw JSON snapshots into a unified timeline CSV</li>
<li>A basic Windows Task Scheduler job to run collection hourly</li>
</ul>

<p>At this stage there was no tracking pool &mdash; each hourly collection simply fetched whatever posts were currently listed as "new." There was no mechanism to follow the same post over time. If a post happened to appear in consecutive hourly scrapes, it would have multiple snapshots, but this was by luck rather than design. Many posts were observed only once. This created significant uncertainty about what kind of dataset was actually being built and whether it would be sufficient for trajectory prediction.</p>


<h2>AI Tools Used</h2>

<p>Two AI tools were used throughout the project. Claude Code (Anthropic, Claude Opus 4.6) was the primary development tool, used for the majority of code generation, data collection infrastructure, comment scraping, analysis, modelling, and the complete topic lifecycle pipeline. OpenAI Codex assisted with initial setup, specific infrastructure tasks, and a major pipeline refactoring on 31 March that separated prediction tracking from live monitoring and redesigned the lifecycle state model. All development was conversational: I directed priorities, questioned results, and pushed exploration when AI suggested premature ceilings.</p>

<h1>3. Critique of Initial Code, Iterative Development &amp; Justification</h1>

<p>With the initial code in place, the first priority was making it actually run. From there, each limitation was discovered and addressed in sequence &mdash; from basic bugs through to the complete topic lifecycle system.</p>

<h2>Phase 1: Getting It Running (March 24&ndash;26)</h2>

<p><strong>Iteration 1: Fixing AI-generated bugs.</strong> The initial code generated by AI could not run. datetime.UTC was used across 12 files but requires Python 3.11+ (the machine ran 3.10). f-string backslash syntax caused runtime failures. The Task Scheduler job crashed silently because AI generated WPF popup windows that fail in non-interactive mode. Unicode em dashes in the code corrupted under Windows cp1252 encoding. All of these required manual identification and correction before any data collection could begin.</p>

<p><strong>Iteration 2: Automated hourly collection.</strong> The initial collector only fetched "new" posts, missing rising, hot, and top listings which capture posts at different lifecycle stages. A schedule system was built with five cadences: hourly (new), every 2 hours (rising), every 4 hours (hot), twice daily (top/day), and daily (top/week). Task Scheduler was installed on two machines, results merged to improve coverage.</p>

<h2>Phase 2: Post-Level Prediction (March 27 &ndash; April 1)</h2>

<p>This phase was driven by the question I asked Claude Code: <em>"i am trying to predict general flow and just by post is this possible?"</em> The answer was yes, but it required building features the initial code did not have.</p>

<p><strong>Iteration 3: History building and gap patching.</strong> build_reddit_history.py merged all raw JSON snapshots into a unified post timeline with 216,944 rows. Each row captures a post at a specific hour: upvotes, comments, velocity (upvotes per hour), comment velocity, and age. patch_snapshot_gaps.py addressed a critical issue: when the collector missed an hour (laptop asleep, subway), the next snapshot showed velocity = 0 even for actively growing posts. This faked a dead-post signal and would corrupt all downstream models. The patcher recalculated velocity from the upvote delta divided by actual gap length, flagging 12.1%% of rows as collection gaps and interpolating 8.6%% of velocities.</p>

<p><strong>Iteration 4: Post lifecycle states.</strong> Each post snapshot was assigned a state based on upvote velocity relative to per-subreddit thresholds: surging (above surging threshold), alive (above alive threshold), cooling (positive but declining), dying (near zero), or dead (no engagement). These states serve as natural labels for classification without manual annotation. I asked <em>"we need to identify much earlier the dead post"</em> and explored whether velocity difference patterns could detect early death. The state model was later redesigned during the Codex refactoring session to provide earlier warning through a more conservative "dead" definition.</p>

<p><strong>Iteration 5: Comment-based prediction.</strong> I asked Claude Code: <em>"can we predict post flow with comments?"</em> This led to two major feature additions. First, 972,353 comments were scored using the VADER sentiment analyser (Hutto and Gilbert, 2014). The finding was counterintuitive: negative comment sentiment correlates with longer post survival (alive posts average sentiment -0.006 vs dead posts +0.045). Controversy drives engagement; apathy kills posts. The initial classifier using sentiment achieved 74.5%% accuracy, but feature importance analysis revealed that comment count alone accounted for 74%% &mdash; the model was counting comments, not analysing sentiment.</p>

<p><strong>Iteration 6: Comment engagement features (Gini coefficient).</strong> The Gini coefficient measures how comment upvotes are distributed within a post's discussion. High Gini (0.63&ndash;0.72) means a few comments dominate &mdash; community consensus. Low Gini (0.36) means diffuse, unfocused discussion. This became the strongest single post-level predictor at 46%% feature importance, surpassing all velocity and upvote features. Classifier accuracy improved from 74.5%% to 77.6%% with this single feature.</p>

<p><strong>Iteration 7: Per-subreddit models.</strong> After seeing the initial 64%% overall accuracy, I noticed that r/politics (72%%) performed much better than r/Games, and linked this to data volume: <em>"72 for politics? wowy and we see it's related to how much data it is."</em> This led to replacing the global model with per-subreddit classifiers. r/politics reached 81%% accuracy while r/Games reached 64%%. r/politics posts have an average lifecycle of 11 hours; r/Games posts survive 49 hours.</p>

<p><strong>Iteration 8: Multi-horizon survival prediction.</strong> I asked about predicting further ahead: <em>"after a couple of hours of observation, would we have general understanding would people discuss it on good level or not?"</em> This led to building seventeen binary classifiers for 1 hour to 7 days:</p>

<table>
<tr><th>Horizon</th><th>ROC AUC</th><th>Accuracy</th></tr>
<tr><td>1 hour</td><td>0.843</td><td>72%%</td></tr>
<tr><td>4 hours</td><td>0.834</td><td>74%%</td></tr>
<tr><td>12 hours</td><td>0.809</td><td>77%%</td></tr>
<tr><td>24 hours</td><td>0.771</td><td>80%%</td></tr>
<tr><td>48 hours</td><td>0.726</td><td>83%%</td></tr>
<tr><td>7 days</td><td>~0.57</td><td>85%%</td></tr>
</table>

<p>At 7 days, accuracy paradoxically rises to 85%% while ROC drops to 0.57 &mdash; predicting "everything dies" achieves high accuracy but zero discriminative power. This demonstrates why ROC AUC is more appropriate than accuracy for imbalanced classification.</p>

<p><strong>Iteration 9: Surging and dead detection.</strong> Specialised binary classifiers for specific states achieved the project's strongest post-level results: surging detection at 0.987 ROC AUC and dead detection at 0.945. State rise prediction achieved 0.947 ROC and barely decayed to 0.926 at 24 hours. Rise is more predictable than fall because it requires detectable signals; falling is the absence of signal.</p>

<p><strong>Iteration 10: Scenario-based prediction and anchoring.</strong> I proposed the idea of injecting external assumptions: <em>"what if we suggested a constant at particular date and parameters that will affect the flow?"</em> This became the scenario layer in predict_post_flow.py, where users can specify event assumptions (quiet, normal, moderate, major, breaking) that shift the initial state distribution. An anchor layer was added to replace historical distributions with actual 2-hour observations when available.</p>

<p><strong>Iteration 11: Post outcome and time-to-death.</strong> predict_post_outcome.py combined empirical growth multipliers per subreddit with state transition matrices from 116,000 observed transitions to produce pop/flop probabilities with estimated peak upvote range. predict_time_to_death.py achieved R&sup2;=0.459 with MAE of 9.1 hours. The top feature was states_seen_count (30%%) &mdash; posts that had been through more state transitions were more predictable.</p>

<p><strong>Iteration 12: Cross-subreddit success prediction.</strong> 1,305 detected cross-posts were analysed using title similarity matching. The politics&rarr;news route had 72%% success rate. worldnews broke stories first 516 times. Median propagation time: 11.1 hours.</p>

<p><strong>Iteration 13: Pipeline refactoring (Codex session 2).</strong> The tracking system was split into two lanes: a fixed cohort for prediction and a rolling shortlist for live monitoring. This separation was critical because a rolling pool biases tracking toward already-strong posts. The lifecycle state model was redesigned. Scheduler BOM bugs were fixed.</p>

<h2>Phase 3: Topic Lifecycle Prediction (April 7&ndash;8)</h2>

<p>Post-level prediction worked well at short horizons but degraded beyond 24 hours. I wanted to predict <em>general</em> flow &mdash; not individual posts but the overall story. This motivated the shift to topic-level prediction, tracking how stories emerge and spread regardless of which individual posts carry them.</p>

<p><strong>Iteration 14: Co-occurrence pair detection.</strong> Two-word pairs from the same title represent specific stories. "birthright+citizenship" is a Supreme Court case. "russian+tanker" is a naval incident. Temporal validation: 0.813 ROC AUC.</p>

""" + img_base64("fig1_topic_trajectories.png") + """
<p class="caption">Figure 3: Topic lifecycle trajectories. Ongoing stories (Hormuz Strait) show repeated surges. One-shot events (Easter+Trump) spike once and die.</p>

<p><strong>Iteration 15: Content-agnostic detection.</strong> The same algorithm detected: "official+trailer" (122 posts, game announcements), "hormuz+strait" (252 posts, geopolitical crisis), "crimson+desert" (65 posts, game launch), "media+social" (127 posts, tech policy). The model operates on engagement patterns alone.</p>

<p><strong>Iteration 16: Growth peak detection.</strong> Predict whether a topic has peaked or will keep growing. ROC AUC: 0.958. Dominant features: post growth day-over-day (52.7%%) and upvote growth (38.1%%).</p>

<p><strong>Iteration 17: Topic death prediction.</strong> "Will this topic die tomorrow?" ROC AUC: 0.890. Top features: post count (46.1%%), subreddit coverage (17.1%%).</p>

<p><strong>Iteration 18: Death speed classification.</strong> Quick death (0&ndash;1 days) vs slow death (2+ days). ROC AUC: 0.996. Decisive feature: decline rate on day after peak (67.8%%).</p>

""" + img_base64("fig5_pipeline_summary.png") + """
<p class="caption">Figure 4: Complete topic lifecycle prediction pipeline. Green = excellent (&gt;0.9), orange = good (&gt;0.8).</p>

<p><strong>Iteration 19: Subreddit spread prediction.</strong> Will a topic appear in a new subreddit tomorrow? r/politics: 0.756 ROC AUC. r/politics breaks stories first (47,580 times), not r/news. Most common route: news &rarr; worldnews (2,354 times).</p>

""" + img_base64("fig8_subreddit_spread.png") + """
<p class="caption">Figure 5: Left: which subreddit breaks stories first. Right: most common cross-subreddit spread routes.</p>

<p><strong>Iteration 20: Ongoing story vs one-shot event.</strong> When a topic drops, will it come back? ROC AUC: 0.970. Strongest feature: multiple peaks (21.5%% importance, 15.5x ratio).</p>

<table>
<tr><th>Signal</th><th>Ongoing story</th><th>One-shot event</th><th>Ratio</th></tr>
<tr><td>Multiple peaks</td><td>0.9</td><td>0.1</td><td>15.5x</td></tr>
<tr><td>Consistency</td><td>0.4</td><td>0.1</td><td>4.6x</td></tr>
<tr><td>Active days</td><td>3.2</td><td>0.8</td><td>4.0x</td></tr>
<tr><td>Peak posts</td><td>3.8</td><td>1.6</td><td>2.3x</td></tr>
</table>

""" + img_base64("fig12_ongoing_vs_oneshot.png") + """
<p class="caption">Figure 6: Feature ratios distinguishing ongoing stories from one-shot events.</p>

<p><strong>Iteration 21: Topic death definition.</strong> 1-day definition has 13.1%% false-death rate. Topics peaking at 8&ndash;11 posts revive 44.8%% of the time. Two-consecutive-day definition reduces false deaths to 6.8%%.</p>

""" + img_base64("fig6_revival_rates.png") + """
<p class="caption">Figure 7: Left: false death rates by definition. Right: revival rates by topic size.</p>

<p><strong>Iteration 22: Noise filter.</strong> 98.6%% of word pairs never grow. At 99%% confidence, filters out 87%% of pairs with 99.5%% precision. ROC AUC: 0.824.</p>

<p><strong>Iteration 23: Model comparison and hyperparameter tuning.</strong></p>

""" + img_base64("fig7_model_heatmap.png") + """
<p class="caption">Figure 8: Model performance across all tasks. Bold = best per task. No single model dominates.</p>

<table>
<tr><th>Model</th><th>Default ROC</th><th>Tuned ROC</th><th>Improvement</th></tr>
<tr><td>Decision Tree</td><td>0.577</td><td>0.842</td><td>+0.265</td></tr>
<tr><td>Gradient Boosting</td><td>0.713</td><td>0.835</td><td>+0.122</td></tr>
<tr><td>Random Forest</td><td>0.820</td><td>0.846</td><td>+0.027</td></tr>
<tr><td>Logistic Regression</td><td>0.859</td><td>0.860</td><td>+0.001</td></tr>
</table>

""" + img_base64("fig10_gbm_learning_rate.png") + """
<p class="caption">Figure 9: Gradient Boosting learning rate tuning. Peaks at 0.02, degrades at higher rates.</p>

<h1>4. Final Code Evaluation and Reflection</h1>

<h2>Post-Level Prediction Results</h2>

<p>Post-level models predict the trajectory of individual posts. These were developed first and provided the foundation for topic-level work.</p>

<table>
<tr><th>Task</th><th>ROC AUC</th><th>Key Feature</th></tr>
<tr><td>Surging detection</td><td>0.987</td><td>Upvote velocity</td></tr>
<tr><td>State rise prediction (1h)</td><td>0.947</td><td>Current velocity + comment rate</td></tr>
<tr><td>Dead detection</td><td>0.945</td><td>Velocity collapse</td></tr>
<tr><td>Survival (1h)</td><td>0.843</td><td>Gini coefficient (46%%)</td></tr>
<tr><td>Survival (4h)</td><td>0.834</td><td>Gini + velocity</td></tr>
<tr><td>Survival (24h)</td><td>0.771</td><td>Per-subreddit models</td></tr>
<tr><td>Survival (7d)</td><td>~0.57</td><td>Base rate dominates</td></tr>
</table>

<p>Post prediction accuracy decays with time horizon: from 0.843 ROC at 1 hour to approximately 0.57 at 7 days. The Gini coefficient of comment upvote distribution was the strongest single feature at 46%% importance. VADER sentiment analysis revealed that negative sentiment correlates with longer survival (alive average -0.006 vs dead +0.045), confirming that controversy drives engagement.</p>

<h2>Topic Lifecycle Pipeline Results</h2>

<table>
<tr><th>Lifecycle Stage</th><th>Task</th><th>Best ROC AUC</th><th>Best Model</th></tr>
<tr><td>Filter</td><td>Discard noise (87%% filtered)</td><td>0.850</td><td>Logistic Regression</td></tr>
<tr><td>Birth</td><td>Detect emerging topic (1-3 to 5+)</td><td>0.860</td><td>Logistic Regression</td></tr>
<tr><td>Growth</td><td>Has it peaked or still growing?</td><td>0.958</td><td>Gradient Boosting</td></tr>
<tr><td>Spread</td><td>Will it reach r/politics?</td><td>0.756</td><td>Random Forest</td></tr>
<tr><td>Decline</td><td>Topic dying state detection</td><td>0.992</td><td>Random Forest</td></tr>
<tr><td>Death</td><td>Will it die tomorrow?</td><td>0.890</td><td>Random Forest</td></tr>
<tr><td>Death speed</td><td>Quick death or slow death?</td><td>0.999</td><td>Logistic Regression</td></tr>
<tr><td>Revival</td><td>Ongoing story or one-shot?</td><td>0.970</td><td>Random Forest</td></tr>
</table>

<p>A key discovery was the inverse relationship between post and topic predictability. Individual posts become chaotic beyond 24 hours (ROC decays from 0.843 to 0.57). Topics become more predictable at longer horizons because they follow momentum. r/politics has the best post prediction (81%%) but the worst topic prediction (R&sup2;=0.551). r/news is the reverse: worst posts (68%%) but best topics (R&sup2;=0.808).</p>

""" + img_base64("fig2_roc_emergence_models.png") + """
<p class="caption">Figure 10: ROC curves comparing five classifiers on topic emergence detection.</p>

<h2>Topic State Transitions</h2>

""" + img_base64("fig13_transition_matrix.png") + """
<p class="caption">Figure 11: Topic state transition matrix. Surging topics have only 4%% chance of remaining surging. Dead topics stay dead 94%% of the time, but 6%% revive.</p>

<h2>What Does Not Work</h2>

<p><strong>Magnitude prediction.</strong> The system detects whether a topic will grow (0.86 ROC) but cannot predict how large. Szabo-Huberman log-linear model achieved R&sup2;=0.22. All regression approaches failed because at 1&ndash;3 posts, 99.5%% of word pairs remain small.</p>

<p><strong>Exact timing.</strong> "Days until death" gives R&sup2;=-0.5 (worse than guessing the mean). Binary classification works; regression does not.</p>

<p><strong>Revival timing.</strong> "When will a dead topic come back?" achieves 0.578 ROC &mdash; barely above random. Revival depends on external events. However, predicting which topics are the type that revives works at 0.970 ROC.</p>

<h2>Limitations</h2>
<ul>
<li>13 days of data limits rare event learning (~62 topic growth events in test sets)</li>
<li>Daily topic granularity; hourly resolution would provide more signal</li>
<li>No NLP at topic level; ongoing stories distinguished by trajectory shape only</li>
<li>Magnitude prediction unsolved from early signals</li>
<li>All findings correlational, not causal</li>
</ul>

<h1>5. Reflection on AI-Assisted Coding</h1>

<h2>Where AI Was Effective</h2>
<p>AI tools enabled rapid prototyping across the entire pipeline. The initial collection system, history builder, and prediction framework were functional within the first session. Algorithm selection and boilerplate code generation were particularly effective.</p>

<h2>Where AI Was Misleading</h2>
<p><strong>Default model assumptions.</strong> Both Codex and Claude Code defaulted to Random Forest for every task. Logistic Regression outperforms it on the core task &mdash; not discovered until I explicitly requested comparison.</p>
<p><strong>Redundant feature suggestions.</strong> Claude Code recommended post-level activity states for topic prediction. These showed no separation (alive ratio: 0.9x). The developer identified this as redundant.</p>
<p><strong>Premature ceiling claims.</strong> AI declared ceilings multiple times. The developer pushed past each, discovering the lifecycle pipeline, spread prediction, and ongoing-vs-one-shot classification.</p>
<p><strong>Hyperparameter negligence.</strong> Default parameters used throughout. Decision Tree improved +0.265 and Gradient Boosting +0.122 with basic tuning.</p>

<h2>Validation Approach</h2>
<ul>
<li>Temporal validation: trained on days 1&ndash;8, tested on days 9&ndash;13</li>
<li>Feature importance analysis to verify real patterns</li>
<li>Multiple classifier comparison to avoid algorithm bias</li>
<li>Testing established research models (Szabo-Huberman) against data</li>
<li>Honest reporting of negative results</li>
</ul>

<h2>Ethical Considerations</h2>
<ul>
<li>1.1-second rate limiting between API requests</li>
<li>Aggregate analysis only; no individual user profiling</li>
<li>Public data only via documented JSON endpoints</li>
<li>Transparent documentation of all AI tools used</li>
</ul>

<h1>References</h1>
<p>Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', <em>Proceedings of the International AAAI Conference on Web and Social Media</em>.</p>
<p>Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', <em>Journal of Machine Learning Research</em>, 12, pp. 2825-2830.</p>
<p>Szabo, G. and Huberman, B.A. (2010) 'Predicting the Popularity of Online Content', <em>Communications of the ACM</em>, 53(8), pp. 80-88.</p>

</body>
</html>
"""

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Saved: %s" % OUTPUT_PATH)
print("Open in browser: file:///%s" % os.path.abspath(OUTPUT_PATH).replace("\\", "/"))
