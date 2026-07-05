# Match Decision Analytics Engine

**Tournaments:** FIFA World Cup 2022 | UEFA Euro 2024 | Copa America 2024
**Dataset:** 294 team-match records across 147 matches
**Data source:** StatsBomb open data

This is a decision analytics system built for coaching staff and analysts. It answers one question: given observable tactical decisions made before and during a match, what outcome should a team expect?

The project was built around the Cape Verde vs Argentina match (FIFA World Cup 2026, Round of 32, July 3) as the central case study. Cape Verde produced just 0.76 xG but scored 2 goals and pushed Argentina to extra time. The engine models why their tactical decisions nearly produced one of the biggest upsets in World Cup history.

---

## Architecture

Three layers work together:

- **SQL layer** - in-memory SQLite database storing all match features per team per match. Every analysis query is written in SQL for auditability.
- **Decision Tree 1** - outcome classifier predicting Win, Draw or Loss from tactical decision inputs.
- **Decision Tree 2** - phase-level goal regressors predicting expected goals scored in H1, H2 and ET separately.

---

## Data Layer

### What StatsBomb open data is

StatsBomb is a football analytics company that publishes a subset of their event data for free. Event data means every single action in a match is logged as a row: every pass, shot, press, duel, substitution, with timestamps, locations and dozens of attributes per event. This is fundamentally different from match stats which only give you final aggregates like "22 shots, 64% possession." Event data lets you compute things like "what was the xG on shots taken while under pressure in the second half."

### Why three tournaments

Using WC 2022 alone risks learning patterns specific to that tournament's conditions. Mixing in Euro 2024 (European style, high pressing teams) and Copa America 2024 (South American style, more compact defending) gives the model breadth and makes it generalise across tactical philosophies.

---

## Feature Engineering

Every match produces two rows, one per team. Each row has 20+ features extracted from raw events.

### xG (expected goals)

Summed from `shot_statsbomb_xg` across all shots by that team. StatsBomb's xG model assigns each shot a probability of being a goal based on location, angle, body part, game state and many other factors. A shot from six yards out might be 0.45 xG. A long-range effort might be 0.04. Summing across all shots gives the total deserved goals - what a team should have scored given the quality of chances they created.

### opp_xg and xg_diff

Same calculation for the opponent. `xg_diff = xg - opp_xg` is the single most predictive feature in the model because it captures whether a team dominated or was dominated in quality terms. A team that creates 2.5 xG and concedes 0.6 xG won the match on merit even if the scoreline was 1-1.

### Possession proxy

Instead of using StatsBomb's possession column directly (which is inconsistent across tournaments), pass count share is used: `passes / (passes + opponent_passes)`. This is a reliable proxy because possession and passing volume are tightly correlated.

### Shots on target ratio (sot_ratio)

`shots_on_target / total_shots`. This is the shot selection discipline metric and the primary decision variable for whether a team prioritises quality over volume. Cape Verde took 16 shots and had 5 on target: 31%. The model shows this discipline is the single highest-value decision for an underdog.

### Press ratio

`team_pressures / (team_pressures + opponent_pressures)`. A pressure event in StatsBomb is logged when a player actively closes down the ball carrier. A team in a deep low block generates very few pressures. A high-pressing team generates many. The ratio tells you which team is doing more of the pressing. Cape Verde's press ratio against Argentina was 0.38, consistent with their low-block shape.

### First substitution minute

The minute of the earliest substitution. A manager who makes their first change at minute 53 is responding tactically - fixing something or executing a planned adjustment to change the match. A manager who waits until minute 78 is managing energy. The data shows early substitutions correlate with wins for underdogs, reflecting tactical agility rather than weakness.

### xG under pressure (up_xg)

Summed xG from shots taken while the StatsBomb `under_pressure` flag was True. A team generating 0.8 xG with half of it under pressure is in a very different position to a team generating 0.8 xG from composed, unmarked efforts.

### Phase xG (xg_h1, xg_h2, xg_et)

StatsBomb labels each event with a period: 1 = first half, 2 = second half, 3 and 4 = extra time halves. Splitting xG by phase reveals where in the match a team generates its threat. Winning teams tend to produce more second-half xG. Underdogs often generate disproportionate ET xG because tired favourites open up.

### Duel ratio

`duels_won / total_duels`. Duels are contested 50-50 ball events. This adds a physical and positional aggression dimension that pressing ratio alone does not capture.

---

## SQL Layer

The full feature DataFrame is loaded into an in-memory SQLite database. Every downstream analysis runs through SQL rather than pandas, making the data transformations auditable and reproducible.

Three core queries power the dashboard:

- **Decision metrics by outcome** - groups all 294 records by Win, Draw, Loss and computes the average of every decision variable per group. Immediately surfaces which tactical inputs correlate with each outcome.
- **Underdog wins** - filters for matches won with under 42% possession. This surfaces the Morocco, Japan and Cape Verde style matches and shows what their tactical profile looked like.
- **Phase xG by tournament** - shows how goal threat varies by match phase across WC 2022, Euro 2024 and Copa America 2024. WC 2022 teams generate more second-half xG than Copa teams, reflecting genuine differences in how those competitions play out.

---

## Models

### Decision Tree 1 - Outcome Classifier

Predicts Win, Draw or Loss from 13 tactical features.

**Why a decision tree and not a more complex model**

For this use case, interpretability matters more than raw accuracy. A gradient boosted tree might get 65% accuracy but you cannot explain to a head coach why it predicted a loss. A decision tree can be walked through step by step: "your SoT ratio was below 0.30, and your first sub came before minute 58, and your press ratio was below 0.44 - therefore Loss." That conversation is possible. A neural network's reasoning is not.

**Training setup**

The model uses an 80/20 train-test split with stratification to preserve the Win/Draw/Loss distribution in both sets. Tree depth is tuned by looping depths 2 through 8 and selecting the depth with the best 5-fold cross-validated accuracy. Best depth came out at 2.

**Performance**

Cross-validated accuracy: **58.8%** on a 3-class problem where a random classifier scores 33.3%. That is a 25.5 percentage point lift over random, which is meaningful given the 294-record dataset size.

The model performs best on Wins (63% precision) and struggles most with Draws, which is expected - a 2-1 win and a 2-1 loss can have very similar tactical profiles on aggregate. Draws are the hardest outcome to predict from tactical inputs alone.

**Feature importance**

`xg_diff` and `sot_ratio` dominate the Gini importance scores. This validates the analytical story: the quality differential and shot selection discipline are the two strongest signals in the data. Pressing ratio and duel ratio contribute meaningfully. Substitution timing has smaller but consistent importance.

### Decision Tree 2 - Phase Goal Regressors

Three separate regression trees, one per phase, predicting how many goals a team will score in H1, H2 and ET.

**Why R2 varies so much across phases**

- H1 R2 is near zero or negative. First-half goals are essentially random given tactical inputs - matches are still forming and teams are reading each other.
- H2 R2 is slightly better but still modest. Patterns emerge but noise remains high.
- ET R2 reaches 0.58. By 90 minutes you know which team has been dominant, which has fatigued, which made proactive substitutions, and which is defending desperately. Extra time goal probability is genuinely predictable from the feature set.

The ET model is the most practically useful for coaching staff: it answers "if this match goes to extra time, how many goals should we realistically expect to score?"

### predict_scenario()

The interface the engine exposes. Pass in your team's tactical parameters and receive outcome probabilities plus phase-level goal forecasts.

```python
predict_scenario(
    xg=0.76, opp_xg=2.28, possession=0.36,
    sot_ratio=0.31, press_ratio=0.38,
    first_sub_min=55, n_subs=5, duel_ratio=0.52
)
# Returns:
# Win: 21.7% | Draw: 20.0% | Loss: 58.3%
# H1: 0.27 goals | H2: 0.29 goals | ET: 0.00 goals
```

---

## Dashboard

Eight charts, all using a shared dark tactical theme. Each one answers a specific question a coaching staff or analyst would actually ask.

### Chart 1 - Decision Radar

A radar chart plotting six decision dimensions simultaneously, normalised to 0-1 so they are comparable on the same scale. The three overlapping polygons show the average profile of winning, drawing and losing teams. Richer than a bar chart because all dimensions are visible simultaneously and overlap is informative.

### Chart 2 - Decision Space Map

The core visualisation. An 80x80 grid of pressing intensity and shot quality combinations is run through the outcome tree and coloured by predicted outcome. The result is a decision boundary map showing exactly where in the tactical space different strategies lead to different outcome regions. Every real team from the dataset is plotted on top, and Cape Verde is marked with a star so you can see where their actual profile sat relative to the decision boundaries. xG and opposing xG scale proportionally with the grid axes so the contour reflects the model's strongest predictors meaningfully.

### Chart 3 - Scenario Comparator

Six named scenarios compared side by side. Left panel shows Win, Draw, Loss probability stacked to 100%. Right panel shows expected goals by phase grouped per scenario. All scenario values are derived directly from StatsBomb event data:

- Cape Verde: actual 2026 match inputs
- Morocco WC22: average across wins vs Canada, Portugal and Belgium
- Japan WC22: average across giant-killing wins vs Spain and Germany
- Argentina: actual 2026 match inputs

The key analytical finding: Japan's giant-killing was built on aggressive high pressing (press ratio 0.70), not a low block. Cape Verde's was built on defensive shape and shot discipline. Two different paths to the same underdog result.

### Chart 4 - xG Efficiency Scatter

Each dot is one team in one match. X axis is xG generated, Y axis is actual goals scored. The dashed reference line is where xG equals goals (perfect calibration). Dots above the line overperformed, dots below underperformed. Annotated zones distinguish high efficiency territory (low xG, goals scored) from volume territory (high xG, fewer goals). Cape Verde sits furthest above the line at 0.76 xG and 2 goals: +1.24 overperformance, the largest in the dataset.

### Chart 5 - Phase xG Heatmap

A three-panel grid, one per tournament. Each cell shows average xG for a given phase and outcome combination. Reading across a row reveals where in the match a team generates its threat. Reading down a column shows how xG in that phase correlates with outcome. The cross-tournament comparison shows genuine tactical differences between competitions.

### Chart 6 - Substitution Timing Bubble Chart

Substitution timing buckets on the x axis, match count on the y axis, bubble size encoding the weight of evidence. The three outcome groups are offset horizontally so no bubble obscures another. The pattern shows early substitutions (pre-46 and 46-60 windows) skew toward wins for underdog teams, while very late substitutions distribute more evenly.

### Chart 7 - Confusion Matrix

Each cell shows both the raw count and the row-normalised percentage. Row normalisation is the key - it shows where the model is systematically wrong rather than just where the volume is. The model is strongest on Wins, reasonable on Losses, and weakest on Draws, which is the expected pattern for a model trained on tactical features.

### Chart 8 - Decision Waterfall

Starting from the tournament average team as baseline (37.6% win probability), one tactical input at a time is swapped from the dataset average to Cape Verde's actual value. The win probability change is recorded at each step.

This chart uses logistic regression rather than the decision tree because the tree at depth 2 has only 4 leaf nodes - too coarse to show meaningful movement when a single feature changes. Logistic regression produces a smooth continuous probability surface where every feature swap registers a real marginal effect.

The waterfall shows which decisions added win probability and which subtracted it. Conceding possession hurt. Absorbing high opp_xg hurt. But shot selection discipline and early tactical substitutions partially offset those losses. The net result: Cape Verde's full tactical profile produces a 22.0% win probability from the model, down 15.6 percentage points from the average team baseline. Given they were a 10% implied probability from the bookmakers, the model says their shape was worth roughly 12 percentage points of additional win probability that the market was not pricing in.

---

## Key Findings

**Shot quality is the most decisive controllable decision.** Teams that generate fewer but higher-quality shots win more often than high-volume, low-precision teams. For an underdog this is the single highest-value decision available.

**Pressing ratio has a non-linear relationship with outcome.** The data does not support the idea that more pressing always helps. Japan pressed at 70% and won. Cape Verde pressed at 38% and nearly won. Morocco pressed at 60% and won. The optimal level depends on the tactical matchup, not a universal rule.

**Early substitutions correlate with underdog wins.** Both Japan giant-killing matches had halftime substitutions. Cape Verde's first change came at minute 55. This reflects tactical agility - adjusting the game plan in real time - rather than injury or panic.

**Extra time is the most predictable phase.** Once 90 minutes of data are observable, the ET goal model achieves R2 of 0.58. A coaching staff that knows the model's ET prediction can make informed decisions about whether to chase a goal in the 91st minute or protect for penalties.

**Cape Verde's bookmaker probability (10%) significantly undervalued their tactical profile.** The model, trained on observable tactical features from three major tournaments, gives them 22% win probability once shot quality and defensive shape are accounted for. The market priced the name on the shirt. The model priced the decisions on the pitch.

---

## Usage

```python
# Predict outcome for any tactical scenario
out, phases = predict_scenario(
    xg=1.2,            # expected goals your team will generate
    opp_xg=1.8,        # expected goals you will concede
    possession=0.42,   # pass share (proxy for possession)
    sot_ratio=0.35,    # shots on target as fraction of total shots
    press_ratio=0.48,  # your pressing share vs opponent
    first_sub_min=58,  # minute of your first substitution
    n_subs=5,          # total substitutions made
    duel_ratio=0.45,   # fraction of duels won
)

print(f"Win: {out['Win']:.1%} | Draw: {out['Draw']:.1%} | Loss: {out['Loss']:.1%}")
print(f"H1 goals: {phases['H1 (0-45)']:.2f}")
print(f"H2 goals: {phases['H2 (45-90)']:.2f}")
print(f"ET goals: {phases['ET (90-120)']:.2f}")
```

---

## Dependencies

```
statsbombpy
scikit-learn
pandas
numpy
sqlite3
plotly
```

---

## Data Notes

All training data is sourced from StatsBomb open data (WC 2022, Euro 2024, Copa America 2024). The Cape Verde vs Argentina 2026 match stats used in the scenario inputs are manually sourced from post-match reporting and are not available through the StatsBomb open data pipeline, which does not yet cover the 2026 World Cup. The Morocco and Japan benchmark scenario values are computed as averages across their respective wins in the StatsBomb dataset and can be verified by querying the `match_features` SQL table directly.

---

## Model Limitations

The outcome tree operates at depth 2 with 4 leaf nodes. This makes the rules fully human-readable but means the probability surface is coarse. For the waterfall analysis, logistic regression is used instead because it produces smooth marginal contributions per feature. A future version could incorporate a random forest or gradient boosted tree for the probability estimates while retaining the decision tree for interpretability.

The dataset of 294 records is sufficient to demonstrate the analytical approach but too small to support confident generalisation across all tactical contexts. Adding club-level data from Champions League or top domestic leagues would substantially improve the model.

xG is not a perfect measure of chance quality and carries its own model assumptions from StatsBomb's underlying shot model. All findings should be interpreted as directional rather than absolute.
