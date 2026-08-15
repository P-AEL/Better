# Better prediction model

## Purpose

The model estimates UFC fight win probabilities from information available before
an event. It is an experimental forecasting system, not an instruction to wager.
The public site only publishes a betting signal after the market residual beats
the no-vig market on the untouched chronological test set and then repeats that
result on at least 100 forward-recorded priced predictions.

## Data and features

- Completed UFCStats fights are processed event by event in chronological order.
- Every training row is created before that event updates either fighter's state.
- Draws update ratings and statistics but are excluded from the binary target.
- No contests and scheduled fights are excluded from model targets.
- Each decisive fight is mirrored, enforcing red/blue orientation symmetry.
- Dynamic Glicko tracks fighter strength and rating uncertainty; inactivity
  increases uncertainty before the next fight.
- Features include Elo/Glicko, opponent quality, recent form, non-linear age and
  division-specific peak age, layoff, finish rate, trends, and exponentially
  weighted per-minute offense and defense.
- Fight statistics are standardized against only the prior observations in the
  same division. The feature ablation compares this core set against the full set.
- Missing physical information is median-imputed inside the fitted pipeline.

## Validation

Events, rather than individual rows, define all chronological boundaries. Model
selection uses expanding walk-forward folds on the first 70% of events. The next
15% calibrates probabilities with Platt scaling. The final 15% remains untouched
until model and market evaluation.

The challengers are spline logistic regression, shallow gradient boosting,
time-aware CatBoost, and dynamic Glicko. Their expanding-window out-of-fold
probabilities train a regularized meta-model. An independent later period fits
Platt calibration; the final 15% of events remains untouched until evaluation.

For priced fights, a regularized market residual uses market log-odds as a fixed
offset and learns only a correction from cross-fitted model predictions and a
small set of pre-fight features. Its regularization is selected on the calibration
period and evaluated once on the untouched test period.

## Betting gate

A signal requires all of the following:

- The market residual improves both test log loss and Brier score.
- Improvement in test log loss exceeds 0.001.
- At least 100 forward-priced predictions have settled, and the residual improves
  forward log loss by at least 0.005 as well as forward Brier score.
- Edge is at least 3 percentage points and expected value is at least 5%.
- Expected value stays positive after subtracting the calibration uncertainty
  margin from the estimated win probability.

Backtest-qualified bets remain `Paper` candidates until the forward gate passes.
Otherwise the site labels discrepancies as experimental and publishes no best
bet. Odds source and observation time are displayed when available. Each
new collection is also appended to `ufc_fight_odds_history.csv`, preserving the
future snapshots needed to measure line movement and closing-line value.

## Reproduction

Run `python train_model.py`, `python forward_test.py --settle-only`,
`python generate_site.py`, and `python forward_test.py`. Training writes
`betting_model.joblib` and `model_metrics.json`; the forward command maintains
`data/forward_predictions.csv` and `site/data/forward-results.json`.

## Known limitations

- Historical odds before model version 2 are not consistently timestamped and
  may represent closing prices rather than executable prices at a fixed decision
  time. New snapshots are timestamped, but this evidence must accumulate.
- UFCStats totals do not contain every technique, injury, camp, or opponent-quality
  signal that can affect a fight.
- Sparse histories make debutants and infrequent fighters particularly uncertain.
- Profitability must be established through forward paper trading and closing-line
  value before any staking policy is considered.
