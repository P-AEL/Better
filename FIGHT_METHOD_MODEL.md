# Fight-method model

`method_model.py` estimates whether a UFC bout ends by `decision` or
`not_decision`. It does not predict the winner or the particular non-decision
method.

Run `python -B method_model.py` to rebuild `fight_method_model.joblib` and
`fight_method_metrics.json`, then run `python -B generate_site.py` to publish
the static prediction endpoint at `site/data/method-predictions.json`.

The training data is processed event by event in chronological order. Every
feature is formed before the event and absolute pair differences make inference
invariant to fighter order. Any completed outcome that is not an official
decision is the `not_decision` target. The report records its exact mapping, cutoff, schema, dependency versions,
chronological split sizes, baseline comparison, calibration, class counts,
confusion matrix, precision/recall, and available segment results.

The current source lacks a trustworthy scheduled-rounds field, so training uses
the standard three-round context. Five-round predictions are consequently shown
as limited-data estimates.
