# Fight-method model

`method_model.py` estimates how a UFC bout will end: `ko_tko`, `submission`,
`decision`, or official `nc`. It does not predict the winner.

Run `python -B method_model.py` to rebuild `fight_method_model.joblib` and
`fight_method_metrics.json`, then run `python -B generate_site.py` to publish
the static prediction endpoint at `site/data/method-predictions.json`.

The training data is processed event by event in chronological order. Every
feature is formed before the event and absolute pair differences make inference
invariant to fighter order. DQ, overturned, blank, and other unmapped methods
are excluded from targets; their completed bouts still update historical fighter
state. The report records its exact mapping, cutoff, schema, dependency versions,
chronological split sizes, baseline comparison, calibration, class counts,
confusion matrix, precision/recall, and available segment results.

The current source lacks a trustworthy scheduled-rounds field, so training uses
the standard three-round context. Five-round predictions are consequently shown
as limited-data estimates. NC has very few examples and is smoothed toward its
historical rate; it should not be treated as a reliable individual-fight forecast.
