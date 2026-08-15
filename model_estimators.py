import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def _logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -30, 30)))


class CatBoostAdapter(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        categorical_columns=(),
        iterations=400,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=8.0,
        random_seed=42,
    ):
        self.categorical_columns = categorical_columns
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def _prepare(self, frame):
        prepared = pd.DataFrame(frame).copy()
        for column in self.categorical_columns:
            prepared[column] = prepared[column].fillna("Unknown").astype(str)
        return prepared

    def _build_model(self, iterations):
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            has_time=True,
            allow_writing_files=False,
            verbose=False,
            thread_count=1,
        )

    def fit(self, X, y):
        frame = self._prepare(X)
        split = max(1, int(len(frame) * 0.9))
        train_x, train_y = frame.iloc[:split], np.asarray(y)[:split]
        validation_x, validation_y = frame.iloc[split:], np.asarray(y)[split:]
        self.model_ = self._build_model(self.iterations)
        fit_kwargs = {"cat_features": list(self.categorical_columns)}
        has_validation = len(validation_x) >= 20 and len(np.unique(validation_y)) == 2
        if has_validation:
            fit_kwargs.update({
                "eval_set": (validation_x, validation_y),
                "early_stopping_rounds": 40,
                "use_best_model": True,
            })
        self.model_.fit(train_x, train_y, **fit_kwargs)
        best_iteration = self.model_.get_best_iteration()
        final_iterations = (
            best_iteration + 1
            if best_iteration is not None and best_iteration >= 0
            else self.iterations
        )
        self.model_ = self._build_model(final_iterations)
        self.model_.fit(
            frame,
            np.asarray(y),
            cat_features=list(self.categorical_columns),
            verbose=False,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prepare(X))


def make_spline_logistic(numeric_columns, categorical_columns, c_value=0.1):
    nonlinear = [
        column for column in (
            "age_diff",
            "age_peak_distance_diff",
            "layoff_diff",
            "glicko_uncertainty_mean",
            "experience_total",
        )
        if column in numeric_columns
    ]
    linear = [column for column in numeric_columns if column not in nonlinear]
    preprocess = ColumnTransformer([
        ("nonlinear", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("scale", StandardScaler()),
        ]), nonlinear),
        ("linear", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), linear),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical_columns)),
    ])
    return Pipeline([
        ("features", preprocess),
        ("classifier", LogisticRegression(C=c_value, max_iter=2500)),
    ])


def make_gradient_model(numeric_columns, depth=3):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", GradientBoostingClassifier(
            max_depth=depth,
            n_estimators=250,
            learning_rate=0.035,
            min_samples_leaf=35,
            subsample=0.8,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


class TemporalEnsemble:
    def __init__(self, models, model_columns, meta_model, calibrator, glicko_column):
        self.models = models
        self.model_columns = model_columns
        self.meta_model = meta_model
        self.calibrator = calibrator
        self.glicko_column = glicko_column
        self.model_names = tuple(sorted(models))

    def meta_features(self, frame):
        columns = []
        for name in self.model_names:
            model = self.models[name]
            selected = frame[self.model_columns[name]]
            columns.append(model.predict_proba(selected)[:, 1])
        columns.append(_sigmoid(frame[self.glicko_column].to_numpy()))
        return np.column_stack(columns)

    def raw_probability(self, frame):
        return self.meta_model.predict_proba(self.meta_features(frame))[:, 1]

    def predict_proba(self, frame):
        raw = self.raw_probability(frame)
        calibrated = self.calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])


class MarketResidualModel:
    def __init__(self, feature_columns, l2=10.0):
        self.feature_columns = tuple(feature_columns)
        self.l2 = float(l2)

    def _matrix(self, model_probability, market_probability, frame, fit=False):
        numeric = frame[list(self.feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        delta = (_logit(model_probability) - _logit(market_probability)).reshape(-1, 1)
        matrix = np.column_stack([delta, numeric])
        if fit:
            self.medians_ = np.nanmedian(matrix, axis=0)
            matrix = np.where(np.isnan(matrix), self.medians_, matrix)
            self.means_ = matrix.mean(axis=0)
            self.scales_ = matrix.std(axis=0)
            self.scales_[self.scales_ < 1e-8] = 1.0
        else:
            matrix = np.where(np.isnan(matrix), self.medians_, matrix)
        return (matrix - self.means_) / self.scales_

    def fit(self, model_probability, market_probability, frame, target):
        matrix = self._matrix(model_probability, market_probability, frame, fit=True)
        market_logit = _logit(market_probability)
        target = np.asarray(target, dtype=float)

        def objective(parameters):
            correction = parameters[0] + matrix @ parameters[1:]
            probability = _sigmoid(market_logit + correction)
            loss = -np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))
            penalty = self.l2 * np.sum(parameters[1:] ** 2) / len(target)
            return loss + penalty

        result = minimize(objective, np.zeros(matrix.shape[1] + 1), method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"Market residual optimization failed: {result.message}")
        self.parameters_ = result.x
        return self

    def predict_probability(self, model_probability, market_probability, frame):
        matrix = self._matrix(model_probability, market_probability, frame)
        correction = self.parameters_[0] + matrix @ self.parameters_[1:]
        return _sigmoid(_logit(market_probability) + correction)
