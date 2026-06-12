from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from IPython.display import display
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"
RANDOM_STATE = 23

pd.options.plotting.backend = "plotly"
pio.templates["palatino_white"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["palatino_white"].layout.font.family = "Palatino, Palatino Linotype, Book Antiqua, serif"
pio.templates["palatino_white"].data.scatter = [go.Scatter(marker={"size": 8})]
pio.templates["palatino_white"].data.scattergl = [go.Scattergl(marker={"size": 8})]
pio.templates.default = "palatino_white"
px.defaults.template = "palatino_white"
px.defaults.width = 840
px.defaults.height = 460


def display_df(df, rows=10):
    return display(df.head(rows))


def make_polynomial_sample(n=80, random_state=23):
    rng = np.random.default_rng(random_state)
    x = np.linspace(-5, 5, n)
    y = 2 + 0.8 * x - 0.35 * x ** 2 + 0.08 * x ** 3 + rng.normal(0, 3, n)
    return pd.DataFrame({"x": x, "y": y})


def one_hot_encoder(**kwargs):
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def find_birthweight_file(year=1971):
    candidates = [
        DATA_DIR / f"{year}.csv.gz",
        ROOT.parent / "DAIR3-Workshop" / "resources" / "unit_3" / "data" / f"{year}.csv.gz",
        ROOT.parent / "resources" / "unit_3" / "data" / f"{year}.csv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the prepared NCHS birthweight CSV. "
        "Run the Unit 3 prep.py script first, or place 1971.csv.gz in materials-draft/data.\n"
        f"Searched:\n{searched}"
    )


def load_birthweight_1971():
    return pd.read_csv(find_birthweight_file(1971))


def prepare_birthweight_modeling_data():
    births = load_birthweight_1971()
    cols = ["birthweight", "sex", "momage", "dadage", "plurality", "birthorder"]
    return births[cols].dropna().copy()


def make_interpretable_pipeline(numeric_cols, categorical_cols, model=None, scale_numeric=False):
    if model is None:
        model = LinearRegression()
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def regression_metrics(y_true, predictions):
    return pd.Series({
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": mean_squared_error(y_true, predictions) ** 0.5,
        "R2": r2_score(y_true, predictions),
    })


def model_diagnostics(fit_pipeline, X_train, X_test, y_train, y_test):
    train_predictions = fit_pipeline.predict(X_train)
    test_predictions = fit_pipeline.predict(X_test)

    def summarize(y_true, predictions):
        mse = mean_squared_error(y_true, predictions)
        return pd.Series({
            "MSE": mse,
            "RMSE": mse ** 0.5,
            "MAE": mean_absolute_error(y_true, predictions),
            "R^2": r2_score(y_true, predictions),
        })

    return pd.DataFrame({
        "train": summarize(y_train, train_predictions),
        "test": summarize(y_test, test_predictions),
    }).loc[["MSE", "RMSE", "MAE", "R^2"]]


def train_test_evaluate(model, X, y, test_size=0.2, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {"model": model, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "predictions": preds, "metrics": regression_metrics(y_test, preds)}


def compute_vif(X_numeric):
    X_clean = X_numeric.dropna().astype(float)
    values = X_clean.to_numpy()
    vifs = []
    for i in range(values.shape[1]):
        y = values[:, i]
        others = np.delete(values, i, axis=1)
        others = np.column_stack([np.ones(len(others)), others])
        fitted = others @ np.linalg.lstsq(others, y, rcond=None)[0]
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vifs.append(np.inf if np.isclose(r2, 1) else 1 / (1 - r2))
    return pd.DataFrame({
        "feature": X_clean.columns,
        "VIF": vifs,
    }).sort_values("VIF", ascending=False)


def cramers_v(x, y, bias_corrected=True):
    table = pd.crosstab(x, y)
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0 or min(observed.shape) < 2:
        return np.nan

    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    r, k = observed.shape
    phi2 = chi2 / n
    if not bias_corrected:
        return np.sqrt(phi2 / min(k - 1, r - 1))

    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2_corr / denom) if denom > 0 else np.nan


def categorical_association_matrix(df, columns):
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for col_a in columns:
        for col_b in columns:
            matrix.loc[col_a, col_b] = 1.0 if col_a == col_b else cramers_v(df[col_a], df[col_b])
    return matrix


def residual_frame(result):
    return result["X_test"].copy().assign(
        actual=result["y_test"].to_numpy(),
        predicted=result["predictions"],
        residual=result["y_test"].to_numpy() - result["predictions"],
    )


def subgroup_metrics(frame, actual_col="actual", pred_col="predicted", group_col="urbanicity"):
    columns = ["n", "MSE", "RMSE", "MAE", "R^2", "mean_residual"]

    def summarize(group):
        mse = mean_squared_error(group[actual_col], group[pred_col])
        return pd.Series({
            "n": len(group),
            "MSE": mse,
            "RMSE": mse ** 0.5,
            "MAE": mean_absolute_error(group[actual_col], group[pred_col]),
            "R^2": r2_score(group[actual_col], group[pred_col]) if len(group) > 1 else np.nan,
            "mean_residual": (group[actual_col] - group[pred_col]).mean(),
        })

    summary = frame.groupby(group_col, observed=True).apply(summarize).sort_values("MAE", ascending=False)
    summary["n"] = summary["n"].astype(int)
    return summary[columns]


def pipeline_coefficients(model, model_step="model", preprocessor_step="preprocessor"):
    fitted = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if hasattr(fitted, "named_steps"):
        preprocessor = fitted.named_steps.get(preprocessor_step)
        estimator = fitted.named_steps.get(model_step, fitted.steps[-1][1])
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            names = preprocessor.get_feature_names_out()
        else:
            names = np.arange(len(estimator.coef_))
    else:
        estimator = fitted
        names = np.arange(len(estimator.coef_))
    return pd.Series(estimator.coef_, index=names, name="coefficient").sort_values(key=np.abs, ascending=False)


def commute_coefs(model):
    fitted = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    names = fitted[0].get_feature_names_out()
    coefs = fitted[-1].coef_
    return pd.Series(coefs, index=names)
