"""Reusable analysis helpers for DSCI 552 Homework 2."""
from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"


def load_data(path: str | Path = "data/Folds5x2_pp.xlsx") -> pd.DataFrame:
    """Load Sheet 1 and normalize legacy T/EP headings to UCI's AT/PE names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python download_data.py` from the repository root."
        )
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=0)
    df = df.rename(columns={"T": "AT", "EP": "PE"})
    expected = FEATURES + [TARGET]
    if list(df.columns) != expected:
        raise ValueError(f"Expected columns {expected}; found {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in the UCI dataset")
    return df


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in df.columns:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        rows[col] = {
            "mean": s.mean(), "median": s.median(), "min": s.min(),
            "max": s.max(), "range": s.max() - s.min(),
            "Q1": q1, "Q3": q3, "IQR": q3 - q1,
        }
    return pd.DataFrame(rows).T


def simple_models(df: pd.DataFrame) -> dict[str, object]:
    return {x: sm.OLS(df[TARGET], sm.add_constant(df[[x]])).fit() for x in FEATURES}


def multiple_model(df: pd.DataFrame):
    return sm.OLS(df[TARGET], sm.add_constant(df[FEATURES])).fit()


def cubic_models(df: pd.DataFrame) -> dict[str, object]:
    models = {}
    for x in FEATURES:
        z = pd.DataFrame({x: df[x], f"{x}^2": df[x] ** 2, f"{x}^3": df[x] ** 3})
        models[x] = sm.OLS(df[TARGET], sm.add_constant(z)).fit()
    return models


def interaction_model(df: pd.DataFrame):
    X = df[FEATURES].copy()
    for a, b in itertools.combinations(FEATURES, 2):
        X[f"{a}:{b}"] = X[a] * X[b]
    return sm.OLS(df[TARGET], sm.add_constant(X)).fit()


def hierarchy_backward_elimination(
    X: pd.DataFrame, y: pd.Series, powers: np.ndarray, alpha: float = 0.05
):
    """Backward p-value selection while enforcing strong hierarchy.

    A main effect cannot be removed while a square or interaction containing it
    remains in the model.
    """
    active = list(range(X.shape[1]))
    while True:
        model = sm.OLS(y, sm.add_constant(X.iloc[:, active], has_constant="add")).fit()
        pvals = model.pvalues.drop("const")
        removable = []
        for pos, original_idx in enumerate(active):
            power = powers[original_idx]
            if power.sum() == 1:  # main effect: retain if a child term remains
                variable = int(np.flatnonzero(power)[0])
                has_child = any(
                    powers[j].sum() > 1 and powers[j][variable] > 0 for j in active
                )
                if has_child:
                    continue
            removable.append((original_idx, pvals.iloc[pos]))
        if not removable:
            return model, active
        worst_idx, worst_p = max(removable, key=lambda pair: pair[1])
        if worst_p <= alpha:
            return model, active
        active.remove(worst_idx)


def regression_comparison(X_train, X_test, y_train, y_test):
    base = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    base_train = mean_squared_error(y_train, base.predict(sm.add_constant(X_train, has_constant="add")))
    base_test = mean_squared_error(y_test, base.predict(sm.add_constant(X_test, has_constant="add")))

    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_arr = poly.fit_transform(X_train)
    test_arr = poly.transform(X_test)
    names = poly.get_feature_names_out(FEATURES)
    train_poly = pd.DataFrame(train_arr, columns=names, index=X_train.index)
    test_poly = pd.DataFrame(test_arr, columns=names, index=X_test.index)
    selected, active = hierarchy_backward_elimination(train_poly, y_train, poly.powers_)
    selected_train = train_poly.iloc[:, active]
    selected_test = test_poly.iloc[:, active]
    selected_train_mse = mean_squared_error(
        y_train, selected.predict(sm.add_constant(selected_train, has_constant="add"))
    )
    selected_test_mse = mean_squared_error(
        y_test, selected.predict(sm.add_constant(selected_test, has_constant="add"))
    )
    result = pd.DataFrame(
        {"Train MSE": [base_train, selected_train_mse],
         "Test MSE": [base_test, selected_test_mse]},
        index=["All main effects", "Selected degree-2 model"],
    )
    return result, selected, list(selected_train.columns)


def knn_sweep(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    records = []
    scaler = StandardScaler().fit(X_train)
    variants = {
        "Raw": (X_train.to_numpy(), X_test.to_numpy()),
        "Normalized": (scaler.transform(X_train), scaler.transform(X_test)),
    }
    for variant, (train_x, test_x) in variants.items():
        for k in range(1, 101):
            model = KNeighborsRegressor(n_neighbors=k).fit(train_x, y_train)
            records.append({
                "Variant": variant, "k": k, "1/k": 1 / k,
                "Train MSE": mean_squared_error(y_train, model.predict(train_x)),
                "Test MSE": mean_squared_error(y_test, model.predict(test_x)),
            })
    return pd.DataFrame(records)
