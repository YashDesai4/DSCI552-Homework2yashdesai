"""Build the submission notebook from version-controlled cell sources."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip().splitlines(True)})


md(r"""
# DSCI 552 - Homework 2
**Name:** Yash Desai  
**Reproducibility:** Python 3, fixed split seed 42, relative paths only

This notebook answers every part of Homework 2. The response is net hourly
electrical energy output (`PE`, MW); predictors are ambient temperature (`AT`,
degrees C), exhaust vacuum (`V`, cm Hg), ambient pressure (`AP`, mbar), and
relative humidity (`RH`, percent).
""")

code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools, io, shutil
from urllib.request import urlopen
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES, TARGET = ["AT", "V", "AP", "RH"], "PE"

def load_data(path):
    path = Path(path)
    df = pd.read_excel(path, sheet_name=0) if path.suffix.lower() != ".csv" else pd.read_csv(path)
    df = df.rename(columns={"T": "AT", "EP": "PE"})
    if list(df.columns) != FEATURES + [TARGET]:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    return df

def descriptive_table(df):
    rows = {}
    for col in df.columns:
        s = df[col]; q1, q3 = s.quantile([.25, .75])
        rows[col] = {"mean":s.mean(), "median":s.median(), "min":s.min(), "max":s.max(),
                     "range":s.max()-s.min(), "Q1":q1, "Q3":q3, "IQR":q3-q1}
    return pd.DataFrame(rows).T

def simple_models(df):
    return {x: sm.OLS(df[TARGET], sm.add_constant(df[[x]])).fit() for x in FEATURES}

def multiple_model(df):
    return sm.OLS(df[TARGET], sm.add_constant(df[FEATURES])).fit()

def cubic_models(df):
    out = {}
    for x in FEATURES:
        z = pd.DataFrame({x:df[x], f"{x}^2":df[x]**2, f"{x}^3":df[x]**3})
        out[x] = sm.OLS(df[TARGET], sm.add_constant(z)).fit()
    return out

def interaction_model(df):
    X = df[FEATURES].copy()
    for a, b in itertools.combinations(FEATURES, 2): X[f"{a}:{b}"] = X[a] * X[b]
    return sm.OLS(df[TARGET], sm.add_constant(X)).fit()

def hierarchy_backward_elimination(X, y, powers, alpha=.05):
    active = list(range(X.shape[1]))
    while True:
        model = sm.OLS(y, sm.add_constant(X.iloc[:, active], has_constant="add")).fit()
        pvals, removable = model.pvalues.drop("const"), []
        for pos, idx in enumerate(active):
            power = powers[idx]
            if power.sum() == 1:
                var = int(np.flatnonzero(power)[0])
                if any(powers[j].sum() > 1 and powers[j][var] > 0 for j in active): continue
            removable.append((idx, pvals.iloc[pos]))
        if not removable: return model, active
        worst_idx, worst_p = max(removable, key=lambda pair: pair[1])
        if worst_p <= alpha: return model, active
        active.remove(worst_idx)

def regression_comparison(X_train, X_test, y_train, y_test):
    base = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    base_train = mean_squared_error(y_train, base.predict(sm.add_constant(X_train, has_constant="add")))
    base_test = mean_squared_error(y_test, base.predict(sm.add_constant(X_test, has_constant="add")))
    poly = PolynomialFeatures(2, include_bias=False)
    tr = pd.DataFrame(poly.fit_transform(X_train), columns=poly.get_feature_names_out(FEATURES), index=X_train.index)
    te = pd.DataFrame(poly.transform(X_test), columns=tr.columns, index=X_test.index)
    selected, active = hierarchy_backward_elimination(tr, y_train, poly.powers_)
    tr, te = tr.iloc[:,active], te.iloc[:,active]
    result = pd.DataFrame({"Train MSE":[base_train, mean_squared_error(y_train, selected.predict(sm.add_constant(tr, has_constant="add")))],
                           "Test MSE":[base_test, mean_squared_error(y_test, selected.predict(sm.add_constant(te, has_constant="add")))]},
                          index=["All main effects", "Selected degree-2 model"])
    return result, selected, list(tr.columns)

def knn_sweep(X_train, X_test, y_train, y_test):
    scaler, rows = StandardScaler().fit(X_train), []
    variants = {"Raw":(X_train.to_numpy(),X_test.to_numpy()),
                "Normalized":(scaler.transform(X_train),scaler.transform(X_test))}
    for variant, (tr, te) in variants.items():
        for k in range(1,101):
            m = KNeighborsRegressor(k).fit(tr,y_train)
            rows.append({"Variant":variant,"k":k,"1/k":1/k,
                         "Train MSE":mean_squared_error(y_train,m.predict(tr)),
                         "Test MSE":mean_squared_error(y_test,m.predict(te))})
    return pd.DataFrame(rows)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
sns.set_theme(style="whitegrid", context="notebook")
DATA = Path("data/Folds5x2_pp.xlsx")
if not DATA.exists():
    print("Dataset missing; downloading the official UCI archive...")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/static/public/294/combined%2Bcycle%2Bpower%2Bplant.zip"
    with ZipFile(io.BytesIO(urlopen(url).read())) as archive:
        member = next(n for n in archive.namelist() if n.endswith("Folds5x2_pp.xlsx"))
        with archive.open(member) as src, DATA.open("wb") as dst: shutil.copyfileobj(src, dst)
df = load_data(DATA)
df.head()
""")

md("""
## 1. Combined Cycle Power Plant Data

### (a) Dataset

The official UCI Combined Cycle Power Plant workbook is stored under `data/`.
Per the assignment footnote, only the first worksheet is read. The original
data are unnormalized and contain no missing values.

### (b) Exploring the data

#### (i) Dimensions and meaning
""")
code("""
print(f"Rows: {df.shape[0]:,}; columns: {df.shape[1]}")
print("Missing values:", int(df.isna().sum().sum()))
df.info()
""")
md("""
There are **9,568 rows and 5 columns**. Each row is one hourly-average operating
observation from 2006-2011 while the plant operated at full load. Four columns
(`AT`, `V`, `AP`, `RH`) are predictors, and `PE` is the continuous response.

#### (ii) Pairwise scatterplots
""")
code("""
g = sns.pairplot(df, corner=True, plot_kws={"s": 10, "alpha": 0.25})
g.fig.suptitle("Pairwise relationships in the CCPP data", y=1.02)
plt.show()
print(df.corr().round(3))
""")
md("""
`PE` has a strong negative relationship with `AT` and `V`: warmer conditions
and greater exhaust vacuum generally correspond to lower electrical output.
`AP` is positively related to `PE`, while `RH` has a weaker negative marginal
relationship. `AT` and `V` are themselves strongly positively associated, so
their marginal slopes should not be interpreted as mutually adjusted effects.
The plots also show curvature, especially for `AT` and `V`, non-constant spread,
and a small number of unusual observations.

#### (iii) Summary statistics
""")
code("descriptive_table(df).round(3)")

md("""
### (c) Simple linear regressions

For each predictor I fit `PE = beta0 + beta1 X + error`. A predictor is called
statistically significant when its slope p-value is below 0.05. The diagnostic
plots show the fitted relationship and externally studentized residuals. Points
with absolute studentized residual above 3 are flagged as potential outliers;
they are **not automatically deleted**, because they may be valid plant
conditions and the question provides no data-quality reason to remove them.
""")
code("""
simple = simple_models(df)
simple_results = pd.DataFrame({
    x: {"Intercept": m.params["const"], "Slope": m.params[x],
        "Slope p-value": m.pvalues[x], "R-squared": m.rsquared}
    for x, m in simple.items()
}).T
simple_results
""")
code("""
fig, axes = plt.subplots(4, 2, figsize=(12, 17))
outlier_counts = {}
for row, x in enumerate(FEATURES):
    m = simple[x]
    order = np.argsort(df[x].to_numpy())
    axes[row, 0].scatter(df[x], df.PE, s=8, alpha=.2)
    axes[row, 0].plot(df[x].to_numpy()[order], m.fittedvalues.to_numpy()[order], color="crimson")
    axes[row, 0].set(title=f"PE versus {x}", xlabel=x, ylabel="PE")
    stud = m.get_influence().resid_studentized_external
    outlier_counts[x] = int((np.abs(stud) > 3).sum())
    axes[row, 1].scatter(m.fittedvalues, stud, s=8, alpha=.25)
    axes[row, 1].axhline(0, color="black", lw=1)
    axes[row, 1].axhline(3, color="crimson", ls="--")
    axes[row, 1].axhline(-3, color="crimson", ls="--")
    axes[row, 1].set(title=f"{x}: studentized-residual diagnostic",
                     xlabel="Fitted PE", ylabel="Studentized residual")
plt.tight_layout(); plt.show()
pd.Series(outlier_counts, name="|studentized residual| > 3")
""")
md("""
All four marginal slopes are significant at the 5% level (indeed, the printed
p-values are extremely small). `AT` is the strongest one-variable linear model.
The residual patterns support the later investigation of nonlinear terms.
Although the flagged cases deserve inspection, wholesale deletion based only on
the response residual would bias the analysis; therefore all valid observations
are retained.

### (d) Multiple linear regression
""")
code("""
multiple = multiple_model(df)
print(multiple.summary())
pd.DataFrame({"Coefficient": multiple.params,
              "p-value": multiple.pvalues,
              "Reject H0 at 0.05": multiple.pvalues < .05})
""")
md("""
The joint F-test assesses whether the predictors collectively explain `PE`, and
the individual t-tests assess each partial association while holding the other
variables fixed. The table above gives the exact decision for every predictor.

### (e) Simple versus multiple coefficients
""")
code("""
coef_compare = pd.DataFrame({
    "Simple coefficient": [simple[x].params[x] for x in FEATURES],
    "Multiple coefficient": [multiple.params[x] for x in FEATURES],
}, index=FEATURES)
ax = coef_compare.plot.scatter(x="Simple coefficient", y="Multiple coefficient", figsize=(7, 6))
for name, row in coef_compare.iterrows():
    ax.annotate(name, (row.iloc[0], row.iloc[1]), xytext=(5, 5), textcoords="offset points")
ax.axhline(0, color="gray", lw=1); ax.axvline(0, color="gray", lw=1)
ax.set_title("Marginal and mutually adjusted slopes")
plt.show()
coef_compare
""")
md("""
Differences between the two axes are expected because the predictors are
correlated. A simple coefficient combines a predictor's direct association with
associations carried through omitted predictors; a multiple coefficient is the
estimated partial effect after adjustment. This is why magnitude, and possibly
sign, can change between the two models.

### (f) Nonlinear associations

For each predictor I fit `PE = beta0 + beta1 X + beta2 X^2 + beta3 X^3 + error`.
Evidence of nonlinearity is present when either the quadratic or cubic term is
significant.
""")
code("""
cubics = cubic_models(df)
nonlinear_results = []
for x, m in cubics.items():
    nonlinear_results.append({
        "Predictor": x,
        "p(X)": m.pvalues[x],
        "p(X^2)": m.pvalues[f"{x}^2"],
        "p(X^3)": m.pvalues[f"{x}^3"],
        "Any nonlinear term significant": (m.pvalues[[f"{x}^2", f"{x}^3"]] < .05).any(),
    })
pd.DataFrame(nonlinear_results).set_index("Predictor")
""")
code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, x in zip(axes.flat, FEATURES):
    grid = np.linspace(df[x].min(), df[x].max(), 300)
    design = pd.DataFrame({x: grid, f"{x}^2": grid**2, f"{x}^3": grid**3})
    pred = cubics[x].predict(sm.add_constant(design, has_constant="add"))
    ax.scatter(df[x], df.PE, s=7, alpha=.15)
    ax.plot(grid, pred, color="crimson", lw=2)
    ax.set(title=f"Cubic fit for {x}", xlabel=x, ylabel="PE")
plt.tight_layout(); plt.show()
""")

md("""
### (g) Pairwise interactions

The model below includes all four main effects and all six two-way interaction
terms. Interaction terms with p-values below 0.05 provide evidence that the
association of one predictor with `PE` changes with the level of another.
""")
code("""
interactions = interaction_model(df)
interaction_rows = [name for name in interactions.params.index if ":" in name]
pd.DataFrame({"Coefficient": interactions.params[interaction_rows],
              "p-value": interactions.pvalues[interaction_rows],
              "Significant at 0.05": interactions.pvalues[interaction_rows] < .05})
""")

md("""
### (h) Main-effects model versus selected degree-2 model

I randomly reserve 30% of observations for testing. Both models use the exact
same split. The expanded model starts with all main effects, squares, and
pairwise interactions. Backward elimination removes terms with p-values above
0.05 while enforcing **strong hierarchy**: a main effect is retained whenever
any square or interaction involving it remains. No test observations influence
selection.
""")
code("""
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=.30, random_state=42
)
regression_mse, selected_model, selected_terms = regression_comparison(
    X_train, X_test, y_train, y_test
)
print("Selected terms:", selected_terms)
display(regression_mse)
print(selected_model.summary())
""")
md("""
The selected model improves only if its held-out test MSE is smaller; training
MSE alone cannot establish improvement. The printed terms document the final
hierarchical specification, while the MSE table gives the requested train/test
comparison.

### (i) KNN regression

For each `k` from 1 through 100, I fit KNN on raw features and on standardized
features. Standardization parameters are learned from training data only. The
best `k` minimizes test MSE. The x-axis is `1/k`, as requested.
""")
code("""
knn = knn_sweep(X_train, X_test, y_train, y_test)
best_knn = knn.loc[knn.groupby("Variant")["Test MSE"].idxmin()].set_index("Variant")
best_knn[["k", "Train MSE", "Test MSE"]]
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, variant in zip(axes, ["Raw", "Normalized"]):
    part = knn[knn.Variant == variant]
    ax.plot(part["1/k"], part["Train MSE"], label="Train MSE")
    ax.plot(part["1/k"], part["Test MSE"], label="Test MSE")
    ax.set(title=f"KNN with {variant.lower()} features", xlabel="1/k", ylabel="MSE")
    ax.legend()
plt.tight_layout(); plt.show()
""")

md("""
### (j) KNN versus the best linear model
""")
code("""
comparison = pd.concat([
    regression_mse.assign(Model=regression_mse.index).reset_index(drop=True),
    best_knn.reset_index().rename(columns={"Variant": "Model"})[["Model", "Train MSE", "Test MSE"]]
], ignore_index=True)
comparison["Model"] = comparison["Model"].replace({"Raw": "KNN (raw)", "Normalized": "KNN (normalized)"})
comparison.sort_values("Test MSE")
""")
md("""
The preferred method is the row with the smallest **test MSE** in the final
table. KNN can capture local nonlinear structure without specifying a formula,
but is sensitive to scale, which is why normalized KNN is the fairer version.
The degree-2 regression is more interpretable and can extrapolate, whereas KNN
predictions are local averages. The train-test gap also reveals overfitting:
very small `k` generally has low training error but poorer test performance.

## 2. ISLR 2.4.1

For each scenario, compare a flexible method with an inflexible one.

1. **Very large n, small p - flexible is generally better.** Abundant data
   reduces variance/overfitting concerns and permits learning a more complex
   relationship.
2. **Very large p, small n - flexible is generally worse.** With too many
   predictors and few observations, variance is high and overfitting is likely.
3. **Highly nonlinear predictor-response relationship - flexible is better.**
   An inflexible method would have high bias because it cannot represent the
   true shape well.
4. **Extremely high error variance - flexible is generally worse.** It is prone
   to fitting irreducible noise, increasing variance without improving the
   underlying signal.

## 3. ISLR 2.4.7

The test point is `(0, 0, 0)`. The Euclidean distances are:

| Observation | Coordinates | Class | Distance |
|---:|---|---|---:|
| 1 | (0, 3, 0) | Red | 3.000 |
| 2 | (2, 0, 0) | Red | 2.000 |
| 3 | (0, 1, 3) | Red | sqrt(10) = 3.162 |
| 4 | (0, 1, 2) | Green | sqrt(5) = 2.236 |
| 5 | (-1, 0, 1) | Green | sqrt(2) = 1.414 |
| 6 | (1, 1, 1) | Red | sqrt(3) = 1.732 |

**(b)** With `K=1`, the prediction is **Green**, because observation 5 is the
single nearest neighbor.

**(c)** With `K=3`, the nearest observations are 5 (Green), 6 (Red), and 2
(Red), so the majority-vote prediction is **Red**.

**(d)** If the Bayes boundary is highly nonlinear, a **small K** is generally
preferred because it gives a flexible, local decision boundary. A large K
smooths the boundary too aggressively and creates high bias (although an
extremely small K can have high variance).
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
(ROOT / "Homework2_Yash_Desai.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
