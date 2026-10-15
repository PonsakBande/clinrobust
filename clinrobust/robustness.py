"""Robustness experiments: MCAR missingness, imputation, noise, elimination.

Design guarantees baked in:
  * imputers are ALWAYS fitted on training data only (no test leakage);
  * KNN distances are computed in standardised space (raw-unit distances
    would be dominated by whichever feature has the largest scale);
  * every random draw uses an explicit integer seed (no hash-based seeds).
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score)


def mcar_mask(X, rate, seed):
    """Mask exactly round(rate * n_cells) individual cells (MCAR)."""
    rng = np.random.RandomState(seed)
    arr = X.to_numpy(dtype=float, copy=True)
    n_cells = arr.size
    n_missing = int(round(n_cells * rate))
    flat = rng.choice(n_cells, size=n_missing, replace=False)
    mask = np.zeros(n_cells, dtype=bool)
    mask[flat] = True
    arr.flat[mask] = np.nan
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def fit_imputer(method, X_train, knn_k=5):
    """Fit an imputer on TRAINING data only."""
    if method == "mean":
        imp = SimpleImputer(strategy="mean")
        imp.fit(X_train)
        return {"method": method, "imputer": imp}
    if method == "median":
        imp = SimpleImputer(strategy="median")
        imp.fit(X_train)
        return {"method": method, "imputer": imp}
    if method == "knn":
        scaler = StandardScaler()
        imp = KNNImputer(n_neighbors=knn_k)
        imp.fit(scaler.fit_transform(X_train))
        return {"method": method, "scaler": scaler, "imputer": imp}
    raise ValueError(f"unknown imputation method: {method}")


def transform_imputer(fitted, X):
    """Apply a train-fitted imputer; returns original units and column order."""
    if fitted["method"] in ("mean", "median"):
        out = fitted["imputer"].transform(X)
    else:
        Xs = fitted["scaler"].transform(X)
        out = fitted["scaler"].inverse_transform(fitted["imputer"].transform(Xs))
    return pd.DataFrame(out, columns=X.columns, index=X.index)


def add_gaussian_noise(X, feature, level, train_std, seed):
    rng = np.random.RandomState(seed)
    Xn = X.astype(float).copy()
    Xn[feature] = Xn[feature] + rng.normal(0.0, level * train_std, size=len(Xn))
    return Xn


def missingness_simulation(model, X_train, X_test, y_test, rates, methods,
                           reps, seed, knn_k=5):
    """Evaluate a FIXED model on MCAR-masked, imputed test data.

    Within each (rate, rep) the SAME mask is used for every method, so methods
    are compared on identical missingness patterns.
    """
    summary, raw = [], []
    for rate in rates:
        for method in methods:
            fitted = fit_imputer(method, X_train, knn_k=knn_k)
            accs, f1s = [], []
            for rep in range(reps):
                mseed = seed + 1000 * int(rate * 100) + rep
                Xi = transform_imputer(fitted, mcar_mask(X_test, rate, mseed))
                pred = model.predict(Xi)
                accs.append(accuracy_score(y_test, pred))
                f1s.append(f1_score(y_test, pred, average="macro",
                                    zero_division=0))
                raw.append({"missingness_pct": int(rate * 100), "method": method,
                            "rep": rep + 1, "accuracy": accs[-1],
                            "macro_f1": f1s[-1]})
            summary.append({
                "missingness_pct": int(rate * 100), "method": method,
                "mean_accuracy": np.mean(accs), "std_accuracy": np.std(accs),
                "acc_p2_5": np.percentile(accs, 2.5),
                "acc_p97_5": np.percentile(accs, 97.5),
                "mean_macro_f1": np.mean(f1s), "std_macro_f1": np.std(f1s),
                "f1_p2_5": np.percentile(f1s, 2.5),
                "f1_p97_5": np.percentile(f1s, 97.5)})
    return pd.DataFrame(summary), pd.DataFrame(raw)


def noise_sensitivity(model, X_test, y_test, features, train_stds, levels,
                      reps, seed, labels=(0, 1, 2)):
    """Per-feature Gaussian noise; reports drops in accuracy, balanced accuracy
    and minority-class (labels[-1]) recall relative to the clean baseline."""
    pred0 = model.predict(X_test)
    base_acc = accuracy_score(y_test, pred0)
    base_bal = balanced_accuracy_score(y_test, pred0)
    base_ad = recall_score(y_test, pred0, labels=[labels[-1]], average="macro",
                           zero_division=0)
    rows = []
    for f_idx, feat in enumerate(features):
        for level in levels:
            accs, bals, ads = [], [], []
            for rep in range(reps):
                nseed = seed + 10000 * f_idx + 100 * int(round(level * 10)) + rep
                pred = model.predict(
                    add_gaussian_noise(X_test, feat, level,
                                       train_stds[feat], nseed))
                accs.append(accuracy_score(y_test, pred))
                bals.append(balanced_accuracy_score(y_test, pred))
                ads.append(recall_score(y_test, pred, labels=[labels[-1]],
                                        average="macro", zero_division=0))
            rows.append({"feature": feat, "level": level,
                         "mean_accuracy": np.mean(accs),
                         "std_accuracy": np.std(accs),
                         "acc_drop": base_acc - np.mean(accs),
                         "balacc_drop": base_bal - np.mean(bals),
                         "ad_recall_drop": base_ad - np.mean(ads)})
    return pd.DataFrame(rows)


def backward_elimination_cv(X, y, removal_order, estimator, n_splits=5,
                            n_repeats=10, seed=42, labels=(0, 1, 2)):
    """Remove features in the given order; evaluate each subset by repeated
    stratified CV with pooled out-of-fold predictions. Hyperparameters are
    held fixed (estimator is cloned per fold)."""
    def cv_once(features, rep):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + rep)
        pred = cross_val_predict(clone(estimator), X[features], y, cv=skf)
        rec = recall_score(y, pred, labels=list(labels), average=None,
                           zero_division=0)
        return {"accuracy": accuracy_score(y, pred),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "recall_ad": rec[labels[-1]]}

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "recall_ad"]
    remaining = list(X.columns)
    removed = "-"
    rows = []
    for step in range(len(remaining)):
        reps = [cv_once(remaining, rep) for rep in range(n_repeats)]
        row = {"n_features": len(remaining), "removed_to_get_here": removed,
               "features_kept": ", ".join(remaining)}
        for m in metrics:
            vals = [r[m] for r in reps]
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sd"] = float(np.std(vals))
        rows.append(row)
        if len(remaining) > 1:
            removed = removal_order[step]
            remaining.remove(removed)
    df = pd.DataFrame(rows)
    full = df.iloc[0]
    for m in metrics:
        df[f"{m}_pct_ref"] = df[f"{m}_mean"] / full[f"{m}_mean"] * 100
    return df
