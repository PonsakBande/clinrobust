"""Model construction, tuning, evaluation and uncertainty quantification."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, precision_score, confusion_matrix)


def build_pipeline(seed=42, class_weight="balanced"):
    """impute(mean) -> scale -> class-weighted Random Forest."""
    rf = RandomForestClassifier(random_state=seed, class_weight=class_weight,
                                n_jobs=-1)
    return Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("scale", StandardScaler()),
                     ("rf", rf)])


def tune_model(X_train, y_train, mcfg, seed=42):
    """Grid-search the pipeline on TRAINING data only."""
    pipe = build_pipeline(seed=seed,
                          class_weight=mcfg.get("class_weight", "balanced"))
    gs = GridSearchCV(pipe, mcfg["param_grid"], cv=mcfg.get("cv_folds", 5),
                      scoring=mcfg.get("scoring", "balanced_accuracy"),
                      n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs


def compute_metrics(y_true, y_pred, labels=(0, 1, 2)):
    """All headline metrics plus per-class recall/precision and confusion matrix."""
    rec = recall_score(y_true, y_pred, labels=list(labels), average=None,
                       zero_division=0)
    prec = precision_score(y_true, y_pred, labels=list(labels), average=None,
                           zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)),
    }
    for i, lab in enumerate(labels):
        out[f"recall_{lab}"] = rec[i]
        out[f"precision_{lab}"] = prec[i]
    return out


def stratified_bootstrap_ci(y_true, y_pred,
                            metric_names=("accuracy", "balanced_accuracy",
                                          "macro_f1", "recall_2"),
                            n_boot=1000, seed=42, labels=(0, 1, 2)):
    """95% CIs from within-class resampling (preserves class sizes exactly).

    Stratification prevents the empty-minority-class artifact that an
    unstratified bootstrap produces at small n.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_idx = {c: np.where(y_true == c)[0] for c in labels}
    values = {m: [] for m in metric_names}
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(class_idx[c], size=len(class_idx[c]), replace=True)
            for c in labels
        ])
        m = compute_metrics(y_true[idx], y_pred[idx], labels=labels)
        for name in metric_names:
            values[name].append(m[name])
    return {name: {"mean": float(np.mean(v)),
                   "lower": float(np.percentile(v, 2.5)),
                   "upper": float(np.percentile(v, 97.5))}
            for name, v in values.items()}
