"""Configuration handling for clinrobust.

A single nested dictionary controls the entire audit. To analyse a new
dataset (e.g. OASIS-3), copy config.yaml, edit it, and call
load_config("path/to/new_config.yaml") -- no code changes required.
"""
import copy

DEFAULT_CONFIG = {
    "project": {"name": "ad-robustness-audit", "seed": 42},

    "data": {
        "path": "/content/drive/MyDrive/oasis_longitudinal.csv",
        "subject_id": "Subject ID",
        # keep only rows matching every {column: value} pair (e.g. baseline visit)
        "filters": {"Visit": 1},
        # rename raw columns to canonical names
        "rename": {"M/F": "Gender"},
        # map raw category labels to numeric codes
        "encode": {"Gender": {"M": 0, "F": 1}},
        "clinical_features": ["Age", "Gender", "EDUC", "MMSE"],
        "imaging_features": ["eTIV", "nWBV", "ASF"],
        "target": {
            "source": "CDR",
            "column": "Target",
            "rules": [
                {"op": "eq", "value": 0.0, "code": 0, "label": "Normal"},
                {"op": "eq", "value": 0.5, "code": 1, "label": "MCI"},
                {"op": "ge", "value": 1.0, "code": 2, "label": "AD"}
            ],
            "class_names": ["Normal", "MCI", "AD"]
        }
    },

    "split": {"test_size": 0.2, "stratify": True},

    "model": {
        "type": "random_forest",
        "class_weight": "balanced",
        "scoring": "balanced_accuracy",
        "cv_folds": 5,
        "param_grid": {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [5, 10, None],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2]
        }
    },

    "missingness": {
        "rates": [0.10, 0.20, 0.30, 0.40, 0.50],
        "methods": ["mean", "median", "knn"],
        "knn_k": 5,
        "reps": 50
    },

    "noise": {"levels": [0.1, 0.3, 0.5], "reps": 50},

    "elimination": {"n_splits": 5, "n_repeats": 10, "retention_threshold": 0.90},

    "bootstrap": {"n_boot": 1000},

    "output": {
        "base_dir": "/content/drive/MyDrive/AD_Robustness_Project",
        "fig_subdir": "figures",
        "tab_subdir": "tables",
        "dpi": 300
    }
}


def get_default_config():
    """Return a deep copy so callers may mutate it safely."""
    return copy.deepcopy(DEFAULT_CONFIG)


def _deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    """Return DEFAULT_CONFIG, optionally overridden by a YAML file."""
    cfg = get_default_config()
    if path is not None:
        import yaml
        with open(path, "r") as fh:
            user_cfg = yaml.safe_load(fh)
        if user_cfg:
            cfg = _deep_update(cfg, user_cfg)
    return cfg
