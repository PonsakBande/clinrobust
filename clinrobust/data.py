"""Data loading and preparation (dataset-agnostic, config-driven)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_OPS = {
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "ge": lambda s, v: s >= v,
    "le": lambda s, v: s <= v,
    "gt": lambda s, v: s > v,
    "lt": lambda s, v: s < v,
}


def load_table(path):
    return pd.read_csv(path)


def apply_filters(df, filters):
    out = df.copy()
    for col, val in (filters or {}).items():
        out = out[out[col] == val]
    return out.reset_index(drop=True)


def rename_columns(df, mapping):
    return df.rename(columns=mapping or {})


def encode_values(df, encodings):
    out = df.copy()
    for col, mapping in (encodings or {}).items():
        out[col] = out[col].map(mapping)
    return out


def derive_target(df, target_cfg):
    """Map a source column to integer class codes via an ordered rule list.

    Rows whose source value matches no rule are dropped (reported to caller
    via the returned frame length).
    """
    out = df.copy()
    src = target_cfg["source"]
    col = target_cfg.get("column", "Target")
    codes = np.full(len(out), np.nan)
    for rule in target_cfg["rules"]:
        mask = _OPS[rule["op"]](out[src], rule["value"]).values
        codes[mask] = rule["code"]
    out[col] = codes
    out = out.dropna(subset=[col]).reset_index(drop=True)
    out[col] = out[col].astype(int)
    return out


def prepare_dataset(cfg):
    """Full preparation chain: load -> filter -> rename -> encode -> target.

    Returns (X, y, df): float feature matrix, integer target vector, and the
    prepared dataframe (for cohort summaries).
    """
    dcfg = cfg["data"]
    df = load_table(dcfg["path"])
    df = apply_filters(df, dcfg.get("filters"))
    df = rename_columns(df, dcfg.get("rename"))
    df = encode_values(df, dcfg.get("encode"))
    df = derive_target(df, dcfg["target"])
    features = dcfg["clinical_features"] + dcfg["imaging_features"]
    X = df[features].astype(float)
    y = df[dcfg["target"]["column"]].astype(int)
    return X, y, df


def stratified_split(X, y, test_size, seed):
    """Single stratified split; identical rows must feed every compared model."""
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=seed)
