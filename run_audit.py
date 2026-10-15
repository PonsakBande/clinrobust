#!/usr/bin/env python3
"""Reproduce the four-experiment robustness audit from a YAML config.

Usage
-----
    python run_audit.py                       # uses ./config.yaml
    python run_audit.py --config other.yaml   # e.g. OASIS-3
    python run_audit.py --skip-slow           # skip 50-rep missingness/noise
                                              # and 10-repeat elimination

Outputs (tables/*.csv, figures/*.png+pdf) are written to the directory
named in config['output']['base_dir'].
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow `python run_audit.py` from the toolkit root without installing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinrobust.config import load_config
from clinrobust.data import prepare_dataset, stratified_split
from clinrobust.models import (tune_model, compute_metrics,
                               stratified_bootstrap_ci)
from clinrobust.robustness import (missingness_simulation, fit_imputer,
                                   transform_imputer, noise_sensitivity,
                                   backward_elimination_cv)
from clinrobust.reporting import (ensure_dirs, set_publication_style,
                                  save_figure, save_table)


def _cohort_table(df, cfg):
    clin = cfg["data"]["clinical_features"]
    img = cfg["data"]["imaging_features"]
    rules = cfg["data"]["target"]["rules"]
    names = cfg["data"]["target"]["class_names"]
    src = cfg["data"]["target"]["source"]
    rows = []
    for rule, name in zip(rules, names):
        sub = df[df[cfg["data"]["target"]["column"]] == rule["code"]]
        rec = {"Stage": name, "N": len(sub)}
        if "Gender" in sub.columns:
            rec["Gender (M/F)"] = (
                f"{int((sub['Gender'] == 0).sum())}/"
                f"{int((sub['Gender'] == 1).sum())}"
            )
        for col in clin + img:
            if col == "Gender" or col not in sub.columns:
                continue
            rec[f"{col} (mean ± std)"] = (
                f"{sub[col].mean():.3f} ± {sub[col].std():.3f}"
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt_ci(point, ci, digits=3):
    return (f"{point:.{digits}f} "
            f"[{ci['lower']:.{digits}f}–{ci['upper']:.{digits}f}]")


def plot_missingness(summary, ref_acc, ref_f1, fig_dir, dpi):
    colors = {"mean": "#1f77b4", "median": "#ff7f0e", "knn": "#2ca02c"}
    markers = {"mean": "o", "median": "s", "knn": "^"}
    labels = {"mean": "Mean", "median": "Median", "knn": "KNN (k=5)"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mean_col, lo_col, hi_col, ylabel, title, href in [
        (axes[0], "mean_accuracy", "acc_p2_5", "acc_p97_5",
         "Accuracy", "Accuracy Degradation with Missing Data", ref_acc),
        (axes[1], "mean_macro_f1", "f1_p2_5", "f1_p97_5",
         "Macro F1-Score", "Macro F1 Degradation with Missing Data", ref_f1),
    ]:
        for method in summary["method"].unique():
            sub = summary[summary["method"] == method].sort_values(
                "missingness_pct")
            x = sub["missingness_pct"].to_numpy()
            ax.plot(x, sub[mean_col], color=colors[method],
                    marker=markers[method], linewidth=2.5, markersize=8,
                    label=labels.get(method, method))
            ax.fill_between(x, sub[lo_col], sub[hi_col],
                            color=colors[method], alpha=0.16)
        ax.axhline(href, color="black", linestyle="--", linewidth=2,
                   label="Reference (0% missing)")
        ax.set_xlabel("Missingness Rate (%)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(sorted(summary["missingness_pct"].unique()))
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylim(0.5, 1.0)
    axes[1].set_ylim(0.3, 1.0)
    fig.text(0.5, -0.02,
             "Shaded regions: empirical 95% simulation intervals (50 MCAR masks).",
             ha="center", fontsize=9)
    plt.tight_layout()
    save_figure(fig, "figure_1_missingness_degradation_curves", fig_dir, dpi)
    plt.close(fig)


def plot_no_scan(scen, fig_dir, dpi):
    names = list(scen.keys())
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    acc = [scen[n]["accuracy"] for n in names]
    f1 = [scen[n]["macro_f1"] for n in names]
    b1 = ax.bar(x - width / 2, acc, width, label="Accuracy",
                color="#1f77b4", edgecolor="black")
    b2 = ax.bar(x + width / 2, f1, width, label="Macro F1",
                color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Performance When MRI Is Unavailable:\n"
                 "Retraining Without Imaging vs Imputing Missing Scan",
                 fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    save_figure(fig, "figure_2_no_scan_comparison", fig_dir, dpi)
    plt.close(fig)


def plot_noise(noise_df, features, clinical, imaging, fig_dir, dpi):
    n05 = (noise_df[np.isclose(noise_df["level"], 0.5)]
           .sort_values("acc_drop", ascending=True))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    colors = ["#d62728" if f in imaging else "#1f77b4" for f in n05["feature"]]
    ax.barh(n05["feature"], n05["acc_drop"], xerr=n05["std_accuracy"],
            color=colors, edgecolor="black", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Drop at 0.5σ Noise", fontweight="bold")
    ax.set_title("(A) Feature Sensitivity Ranking", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1f77b4", edgecolor="black",
                             label="Clinical"),
                       Patch(facecolor="#d62728", edgecolor="black",
                             label="Neuroimaging")], loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    # reconstruct a clean baseline from any 0.5-sigma row: acc = baseline - drop
    # We plot accuracy vs level using mean_accuracy directly.
    for feat in features:
        sub = noise_df[noise_df["feature"] == feat].sort_values("level")
        style = "--" if feat in imaging else "-"
        ax.plot(sub["level"], sub["mean_accuracy"], marker="o",
                linestyle=style, linewidth=2, label=feat)
    ax.set_xlabel("Noise Level (× feature σ)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("(B) Accuracy vs Noise Level per Feature", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "figure_3_noise_sensitivity", fig_dir, dpi)
    plt.close(fig)


def plot_elimination(elim_df, fig_dir, dpi):
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "recall_ad"]
    labels = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
              "macro_f1": "Macro F1", "recall_ad": "AD recall"}
    cols = {"accuracy": "#1f77b4", "balanced_accuracy": "#2ca02c",
            "macro_f1": "#ff7f0e", "recall_ad": "#d62728"}
    n_feats = elim_df["n_features"].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for m in metric_cols:
        mu, sd = elim_df[f"{m}_mean"].values, elim_df[f"{m}_sd"].values
        ax.plot(n_feats, mu, marker="o", linewidth=2, color=cols[m],
                label=labels[m])
        ax.fill_between(n_feats, mu - sd, mu + sd, color=cols[m], alpha=0.15)
    ax.set_xlabel("Number of features retained", fontweight="bold")
    ax.set_ylabel("Performance (repeated stratified CV)", fontweight="bold")
    ax.set_title("(A) Performance vs Number of Features", fontweight="bold")
    ax.set_xticks(n_feats)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for m in metric_cols:
        ax.plot(n_feats, elim_df[f"{m}_pct_ref"].values, marker="o",
                linewidth=2, color=cols[m], label=labels[m])
    ax.axhline(90, color="black", linestyle="--", linewidth=1.5,
               label="90% threshold")
    ax.set_xlabel("Number of features retained", fontweight="bold")
    ax.set_ylabel("% of full-model performance retained", fontweight="bold")
    ax.set_title("(B) Performance Retained vs Number of Features",
                 fontweight="bold")
    ax.set_xticks(n_feats)
    ax.invert_xaxis()
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    for n, rem in zip(n_feats, elim_df["removed_to_get_here"]):
        if rem != "-":
            ax.annotate(f"-{rem}", (n, 3), ha="center", fontsize=8,
                        rotation=90, color="gray")
    plt.tight_layout()
    save_figure(fig, "figure_4_minimum_viable_features", fig_dir, dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run the clinrobust four-experiment audit.")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (default: ./config.yaml)")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip 50-rep missingness/noise and 10-repeat CV.")
    args = parser.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "config.yaml")
    cfg = load_config(cfg_path)
    seed = cfg["project"]["seed"]
    dpi = cfg["output"].get("dpi", 300)
    fig_dir, tab_dir = ensure_dirs(
        cfg["output"]["base_dir"],
        cfg["output"].get("fig_subdir", "figures"),
        cfg["output"].get("tab_subdir", "tables"))
    set_publication_style()

    t0 = time.time()
    print("=" * 72)
    print("clinrobust audit")
    print("  config :", cfg_path)
    print("  data   :", cfg["data"]["path"])
    print("  output :", cfg["output"]["base_dir"])
    print("=" * 72)

    # ---- Data ----
    X, y, df = prepare_dataset(cfg)
    clin = cfg["data"]["clinical_features"]
    img = cfg["data"]["imaging_features"]
    names = cfg["data"]["target"]["class_names"]
    print(f"\nCohort: n={len(y)} | classes={y.value_counts().sort_index().to_dict()}")
    save_table(_cohort_table(df, cfg), "table_1_baseline_characteristics", tab_dir)

    X_train, X_test, y_train, y_test = stratified_split(
        X, y, cfg["split"]["test_size"], seed)
    split_rows = []
    for code, lab in enumerate(names):
        split_rows.append({
            "class": lab,
            "full_n": int((y == code).sum()),
            "train_n": int((y_train == code).sum()),
            "test_n": int((y_test == code).sum()),
        })
    save_table(pd.DataFrame(split_rows), "table_2_train_test_class_distribution",
               tab_dir)

    # ---- Models ----
    print("\nTuning reference model...")
    ref_gs = tune_model(X_train, y_train, cfg["model"], seed=seed)
    print("  best params:", ref_gs.best_params_,
          f"| CV {cfg['model']['scoring']}={ref_gs.best_score_:.4f}")
    print("Tuning clinical-only model...")
    clin_gs = tune_model(X_train[clin], y_train, cfg["model"], seed=seed)
    print("  best params:", clin_gs.best_params_,
          f"| CV {cfg['model']['scoring']}={clin_gs.best_score_:.4f}")

    ref_pred = ref_gs.predict(X_test)
    clin_pred = clin_gs.predict(X_test[clin])
    ref_m = compute_metrics(y_test, ref_pred)
    clin_m = compute_metrics(y_test, clin_pred)
    print(f"  reference   acc={ref_m['accuracy']:.4f}  "
          f"balAcc={ref_m['balanced_accuracy']:.4f}  "
          f"ADrec={ref_m['recall_2']:.4f}")
    print(f"  clinical    acc={clin_m['accuracy']:.4f}  "
          f"balAcc={clin_m['balanced_accuracy']:.4f}  "
          f"ADrec={clin_m['recall_2']:.4f}")

    n_boot = cfg["bootstrap"]["n_boot"]
    ref_ci = stratified_bootstrap_ci(y_test, ref_pred, n_boot=n_boot, seed=seed)
    clin_ci = stratified_bootstrap_ci(y_test, clin_pred, n_boot=n_boot, seed=seed)

    t3 = pd.DataFrame([
        {"Metric": lab,
         "Point Estimate": f"{ref_m[k]:.4f}",
         "Bootstrap Mean": f"{ref_ci[k]['mean']:.4f}",
         "CI 95% Lower": f"{ref_ci[k]['lower']:.4f}",
         "CI 95% Upper": f"{ref_ci[k]['upper']:.4f}"}
        for k, lab in [("accuracy", "Accuracy"),
                       ("balanced_accuracy", "Balanced Accuracy"),
                       ("macro_f1", "Macro F1"),
                       ("recall_2", "AD Recall")]
    ])
    save_table(t3, "table_3_reference_model_bootstrap_ci", tab_dir)

    t4_rows = []
    for k, lab in [("accuracy", "Accuracy"),
                   ("balanced_accuracy", "Balanced Accuracy"),
                   ("macro_f1", "Macro F1"),
                   ("recall_2", "AD Recall")]:
        pct = (clin_m[k] / ref_m[k] * 100) if ref_m[k] > 0 else 0.0
        t4_rows.append({
            "Metric": lab,
            "Reference (95% CI)": _fmt_ci(ref_m[k], ref_ci[k]),
            "Clinical-Only (95% CI)": _fmt_ci(clin_m[k], clin_ci[k]),
            "% Retained": f"{pct:.1f}%",
        })
    save_table(pd.DataFrame(t4_rows),
               "table_4_ablation_reference_vs_clinical_only", tab_dir)

    if args.skip_slow:
        print("\n--skip-slow set: omitting missingness, noise and elimination.")
        print(f"Done in {time.time() - t0:.1f}s")
        return 0

    # ---- Missingness ----
    print("\nMissingness simulation...")
    mcfg = cfg["missingness"]
    summary, raw = missingness_simulation(
        ref_gs, X_train, X_test, y_test,
        rates=mcfg["rates"], methods=mcfg["methods"], reps=mcfg["reps"],
        seed=seed, knn_k=mcfg["knn_k"])
    save_table(summary, "table_5_missingness_simulation_results", tab_dir)
    save_table(raw, "table_5b_missingness_simulation_raw", tab_dir)
    plot_missingness(summary, ref_m["accuracy"], ref_m["macro_f1"], fig_dir, dpi)

    # ---- No-scan ----
    print("No-scan scenario...")
    X_ns = X_test.astype(float).copy()
    X_ns[img] = np.nan
    scen = {
        "Reference (full data)": ref_pred,
        "Clinical-only (retrained)": clin_pred,
    }
    scen_point = {
        "Reference (full data)": ref_m,
        "Clinical-only (retrained)": clin_m,
    }
    for method in mcfg["methods"]:
        fitted = fit_imputer(method, X_train, knn_k=mcfg["knn_k"])
        pred = ref_gs.predict(transform_imputer(fitted, X_ns))
        label = f"No scan + {method.capitalize()}"
        scen[label] = pred
        scen_point[label] = compute_metrics(y_test, pred)

    t6_rows = []
    for name, pred in scen.items():
        pt = scen_point[name]
        boot = stratified_bootstrap_ci(y_test, pred, n_boot=n_boot, seed=seed)
        t6_rows.append({
            "Scenario": name,
            "Accuracy": _fmt_ci(pt["accuracy"], boot["accuracy"]),
            "Balanced_Acc": _fmt_ci(pt["balanced_accuracy"],
                                    boot["balanced_accuracy"]),
            "Macro_F1": _fmt_ci(pt["macro_f1"], boot["macro_f1"]),
            "AD_Recall": _fmt_ci(pt["recall_2"], boot["recall_2"]),
        })
    save_table(pd.DataFrame(t6_rows),
               "table_6_neuroimaging_missing_comparison", tab_dir)
    plot_no_scan(scen_point, fig_dir, dpi)

    # ---- Noise ----
    print("Noise sensitivity...")
    ncfg = cfg["noise"]
    noise_df = noise_sensitivity(
        ref_gs, X_test, y_test, features=list(X.columns),
        train_stds=X_train.std(ddof=0), levels=ncfg["levels"],
        reps=ncfg["reps"], seed=seed)
    save_table(noise_df, "table_7_noise_sensitivity_raw", tab_dir)
    n05 = (noise_df[np.isclose(noise_df["level"], 0.5)]
           .sort_values("acc_drop", ascending=False).reset_index(drop=True))
    n05.insert(0, "Rank", range(1, len(n05) + 1))
    n05["Group"] = n05["feature"].map(
        lambda f: "Clinical" if f in clin else "Neuroimaging")
    save_table(n05, "table_8_feature_sensitivity_ranking", tab_dir)
    plot_noise(noise_df, list(X.columns), clin, img, fig_dir, dpi)

    # ---- Elimination ----
    print("Backward elimination (repeated CV)...")
    removal_order = (noise_df[np.isclose(noise_df["level"], 0.5)]
                     .sort_values("acc_drop")["feature"].tolist())
    print("  removal order:", removal_order)
    ecfg = cfg["elimination"]
    elim_df = backward_elimination_cv(
        X, y, removal_order, ref_gs.best_estimator_,
        n_splits=ecfg["n_splits"], n_repeats=ecfg["n_repeats"], seed=seed)
    save_table(elim_df, "table_10_minimum_viable_input_elimination", tab_dir)
    plot_elimination(elim_df, fig_dir, dpi)

    print(f"\nAudit complete in {time.time() - t0:.1f}s")
    print("  tables  ->", tab_dir)
    print("  figures ->", fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
