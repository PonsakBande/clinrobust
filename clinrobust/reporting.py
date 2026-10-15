"""Publication-quality output helpers."""
import os
import matplotlib.pyplot as plt


def ensure_dirs(base_dir, fig_subdir="figures", tab_subdir="tables"):
    fig_dir = os.path.join(base_dir, fig_subdir)
    tab_dir = os.path.join(base_dir, tab_subdir)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    return fig_dir, tab_dir


def set_publication_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.3, "figure.figsize": (7, 5),
    })


def save_figure(fig, name, fig_dir, dpi=300):
    png = os.path.join(fig_dir, name + ".png")
    pdf = os.path.join(fig_dir, name + ".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def save_table(df, name, tab_dir, index=False):
    path = os.path.join(tab_dir, name + ".csv")
    df.to_csv(path, index=index)
    return path


def audit_manifest(base_dir, tables, figures, fig_subdir="figures",
                   tab_subdir="tables"):
    """Verify every expected deliverable exists; return a status dataframe."""
    import pandas as pd
    rows = []
    for t in tables:
        p = os.path.join(base_dir, tab_subdir, t)
        rows.append({"item": t, "type": "table", "exists": os.path.exists(p),
                     "size_kb": round(os.path.getsize(p) / 1024, 1)
                     if os.path.exists(p) else None})
    for figname in figures:
        for ext in ("png", "pdf"):
            p = os.path.join(base_dir, fig_subdir, figname + "." + ext)
            rows.append({"item": figname + "." + ext, "type": "figure",
                         "exists": os.path.exists(p),
                         "size_kb": round(os.path.getsize(p) / 1024, 1)
                         if os.path.exists(p) else None})
    return pd.DataFrame(rows)
