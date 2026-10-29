"""Static figures for the report (matplotlib, saved as PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

# Validated categorical palette (fixed order) and neutral inks
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110,
})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=150)
    plt.close(fig)


def national_series(panels):
    nat = panels.national
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(nat.index, nat.values, color=BLUE, marker="o", ms=6)
    dup = pd.Timestamp(C.DUPLICATE_WEEK)
    ax.scatter([dup], [nat.loc[dup]], s=120, facecolor="none", edgecolor=RED, lw=2, zorder=5)
    ax.annotate("Week 26: source re-printed\nweek 25 counts", (dup, nat.loc[dup]),
                xytext=(10, -40), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_title("Dengue cases notified per week, Sri Lanka (weeks 24-33, 2026)")
    ax.set_ylabel("Cases")
    ax.set_ylim(0, None)
    fig.autofmt_xdate()
    _save(fig, "01_national_weekly.png")


def division_heatmap(panels):
    w = panels.weekly.T
    w = w.loc[w.iloc[:, 1:].mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(np.log1p(w.values), aspect="auto", cmap="Blues")
    ax.set_xticks(range(w.shape[1]))
    ax.set_xticklabels([f"W{24+i}" for i in range(w.shape[1])])
    ax.set_yticks(range(w.shape[0]))
    ax.set_yticklabels(w.index, fontsize=8)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(w.values):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6.5,
                color="white" if np.log1p(v) > 5.5 else INK)
    ax.set_title("Weekly dengue cases by RDHS division (colour = log scale)")
    ax.set_xlabel("Epidemiological week of 2026 (W24 flagged as misaligned, W26 duplicate)")
    fig.colorbar(im, ax=ax, label="log(1 + cases)", shrink=0.6)
    _save(fig, "02_division_heatmap.png")


def rank_consistency(rank_df):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = pd.to_datetime(rank_df["week_end"])
    ax.plot(x, rank_df["spearman_vs_hist_annual_mean"], color=BLUE, marker="o", ms=6,
            label="vs mean annual cases 2023-25")
    ax.plot(x, rank_df["spearman_vs_next_week"], color=ORANGE, marker="s", ms=6,
            label="vs following week")
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Cross-division consistency of each week's counts")
    ax.legend(loc="lower right")
    fig.autofmt_xdate()
    _save(fig, "03_rank_consistency.png")


def model_comparison(metrics):
    t = metrics[metrics["split"] == "test"].sort_values("MAE_log", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in ((axes[0], "MAE_log", "MAE on log(1+cases) - lower is better"),
                           (axes[1], "WAPE_%", "WAPE (%) - lower is better")):
        colors = [ORANGE if "NGBoost (Normal" in m else BLUE for m in t["model"]]
        ax.barh(t["model"], t[col], color=colors, height=0.6)
        for y, v in enumerate(t[col]):
            ax.text(v, y, f" {v:.3f}" if col == "MAE_log" else f" {v:.1f}", va="center", color=INK2, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, t[col].max() * 1.25)
    axes[1].set_yticklabels([])
    fig.suptitle("Hold-out test (target weeks 32-33, n = 52 division-weeks)", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, "04_model_comparison_test.png")


def actual_vs_pred(preds, model_name):
    p = preds[(preds["model"] == model_name) & (preds["split"] == "test")]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = [0.8, max(p["y_next"].max(), p["y_hat"].max()) * 1.4]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.errorbar(p["y_next"] + 1, p["y_hat"] + 1,
                yerr=[p["y_hat"] - p["lo80"], p["hi80"] - p["y_hat"]],
                fmt="o", color=BLUE, ecolor=BLUE, alpha=0.8, ms=6, elinewidth=1, capsize=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual cases + 1 (log scale)")
    ax.set_ylabel("Predicted cases + 1 (log scale)")
    ax.set_title("NGBoost: test predictions with 80% intervals", fontsize=11)
    _save(fig, "05_actual_vs_predicted_test.png")


def division_forecasts(sup, preds, model_name, divisions):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    p = preds[preds["model"] == model_name]
    for ax, div in zip(axes.ravel(), divisions):
        hist = sup[sup["division"] == div]
        obs = pd.concat([hist[["week_end", "cases_t"]].rename(columns={"week_end": "w", "cases_t": "y"}),
                         hist[["target_week_end", "y_next"]].rename(columns={"target_week_end": "w", "y_next": "y"})]
                        ).drop_duplicates("w").sort_values("w")
        ax.plot(obs["w"], obs["y"], color=INK2, marker="o", ms=4, lw=1.5, label="Actual")
        q = p[p["division"] == div].sort_values("target_week_end")
        ax.fill_between(q["target_week_end"], q["lo80"], q["hi80"], color=ORANGE, alpha=0.18, lw=0)
        ax.plot(q["target_week_end"], q["y_hat"], color=ORANGE, marker="s", ms=4, label="NGBoost one-week-ahead")
        ax.axvline(pd.Timestamp(C.TEST_TARGET_WEEKS[0]) - pd.Timedelta(days=3.5), color=MUTED, lw=1, ls=":")
        ax.set_title(div, fontsize=10)
        ax.set_ylim(0, None)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes[1]:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Rolling one-week-ahead forecasts (shaded = 80% interval; dotted line = start of test)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, "06_division_forecasts.png")


def importance_bars(df, value_col, title, fname, err_col=None, xlabel=""):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(d["label"], d[value_col], color=BLUE, height=0.6,
            xerr=d[err_col] if err_col else None, error_kw={"ecolor": MUTED, "lw": 1})
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    _save(fig, fname)


def shap_beeswarm(sv):
    import shap
    shap.plots.beeswarm(sv, show=False, max_display=10, plot_size=(9, 4.2))
    fig = plt.gcf()
    fig.suptitle("SHAP values: effect of each feature on predicted log growth", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, "08_shap_beeswarm.png")


def shap_waterfall(sv, idx, title, fname):
    import shap
    shap.plots.waterfall(sv[idx], show=False, max_display=8)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 4.6)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, fname)


def pdp_grid(curves):
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.6), sharey=True)
    for ax, (feat, (grid, ice, pdp)) in zip(np.atleast_1d(axes), curves.items()):
        for row in ice[:: max(1, len(ice) // 40)]:
            ax.plot(grid, row, color=BLUE, alpha=0.12, lw=1)
        ax.plot(grid, pdp, color=ORANGE, lw=2.5, label="Average (PDP)")
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_xlabel(C.FEATURE_LABELS[feat], fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("Predicted log growth")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Partial dependence (orange) and individual conditional expectation (blue)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, "10_partial_dependence.png")
