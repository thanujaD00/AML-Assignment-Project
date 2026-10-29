"""End-to-end pipeline: audit -> features -> tuning/testing -> explainability -> forecast.

Run from the project root:
    python -m dengue.pipeline          (after `pip install -e .`)
or  python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data, data_quality, evaluate, explain, features, plots
from .models import NGBoostGrowth, RFGrowth

warnings.filterwarnings("ignore")
PROPOSED = NGBoostGrowth.name


def _write(df: pd.DataFrame, name: str, **kw):
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLE_DIR / name, index=False, **kw)


def load_all(week26_policy: str = C.WEEK26_POLICY):
    long = data.load_long()
    panels = data.build_panels(long)
    weekly = data.clean_weekly(panels, week26_policy)
    sup = features.build_supervised(panels, weekly, week26_policy)
    sup_future = features.build_supervised(panels, weekly, week26_policy, include_future=True)
    return long, panels, weekly, sup, sup_future


def fit_final_model(sup: pd.DataFrame, params: dict) -> NGBoostGrowth:
    return NGBoostGrowth(params=params).fit(sup)


def main() -> dict:
    t0 = time.time()
    for d in (C.PROCESSED_DIR, C.FIG_DIR, C.TABLE_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. data
    print("[1/7] Loading and auditing data ...")
    long, panels, weekly, sup, sup_future = load_all()
    _write(data_quality.describe_dataset(long), "dataset_overview.csv")
    audit = data_quality.audit(panels)
    _write(audit["checks"], "data_quality_checks.csv")
    _write(audit["rank_consistency"], "data_quality_rank_consistency.csv")
    print(audit["checks"].to_string(index=False))

    data.tidy_weekly_csv(panels).to_csv(C.PROCESSED_DIR / "dengue_weekly_by_division.csv", index=False)
    panels.annual.T.rename(columns=lambda d: f"total_{d.year}").rename_axis("division").reset_index() \
        .to_csv(C.PROCESSED_DIR / "dengue_annual_by_division.csv", index=False)
    sup.to_csv(C.PROCESSED_DIR / "supervised_dataset.csv", index=False)

    # ------------------------------------------------------------------ 2. EDA
    print("[2/7] EDA figures ...")
    plots.national_series(panels)
    plots.division_heatmap(panels)
    plots.rank_consistency(audit["rank_consistency"])
    desc = sup[["cases_t", "y_next"] + C.FEATURES].describe().T.reset_index().rename(columns={"index": "variable"})
    _write(desc.round(4), "supervised_descriptive_stats.csv")

    # ------------------------------------------------------------------ 3. main experiment
    print("[3/7] Tuning and testing all models (rolling origin) ...")
    res = evaluate.run_experiment(sup)
    metrics = res["metrics"]
    _write(metrics.round(4), "model_metrics.csv")
    _write(evaluate.paired_bootstrap(res["predictions"], PROPOSED).round(4), "paired_bootstrap_test.csv")
    res["predictions"].to_csv(C.TABLE_DIR / "predictions_validation_test.csv", index=False)
    for name, grid in res["tuning"].items():
        slug = name.split(" (")[0].lower().replace(" ", "_") + ("_" + name.split("(")[1].split(",")[0].lower() if "(" in name else "")
        _write(grid.round(4), f"tuning_{slug}.csv")
    print(metrics.round(3).to_string(index=False))

    # ------------------------------------------------------------------ 4. sensitivity + ablation
    print("[4/7] Sensitivity analysis (week 26 as printed) and feature ablation ...")
    _, _, _, sup_ap, _ = load_all("as_printed")
    sens = evaluate.run_experiment(sup_ap, min_train_weeks=2)["metrics"]
    sens.insert(0, "week26_policy", "as_printed")
    _write(sens.round(4), "model_metrics_sensitivity_week26_as_printed.csv")

    best = res["best_params"][PROPOSED]
    test_weeks = res["weeks"]["test"]
    abl_rows = []
    for label, feats in (("All features", C.FEATURES),
                         ("Without national features", [f for f in C.FEATURES if "national" not in f]),
                         ("Division lags only", ["log_cases_t", "log_cases_t1", "growth_t"])):
        cls = type("Abl", (NGBoostGrowth,), {})
        preds = []
        for wk in test_weeks:
            m = cls(params=best, features=list(feats)).fit(sup[sup["target_week_end"] < wk])
            te = sup[sup["target_week_end"] == wk]
            preds.append(pd.DataFrame({"y": te["y_next"], "yhat": m.predict(te)}))
        p = pd.concat(preds)
        abl_rows.append({"feature_set": label, "n_features": len(feats),
                         **evaluate.point_metrics(p["y"], p["yhat"])})
    ablation = pd.DataFrame(abl_rows)
    _write(ablation.round(4), "ablation_ngboost_test.csv")

    # ------------------------------------------------------------------ 5. figures for results
    print("[5/7] Result figures ...")
    plots.model_comparison(metrics)
    plots.actual_vs_pred(res["predictions"], PROPOSED)
    top_divs = ["Colombo", "Gampaha", "Kandy", "Kalutara", "Galle", "Jaffna"]
    plots.division_forecasts(sup, res["predictions"], PROPOSED, top_divs)

    # ------------------------------------------------------------------ 6. explainability
    print("[6/7] Explainability (SHAP, permutation importance, PDP/ICE) ...")
    final = fit_final_model(sup, best)
    sv = explain.shap_values(final, sup)
    shap_imp = explain.shap_importance(sv, final.features)
    _write(shap_imp.round(5), "xai_shap_importance_ngboost.csv")
    plots.importance_bars(shap_imp, "mean_abs_shap", "Global importance: mean |SHAP| (NGBoost, all 156 rows)",
                          "07_shap_importance.png", xlabel="mean |SHAP value| (log-growth units)")
    plots.shap_beeswarm(sv)

    dev_model = NGBoostGrowth(params=best).fit(sup[sup["target_week_end"] < test_weeks[0]])
    perm = explain.permutation_importance(dev_model, sup[sup["target_week_end"].isin(test_weeks)])
    _write(perm.round(5), "xai_permutation_importance_test.csv")
    plots.importance_bars(perm, "mean_increase_MAE_log",
                          "Permutation importance on hold-out test (increase in MAE_log)",
                          "09_permutation_importance.png", err_col="std",
                          xlabel="increase in MAE on log scale when feature is shuffled")

    rf = RFGrowth(params=res["best_params"][RFGrowth.name]).fit(sup)
    rf_imp = explain.rf_shap_importance(rf, sup)
    _write(rf_imp.round(5), "xai_shap_importance_random_forest.csv")

    top3 = shap_imp["feature"].head(3).tolist()
    curves = {f: explain.partial_dependence(final, sup, f) for f in top3}
    plots.pdp_grid(curves)
    pdp_tbl = pd.concat([pd.DataFrame({"feature": f, "value": g, "pdp_log_growth": p})
                         for f, (g, _, p) in curves.items()])
    _write(pdp_tbl.round(5), "xai_partial_dependence.csv")

    # SHAP for an individual case: the genuine next-week forecast for Colombo
    fut_rows = sup_future[sup_future["y_next"].isna()].reset_index(drop=True)
    sv_fut = explain.shap_values(final, fut_rows)
    ci = int(fut_rows.index[fut_rows["division"] == "Colombo"][0])
    target = fut_rows.loc[ci, "target_week_end"].date()
    plots.shap_waterfall(sv_fut, ci, f"SHAP explanation: Colombo, forecast for week ending {target} "
                         "(output = predicted log growth)", "11_shap_waterfall_colombo.png")

    # ------------------------------------------------------------------ 7. final model + forecast
    print("[7/7] Final model and next-week forecast ...")
    joblib.dump({"model": final, "params": best, "trained_rows": len(sup),
                 "trained_target_weeks": [str(d.date()) for d in res["weeks"]["all"]]},
                C.MODEL_DIR / "ngboost_dengue.joblib")
    fut = sup_future[sup_future["y_next"].isna()].copy()
    fut["forecast"] = final.predict(fut)
    for lvl in C.PI_LEVELS:
        lo, hi = final.predict_interval(fut, lvl)
        fut[f"lo{int(lvl*100)}"], fut[f"hi{int(lvl*100)}"] = lo, hi
    fc = fut[["division", "week_end", "target_week_end", "cases_t", "forecast", "lo80", "hi80", "lo90", "hi90"]]
    fc = fc.sort_values("forecast", ascending=False)
    _write(fc.round(1), "forecast_next_week.csv")

    summary = {
        "n_supervised_rows": int(len(sup)),
        "n_divisions": int(sup["division"].nunique()),
        "target_weeks": [str(d.date()) for d in res["weeks"]["all"]],
        "tuning_weeks": [str(d.date()) for d in res["weeks"]["tune"]],
        "test_weeks": [str(d.date()) for d in test_weeks],
        "best_params": res["best_params"],
        "forecast_week": str(fc["target_week_end"].iloc[0].date()),
        "forecast_national_sum": float(fc["forecast"].sum()),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(C.TABLE_DIR / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"Done in {summary['runtime_seconds']} s. Outputs in reports/ and models/.")
    return summary


if __name__ == "__main__":
    main()
