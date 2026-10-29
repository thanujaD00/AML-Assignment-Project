"""Rolling-origin (expanding window) tuning and testing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .models import MODEL_CLASSES


def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    ly, lyh = np.log1p(y), np.log1p(yhat)
    ss_res = np.sum((ly - lyh) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE_%": float(100 * np.sum(np.abs(err)) / np.sum(y)),
        "MAE_log": float(np.mean(np.abs(ly - lyh))),
        "R2_log": float(1 - ss_res / ss_tot),
        "Bias": float(np.mean(err)),
    }


def interval_metrics(y, lo, hi, level) -> dict[str, float]:
    y = np.asarray(y, float)
    inside = (y >= lo) & (y <= hi)
    return {f"PI{int(level*100)}_coverage_%": float(100 * inside.mean()),
            f"PI{int(level*100)}_mean_width": float(np.mean(hi - lo))}


def _weeks(sup: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(sup["target_week_end"].unique())


def rolling_predictions(model_cls, params, sup, eval_weeks) -> pd.DataFrame:
    """One-step-ahead: for each eval week, train on all earlier target weeks."""
    out = []
    for wk in eval_weeks:
        train = sup[sup["target_week_end"] < wk]
        test = sup[sup["target_week_end"] == wk]
        if train.empty or test.empty:
            continue
        m = model_cls(params=params) if params is not None else model_cls()
        m.fit(train)
        pred = test[["division", "week_end", "target_week_end", "cases_t", "y_next"]].copy()
        pred["y_hat"] = m.predict(test)
        if getattr(m, "probabilistic", False):
            for lvl in C.PI_LEVELS:
                lo, hi = m.predict_interval(test, lvl)
                pred[f"lo{int(lvl*100)}"], pred[f"hi{int(lvl*100)}"] = lo, hi
        pred["n_train"] = len(train)
        out.append(pred)
    return pd.concat(out, ignore_index=True)


def tune(model_cls, sup, tune_weeks) -> tuple[dict | None, pd.DataFrame]:
    """Grid search scored by pooled MAE on the log scale over the inner folds."""
    grid = getattr(model_cls, "grid", None)
    if not grid:
        return None, pd.DataFrame()
    rows = []
    for params in grid:
        p = rolling_predictions(model_cls, params, sup, tune_weeks)
        m = point_metrics(p["y_next"], p["y_hat"])
        rows.append({**{f"param_{k}": v for k, v in params.items()},
                     "cv_MAE_log": m["MAE_log"], "cv_MAE": m["MAE"]})
    res = pd.DataFrame(rows).sort_values("cv_MAE_log").reset_index(drop=True)
    best = {k.replace("param_", ""): v for k, v in res.iloc[0].items() if k.startswith("param_")}
    best = {k: (None if pd.isna(v) else (int(v) if k in ("max_depth", "min_samples_leaf", "n_estimators") else v))
            for k, v in best.items()}
    return best, res


def run_experiment(sup: pd.DataFrame, min_train_weeks: int = 1) -> dict:
    """Tune on early target weeks (rolling folds), then test on the hold-out weeks.

    min_train_weeks: number of earliest target weeks that are only ever used for
    training (each validation fold needs at least this many training weeks).
    """
    weeks = _weeks(sup)
    test_weeks = [pd.Timestamp(w) for w in C.TEST_TARGET_WEEKS]
    dev_weeks = [w for w in weeks if w < test_weeks[0]]
    tune_weeks = dev_weeks[min_train_weeks:]
    results, preds, tuning, best_params = [], [], {}, {}
    for cls in MODEL_CLASSES:
        params, grid_res = tune(cls, sup, tune_weeks)
        best_params[cls.name] = params
        if not grid_res.empty:
            tuning[cls.name] = grid_res
        cv = rolling_predictions(cls, params, sup, tune_weeks)
        te = rolling_predictions(cls, params, sup, test_weeks)
        cv["split"], te["split"] = "validation", "test"
        p = pd.concat([cv, te], ignore_index=True)
        p["model"] = cls.name
        preds.append(p)
        for split, d in (("validation", cv), ("test", te)):
            row = {"model": cls.name, "split": split, "n": len(d), **point_metrics(d["y_next"], d["y_hat"])}
            if cls.probabilistic:
                for lvl in C.PI_LEVELS:
                    row.update(interval_metrics(d["y_next"], d[f"lo{int(lvl*100)}"], d[f"hi{int(lvl*100)}"], lvl))
            results.append(row)
    return {
        "metrics": pd.DataFrame(results),
        "predictions": pd.concat(preds, ignore_index=True),
        "tuning": tuning,
        "best_params": best_params,
        "weeks": {"all": weeks, "tune": tune_weeks, "test": test_weeks},
    }


def paired_bootstrap(preds: pd.DataFrame, reference: str, n_boot: int = 5000,
                     seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Difference in test MAE_log (reference minus other model) with a 95% CI.

    Divisions are resampled with replacement (both test weeks of a division stay
    together), so the interval reflects division-to-division variability.
    Negative values favour the reference model.
    """
    t = preds[preds["split"] == "test"].copy()
    t["abs_log_err"] = np.abs(np.log1p(t["y_next"]) - np.log1p(t["y_hat"]))
    wide = t.pivot_table(index="division", columns="model", values="abs_log_err", aggfunc="mean")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wide), size=(n_boot, len(wide)))
    rows = []
    for other in wide.columns:
        if other == reference:
            continue
        d = (wide[reference] - wide[other]).to_numpy()
        boots = d[idx].mean(axis=1)
        rows.append({"reference": reference, "compared_with": other,
                     "mean_diff_MAE_log": d.mean(),
                     "ci95_low": np.percentile(boots, 2.5), "ci95_high": np.percentile(boots, 97.5),
                     "divisions_where_reference_better": int((d < 0).sum()),
                     "n_divisions": len(d)})
    return pd.DataFrame(rows)
