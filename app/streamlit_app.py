"""Streamlit front-end: next-week dengue forecasts with explanations.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dengue import config as C  # noqa: E402
from dengue import data, explain, features  # noqa: E402
from dengue.models import NGBoostGrowth  # noqa: E402

DEFAULT_PARAMS = {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2}

st.set_page_config(page_title="Sri Lanka Dengue Forecaster", page_icon="🦟", layout="wide")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data():
    long = data.load_long()
    panels = data.build_panels(long)
    weekly = data.clean_weekly(panels)
    sup = features.build_supervised(panels, weekly)
    fut = features.build_supervised(panels, weekly, include_future=True)
    fut = fut[fut["y_next"].isna()].reset_index(drop=True)
    return panels, sup, fut


@st.cache_resource
def load_model(_sup: pd.DataFrame):
    path = C.MODEL_DIR / "ngboost_dengue.joblib"
    try:
        bundle = joblib.load(path)
        return bundle["model"], bundle["params"], "loaded from models/ngboost_dengue.joblib"
    except Exception as exc:  # missing file or library-version mismatch -> retrain (takes ~1 s)
        params = DEFAULT_PARAMS
        summary = C.TABLE_DIR / "run_summary.json"
        if summary.exists():
            params = json.loads(summary.read_text())["best_params"][NGBoostGrowth.name]
        model = NGBoostGrowth(params=params).fit(_sup)
        return model, params, f"retrained at start-up ({type(exc).__name__})"


def read_table(name):
    p = C.TABLE_DIR / name
    return pd.read_csv(p) if p.exists() else None


panels, sup, fut = load_data()
model, params, model_source = load_model(sup)
hist_annual = panels.annual.mean(axis=0)


def predict_with_interval(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["division", "cases_t"]].copy()
    out["forecast"] = model.predict(rows)
    for lvl in C.PI_LEVELS:
        lo, hi = model.predict_interval(rows, lvl)
        out[f"lo{int(lvl*100)}"], out[f"hi{int(lvl*100)}"] = lo, hi
    out["change_%"] = 100 * (out["forecast"] - out["cases_t"]) / out["cases_t"].clip(lower=1)
    return out


def waterfall(rows: pd.DataFrame, i: int = 0):
    sv = explain.shap_values(model, rows)
    plt.close("all")
    shap.plots.waterfall(sv[i], show=False, max_display=8)
    fig = plt.gcf()
    fig.set_size_inches(9, 4.2)
    fig.tight_layout()
    return fig, sv


def explain_in_words(sv, i: int = 0) -> str:
    vals = sv.values[i]
    order = np.argsort(-np.abs(vals))[:3]
    parts = []
    for j in order:
        direction = "pushes the forecast **up**" if vals[j] > 0 else "pushes the forecast **down**"
        parts.append(f"- *{sv.feature_names[j]}* {direction} ({vals[j]:+.3f} log-growth)")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("Sri Lanka dengue: next-week case forecaster")
st.caption(
    "Model: NGBoost (natural gradient boosting) predicting the log growth of weekly notified cases "
    "per RDHS division. Data: Epidemiology Unit, Ministry of Health, Weekly Epidemiological Report, "
    "Table 1 (weeks 24-33 of 2026), compiled by research.dulajmendis.com. "
    f"Model {model_source}."
)
st.warning(
    "Research prototype trained on only 10 weekly reports (156 usable division-weeks). "
    "Prediction intervals were too narrow in testing (nominal 90% intervals covered ~67% of outcomes). "
    "Do not use for operational public-health decisions.", icon="⚠️")

tab_fc, tab_div, tab_whatif, tab_perf, tab_data = st.tabs(
    ["Next-week forecast", "Division explorer", "What-if prediction", "Model performance", "Data & limitations"])

# ---------------------------------------------------------------- forecast tab
with tab_fc:
    target = fut["target_week_end"].iloc[0].date()
    last = fut["week_end"].iloc[0].date()
    res = predict_with_interval(fut).sort_values("forecast", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Notified week ending {last}", f"{int(res['cases_t'].sum()):,}")
    c2.metric(f"Forecast, week ending {target}", f"{res['forecast'].sum():,.0f}",
              f"{100*(res['forecast'].sum()/res['cases_t'].sum()-1):+.1f}%", delta_color="inverse")
    c3.metric("Divisions forecast to rise", int((res["forecast"] > res["cases_t"]).sum()))
    st.caption("The forecast week is beyond the dataset - there is no actual value to compare against yet.")
    nat_now = fut["log_national_t"].iloc[0]
    if nat_now < sup["log_national_t"].min():
        st.info(f"The latest national count ({np.expm1(nat_now):,.0f}) is lower than any week the model was "
                f"trained on (minimum {np.expm1(sup['log_national_t'].min()):,.0f}). Tree-based models cannot "
                "extrapolate, so the model treats it like the lowest week it has seen; the forecast assumes the "
                "recent decline continues at a similar pace.")

    chart_df = res.melt(id_vars=["division"], value_vars=["cases_t", "forecast"],
                        var_name="series", value_name="cases")
    chart_df["series"] = chart_df["series"].map({"cases_t": f"Notified ({last})", "forecast": f"Forecast ({target})"})
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("division:N", sort=list(res["division"]), title=None),
        x=alt.X("cases:Q", title="Cases"),
        yOffset="series:N",
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=[f"Notified ({last})", f"Forecast ({target})"],
                                        range=["#8a8984", "#2a78d6"]),
                        legend=alt.Legend(orient="top")),
        tooltip=["division", "series", alt.Tooltip("cases:Q", format=",.0f")],
    ).properties(height=26 * 26)
    st.altair_chart(bars, width="stretch")

    show = res.rename(columns={"cases_t": f"cases ({last})", "forecast": f"forecast ({target})"})
    st.dataframe(show.round(1), hide_index=True, width="stretch")
    st.download_button("Download forecast CSV", show.to_csv(index=False).encode(),
                       file_name=f"dengue_forecast_{target}.csv", mime="text/csv")

# ---------------------------------------------------------------- division tab
with tab_div:
    div = st.selectbox("RDHS division", sorted(sup["division"].unique()),
                       index=sorted(sup["division"].unique()).index("Colombo"))
    obs = panels.weekly[div].rename("cases").reset_index().rename(columns={"date": "week_end"})
    obs["status"] = "ok"
    obs.loc[obs["week_end"] == pd.Timestamp(C.MISALIGNED_WEEK), "status"] = "excluded: suspected misaligned"
    obs.loc[obs["week_end"] == pd.Timestamp(C.DUPLICATE_WEEK), "status"] = "excluded: duplicate print"
    preds = read_table("predictions_validation_test.csv")
    layers = [alt.Chart(obs[obs["status"] == "ok"]).mark_line(point=alt.OverlayMarkDef(size=60), color="#52514e").encode(
        x=alt.X("week_end:T", title="Week ending"), y=alt.Y("cases:Q", title="Cases"),
        tooltip=["week_end:T", "cases:Q", "status:N"])]
    bad = obs[obs["status"] != "ok"]
    layers.append(alt.Chart(bad).mark_point(shape="cross", size=140, color="#e34948", filled=True).encode(
        x="week_end:T", y="cases:Q", tooltip=["week_end:T", "cases:Q", "status:N"]))
    if preds is not None:
        p = preds[(preds["model"] == NGBoostGrowth.name) & (preds["division"] == div)].copy()
        p["target_week_end"] = pd.to_datetime(p["target_week_end"])
        layers.append(alt.Chart(p).mark_area(opacity=0.18, color="#eb6834").encode(
            x="target_week_end:T", y="lo80:Q", y2="hi80:Q"))
        layers.append(alt.Chart(p).mark_line(point=True, color="#eb6834", strokeDash=[4, 3]).encode(
            x="target_week_end:T", y="y_hat:Q",
            tooltip=[alt.Tooltip("target_week_end:T", title="week"), alt.Tooltip("y_hat:Q", format=".0f", title="predicted"),
                     alt.Tooltip("y_next:Q", title="actual"), "split:N"]))
    frow = fut[fut["division"] == div].reset_index(drop=True)
    fres = predict_with_interval(frow)
    fpt = pd.DataFrame({"week_end": [pd.Timestamp(frow["target_week_end"].iloc[0])],
                        "forecast": fres["forecast"], "lo": fres["lo80"], "hi": fres["hi80"]})
    layers.append(alt.Chart(fpt).mark_rule(color="#2a78d6", strokeWidth=2).encode(x="week_end:T", y="lo:Q", y2="hi:Q"))
    layers.append(alt.Chart(fpt).mark_point(color="#2a78d6", size=120, filled=True).encode(
        x="week_end:T", y="forecast:Q", tooltip=[alt.Tooltip("forecast:Q", format=".0f")]))
    st.altair_chart(alt.layer(*layers).properties(height=360), width="stretch")
    st.caption("Grey = notified cases (red crosses = weeks excluded from modelling). Orange dashed = rolling "
               "one-week-ahead predictions with 80% interval. Blue = forecast for the next, unobserved week.")
    c1, c2 = st.columns([1, 2])
    c1.metric(f"Forecast for {frow['target_week_end'].iloc[0].date()}", f"{fres['forecast'].iloc[0]:.0f}",
              f"{fres['change_%'].iloc[0]:+.1f}% vs last week", delta_color="inverse")
    c1.write(f"80% interval: **{fres['lo80'].iloc[0]:.0f} - {fres['hi80'].iloc[0]:.0f}**  \n"
             f"90% interval: **{fres['lo90'].iloc[0]:.0f} - {fres['hi90'].iloc[0]:.0f}**")
    fig, sv = waterfall(frow)
    c1.markdown("**Main drivers**\n" + explain_in_words(sv))
    c2.pyplot(fig, width="stretch")

# ---------------------------------------------------------------- what-if tab
with tab_whatif:
    st.write("Enter the latest surveillance figures to obtain a one-week-ahead prediction and its explanation.")
    divs = sorted(sup["division"].unique())
    c1, c2 = st.columns(2)
    wdiv = c1.selectbox("Division (sets its 2023-2025 historical burden)", divs, index=divs.index("Kandy"))
    latest = panels.weekly.iloc[-1]
    prev = panels.weekly.iloc[-2]
    cases_t = c1.number_input("Cases notified this week", 0, 20000, int(latest[wdiv]))
    cases_t1 = c1.number_input("Cases notified last week", 0, 20000, int(prev[wdiv]))
    nat_t = c2.number_input("National cases this week", 1, 100000, int(panels.national.iloc[-1]))
    nat_t1 = c2.number_input("National cases last week", 1, 100000, int(panels.national.iloc[-2]))
    c2.number_input("Historical mean annual cases 2023-25 (from dataset)", value=float(round(hist_annual[wdiv], 1)),
                    disabled=True)
    row = features.make_feature_row(wdiv, cases_t, cases_t1, nat_t, nat_t1, hist_annual[wdiv])
    r = predict_with_interval(row)
    lo_obs, hi_obs = sup["log_national_t"].min(), sup["log_national_t"].max()
    if not (lo_obs <= row["log_national_t"].iloc[0] <= hi_obs):
        st.info(f"National cases outside the training range ({np.expm1(lo_obs):,.0f}-{np.expm1(hi_obs):,.0f}); "
                "tree models cannot extrapolate, so treat this prediction with extra caution.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted cases next week", f"{r['forecast'].iloc[0]:.0f}", f"{r['change_%'].iloc[0]:+.1f}%",
              delta_color="inverse")
    m2.metric("80% interval", f"{r['lo80'].iloc[0]:.0f} - {r['hi80'].iloc[0]:.0f}")
    m3.metric("90% interval", f"{r['lo90'].iloc[0]:.0f} - {r['hi90'].iloc[0]:.0f}")
    fig, sv = waterfall(row)
    st.pyplot(fig, width="stretch")
    st.markdown("**Main drivers**\n" + explain_in_words(sv))
    st.caption("SHAP values are in log-growth units: f(x) is the predicted log(1+next week) - log(1+this week); "
               "E[f(X)] is the average over the training data.")

# ---------------------------------------------------------------- performance tab
with tab_perf:
    m = read_table("model_metrics.csv")
    if m is None:
        st.info("Run `python scripts/run_pipeline.py` to generate evaluation results.")
    else:
        st.subheader("Rolling-origin evaluation")
        st.write("Validation = target weeks ending 2026-07-19, 07-26, 08-02 (used for tuning). "
                 "Test = target weeks ending 2026-08-09 and 08-16 (never used for any decision).")
        st.dataframe(m.round(3), hide_index=True, width="stretch")
        b = read_table("paired_bootstrap_test.csv")
        if b is not None:
            st.subheader("Is NGBoost significantly different? (paired bootstrap over divisions)")
            st.dataframe(b.round(4), hide_index=True, width="stretch")
        for f in ["04_model_comparison_test.png", "05_actual_vs_predicted_test.png", "06_division_forecasts.png",
                  "07_shap_importance.png", "08_shap_beeswarm.png", "09_permutation_importance.png",
                  "10_partial_dependence.png"]:
            if (C.FIG_DIR / f).exists():
                st.image(str(C.FIG_DIR / f))

# ---------------------------------------------------------------- data tab
with tab_data:
    st.subheader("Source")
    st.markdown(
        "- **Primary source:** Epidemiology Unit, Ministry of Health, Sri Lanka - Weekly Epidemiological Report, "
        "Table 1 (cases notified by Medical Officers of Health, by RDHS division).\n"
        "- **Compiled by:** research.dulajmendis.com, dataset *disease-surveillance*, version 2026-10-10 "
        "(file SHA-256 verified against the published hash).\n"
        "- **Content used:** dengue cases per week for 26 RDHS divisions, weeks 24-33 of 2026; annual totals 2023-2025.\n"
        "- **Personal data:** none - aggregated counts only.")
    q = read_table("data_quality_checks.csv")
    if q is not None:
        st.subheader("Automated data-quality checks")
        st.dataframe(q, hide_index=True, width="stretch")
    rc = read_table("data_quality_rank_consistency.csv")
    if rc is not None:
        st.subheader("Rank consistency of division counts between weeks")
        st.dataframe(rc, hide_index=True, width="stretch")
    st.subheader("Known limitations")
    st.markdown(
        "1. Only 10 weekly reports, all in one season (June-August 2026), so no seasonality can be learned.\n"
        "2. Week 26 counts were re-printed from week 25 by the source; treated as missing.\n"
        "3. Week 24 division counts do not match the pattern of any other week (rank correlation 0.08) although "
        "they sum to the national total; treated as missing.\n"
        "4. Notified cases depend on reporting effort and health-seeking behaviour, not only on true infections.\n"
        "5. No weather, mosquito-index or population data are included (none in the verified dataset).")
