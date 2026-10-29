# Sri Lankan Dengue Case Prediction with NGBoost

Applied Machine Learning assignment — M.Sc. in Artificial Intelligence, University of Moratuwa.

**Student:** T.D.B. Wickramasinghe (Index No. 269216X)

This project predicts **next week's notified dengue cases for each of Sri Lanka's 26 RDHS health
divisions**. It uses official surveillance data from the Epidemiology Unit's *Weekly Epidemiological
Report* and **NGBoost** (Natural Gradient Boosting), a boosting algorithm that predicts a full
probability distribution, so every forecast comes with an uncertainty range. The predictions are
explained with SHAP, permutation importance and partial dependence, and are served through a Streamlit
web app.

> **Status:** research prototype. The verified dataset holds only 10 weekly reports (weeks 24–33 of
> 2026), and two of those weeks have data-quality problems. Read [Limitations](#limitations) before
> using any result.

## Results at a glance (hold-out test: weeks ending 9 and 16 Aug 2026, 52 division-weeks)

| Model | MAE (cases) | WAPE % | MAE on log scale |
|---|---|---|---|
| Naive persistence (next week = this week) | 40.5 | 32.2 | 0.371 |
| Naive drift | 27.3 | 21.7 | 0.430 |
| Ridge regression | 35.6 | 28.3 | 0.404 |
| Random forest | 19.7 | 15.6 | 0.297 |
| NGBoost (Poisson) | 21.0 | 16.7 | 0.296 |
| **NGBoost (Normal, log-growth) — proposed** | **20.4** | **16.2** | **0.299** |

* NGBoost was better than persistence in 22 of 26 divisions. The 95 % bootstrap CI of the improvement
  (−0.129 to +0.015 MAE_log) still includes 0, because the test set is small.
* NGBoost, the random forest and NGBoost-Poisson are statistically indistinguishable.
* The 90 % prediction intervals covered only 67 % of test outcomes, so they are too narrow.
* SHAP: the national epidemic level and the division's recent growth are the strongest drivers. Sudden
  jumps tend to partly reverse.

The full write-up is in [`reports/AML_Dengue_Report.pdf`](reports/AML_Dengue_Report.pdf), with its
source in `reports/report.md`.

## Project structure

```
AML-Assignment-Project/
├── app/
│   └── streamlit_app.py          # Front-end: forecasts, division explorer, what-if, SHAP
├── data/
│   ├── raw/                      # Original Excel file (unchanged) + DATA_SOURCE.md (provenance)
│   └── processed/                # Tidy CSVs written by the pipeline
├── models/
│   └── ngboost_dengue.joblib     # Final model trained on all 156 rows
├── reports/
│   ├── AML_Dengue_Report.pdf     # Written report
│   ├── report.md / report.css    # Report source
│   ├── figures/                  # All figures (PNG)
│   └── tables/                   # All metrics, predictions, XAI tables, forecast (CSV/JSON)
├── scripts/
│   ├── run_pipeline.py           # Runs everything end-to-end
│   └── build_report.py           # Rebuilds the PDF (needs pandoc + playwright)
├── src/dengue/
│   ├── config.py                 # Paths, data-quality decisions, features, split design
│   ├── data.py                   # Load + tidy the workbook, exclude unreliable weeks
│   ├── data_quality.py           # Automated audit (hash, sums, YTD, duplicates, rank consistency)
│   ├── features.py               # One-week-ahead supervised dataset (no leakage)
│   ├── models.py                 # Persistence, drift, ridge, random forest, NGBoost x2
│   ├── evaluate.py               # Rolling-origin tuning/testing, metrics, paired bootstrap
│   ├── explain.py                # SHAP, permutation importance, PDP/ICE
│   ├── plots.py                  # Report figures
│   └── pipeline.py               # Orchestrates all steps
├── tests/                        # pytest: data integrity, leakage, model sanity (10 tests)
├── .vscode/                      # Launch configs, tasks, settings, recommended extensions
├── requirements.txt
└── pyproject.toml
```

## How to run it in VS Code (Windows)

**Prerequisites:** [Python 3.10–3.13](https://www.python.org/downloads/) (tick *Add python.exe to PATH*),
[VS Code](https://code.visualstudio.com/) and [Git](https://git-scm.com/).

1. **Open the project.** In VS Code use *File → Open Folder…* and choose `AML-Assignment-Project`.
   Install the recommended extensions when asked (Python, Pylance, Python Debugger).
2. **Create the environment.** Open a terminal (*Terminal → New Terminal*) and run:
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```
   If PowerShell blocks the activation script, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
   You can also run *Terminal → Run Task… → Setup: create venv + install*.
3. **Select the interpreter.** Press `Ctrl+Shift+P`, choose *Python: Select Interpreter*, then pick
   `.venv`.
4. **Run the ML pipeline** (about 5 minutes). It re-creates every table, figure and the model:
   ```powershell
   python scripts/run_pipeline.py
   ```
   Or use the *Run and Debug* panel (`Ctrl+Shift+D`) → **1. Run full ML pipeline** → ▶.
5. **Run the tests:**
   ```powershell
   python -m pytest -q
   ```
   Or use the *Testing* panel (beaker icon).
6. **Start the web app:**
   ```powershell
   streamlit run app/streamlit_app.py
   ```
   Or use *Run and Debug* → **2. Launch Streamlit app**. The app opens at <http://localhost:8501>.
   It also works before step 4, because a trained model and all results are already committed. If the
   saved model can't be loaded with your library versions, the app retrains it automatically in about
   one second.
7. *(Optional)* **Rebuild the PDF report** after changing `reports/report.md`: install
   [pandoc](https://pandoc.org/installing.html), run `pip install playwright` and
   `playwright install chromium`, then run `python scripts/build_report.py`.

macOS/Linux: use `python3 -m venv .venv && source .venv/bin/activate`. The other commands are the same.

## Method in brief

* **Target:** cases in week *t + 1* for each division. The learned models predict the log growth
  `log(1+y[t+1]) − log(1+y[t])` so that divisions of very different size can share one model.
* **Features (7):** this week's and last week's cases (log), division growth, historical annual burden
  (2023–25), this week relative to the historical weekly mean, national cases this week and national
  growth. All are computed from the dataset only.
* **Evaluation:** rolling-origin, one week ahead. Tuning used validation weeks W29–W31; W32–W33 were held
  out as the test set. Metrics: MAE, RMSE, WAPE, MAE on log scale, R² (log), bias, and interval coverage.
* **Explainability:** TreeSHAP (global and local), permutation importance on the test set, and
  PDP/ICE curves.
* **Extra checks:** sensitivity to how week 26 is treated, feature ablation, and a paired bootstrap
  comparison between models.

## Data

Epidemiology Unit, Ministry of Health, Sri Lanka — *Weekly Epidemiological Report*, Table 1. Compiled by
[research.dulajmendis.com](https://research.dulajmendis.com/data/disease-surveillance/) (dataset
*disease-surveillance*, version 2026-10-10). The file's SHA-256 matches the published hash. The data
are aggregated counts with no personal data. See [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md)
for full provenance.

## Limitations

1. **Very small dataset:** 10 weekly reports from one season, giving 156 usable division-weeks. The model
   cannot learn seasonality or predict when a new rise will start.
2. **Week 26 (28 Jun 2026)** repeats week 25 in the source report. It is treated as missing.
3. **Week 24 (14 Jun 2026)** division counts look misaligned (rank correlation 0.08 with the next week,
   against ≥ 0.95 for all other weeks). They are treated as missing; nothing was corrected or invented.
4. Tree models cannot extrapolate. The latest national count (2,682) is below every training week, so
   the week-34 forecast assumes the decline continues.
5. Prediction intervals are under-calibrated (67 % coverage at a nominal 90 %).
6. Notified cases reflect reporting capacity as well as true infections, which can bias results against
   under-served divisions.
7. No weather, vector-index or population data are used, because none are in the verified dataset.

## Citation

Epidemiology Unit, Ministry of Health, Sri Lanka, Weekly Epidemiological Report, Table 1. Compiled by
research.dulajmendis.com, dataset disease-surveillance, version 2026-10-10.
Duan, T. et al. (2020). *NGBoost: Natural Gradient Boosting for Probabilistic Prediction.* ICML.
