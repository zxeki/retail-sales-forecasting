# Retail Sales Forecasting

End-to-end retail sales forecasting using time-series analysis, machine learning, and an interactive Streamlit dashboard to analyze sales trends, predict future demand, and support data-driven business decisions.

This repository contains the full pipeline: environment setup, a data loader, basic validation, leakage-safe feature engineering, an evaluation suite, baseline plus machine-learning models (LightGBM and Ridge), and an interactive Streamlit dashboard.

## Quick start

From the project root, after [Setup](#setup) and [downloading the dataset](#download-the-dataset):

```powershell
# 1. Activate the environment
.\.venv\Scripts\Activate.ps1

# 2. Prebuild the cached artifacts (trains LightGBM once)
python scripts/build_forecast_artifacts.py

# 3. Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser. If you prefer to run without activating the environment, call the executables directly:

```powershell
.\.venv\Scripts\python.exe scripts\build_forecast_artifacts.py
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

To generate a Kaggle submission, run:

```powershell
python scripts/generate_submission.py
```

See [Run the dashboard](#run-the-dashboard) and [Generate a Kaggle submission](#generate-a-kaggle-submission) for details.

## Project structure

```text
retail-sales-forecasting/
├── app/
│   └── streamlit_app.py      # Interactive Streamlit dashboard
├── data/
│   ├── raw/                  # Kaggle CSV files (not committed)
│   └── processed/            # Forecast artifacts (not committed)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_features_and_baseline.ipynb
│   ├── 04_ml_and_evaluation.ipynb
│   └── 05_release_and_submission.ipynb
├── reports/
│   └── figures/              # Generated plots (not committed)
├── scripts/
│   ├── build_forecast_artifacts.py  # Prebuild dashboard artifacts
│   ├── download_data.py             # kagglehub download helper
│   └── generate_submission.py       # Kaggle submission generator
├── src/
│   └── retail_sales_forecasting/
│       ├── __init__.py
│       ├── analysis.py
│       ├── config.py
│       ├── dashboard_data.py
│       ├── data_loader.py
│       ├── datasets.py
│       ├── evaluation.py
│       ├── features.py
│       ├── forecast.py
│       ├── models.py
│       ├── splits.py
│       ├── submission.py
│       └── validation.py
├── tests/
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_dashboard_data.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_features.py
│   ├── test_forecast.py
│   ├── test_models.py
│   ├── test_splits.py
│   ├── test_submission.py
│   └── test_validation.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.10 or newer (tested with Python 3.13)
- A free Kaggle account
- Kaggle access configured for `kagglehub`, or the optional Kaggle CLI

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download the dataset

The project uses the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

### Option 1: Manual download

1. Open the competition data page and accept the competition rules while signed in to Kaggle.
2. Download the competition ZIP archive.
3. Extract these files directly into `data/raw/`:
   - `train.csv`
   - `test.csv`
   - `stores.csv`
   - `oil.csv`
   - `holidays_events.csv`
   - `transactions.csv`

The final layout should look like:

```text
data/raw/
├── train.csv
├── test.csv
├── stores.csv
├── oil.csv
├── holidays_events.csv
└── transactions.csv
```

### Option 2: kagglehub helper

After installing the requirements and accepting the competition rules on Kaggle, run:

```powershell
python scripts/download_data.py
```

The script downloads the competition through `kagglehub` and copies the six required CSV files into `data/raw/`. To inspect the cached download location directly:

```python
import kagglehub

path = kagglehub.competition_download("store-sales-time-series-forecasting")
print("Path to competition files:", path)
```

### Option 3: Kaggle CLI

Install and configure the CLI first, then download and extract:

```bash
pip install kaggle
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
```

```powershell
# Windows PowerShell
Expand-Archive data/raw/store-sales-time-series-forecasting.zip -DestinationPath data/raw
```

```bash
# macOS / Linux
unzip data/raw/store-sales-time-series-forecasting.zip -d data/raw
```

Kaggle requires credentials in `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows). Never commit this file; it is excluded by `.gitignore`.

## Usage

### Run the tests

```powershell
pytest -v
```

The tests use small temporary CSV files, so they work without the Kaggle dataset.

### Launch the notebook

```powershell
jupyter lab
```

Open a notebook and run the cells from top to bottom:

- `notebooks/01_data_exploration.ipynb` — data loading, structure, and validation.
- `notebooks/02_exploratory_data_analysis.ipynb` — trends, seasonality, stores, families, promotions, holidays, transactions, oil, and data-quality findings.
- `notebooks/03_features_and_baseline.ipynb` — chronological split, leakage-safe features, seasonal-naive baseline, and evaluation metrics.
- `notebooks/04_ml_and_evaluation.ipynb` — trains LightGBM and Ridge on the leakage-safe features, compares them against the baseline, and reports error breakdowns.
- `notebooks/05_release_and_submission.ipynb` — validates the Kaggle submission format, reviews the final comparison, and summarises methodology, limitations, and reproducibility.

Notebook 01 reports:

- dataset inventory, dimensions, and column types
- training date range
- missing values across all loaded datasets
- duplicate rows and keys
- sales distribution, including zero-sales share
- first rows and basic validation results

If `train.csv` is missing, the notebook prints download instructions and skips data-dependent cells instead of failing with an unclear error.

### Run the dashboard

The Streamlit dashboard explores historical sales trends and the LightGBM forecast against the seasonal-naive baseline. It loads cached artifacts from `data/processed/`; if they are missing it builds them on first load.

To prebuild the artifacts explicitly (recommended once after downloading the data):

```powershell
python scripts/build_forecast_artifacts.py
```

Then launch the app (make sure the virtual environment is activated, or use `.\.venv\Scripts\streamlit.exe`):

```powershell
streamlit run app/streamlit_app.py
```

Streamlit prints a local URL and normally opens your browser automatically. Open **http://localhost:8501** if it does not.

Common options:

```powershell
# Use a different port
streamlit run app/streamlit_app.py --server.port 8502

# Run without opening a browser (headless) and bind to all interfaces
streamlit run app/streamlit_app.py --server.headless true --server.address 0.0.0.0

# Skip the browser prompt and usage stats
streamlit run app/streamlit_app.py --browser.gatherUsageStats false
```

Stop the app with `Ctrl+C` in the terminal where it runs.

The dashboard provides:

- sidebar filters for product family, store number, and validation date range
- KPIs for the selected slice (RMSLE, baseline comparison, MAE, rows)
- **Sales trends** tab — monthly totals and the top families by sales
- **Forecast** tab — daily actual vs LightGBM forecast vs seasonal naive, clearly distinguished
- **Error analysis** tab — mean absolute error by family and store
- **Model comparison** tab — the full metrics table on the identical validation window

Troubleshooting:

- **`streamlit: command was not found`** — the environment is not activated. Run `.\.venv\Scripts\Activate.ps1` or use `.\.venv\Scripts\streamlit.exe run app\streamlit_app.py`.
- **"The Kaggle dataset was not found"** — download the data into `data/raw/` first (see [Download the dataset](#download-the-dataset)), then reload the page.
- **First load is slow** — the app trains LightGBM if no cached artifacts exist. Run `python scripts/build_forecast_artifacts.py` once to avoid the wait.
- **Port already in use** — start with `--server.port 8502`.
- **Page is blank** — hard-refresh the browser (Ctrl+F5) or open a private window.

### Generate a Kaggle submission

```powershell
python scripts/generate_submission.py
```

This trains the selected LightGBM model on the full training history, forecasts the official test window (2017-08-16 to 2017-08-31), validates the format, and writes `data/processed/submission.csv` with the required `id`, `sales` columns in the original test-row order.

Because the test horizon is longer than the shortest lag (7 days), later test days have no observed sales to lag on. The script forecasts **recursively**: it predicts one day at a time and feeds each day's predictions back as lag/rolling inputs for the next day, so no future information is ever used.

### Load data from Python

```python
from retail_sales_forecasting.data_loader import load_dataset, load_available_datasets
from retail_sales_forecasting.validation import validate_dataset, format_validation_report

train = load_dataset("train")
print(format_validation_report(validate_dataset(train, "train")))

datasets = load_available_datasets()
print(datasets.keys())
```

When `src` is not on the Python path, either activate the virtual environment and install the project in editable mode, or add `src` to `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

## Data loader and validation behavior

- `load_dataset(name)` loads one CSV from `data/raw/` and parses its date columns.
- `load_datasets(names, skip_missing=True)` loads several files and skips absent ones.
- `load_available_datasets()` loads every available competition file.
- `available_datasets()` and `missing_datasets()` report file availability.
- Missing files raise `DatasetNotFoundError` with the Kaggle URL and download instructions.
- Validation checks required columns, date parsing, missing values, and dataset dimensions.
- Missing values are reported but do not fail validation; missing required columns or unparseable dates do.

## Forecasting components

- `chronological_split(data, validation_days=16)` splits by date so validation always follows training; never shuffle time-series data.
- `build_feature_frame(data)` adds calendar, promotion, lag (7/14/28), and rolling-mean (7/28) features. Lags and rolling means are shifted so no row uses its own or future sales.
- `seasonal_naive_forecast(train, target_dates)` produces a leakage-safe weekly baseline.
- `evaluate_forecast(y_true, y_pred)` returns MAE, RMSE, and RMSLE (the Kaggle metric).
- `train_lightgbm(...)` and `train_ridge(...)` train candidates on `log1p(sales)`; `predict(...)` returns sales on the original scale.
- `compare_models(train, validation, candidates)` ranks models by RMSLE; `error_breakdown(...)` aggregates error by family or store.
- `run_forecast(train, holidays)` trains the selected model, scores it against the baseline, and returns predictions plus a metrics table; `save_forecast_run`/`load_forecast_run` cache the result for the dashboard.
- Baseline reference (16-day validation window): seasonal naive (last week) RMSLE ≈ 0.617, zero baseline RMSLE ≈ 4.42.
- Model results (same 16-day validation window): LightGBM RMSLE ≈ 0.396 / MAE ≈ 67.7, Ridge RMSLE ≈ 0.427 / MAE ≈ 78.0 — both beat the baseline (~36% and ~31% RMSLE improvement).

## Methodology

1. **Data understanding.** Load and validate the six competition files; inspect ranges, missing values, duplicates, and the zero-sales share.
2. **Exploratory analysis.** Study trends, weekday/monthly seasonality, store and family differences, promotions, holidays, transactions, and oil prices.
3. **Chronological validation.** Hold out the last 16 days (2017-07-31 to 2017-08-15) to mirror the Kaggle test horizon. Never shuffle time-series data.
4. **Leakage-safe features.** Calendar, promotion, lags (7/14/28), and rolling means (7/28), all shifted so a row never sees its own or future sales.
5. **Baseline.** A seasonal-naive (last week) forecast is the reference every model must beat.
6. **Candidate models.** LightGBM (gradient-boosted trees) and Ridge (scaled linear), trained on `log1p(sales)` so errors align with the RMSLE metric.
7. **Evaluation.** MAE, RMSE, and RMSLE on the identical split, plus error breakdowns by family and store.
8. **Submission.** Train on the full history and recursively forecast the official test window.

## Results

Validation window: 2017-07-31 to 2017-08-15 (28,512 rows, 54 stores, 33 families).

| Model | MAE | RMSE | RMSLE |
| --- | ---: | ---: | ---: |
| **LightGBM (selected)** | **67.66** | **228.69** | **0.3955** |
| Ridge | 78.03 | 253.69 | 0.4267 |
| Seasonal naive (last week) | 96.54 | 348.38 | 0.6170 |
| Zero baseline | 467.14 | 1332.91 | 4.4195 |

LightGBM improves on the seasonal-naive baseline by roughly **36% RMSLE**. Errors concentrate in high-volume families (BEVERAGES, GROCERY I); low-volume families (BOOKS) are near zero.

## Limitations

- Validation covers a single 16-day late-summer window; seasonality beyond that is untested.
- No hyperparameter tuning or time-based cross-validation was performed.
- Oil prices were excluded following the Week 2 finding of a confounded relationship.
- Recursive multi-step forecasts accumulate error, so later test days are less certain than earlier ones.
- The model does not use store metadata (city, state, type, cluster) or longer-range seasonality.

## Reproducibility

- Install with `pip install -r requirements.txt`; download the data with `python scripts/download_data.py`.
- LightGBM uses a fixed random seed; validation, dashboard, and submission all reuse the same split and metric functions.
- Run `pytest -q` for the automated suite covering loading, features, evaluation, models, the forecast pipeline, dashboard helpers, and submission format.
- Raw data, processed artifacts, figures, environments, and secrets are excluded from version control.

## Dataset attribution

Data comes from the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), provided by Corporación Favorita. The dataset is used here for educational purposes and is not redistributed; download it directly from Kaggle under the competition rules.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
