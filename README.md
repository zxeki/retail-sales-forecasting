# Retail Sales Forecasting

End-to-end retail sales forecasting using time-series analysis, machine learning, and an interactive Streamlit dashboard to analyze sales trends, predict future demand, and support data-driven business decisions.

This repository currently contains the **project foundation**: environment setup, a data loader, basic validation, tests, and a first exploration notebook. No machine learning model is trained yet.

## Project structure

```text
retail-sales-forecasting/
├── data/
│   ├── raw/                  # Kaggle CSV files (not committed)
│   └── processed/            # Cleaned datasets (not committed)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_features_and_baseline.ipynb
├── reports/
│   └── figures/              # Generated plots (not committed)
├── scripts/
│   └── download_data.py      # kagglehub download helper
├── src/
│   └── retail_sales_forecasting/
│       ├── __init__.py
│       ├── analysis.py
│       ├── config.py
│       ├── data_loader.py
│       ├── datasets.py
│       ├── evaluation.py
│       ├── features.py
│       ├── splits.py
│       └── validation.py
├── tests/
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_features.py
│   ├── test_splits.py
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

Notebook 01 reports:

- dataset inventory, dimensions, and column types
- training date range
- missing values across all loaded datasets
- duplicate rows and keys
- sales distribution, including zero-sales share
- first rows and basic validation results

If `train.csv` is missing, the notebook prints download instructions and skips data-dependent cells instead of failing with an unclear error.

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
- Baseline reference (16-day validation window): seasonal naive (last week) RMSLE ≈ 0.617, vs zero baseline RMSLE ≈ 4.42. Week 4 models must beat this.

## Notes

- Raw data, processed data, generated figures, virtual environments, and secrets are excluded from version control.
- No data is fabricated. The project fails gracefully when the Kaggle files are absent.
- No model is trained in this foundation phase.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
