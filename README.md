# SalesPulse – Automated Retail Sales Prediction System

**CSCE 5210 – Fundamentals of Artificial Intelligence**  
**Project:** Retail Sales Forecasting and Demand Intelligence

SalesPulse is an AI-powered retail sales prediction system that uses historical sales data and machine learning models to forecast future product demand. It provides an interactive dashboard, a REST API, model evaluation, and support for uploading custom sales datasets.

---

## 1. Project Overview

Retail sales forecasting helps businesses make better inventory, replenishment, and demand-planning decisions.

SalesPulse uses historical daily sales data to train machine learning models and generate sales predictions for individual products and stores.

### Key Features

- Historical retail sales data processing using the M5 dataset.
- Automated data preprocessing and feature engineering.
- Lag features and rolling averages for time-series forecasting.
- Multiple machine learning models:
  - Linear Regression
  - Random Forest Regression
  - Histogram-based Gradient Boosting
- Model performance comparison using MAE, RMSE, MAPE, and WAPE.
- Recursive 28-day sales forecasting evaluation.
- FastAPI REST API for accessing prediction results.
- Interactive Streamlit dashboard.
- Custom CSV dataset upload and validation.
- Prediction generation for uploaded datasets.
- Dataset management and prediction downloads.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Evaluation | MAE, RMSE, MAPE, WAPE |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| API Server | Uvicorn |
| Testing | Pytest |
| Data Storage | CSV, Parquet |
| Development Environment | GitHub Codespaces / VS Code |

---

## 3. Project Structure

```text
SalesPulse/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── prediction/
│   │   ├── predict.py
│   │   └── custom_dataset.py
│   │
│   └── evaluation/
│       └── evaluate_28day.py
│
├── tests/
│   ├── test_api.py
│   └── test_custom_dataset.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── predictions/
│   │   ├── model_results/
│   │   └── features/
│   │
│   └── user_datasets/
│
├── models/
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Some data and model directories are generated or populated during project execution. Large datasets and generated artifacts may not be included in the Git repository.

---

## 4. Installation and Setup

### Prerequisites

Install the following:

- Python 3.12
- Git
- Visual Studio Code or GitHub Codespaces

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chathrapathi27/SalesPulse.git
cd SalesPulse
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
```

**Linux / macOS / GitHub Codespaces:**

```bash
python3.12 -m venv venv
source venv/bin/activate
```

If Python 3.12 is already the default Python version, you can use:

```bash
python -m venv venv
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Dataset Setup

SalesPulse uses the Walmart M5 Forecasting dataset.
Dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data 
The original dataset includes:

- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

Place the required raw dataset files in the project's data directory according to the data-processing scripts.

The project uses processed Parquet feature files and generated prediction artifacts.

Important generated output files include:

```text
data/processed/predictions/latest_predictions.csv

data/processed/model_results/
    model_comparison.csv
    recursive_28day_summary.csv
    recursive_28day_daily_metrics.csv
    recursive_28day_predictions.csv
```

**Important:** The dashboard and API rely on their required processed prediction and model-metrics files. If these artifacts are not included in your repository, generate them using the project's data-processing and prediction scripts before starting the application.

---

## 6. Running the Project

SalesPulse consists of two main applications:

1. FastAPI backend
2. Streamlit dashboard

Both applications should run simultaneously.

### Step 1: Start the FastAPI Backend

Open the first terminal from the project root directory.

Activate the virtual environment if it is not already active.

Run:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI interactive documentation:

```text
http://localhost:8000/docs
```

### Step 2: Start the Streamlit Dashboard

Open a second terminal from the project root directory.

Activate the virtual environment if necessary.

Run:

```bash
python -m streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

The dashboard will be available at:

```text
http://localhost:8501
```

**GitHub Codespaces users:** Open the Ports tab and access port `8501` using the provided browser link. Port `8000` must also be accessible to the dashboard if it communicates with the API through the Codespaces environment.

### Step 3: Explore the Dashboard

The dashboard provides the following sections:

| Section | Description |
|---|---|
| Overview | Project overview and system information |
| M5 Forecasts | View generated retail sales forecasts |
| Forecast Evaluation | Analyze recursive 28-day forecasting results |
| Upload & Predict | Upload a custom sales CSV and generate predictions |
| My Datasets | View and manage uploaded datasets |
| Model Comparison | Compare machine learning model performance |
| About & Settings | Project information and application settings |

---

## 7. API Endpoints

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API home endpoint |
| GET | `/health` | Check API and required data-file availability |
| GET | `/predict` | Retrieve sales prediction records |
| GET | `/metrics` | Retrieve model comparison metrics |

### Example API Requests

Check API health:

```bash
curl http://localhost:8000/health
```

Retrieve predictions:

```bash
curl "http://localhost:8000/predict?limit=100"
```

Retrieve model metrics:

```bash
curl http://localhost:8000/metrics
```

The prediction endpoint supports a configurable limit from 1 to 50,000 records.

You can also explore and test the API through:

```text
http://localhost:8000/docs
```

---

## 8. Custom Dataset Upload

SalesPulse supports uploading user-provided sales data through the Streamlit dashboard.

### Required CSV Format

The uploaded CSV must contain these columns:

| Column | Description | Example |
|---|---|---|
| `id` | Product or product-store identifier | `PRODUCT_001` |
| `date` | Daily observation date | `2016-01-01` |
| `sales` | Number of units sold | `25` |

### Example CSV

```csv
id,date,sales
PRODUCT_001,2016-01-01,20
PRODUCT_001,2016-01-02,25
PRODUCT_001,2016-01-03,18
PRODUCT_001,2016-01-04,30
PRODUCT_001,2016-01-05,27
```

### Dataset Requirements

- Required columns: `id`, `date`, and `sales`.
- Dates must be valid and consecutive daily observations for each product.
- Sales values must be numeric, finite, and non-negative.
- Duplicate product/date records are not allowed.
- At least 29 consecutive observations are required for one-day-ahead prediction.
- At least 57 consecutive observations are required for the recursive 28-day evaluation.

### Upload Workflow

1. Open the **Upload & Predict** section.
2. Select a CSV file containing historical sales data.
3. Upload the dataset.
4. The system validates the dataset.
5. The selected trained model generates predictions.
6. View the saved dataset in **My Datasets**.
7. Review available evaluation results and download predictions.

Custom dataset predictions use the existing trained models; uploading a dataset does not automatically retrain the model.

---

## 9. Machine Learning and Evaluation

SalesPulse compares multiple regression algorithms.

### Models

- Linear Regression
- Random Forest Regression
- HistGradientBoosting Regression

A seven-day historical average is also used as a simple baseline in the initial model comparison.

### Engineered Features

| Feature | Description |
|---|---|
| `lag_1` | Sales one day earlier |
| `lag_7` | Sales seven days earlier |
| `lag_14` | Sales fourteen days earlier |
| `lag_28` | Sales twenty-eight days earlier |
| `rolling_mean_7` | Average sales over the preceding seven days |
| `rolling_mean_28` | Average sales over the preceding twenty-eight days |
| Calendar features | Date-related information, where available |

### Evaluation Metrics

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **WAPE:** Weighted Absolute Percentage Error

Lower values indicate better performance for these metrics.

### Recursive 28-Day Evaluation

The project evaluates multi-step forecasting using a 28-day holdout period.

The evaluation process:

1. Uses historical observations as the initial input.
2. Predicts the next day.
3. Appends the predicted value to the available history.
4. Repeats the process for 28 days.
5. Compares predictions against actual sales.

The recorded experiment used 111 sampled product-store series, with 3,108 forecasts per model.

---

## 10. Running Automated Tests

SalesPulse uses Pytest to validate API functionality and custom dataset input validation.

From the project root, run:

```bash
python -m pytest -v
```

To run only API tests:

```bash
python -m pytest tests/test_api.py -v
```

To run only custom dataset tests:

```bash
python -m pytest tests/test_custom_dataset.py -v
```

To check Python syntax:

```bash
python -m py_compile dashboard/app.py src/prediction/custom_dataset.py
```

The latest recorded test run completed successfully with:

```text
19 passed, 1 warning
```

The warning concerns a Starlette/httpx TestClient deprecation and did not cause test failures.

---

## 11. Troubleshooting

### Problem: Dashboard cannot connect to the API

Make sure the FastAPI server is running in the first terminal:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Check the API health endpoint:

```text
http://localhost:8000/health
```

Also verify that the dashboard's API URL points to the correct backend address.

### Problem: Prediction file not found

Verify that the required file exists:

```text
data/processed/predictions/latest_predictions.csv
```

Generate the prediction artifact using the project's prediction pipeline if it is missing.

### Problem: Model metrics not available

Check whether the model comparison file exists:

```text
data/processed/model_results/model_comparison.csv
```

Ensure that the model evaluation process has generated the required output.

### Problem: Python module not found

Activate the virtual environment and install dependencies again:

```bash
pip install -r requirements.txt
```

### Problem: Port already in use

Use another available port, for example:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

If you change the API port, update the dashboard API configuration accordingly.

---

## 12. Future Improvements

Potential future enhancements include:

- Training and evaluating models across all product-store series.
- Hyperparameter optimization.
- Integration of promotions, holidays, and pricing features.
- Additional forecasting algorithms.
- Prediction intervals and uncertainty estimation.
- Automated model retraining and model version management.
- Production deployment with Docker and cloud infrastructure.
- Model drift monitoring and performance tracking.

---

## 13. Project Information

**Project Name:** SalesPulse – Automated Retail Sales Prediction System  
**Course:** CSCE 5210 – Fundamentals of Artificial Intelligence  
**Repository:** https://github.com/Chathrapathi27/SalesPulse

**Developed using:** Python, Scikit-learn, Pandas, FastAPI, and Streamlit.

---

*SalesPulse demonstrates how artificial intelligence and machine learning can support retail demand forecasting through an integrated data-processing, model-evaluation, API, and dashboard workflow.*
