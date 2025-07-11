# Churn Prediction Project

This project focuses on predicting customer churn based on demographic and behavioral data. The goal is to identify clients who are at risk of leaving, enabling the company to take proactive retention actions.

---

## 📦 Usage

All steps are included in `01_notebook.ipynb`, including:

- EDA (`Exploratory Data Analysis`)
- Data loading (`data_loading.py`)
- Feature engineering & scaling (`preprocessing.py`)
- Model training (`modeling.py`)
- Cross-validation (`evaluation.py`)

---

## 🧠 Models Used

- Logistic Regression (initial testing)
- Random Forest
- Gradient Boosting
- Decision Tree
- K-Nearest Neighbors
- XGBoost

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation (5-fold)

---

## 📁 Data

**Dataset:** `data/raw/Klienci_DB.csv`

**Columns include:**

- `klient_id`
- `wiek`
- `miasto`
- `plec`
- `dochód_miesięczny`
- `dni_od_ostatniego_zakupu`
- `średnia_wartość_zakupu`
- `liczba_reklamacji`
- `czy_konto_firmowe`
- `czy_churn` (target)

---

## 🔍 Goals

- Build a clean machine learning pipeline.
- Compare multiple models.
- Evaluate results using cross-validation.
- Prepare code structure for further deployment or experimentation.

---

## 🗂️ Project Structure

```
churn-prediction-project/
│
├── data/
│   └── raw/
│       └── Klienci_DB.csv         # Input dataset
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   └── 02_model_pipeline.ipynb    # Main pipeline: preprocessing, modeling, evaluation
│
├── src/
│   ├── data_loading.py            # Data loading function
│   ├── preprocessing.py           # Feature engineering and scaling
│   ├── modeling.py                # Model training and comparison
│   └── evaluation.py              # Cross-validation evaluation
│
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git https://github.com/Wykoo/churn-prediction-project.git
cd churn-prediction-project
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # For Windows use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

*Note: If `requirements.txt` is missing, manually install required libraries like:*
```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
```

---
