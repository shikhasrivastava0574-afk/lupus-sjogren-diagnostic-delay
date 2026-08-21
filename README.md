# 🩸 Explainable Longitudinal Machine Learning Framework for Predicting Prolonged Diagnostic Delay in Systemic Lupus Erythematosus and Sjögren's Disease

An end-to-end explainable longitudinal machine learning framework designed to predict and prevent **Prolonged Diagnostic Delay ($>24$ Months)** in **Systemic Lupus Erythematosus (SLE)** and **Sjögren's Disease (SjD)** using patient EHR trajectories.

---

## 📌 Problem & Motivation

Patients suffering from SLE and Sjögren's Disease routinely experience diagnostic delays ranging from **3 to 7+ years**. Early symptoms—such as chronic fatigue, arthralgia, dry eyes/mouth (Sicca), malar rash, and photosensitivity—are non-specific and overlap with common conditions like fibromyalgia and osteoarthritis.

### 🌟 Key Novelty & Distinction
Rather than building another standard disease classifier (*"Does patient X have Lupus today?"*), this framework models **longitudinal patient clinical history** to address a fundamental clinical bottleneck:
1. **Predicting Prolonged Delay Risk**: Identifies patients at high risk of getting stuck in a $>24$-month diagnostic journey.
2. **Explainable AI (XAI)**: Uses SHAP (SHapley Additive exPlanations) to uncover global population drivers and local patient-level risk factors (e.g. *high PCP encounter density without rheumatology referral*).
3. **Actionable Clinical Decision Support**: Features a counterfactual engine quantifying diagnostic delay months saved by early serology panel ordering or early specialist referral.

---

## 🏗️ Repository Architecture

```
├── app_flask.py                       # Flask REST API and web application server
├── templates/
│   └── index.html                     # Modern HTML5 + Tailwind CSS + Chart.js web dashboard
├── data_simulator.py                  # Clinically realistic longitudinal synthetic EHR generator
├── preprocessing_feature_engineering.py# 52-dimensional temporal feature extraction pipeline
├── models.py                          # PyTorch LSTM, RandomForest, HistGradientBoosting classifiers/regressors
├── explainability.py                  # Model-agnostic SHAP local & global attribution engine
├── evaluation.py                      # Multi-window benchmark evaluator & delay reduction calculator
├── train_and_save_pipeline.py         # Offline pipeline training & serialization script
├── requirements.txt                   # Python environment dependencies
├── benchmark_results.json             # Cross-validation performance outputs
├── tests/
│   └── test_framework.py              # Automated unit test suite
└── simulated_data/
    ├── patients.csv                   # Patient static demographics & ground-truth delay labels
    ├── encounters.csv                 # Longitudinal visit logs with temporal symptoms & lab markers
    └── trained_pipeline.pkl           # Pre-trained serialized model pipeline
```

---

## 📊 Benchmark Performance Results

Evaluated using 5-Fold Stratified Cross-Validation across 6, 12, and 24-month observation windows:

| Observation Window | Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Avg Delay Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **6 Months** | `PyTorch_LSTM` | **0.8725** | **0.8597** | **83.52%** | 75.00% | 11.2 Months saved |
| **12 Months** | `PyTorch_LSTM` | **0.9195** | **0.9209** | **87.78%** | **81.09%** | **14.8 Months saved** |
| **12 Months** | `RandomForest` | 0.8763 | 0.8849 | 83.70% | 74.35% | 12.1 Months saved |
| **24 Months** | `PyTorch_LSTM` | **0.9152** | **0.9225** | **87.78%** | **80.22%** | 18.3 Months saved |

- **Continuous Delay Duration Estimation**: `RandomForestRegressor` estimates total diagnostic delay in months with a Mean Absolute Error (MAE) of **8.01 Months**.
- **Clinical Impact**: Early ML detection at $T_{obs} = 12$ months reduces diagnostic delay by an average of **14.8 Months per flagged patient**.

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/shikhasrivastava0574-afk/lupus-sjogren-diagnostic-delay.git
cd lupus-sjogren-diagnostic-delay
pip install -r requirements.txt
```

### 2. Generate Synthetic EHR Data & Pre-train Pipeline
```bash
python3 data_simulator.py
python3 train_and_save_pipeline.py
```

### 3. Run Automated Tests & Evaluation Benchmarks
```bash
python3 tests/test_framework.py
python3 evaluation.py
```

### 4. Launch Custom Modern Web Dashboard
```bash
python3 app_flask.py
```
Open **`http://localhost:5050`** in your browser.

---

## ☁️ Free Hosting on Streamlit Community Cloud

You can deploy this dashboard online for free in **3 simple steps**:
1. Push this repository to GitHub.
2. Log into [Streamlit Community Cloud](https://streamlit.io/cloud) using your GitHub account.
3. Click **"New App"**, select your repository, set Main file path to `app.py`, and click **Deploy**!

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
