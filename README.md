# 🩺 Diabetes Risk Prediction: An End-to-End Machine Learning Portfolio Project

> **A rigorous, end-to-end machine learning workflow exploring tabular physiological data, leak-free preprocessing pipelines, stratified cross-validation hyperparameter tuning, model trade-off evaluation, and interactive Streamlit deployment.**

---

> [!CAUTION]
> ### ⚠️ Critical Educational & Healthcare Disclaimer
> **This software and machine learning project are strictly for educational, research, and portfolio demonstration purposes.**
> - **Not Clinically Validated:** The predictive models, preprocessing rules, risk categories, and calculated probability estimates are **not clinically validated**, tested in healthcare trials, or certified by any regulatory agency (e.g., FDA, CE-IVD, WHO, or local medical authorities).
> - **Must Not Be Used for Medical Decisions:** This project **must never be used for clinical diagnosis, patient screening, triage, treatment selection, or medical management of diabetes or any other health condition.**
> - **Demographic and Population Limitations:** The underlying historical dataset (the Pima Indians Diabetes Database) was collected from adult female patients of Pima Indian heritage in the Phoenix, Arizona area prior to 1990. Because genetic predispositions, modern lifestyle factors, environmental contexts, and baseline physiological norms differ across cohorts, **the findings and statistical relationships cannot be generalized** across diverse demographic groups, male patients, pediatric populations, or modern clinical environments.
> - **Consult Healthcare Professionals:** Anyone seeking medical advice, diagnosis, or treatment for diabetes or related symptoms should immediately consult a licensed medical doctor or healthcare provider.

---

## 📑 Table of Contents

1. [Project Overview & Architecture](#-project-overview--architecture)
2. [Project Objectives](#-project-objectives)
3. [Features of the Streamlit Application](#-features-of-the-streamlit-application)
4. [Dataset Description & Limitations](#-dataset-description--limitations)
5. [The Eight Input Features & Clinical Meaning](#-the-eight-input-features--clinical-meaning)
6. [Technologies & Libraries Used](#-technologies--libraries-used)
7. [End-to-End Machine Learning Workflow](#-end-to-end-machine-learning-workflow)
8. [Data Preprocessing & Leakage Prevention](#-data-preprocessing--leakage-prevention)
9. [Models Compared](#-models-compared)
10. [Cross-Validation Results](#-cross-validation-results)
11. [Held-Out Test Set Results](#-held-out-test-set-results)
12. [Model Selection & Evaluation Analysis](#-model-selection--evaluation-analysis)
13. [Project Directory Structure](#-project-directory-structure)
14. [Windows Installation Guide](#-windows-installation-guide)
15. [Running the Analysis & Streamlit Application](#-running-the-analysis--streamlit-application)
16. [Saved Artifacts & Outputs Guide](#-saved-artifacts--outputs-guide)
17. [Known Limitations & Responsible-Use Statement](#-known-limitations--responsible-use-statement)
18. [Planned Future Improvements](#-planned-future-improvements)

---

## 🎯 Project Overview & Architecture

This repository demonstrates how to build, benchmark, and deploy a machine learning classification system following clean software engineering and data science best practices. Rather than training models on raw, unvalidated CSV records, this project addresses the real-world statistical and physiological challenges typical of medical tabular data: identifying hidden missingness disguised as numerical zeros, preventing information leakage across train/validation/test partitions, prioritizing clinically meaningful metrics over raw accuracy, and serving the model through an accessible web interface.

```
                    ┌────────────────────────────────────────┐
                    │      Raw Dataset (data/diabetes.csv)   │
                    │         768 Rows x 9 Columns           │
                    └───────────────────┬────────────────────┘
                                        │
                         [Exploratory Data Analysis]
                         - Class Distribution (65.1% / 34.9%)
                         - Zero-as-Missing Detection
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │         Stratified 80 / 20 Train-Test Split         │
             │   Train Set: 614 Patients  |  Test Set: 154 Patients │
             └──────────┬───────────────────────────────┬──────────┘
                        │ (Isolated Partition)          │ (Strictly Held Out)
                        ▼                               ▼
       ┌─────────────────────────────────┐      ┌───────────────┐
       │     Modular Pipeline Setup      │      │               │
       │    ColumnTransformer Engine:    │      │               │
       │ 1. Zero-to-NaN Conversion       │      │               │
       │ 2. Median Imputation            │      │               │
       │ 3. StandardScaler               │      │               │
       └────────────────┬────────────────┘      │               │
                        │                       │               │
                        ▼                       │               │
       ┌─────────────────────────────────┐      │               │
       │   5-Fold Stratified CV Tuning   │      │               │
       │   - Logistic Regression         │      │               │
       │   - Random Forest               │      │               │
       │   - Support Vector Machine      │      │               │
       │   Selection Criterion: Mean F1  │      │               │
       └────────────────┬────────────────┘      │               │
                        │                       │               │
            (Selected: Logistic Regression)     │               │
                        │                       │               │
                        ▼                       │               │
       ┌─────────────────────────────────┐      │               │
       │ Export: best_tuned_pipeline     │      │               │
       │           (.joblib)             │      │               │
       └────────────────┬────────────────┘      │               │
                        │                       │               │
                        └───────────────┬───────┴───────────────┘
                                        │
                                        ▼
                         [Final Test Evaluation & Audit]
                         - Generalization vs. Single Split
                                        │
                                        ▼
                         [Interactive Streamlit App]
                         - Live Input Preprocessing
                         - Probability Gauge & Explanations
```

---

## 🎯 Project Objectives

1. **Implement Leak-Free Preprocessing:** Guarantee that all statistical estimators (imputation medians and scaling parameters) are calculated solely from training partitions, preventing test set contamination.
2. **Handle Biological Impossibilities:** Properly identify and treat biologically invalid zero entries in diagnostic columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) as missing values rather than true biological zeros.
3. **Conduct Rigorous Model Benchmarking:** Compare linear models (`Logistic Regression`), tree ensembles (`Random Forest`), and kernel methods (`Support Vector Machine`) using identical preprocessing and validation folds.
4. **Prioritize Clinically Relevant Metrics:** Mitigate the distorting effects of class imbalance (~65% negative / ~35% positive) by optimizing for **F1-Score** (balancing precision and recall) and **ROC-AUC** rather than raw classification accuracy.
5. **Demonstrate Methodological Integrity:** Provide transparent reasoning for why multi-fold cross-validation is used as the selection authority over single held-out test splits.
6. **Deploy an Interactive, Educational Application:** Encapsulate the complete end-to-end pipeline into an interactive Streamlit tool that exposes real-time predictions, probabilities, and educational warnings.

---

## 💻 Features of the Streamlit Application

The interactive web interface is implemented in [`app.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/app.py) using Streamlit. It loads the pre-fit serialized pipeline (`models/best_tuned_pipeline.joblib`) directly without retraining. Key features include:

- **Interactive Physiological Inputs:** Numeric input widgets for all eight patient features, with configured defaults, ranges, and units.
- **Identical Zero-to-NaN Preprocessing:** Mirrors the offline training pipeline by converting biological zero entries in physiological fields (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) to `NaN`, allowing the pipeline's median imputer to handle them consistently.
- **Dual Classification Output:**
  - **Categorical Risk Level:** Clear visual card classifying the inputs as **Non-Diabetic (Class 0)** or **At-Risk Diabetic (Class 1)**.
  - **Estimated Class Probability:** The model's estimated probability score (e.g., `42.8% estimated risk of diabetes`) displayed alongside a visual progress bar.
- **Submitted Feature Values:** An expandable table displays the eight values submitted by the user for the current prediction.
- **Persistent Healthcare Disclaimers:** Visible callout banners and sidebar notes emphasizing the educational scope and cautioning against real-world self-diagnosis.
- **Self-Contained Execution:** Operates entirely locally without external API keys or remote cloud dependencies.

---

## 📊 Dataset Description & Limitations

### Dataset Overview
- **Dataset Title:** Pima Indians Diabetes Database
- **Originating Body:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Hosted By:** UCI Machine Learning Repository / Kaggle
- **Dataset Size:** 768 patient records (rows)
- **Dimensionality:** 8 continuous/integer input features and 1 binary target outcome
- **Target Variable (`Outcome`):**
  - `0`: Non-diabetic (500 records — 65.10%)
  - `1`: Diabetic (268 records — 34.90%)

### Dataset Limitations
1. **Demographic and Gender Restriction:** All participants in this historical study were females of Pima Indian descent aged 21 years or older residing near Phoenix, Arizona. Consequently, models trained on this cohort cannot represent adult males, pediatric populations, or individuals with different ancestral genetic profiles.
2. **Substantial Unrecorded Observations:** Diagnostic tests that were not recorded were encoded as `0`. Notably, **48.7% of insulin** values and **29.6% of skin thickness** values are zeros. While median imputation allows numerical training, in clinical reality missingness is non-random (Missing Not At Random — MNAR).
3. **Modest Sample Size:** With only 768 total observations, single validation splits are susceptible to high sampling variance, underscoring the necessity of multi-fold cross-validation.
4. **Historical Testing Standards:** The clinical assays and thresholds used prior to 1990 may not reflect modern diagnostic practices (e.g., standard HbA1c testing).

---

## 🔬 The Eight Input Features & Clinical Meaning

The dataset contains eight diagnostic measurements. A crucial step in this project is recognizing which zeros represent legitimate physiological counts versus unrecorded tests:

| # | Feature Name | Data Type | Units / Range | Physiological Meaning & Clinical Interpretation | Handling of Zero Values |
|:---:|:---|:---:|:---:|:---|:---|
| 1 | `Pregnancies` | `int64` | Count (0–17) | Number of times the patient has been pregnant. | **0 is biologically valid** (represents nulliparous patients). Preserved as 0. |
| 2 | `Glucose` | `int64` | mg/dL (0–199) | Plasma glucose concentration measured after a 2-hour oral glucose tolerance test (OGTT). Normal fasting levels are typically <100 mg/dL. | **0 is biologically impossible** in a living patient (5 occurrences). Converted to `NaN` and imputed. |
| 3 | `BloodPressure` | `int64` | mm Hg (0–122) | Diastolic blood pressure (pressure in arteries between heartbeats). Typical normal range is 60–80 mm Hg. | **0 is biologically impossible** (35 occurrences). Converted to `NaN` and imputed. |
| 4 | `SkinThickness` | `int64` | mm (0–99) | Triceps skin fold thickness, an anthropometric measurement used to estimate subcutaneous body fat. | **0 indicates unrecorded test** (227 occurrences / 29.6%). Converted to `NaN` and imputed. |
| 5 | `Insulin` | `int64` | $\mu\text{U/mL}$ (0–846) | 2-hour serum insulin level following glucose administration. | **0 indicates unrecorded test** (374 occurrences / 48.7%). Converted to `NaN` and imputed. |
| 6 | `BMI` | `float64` | $\text{kg/m}^2$ (0.0–67.1) | Body Mass Index ($\text{weight in kg} / (\text{height in m})^2$). Clinical categories: Normal (18.5–24.9), Overweight (25–29.9), Obese ($\ge 30$). | **0 is biologically impossible** (11 occurrences). Converted to `NaN` and imputed. |
| 7 | `DiabetesPedigreeFunction` | `float64` | Score (0.078–2.42) | A mathematical genetic score assessing diabetes family history based on pedigree kinship and age of onset in relatives. | Non-zero continuous score. Preserved as-is. |
| 8 | `Age` | `int64` | Years (21–81) | Chronological age of the patient. In this dataset, all individuals are adults $\ge 21$. | Valid non-zero integer. Preserved as-is. |
| — | **`Outcome` (Target)** | `int64` | Binary (0 or 1) | Diagnostic determination: **0 = Negative / Non-Diabetic**, **1 = Positive / Diabetic**. | Target label. |

---

## 🛠️ Technologies & Libraries Used

| Category | Technology / Library | Purpose in Project |
|:---|:---|:---|
| **Core Environment** | **Python 3.10 – 3.12** | Core programming language environment |
| **Data Manipulation** | **pandas** | Tabular data ingestion, missing-value masking, summary metrics, and CSV reporting |
| **Numeric Computing** | **numpy** | Array manipulations, vector operations, and `NaN` representation |
| **Machine Learning** | **scikit-learn** | End-to-end modeling framework: <br>• Preprocessing: `SimpleImputer`, `StandardScaler`, `ColumnTransformer`, `Pipeline`<br>• Algorithms: `LogisticRegression`, `RandomForestClassifier`, `SVC`, `DecisionTreeClassifier`<br>• Validation: `StratifiedKFold`, `GridSearchCV`<br>• Metrics: `f1_score`, `roc_auc_score`, `precision_score`, `recall_score`, `accuracy_score`, `confusion_matrix` |
| **Data Visualization** | **matplotlib** | Canvas rendering and figure layout generation |
| **Statistical Plotting** | **seaborn** | High-level statistical charts: KDE distributions, boxplots, correlation heatmaps |
| **Model Persistence** | **joblib** | Serializing and loading complete fitted scikit-learn pipelines without retraining |
| **Web Dashboard** | **streamlit** | Interactive, browser-based web application with reactive controls and metric cards |
| **Version Control** | **Git & GitHub** | Source code management, repository tracking, and portfolio presentation |

---

## 🔄 End-to-End Machine Learning Workflow

The project is structured into sequential, reproducible stages:

```text
Stage 1: Environment & Setup ──► Verify Python runtime and packages (virtual environment verification)
Stage 2: Exploratory Data Analysis ──► Validate schema, missingness, and zero counts (explore_data.py)
Stage 3: Statistical Visualizations ──► Export class counts, distributions, heatmaps (visualize_data.py)
Stage 4: Preprocessing & Baseline Training ──► 80/20 Stratified split & Baseline Logistic Regression (train_model.py)
Stage 5: Multi-Model Benchmark ──► Compare 4 baseline architectures on test set (compare_models.py)
Stage 6: Hyperparameter Optimization ──► 5-Fold Stratified CV tuning & export best pipeline (tune_models.py)
Stage 7: Web Application Deployment ──► Interactive Streamlit diagnosis simulator (app.py)
```

1. **Stage 1 — Environment Setup:** Initialize directory hierarchy, configure virtual environment (`.venv`), install pinned dependencies via `requirements.txt`, and verify the Python interpreter and required libraries.
2. **Stage 2 — Exploratory Data Analysis (EDA):** Run [`src/explore_data.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/src/explore_data.py) to audit the dataset shape `(768, 9)`, confirm that no native nulls exist in the CSV, analyze the 65.1% / 34.9% target distribution, and quantify biological zeros.
3. **Stage 3 — Data Visualization:** Run [`src/visualize_data.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/src/visualize_data.py) to render and save distribution charts, boxplots, and correlation heatmaps to `outputs/`.
4. **Stage 4 — Preprocessing & Baseline Pipeline:** Run [`src/prepare_data.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/src/prepare_data.py) and [`src/train_model.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/src/train_model.py). Implement stratified 80/20 partitioning, assemble the `ColumnTransformer`, fit a baseline Logistic Regression model, evaluate on the test split, and export `outputs/model_evaluation.txt`.
5. **Stage 5 — Baseline Model Comparison:** Run [`src/compare_models.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/src/compare_models.py) to benchmark Logistic Regression, Decision Tree, Random Forest, and SVM under uniform test conditions, saving `outputs/model_comparison.csv` and `outputs/06_model_comparison.png`.
6. **Stage 6 — Hyperparameter Optimization & Cross-Validation:** Run [`src/tune_models.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/src/tune_models.py). Perform 5-Fold Stratified Cross-Validation strictly on training data using `GridSearchCV`. Optimize for F1-score, export the winning pipeline to `models/best_tuned_pipeline.joblib`, and benchmark all tuned models on the unseen 20% test split.
7. **Stage 7 — Web Application Deployment:** Run [`app.py`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/app.py) using `streamlit run app.py` to launch the local web UI for interactive scenario testing and prediction.

---

## 🛡️ Data Preprocessing & Leakage Prevention

Data leakage occurs when information from outside the training partition influences feature transformations, resulting in overly optimistic evaluation metrics that fail in production. This project implements strict safeguards:

1. **Split-First Architecture:**
   - The dataset of 768 rows is split into **80% Training (614 patients)** and **20% Test (154 patients)** using `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
   - The test set is set aside immediately and is never examined or accessed during imputation parameter fitting, scaling parameter fitting, or hyperparameter searching.
2. **Selective Zero-to-NaN Conversion:**
   - Diagnostic columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) have values of `0` replaced with `np.nan`.
   - Non-diagnostic variables (`Pregnancies`, `Age`, `DiabetesPedigreeFunction`) are preserved because zero is either valid or absent.
3. **Encapsulation in `Pipeline` and `ColumnTransformer`:**
   - Imputation (`SimpleImputer(strategy='median')`) and standardization (`StandardScaler()`) are enclosed inside an automated scikit-learn `Pipeline`.
   - In cross-validation folds, median values and standard deviation parameters are calculated **only from the training folds** and then applied to the validation fold. Information from validation folds never bleeds into the imputation calculations.

---

## 🤖 Models Compared

Three distinct model families with different inductive biases were evaluated and tuned:

1. **Logistic Regression (Linear Probabilistic Classifier):**
   - Fits a linear decision boundary using the logistic sigmoid function.
   - Highly interpretable: model coefficients directly map to feature odds ratios.
   - Hyperparameter grid: Regularization strength $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$.
   - **Best Tuned Parameter:** `C = 10.0`.
2. **Random Forest Classifier (Ensemble Bagging Method):**
   - A collection of randomized, decorrelated decision trees that aggregate predictions to reduce variance.
   - Non-linear: naturally captures complex feature interactions and non-monotonic boundaries.
   - Hyperparameter grid: `n_estimators` $\in [50, 100, 200]$, `max_depth` $\in [\text{None}, 5, 10]$, `min_samples_split` $\in [2, 5, 10]$.
   - **Best Tuned Parameters:** `n_estimators = 100, max_depth = 10, min_samples_split = 5`.
3. **Support Vector Machine (Margin Maximization):**
   - Finds the optimal hyperplane separating classes with maximal geometric margin.
   - Hyperparameter grid: Kernel $\in [\text{'linear'}, \text{'rbf'}]$, $C \in [0.1, 1.0, 10.0]$, $\gamma \in [\text{'scale'}, \text{'auto'}]$.
   - **Best Tuned Parameters:** `kernel = 'linear', C = 1.0, gamma = 'scale'`.

---

## 📊 Cross-Validation Results

To ensure unbiased hyperparameter selection, 5-Fold Stratified Cross-Validation was run **exclusively on the 614 training records**. Preprocessing pipelines were refitted independently on every fold.

*Recorded in [`outputs/cross_validation_results.csv`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/outputs/cross_validation_results.csv):*

| Model Architecture | Selected Hyperparameters | Mean CV F1-Score | Std CV F1-Score | Mean CV ROC-AUC | Std CV ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Tuned)** 🏆 | `{'C': 10.0}` | **0.6563** | $\pm 0.0291$ | 0.8433 | $\pm 0.0188$ |
| **Support Vector Machine (Tuned)** | `{'C': 1.0, 'gamma': 'scale', 'kernel': 'linear'}` | 0.6524 | $\pm 0.0233$ | **0.8446** | $\pm 0.0158$ |
| **Random Forest (Tuned)** | `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}` | 0.6496 | $\pm 0.0209$ | 0.8251 | $\pm 0.0290$ |

*Primary Selection Metric: Mean Cross-Validation F1-score.*

---

## 🎯 Held-Out Test Set Results

After hyperparameter tuning and model selection were finalized, all three optimized pipelines were evaluated **once** on the untouched 20% test partition (154 total patients: 100 non-diabetic, 54 diabetic).

*Recorded in [`outputs/tuned_model_test_results.csv`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/outputs/tuned_model_test_results.csv):*

| Model | Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TN | FP | FN | TP |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Tuned)** | `max_depth=10, min_samples_split=5, n_estimators=100` | **0.7403** | **0.6458** | **0.5741** | **0.6078** | 0.8065 | 83 | 17 | 23 | **31** |
| **Logistic Regression (Tuned)** | `C=10.0` | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | 82 | 18 | 27 | 27 |
| **Support Vector Machine (Tuned)** | `C=1.0, gamma='scale', kernel='linear'` | 0.7013 | 0.5909 | 0.4815 | 0.5306 | **0.8133** | 82 | 18 | 28 | 26 |

*Confusion matrix key: TN = True Negatives (healthy correctly identified), FP = False Positives (Type I error), FN = False Negatives (Type II error), TP = True Positives (diabetic correctly identified).*

---

## ⚖️ Model Selection & Evaluation Analysis

An honest and critical discussion of why **Logistic Regression** was saved as the final production model despite **Random Forest** scoring higher on the test split:

### 1. Selection Authority: Cross-Validation vs. Single Split
- In accordance with disciplined machine learning practices, **Logistic Regression was selected because it achieved the highest Mean F1-Score (0.6563)** across the 5-fold cross-validation folds.
- Cross-validation tests an algorithm across 5 distinct partitions of the training data. This reduces the risk that performance is an artifact of one favorable data split.
- In addition, Logistic Regression demonstrated excellent discriminative ranking with a Mean CV ROC-AUC of **0.8433** and a low standard deviation ($\pm 0.0188$).

### 2. Why Random Forest Won the Held-Out Test Set
- On the single test set (154 patients), Random Forest achieved an F1-score of **0.6078** compared to **0.5455** for Logistic Regression, catching 4 additional positive cases (31 vs. 27 true positives).
- Ensemble tree models can capitalize on specific non-linear relationships or subgroup interactions that happened to be concentrated in this particular test partition.

### 3. Why the Test Set Must Never Be Used to Choose the Model
> [!IMPORTANT]
> **Data Science Rule:** Using test set performance to select which model to deploy introduces **post-hoc data snooping leakage**.
> - The held-out test set exists solely to measure how well the already-selected model generalizes to unseen data.
> - If an engineer evaluates multiple models on the test set and subsequently chooses the test-set winner, the test set ceases to be an independent out-of-sample benchmark and effectively functions as a secondary training/validation set.
> - Because this repository adheres to sound methodology, the selection rule (Mean CV F1) was maintained, and the trade-off is documented transparently rather than obscured.

---

## 📁 Project Directory Structure

```text
diabetes-risk-prediction/
│
├── .gitignore                          # Ignores .venv, cache, raw data, while tracking code & outputs
├── README.md                           # Comprehensive portfolio documentation (this file)
├── requirements.txt                    # Pinned project dependencies for reproduction
├── app.py                              # Interactive Streamlit web application
│
├── data/
│   ├── .gitkeep                        # Tracks data directory in version control
│   └── diabetes.csv                    # Pima Indians Diabetes dataset (768 rows, 9 columns; git-ignored)
│
├── models/
│   ├── .gitkeep                        # Tracks models directory in version control
│   ├── best_tuned_pipeline.joblib      # Production pipeline (CV Winner: Tuned Logistic Regression, C=10.0)
│   └── diabetes_logistic_regression.joblib  # Baseline Logistic Regression pipeline (git-ignored)
│
├── notebooks/
│   └── .gitkeep                        # Reserved for interactive Jupyter notebook development
│
├── outputs/
│   ├── 01_outcome_countplot.png         # Bar chart: Class 0 (500) vs Class 1 (268) distribution
│   ├── 02_histograms_glucose_age_bmi.png # Histograms and KDE curves for primary features
│   ├── 03_boxplots_glucose_bmi.png      # Boxplots comparing Glucose and BMI by diagnosis
│   ├── 04_correlation_heatmap.png       # Pearson correlation matrix across all features
│   ├── 06_model_comparison.png          # Visual bar chart comparing baseline algorithms
│   ├── cross_validation_results.csv     # 5-Fold Stratified CV scores on training data
│   ├── model_comparison.csv             # Baseline model metrics on the 20% test split
│   ├── model_evaluation.txt             # Baseline evaluation report and learned coefficients
│   └── tuned_model_test_results.csv     # Tuned model metrics and confusion matrices on test split
│
└── src/
    ├── .gitkeep                        # Tracks src directory in version control
    ├── explore_data.py                 # EDA: audits missingness, stats, and zero values
    ├── visualize_data.py               # Generates and saves exploratory charts to outputs/
    ├── prepare_data.py                 # Preprocessing pipeline: zero-to-NaN, ColumnTransformer, split logic
    ├── train_model.py                  # Baseline trainer: fits baseline Logistic Regression pipeline
    ├── compare_models.py               # Comparative benchmark of 4 baseline algorithms on test set
    └── tune_models.py                  # 5-fold CV hyperparameter search, selects & exports best pipeline
```

---

## 💻 Windows Installation Guide

Follow these commands in **Windows PowerShell** or **Command Prompt** to set up the project:

### 1. Navigate to Project Directory
```powershell
cd c:\Users\ASUS\Desktop\Projects\diabetes-risk-prediction
```

### 2. Create the Virtual Environment
Create an isolated environment named `.venv`:
```powershell
py -m venv .venv
```
*(If the `py` launcher is not installed, run `python -m venv .venv`)*

### 3. Activate the Virtual Environment
- **In PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If you encounter a script execution restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then re-run the activation script).*
- **In Command Prompt (`cmd.exe`):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **In Git Bash:**
  ```bash
  source .venv/Scripts/activate
  ```

Your terminal prompt will now display the `(.venv)` prefix.

### 4. Install Dependencies
Install all required libraries specified in [`requirements.txt`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/requirements.txt):
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Analysis & Streamlit Application

With your virtual environment active (`(.venv)`), run the project commands:

### Step 1: Environment Sanity Check
Confirm that the Python interpreter and required packages are functioning inside the virtual environment:
```powershell
python -c "import pandas, sklearn, streamlit; print('Environment OK')"
```

### Step 2: Run Exploratory Data Analysis
Examine dataset summary statistics, null counts, and zero distributions:
```powershell
python src/explore_data.py
```

### Step 3: Generate Exploratory Visualizations
Re-create and export distribution plots, boxplots, and correlation heatmaps to `outputs/`:
```powershell
python src/visualize_data.py
```

### Step 4: Run Baseline Data Preparation & Training
Fit the baseline Logistic Regression pipeline and generate the initial evaluation report:
```powershell
python src/prepare_data.py
python src/train_model.py
```

### Step 5: Benchmark Baseline Algorithms
Train and evaluate 4 baseline algorithms on the test split:
```powershell
python src/compare_models.py
```

### Step 6: Execute 5-Fold Cross-Validation & Hyperparameter Tuning
Run GridSearchCV over candidate models, evaluate on the test split, and export the best pipeline:
```powershell
python src/tune_models.py
```

### Step 7: Launch the Streamlit Web Application
Start the interactive application locally:
```powershell
streamlit run app.py
```

Streamlit will output local access URLs in your terminal:
```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open **`http://localhost:8501`** in any web browser to interact with the model. To stop the application server, press `Ctrl + C` in your terminal.

---

## 📦 Saved Artifacts & Outputs Guide

### Model Artifacts (`models/`)
- **[`best_tuned_pipeline.joblib`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/models/best_tuned_pipeline.joblib) (3.7 KB):** The deployment artifact selected via cross-validation. Encapsulates the fitted `ColumnTransformer` (median imputation + standard scaling) and tuned `LogisticRegression(C=10.0)`. Used directly by `app.py`.
- **`diabetes_logistic_regression.joblib` (3.7 KB):** The initial baseline pipeline (`C=1.0`) created during Stage 4.

### Evaluation & Metric Tables (`outputs/`)
- **[`cross_validation_results.csv`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/outputs/cross_validation_results.csv):** Mean and standard deviation for F1-score and ROC-AUC across 5 cross-validation training folds for Logistic Regression, Random Forest, and SVM.
- **[`model_comparison.csv`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/outputs/model_comparison.csv):** Benchmark metrics (Accuracy, Precision, Recall, F1, ROC-AUC) across 4 baseline algorithms on the 20% test split.
- **[`tuned_model_test_results.csv`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/outputs/tuned_model_test_results.csv):** Comprehensive test split performance metrics for tuned models, including full confusion matrix counts (TN, FP, FN, TP).
- **[`model_evaluation.txt`](file:///c:/Users/ASUS/Desktop/Projects/diabetes-risk-prediction/outputs/model_evaluation.txt):** Detailed plaintext evaluation report including classification breakdown and learned logistic regression weights.

### Visualizations (`outputs/`)
- **`01_outcome_countplot.png`:** Class balance bar chart illustrating the 500 / 268 target ratio.
- **`02_histograms_glucose_age_bmi.png`:** Three-panel histogram displaying right-skewed Age distributions and unimodal Glucose and BMI curves.
- **`03_boxplots_glucose_bmi.png`:** Grouped boxplots demonstrating significant upward distribution shifts in glucose and BMI for positive diabetic outcomes.
- **`04_correlation_heatmap.png`:** Annotated correlation matrix highlighting Glucose ($r = 0.47$) and BMI ($r = 0.29$) as primary linear correlates with diabetes.
- **`06_model_comparison.png`:** Comparative chart illustrating test-set F1 performance across baseline candidate architectures.

---

## ⚠️ Known Limitations & Responsible-Use Statement

1. **Not for Clinical Diagnosis:** This application and its trained parameters do not constitute medical advice, a diagnostic device, or clinical decision support software.
2. **Missing Not At Random (MNAR):** Missing clinical tests were recorded as `0`. While median imputation enables algorithm convergence, in medical settings tests like insulin are typically ordered for specific symptomatic indications. Treating these omissions as random misses biological nuance.
3. **Threshold Calibration:** The default classification decision threshold ($0.50$) was used. In real-world medical screening, decision thresholds are often lowered to maximize sensitivity (recall), trading off precision to reduce false negatives.
4. **Generalizability Gap:** The statistical patterns captured herein reflect a specific demographic cohort collected over three decades ago and cannot be presumed valid across general modern populations.

---

## 🔮 Planned Future Improvements

The following improvements are **planned future extensions** and are not yet implemented in the current repository:

- [ ] **Advanced Imputation Methods:** Implement iterative multivariate imputation (`IterativeImputer` / MICE) or $k$-nearest neighbors imputation (`KNNImputer`) to better handle high-missingness columns like `Insulin` and `SkinThickness`.
- [ ] **Clinical Cost-Sensitive Threshold Tuning:** Optimize the classification decision threshold specifically for recall to lower the false negative rate (missed diabetic patients) based on medical risk preferences.
- [ ] **SHAP & LIME Interpretability Integration:** Add local feature attribution explanations (SHAP waterfall plots) into the Streamlit interface to show individualized prediction drivers.
- [ ] **Modern Gradient Boosting Integration:** Benchmark modern gradient boosting frameworks (XGBoost, LightGBM, and CatBoost) alongside linear and forest models.
- [ ] **Automated CI/CD Test Pipeline:** Implement GitHub Actions to automate unit testing of data transformations, pipeline serialization, and Streamlit component loading on every commit.

---

## 📜 License & Acknowledgments

- **Dataset Credit:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), via the UCI Machine Learning Repository.
- **License:** Distributed under the MIT License for educational and portfolio demonstration use.
