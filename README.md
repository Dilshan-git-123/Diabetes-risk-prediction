# 🩺 Diabetes Risk Prediction: An End-to-End Machine Learning Portfolio Project

> **An applied machine learning workflow covering exploratory data analysis, physiological data preprocessing, cross-validation hyperparameter tuning, model trade-off evaluation, and interactive Streamlit deployment.**

---

> [!CAUTION]
> ### ⚠️ Critical Educational & Clinical Disclaimer
> **This project is strictly for educational, algorithmic study, and machine learning portfolio demonstration purposes.**
> - **Not Clinically Validated:** The models, preprocessing logic, and probability scores produced by this project are **not** certified, validated, or approved by any medical regulatory agency (e.g., FDA, CE-IVD, WHO).
> - **Do Not Use for Clinical Decisions:** This software **must never be used to diagnose, screen, treat, monitor, or manage diabetes or any other medical condition in real patients.**
> - **Demographic Limitation:** The underlying historical dataset (Pima Indians Diabetes Database) was collected from a specific demographic cohort (adult females of Pima Indian heritage in 1990) and **cannot generalize** across diverse ethnicities, modern clinical diagnostic standards, pediatric populations, or different biological sexes.
> - **Seek Professional Care:** Anyone with medical concerns or symptoms should seek direct guidance from a licensed healthcare provider.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement & Educational Objectives](#-problem-statement--educational-objectives)
3. [Dataset Source, Features & Limitations](#-dataset-source-features--limitations)
4. [Technologies & Tools Used](#-technologies--tools-used)
5. [End-to-End Project Workflow](#-end-to-end-project-workflow)
6. [Model Evaluation & Cross-Validation Results](#-model-evaluation--cross-validation-results)
7. [Why Logistic Regression Was Selected (vs. Random Forest)](#-why-logistic-regression-was-selected-vs-random-forest)
8. [Project Folder Tree](#-project-folder-tree)
9. [Saved Artifacts & Outputs Guide](#-saved-artifacts--outputs-guide)
10. [Setup Guide for Windows](#-setup-guide-for-windows)
11. [Execution Guide: Reproducing Analysis & Launching App](#-execution-guide-reproducing-analysis--launching-app)
12. [Key Machine Learning Lessons & Portfolio Takeaways](#-key-machine-learning-lessons--portfolio-takeaways)

---

## 🎯 Project Overview

This repository demonstrates a complete, production-structured data science lifecycle applied to binary classification on tabular healthcare data:

```
[Raw CSV Dataset]
       │
       ▼
[Exploratory Data Analysis] ──► [Distribution & Correlation Plots]
       │
       ▼
[Zero-to-NaN Medical Cleaning]
       │
       ▼
[Stratified 80/20 Train-Test Split] (Data Leakage Safeguard)
       │
       ▼
[Modular ColumnTransformer Pipeline] (Median Imputation + StandardScaler)
       │
       ▼
[5-Fold Stratified Cross-Validation & Grid Search]
  ├── Logistic Regression (Tuned: C=10.0) ──► Selected Model (Mean CV F1: 0.6563)
  ├── Random Forest (Tuned: max_depth=10, min_samples_split=5, n_estimators=100)
  └── Support Vector Machine (Tuned: C=1.0, gamma='scale', kernel='linear')
       │
       ▼
[Single Unseen Test Set Evaluation (154 Patients)] ──► Trade-off Analysis
       │
       ▼
[Exported Pipeline Artifact: best_tuned_pipeline.joblib]
       │
       ▼
[Interactive Streamlit Web Application (app.py)]
```

The application allows users to enter eight physiological and demographic variables, applies identical zero-value preprocessing, normalizes features through the pre-fit scaler, and produces both an algorithmic risk category (Non-Diabetic / At-Risk Diabetic) and an estimated probability score alongside educational context.

---

## 💡 Problem Statement & Educational Objectives

### Problem Statement
Type 2 diabetes is a chronic metabolic condition characterized by elevated blood glucose levels due to insulin resistance or insufficient insulin production. In clinical research, early risk stratification can identify individuals who would benefit from lifestyle interventions and formal diagnostic testing (such as fasting plasma glucose or oral glucose tolerance tests).

Building a statistical classifier for this task presents common real-world machine learning challenges:
1. **Biological implausibility in raw data:** Medical tests that were not performed were recorded as literal `0` values (e.g., Blood Pressure = 0 mm Hg, BMI = 0 kg/m²).
2. **Moderate class imbalance:** In the dataset, ~65.1% of patients are non-diabetic and ~34.9% are diabetic, making standard classification accuracy misleading.
3. **Data leakage hazards:** Imputing missing values or standardizing features on the entire dataset prior to splitting artificially inflates evaluation metrics.
4. **Generalization vs. Single-Split variance:** A model that scores highest on a single test split may not be the most stable model across multi-fold cross-validation.

### Educational Objectives
This portfolio project was designed to demonstrate:
- **Clean modular software architecture** separating exploratory analysis, preprocessing, baseline training, comparative modeling, hyperparameter optimization, and interactive deployment.
- **Strict data leakage prevention** by encapsulating imputation and scaling inside scikit-learn `Pipeline` and `ColumnTransformer` constructs, fit solely on training folds.
- **Metric-driven model selection** focused on **F1-score** (balancing precision and recall for positive diabetic cases) and **ROC-AUC** rather than raw accuracy.
- **Transparent documentation of model trade-offs**, analyzing why cross-validation and single test splits can yield divergent rankings.

---

## 📊 Dataset Source, Features & Limitations

### Source
- **Database:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Dataset Title:** Pima Indians Diabetes Database
- **Observations:** 768 patient records
- **Attributes:** 8 numeric physiological/demographic input features + 1 binary target outcome

### Feature Glossary & Zero-Value Processing

| # | Feature | Data Type | Units / Meaning | Clinical Context & Zero-Handling Strategy |
|:---|:---|:---|:---|:---|
| 1 | `Pregnancies` | `int64` | Count (0–17) | Number of times pregnant. **0 is biologically valid**; left untouched. |
| 2 | `Glucose` | `int64` | mg/dL (0–199) | 2-hour oral glucose tolerance test. **0 is biologically impossible** (5 zeros = 0.7%); converted to `NaN` and imputed with median. |
| 3 | `BloodPressure` | `int64` | mm Hg (0–122) | Diastolic blood pressure. **0 is biologically impossible** (35 zeros = 4.6%); converted to `NaN` and imputed with median. |
| 4 | `SkinThickness` | `int64` | mm (0–99) | Triceps skin fold thickness (indicator of body fat). **0 is unrecorded** (227 zeros = 29.6%); converted to `NaN` and imputed with median. |
| 5 | `Insulin` | `int64` | mu U/ml (0–846) | 2-hour serum insulin. **0 is unrecorded** (374 zeros = 48.7%); converted to `NaN` and imputed with median. |
| 6 | `BMI` | `float64` | kg/m² (0.0–67.1) | Body Mass Index. **0 is biologically impossible** (11 zeros = 1.4%); converted to `NaN` and imputed with median. |
| 7 | `DiabetesPedigreeFunction` | `float64` | Score (0.078–2.42) | Genetic pedigree score quantifying family history of diabetes. 0 is not present; left untouched. |
| 8 | `Age` | `int64` | Years (21–81) | Patient age. All patients are 21 years or older; left untouched. |
| **Target** | `Outcome` | `int64` | Binary (0 or 1) | **0 = Non-Diabetic** (500 patients, 65.10%), **1 = Diabetic** (268 patients, 34.90%). |

### Dataset Limitations
1. **Demographic Homogeneity:** All records represent females aged 21 and older of Pima Indian heritage living near Phoenix, Arizona, collected before 1990. Results cannot be generalized across men, children, or non-Pima populations.
2. **Missing Test Representation:** Nearly half of the insulin values (48.7%) and nearly a third of skin thickness values (29.6%) are missing tests coded as zeros. While median imputation allows algorithms to train, real clinical practice would account for why certain tests were omitted.
3. **Sample Size:** 768 rows is a modest sample size. While well-suited for demonstrating core ML concepts, clinical models typically require tens or hundreds of thousands of diverse records.

---

## 🛠️ Technologies & Tools Used

- **Language:** Python 3.12 (compatible with Python 3.10+)
- **Data Manipulation & Preprocessing:**
  - `pandas` (v2.3+) – Data loading, exploration, and tabular reporting
  - `numpy` (v2.2+) – Matrix operations and NaN conversions
- **Machine Learning & Pipeline Construction:**
  - `scikit-learn` (v1.6+)
    - Imputation: `SimpleImputer(strategy='median')`
    - Normalization: `StandardScaler()`
    - Pipelines: `Pipeline`, `ColumnTransformer`
    - Algorithms: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`
    - Tuning: `GridSearchCV`, `StratifiedKFold`
    - Evaluation: `f1_score`, `roc_auc_score`, `precision_score`, `recall_score`, `accuracy_score`, `confusion_matrix`, `classification_report`
- **Data Visualization:**
  - `matplotlib` (v3.10+) – Plotting backend and figure rendering
  - `seaborn` (v0.13+) – Statistical plotting (countplots, KDE histograms, boxplots, heatmaps)
- **Interactive Web UI:**
  - `streamlit` (v1.66+) – Reactive dashboard with input forms, probability metrics, and responsive layouts
- **Artifact Serialization:**
  - `joblib` (v1.4+) – Saving and reloading trained preprocessing and estimator pipelines

---

## 🔄 End-to-End Project Workflow

The project is structured into sequential stages:

### Stage 1: Setup & Environment Verification
- Verified directory scaffolding, created `.venv`, installed pinned dependencies, and ran `first_step.py` to ensure proper Python interpreter discovery.

### Stage 2: Exploratory Data Analysis (EDA)
- Script: `src/explore_data.py`
- Confirmed dataset shape `(768, 9)`, validated 0 formal CSV nulls, verified class balance (65.1% / 34.9%), and documented physiological zero occurrences.

### Stage 3: Exploratory Visualizations
- Script: `src/visualize_data.py`
- Generated publication-quality visualizations saved to `outputs/`:
  - `01_outcome_countplot.png`: Bar chart of class frequencies.
  - `02_histograms_glucose_age_bmi.png`: Feature distributions with KDE overlays.
  - `03_boxplots_glucose_bmi.png`: Boxplots showing higher median glucose and BMI in diabetic patients.
  - `04_correlation_heatmap.png`: Correlation matrix highlighting Glucose ($r = 0.47$) and BMI ($r = 0.29$) as primary linear correlates with Outcome.

### Stage 4: Modular Preprocessing & Baseline Training
- Scripts: `src/prepare_data.py`, `src/train_model.py`
- Split data into **80% training (614 patients)** and **20% held-out test (154 patients)** with `stratify=y` to preserve the 65/35 ratio.
- Configured a `ColumnTransformer` applying median imputation and standard scaling to medical features.
- Trained a baseline Logistic Regression pipeline, writing comprehensive evaluation metrics to `outputs/model_evaluation.txt` and saving `models/diabetes_logistic_regression.joblib`.

### Stage 5: Baseline Algorithm Comparison
- Script: `src/compare_models.py`
- Compared four baseline models (Logistic Regression, Decision Tree, Random Forest, SVM) on the identical test split, saving tabular metrics to `outputs/model_comparison.csv` and a bar chart to `outputs/06_model_comparison.png`.

### Stage 6: Cross-Validation & Hyperparameter Tuning
- Script: `src/tune_models.py`
- Executed 5-Fold Stratified Cross-Validation on the training set only (preventing test set contamination) with `GridSearchCV`.
- Preprocessing (median imputation + standard scaling) was fit separately within each fold to guarantee zero leakage.
- Optimized Logistic Regression, Random Forest, and SVM for F1-score while tracking ROC-AUC.
- Exported the cross-validation winner to `models/best_tuned_pipeline.joblib`.
- Evaluated all tuned models on the unseen 20% test split, saving `outputs/cross_validation_results.csv` and `outputs/tuned_model_test_results.csv`.

### Stage 7: Interactive Streamlit Web Application
- Script: `app.py`
- Built an educational web app allowing users to input patient measurements.
- Uses `joblib.load()` to load `models/best_tuned_pipeline.joblib` without retraining.
- Applies identical zero-to-NaN handling for physiological inputs.
- Displays prediction outcomes, class probability gauge, feature explanations, and clinical disclaimers.

### Stage 8: Portfolio Documentation & Project Polish
- Current stage: Standardized README, audited directory structure, verified gitignore rules, and verified reproduction steps.

---

## 📈 Model Evaluation & Cross-Validation Results

All figures presented below represent **actual, recorded outputs** generated during project execution.

### 1. Baseline Model Comparison (Stage 5 — Held-Out Test Set)
*Source: `outputs/model_comparison.csv` (Evaluated on 154 unseen patients)*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (Baseline)** | **0.7403** | **0.6522** | **0.5556** | **0.6000** | 0.7963 |
| **Random Forest (Baseline)** | 0.7273 | 0.6250 | **0.5556** | 0.5882 | 0.8076 |
| **Logistic Regression (Baseline)** | 0.7078 | 0.6000 | 0.5000 | 0.5455 | **0.8130** |
| **Decision Tree (Baseline)** | 0.6688 | 0.5333 | 0.4444 | 0.4848 | 0.6172 |

---

### 2. Hyperparameter Tuning via 5-Fold Stratified CV (Stage 6 — Training Set Only)
*Source: `outputs/cross_validation_results.csv` (614 training patients, 5 stratified folds)*

| Model | Best Hyperparameters Selected | Mean CV F1 | Std CV F1 | Mean CV ROC-AUC | Std CV ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Tuned)** 🏆 | `{'C': 10.0}` | **0.6563** | ±0.0291 | 0.8433 | ±0.0188 |
| **Support Vector Machine (Tuned)** | `{'C': 1.0, 'gamma': 'scale', 'kernel': 'linear'}` | 0.6524 | ±0.0233 | **0.8446** | ±0.0158 |
| **Random Forest (Tuned)** | `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}` | 0.6496 | ±0.0209 | 0.8251 | ±0.0290 |

*Primary Selection Metric: Mean Cross-Validation F1-score.*

---

### 3. Tuned Models on Unseen Held-Out Test Split (Stage 6 — Test Set)
*Source: `outputs/tuned_model_test_results.csv` (154 held-out patients: 100 non-diabetic, 54 diabetic)*

| Model | Best Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TN | FP | FN | TP |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Tuned)** | `max_depth=10, min_samples_split=5, n_estimators=100` | **0.7403** | **0.6458** | **0.5741** | **0.6078** | 0.8065 | 83 | 17 | 23 | **31** |
| **Logistic Regression (Tuned)** | `C=10.0` | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | 82 | 18 | 27 | 27 |
| **Support Vector Machine (Tuned)** | `C=1.0, gamma='scale', kernel='linear'` | 0.7013 | 0.5909 | 0.4815 | 0.5306 | **0.8133** | 82 | 18 | 28 | 26 |

*Confusion Matrix Abbreviations: TN = True Negatives, FP = False Positives, FN = False Negatives, TP = True Positives.*

---

## ⚖️ Why Logistic Regression Was Selected (vs. Random Forest)

A key finding in this project is that **different validation frameworks can favor different algorithms**:

### 1. Cross-Validation Winner: Logistic Regression
- Across the 5-fold stratified cross-validation on the 614 training records, **Logistic Regression achieved the highest Mean F1-score (0.6563)** compared to Random Forest (0.6496) and SVM (0.6524).
- Because cross-validation evaluates performance across five distinct validation partitions, it provides a more generalized, lower-variance estimate of how the algorithm adapts across the full training distribution.
- **Model Parsimony & Interpretability:** Logistic Regression produces clear, monotonic feature coefficients (e.g., Glucose: $+1.18$, BMI: $+0.69$, Pregnancies: $+0.38$), making every prediction auditable and transparent for educational analysis.
- Accordingly, the automated model selection routine in `src/tune_models.py` chose Logistic Regression as the pipeline saved to `models/best_tuned_pipeline.joblib`.

### 2. Single Test-Split Winner: Random Forest
- When both tuned models were evaluated on the single held-out test split of 154 patients, **Random Forest outperformed Logistic Regression**:
  - Test F1-Score: **0.6078** (Random Forest) vs. **0.5455** (Logistic Regression)
  - Test Accuracy: **0.7403** vs. **0.7078**
  - Sensitivity / Recall: **57.41%** (31 true positive patients detected) vs. **50.00%** (27 detected)
  - False Negatives: **23** missed cases vs. **27** missed cases

### 3. The Core Machine Learning Lesson
This discrepancy illustrates a fundamental machine learning principle:
> A single test split is subject to sampling variance. An ensemble model like Random Forest can capitalize on specific non-linear patterns within an individual test partition, while a linear model like Logistic Regression may exhibit superior stability averaged across multiple cross-validation folds.

Documenting both perspectives rather than hiding test-set discrepancies demonstrates methodological integrity and real-world data science maturity.

---

## 📁 Project Folder Tree

```text
diabetes-risk-prediction/
│
├── .gitignore                          # Excludes .venv, cache, IDE files, while tracking code & outputs
├── README.md                           # Comprehensive portfolio documentation (this file)
├── requirements.txt                    # Pinned project dependencies for reproduction
├── first_step.py                       # Initial environment sanity check script
├── app.py                              # Interactive Streamlit web application
│
├── data/
│   ├── .gitkeep                        # Ensures data folder structure is preserved in git
│   └── diabetes.csv                    # Pima Indians Diabetes dataset (768 rows, 9 columns)
│
├── models/
│   ├── .gitkeep                        # Preserves models directory structure
│   ├── best_tuned_pipeline.joblib      # Primary tuned model pipeline (CV Winner: Logistic Regression C=10.0)
│   └── diabetes_logistic_regression.joblib  # Baseline trained Logistic Regression pipeline
│
├── notebooks/
│   └── .gitkeep                        # Reserved for interactive Jupyter notebook prototyping
│
├── outputs/
│   ├── 01_outcome_countplot.png         # Target distribution visualization (Class 0 vs Class 1)
│   ├── 02_histograms_glucose_age_bmi.png # Histograms and KDE distributions for core features
│   ├── 03_boxplots_glucose_bmi.png      # Boxplots comparing Glucose & BMI across diagnoses
│   ├── 04_correlation_heatmap.png       # Correlation matrix across all features and target
│   ├── 06_model_comparison.png          # Visual bar chart comparing baseline models
│   ├── cross_validation_results.csv    # 5-Fold Stratified CV scores on training data
│   ├── model_comparison.csv             # Baseline model metrics on 20% test split
│   ├── model_evaluation.txt             # Detailed baseline evaluation report & coefficients
│   └── tuned_model_test_results.csv     # Final evaluation of tuned models on 20% test split
│
└── src/
    ├── .gitkeep                        # Preserves src directory structure
    ├── explore_data.py                 # EDA script: summary stats, missingness, physiological zeros
    ├── visualize_data.py               # Plotting script: generates charts in outputs/
    ├── prepare_data.py                 # Preprocessing pipeline: zero-to-NaN, ColumnTransformer, imputer/scaler
    ├── train_model.py                  # Baseline trainer: fits Logistic Regression, exports evaluation report
    ├── compare_models.py               # Comparative benchmark of 4 baseline algorithms
    └── tune_models.py                  # 5-fold CV hyperparameter search, selects & exports best pipeline
```

---

## 📦 Saved Artifacts & Outputs Guide

### Model Artifacts (`models/`)
- **`best_tuned_pipeline.joblib` (3.7 KB):** The deployment-ready pipeline produced by `src/tune_models.py`. Contains the fitted `ColumnTransformer` (median imputer and standard scaler) coupled with tuned `LogisticRegression(C=10.0)`. Used directly by `app.py`.
- **`diabetes_logistic_regression.joblib` (3.7 KB):** The baseline pipeline produced by `src/train_model.py` with default hyperparameters (`C=1.0`).

### Evaluation & Benchmark Tables (`outputs/`)
- **`cross_validation_results.csv`:** Mean and standard deviation of F1 and ROC-AUC across 5 stratified training folds for Logistic Regression, Random Forest, and SVM.
- **`model_comparison.csv`:** Baseline test-set comparison across Accuracy, Precision, Recall, F1-Score, and ROC-AUC for 4 candidate models.
- **`tuned_model_test_results.csv`:** Unseen test-set evaluation for all three hyperparameter-tuned pipelines, including full confusion matrix tallies (TN, FP, FN, TP).
- **`model_evaluation.txt`:** Plaintext baseline report detailing class distributions, confusion matrix interpretation, classification report, and feature coefficients.

### Visualizations (`outputs/`)
- **`01_outcome_countplot.png`:** Class balance diagram showing 500 healthy and 268 diabetic patients.
- **`02_histograms_glucose_age_bmi.png`:** Three-panel histogram displaying right-skewed Age and unimodal Glucose/BMI.
- **`03_boxplots_glucose_bmi.png`:** Grouped boxplots demonstrating significant distribution shifts in glucose and BMI for positive diagnoses.
- **`04_correlation_heatmap.png`:** Seaborn annotated heatmap showing inter-feature correlations.
- **`06_model_comparison.png`:** Comparative bar chart contrasting F1-scores across baseline algorithms.

---

## 💻 Setup Guide for Windows

Follow these step-by-step instructions in PowerShell or Command Prompt.

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- On Windows, the Python Launcher (`py`) is recommended because it automatically locates modern Python installations even if legacy versions exist in PATH.

### 2. Open Project Folder
Open PowerShell or Command Prompt and navigate to the project directory:
```powershell
cd c:\Users\ASUS\Desktop\Projects\diabetes-risk-prediction
```

### 3. Create Virtual Environment
Create an isolated virtual environment named `.venv`:
```powershell
py -m venv .venv
```
*(If `py` is not installed, use `python -m venv .venv`).*

### 4. Activate the Virtual Environment
Activate the environment so that all subsequent commands use the isolated project dependencies:

- **In PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  > [!TIP]
  > **PowerShell Execution Policy Error?** If you see an error stating that script execution is disabled on your machine, run this one-time command in PowerShell to permit local scripts for your current session:
  > ```powershell
  > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  > ```
  > Then rerun `.\.venv\Scripts\Activate.ps1`.

- **In Command Prompt (`cmd.exe`):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **In Git Bash:**
  ```bash
  source .venv/Scripts/activate
  ```

Once activated, your terminal prompt will display the prefix: `(.venv)`.

### 5. Install Dependencies
Upgrade pip and install all required libraries:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution Guide: Reproducing Analysis & Launching App

Ensure your virtual environment is active (`(.venv)`) before running these commands.

### Step 1: Verify Python Environment
Confirm that basic Python variables and imports execute correctly:
```powershell
python first_step.py
```
*Expected output:* Prints project name and sample patient age.

---

### Step 2: Run Exploratory Data Analysis (EDA)
Inspect dataset dimensions, feature types, missingness, class breakdown, and physiological zero counts:
```powershell
python src/explore_data.py
```

---

### Step 3: Re-Generate Exploratory Visualizations
Create and export all distribution charts, boxplots, and heatmaps to the `outputs/` folder:
```powershell
python src/visualize_data.py
```

---

### Step 4: Run Baseline Data Preparation & Training
Run data preprocessing checks, fit the baseline Logistic Regression pipeline, evaluate on the 20% test split, and generate `outputs/model_evaluation.txt`:
```powershell
python src/prepare_data.py
python src/train_model.py
```

---

### Step 5: Compare Four Baseline Algorithms
Train and benchmark Logistic Regression, Decision Tree, Random Forest, and SVM on the held-out test set:
```powershell
python src/compare_models.py
```
*Outputs generated:* `outputs/model_comparison.csv` and `outputs/06_model_comparison.png`.

---

### Step 6: Execute 5-Fold Cross-Validation & Hyperparameter Tuning
Run GridSearchCV with Stratified 5-Fold CV across hyperparameter grids, evaluate tuned models on the unseen test set, and export the best pipeline to `models/best_tuned_pipeline.joblib`:
```powershell
python src/tune_models.py
```
*Outputs generated:* `outputs/cross_validation_results.csv`, `outputs/tuned_model_test_results.csv`, and `models/best_tuned_pipeline.joblib`.

---

### Step 7: Launch the Interactive Streamlit Web App
Launch the interactive web application in your local browser:
```powershell
streamlit run app.py
```

Streamlit will print local access URLs in your terminal:
```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open `http://localhost:8501` in your web browser. You can now adjust sliders and numeric inputs for all 8 patient features and observe real-time model risk predictions and probability estimates.

To stop the Streamlit server, press `Ctrl + C` in your terminal.

---

## 🧠 Key Machine Learning Lessons & Portfolio Takeaways

1. **Never Impute Before Splitting:**
   Imputing missing values using the full dataset leaks information from the test set into the training process. Fitting `SimpleImputer` solely on training folds guarantees that performance metrics reflect true out-of-sample generalization.

2. **Domain Understanding Dictates Preprocessing:**
   Mechanical preprocessing treats all zeros identically. Domain knowledge revealed that `Pregnancies = 0` is a normal, valid medical value, whereas `Glucose = 0` or `BMI = 0` indicates an unrecorded test that must be treated as missing (`NaN`).

3. **Accuracy is Insufficient for Skewed Classes:**
   In a dataset with 65% non-diabetic records, a naive classifier that predicts "healthy" for every patient achieves 65% accuracy while missing 100% of actual diabetics. Prioritizing **F1-Score** balances precision (avoiding false alarms) and recall (catching true positive patients).

4. **Multi-Fold Stability vs. Single Test Splits:**
   Evaluating on a single test set can be noisy. Using **Stratified 5-Fold Cross-Validation** during hyperparameter search provides an objective estimate of algorithmic stability and protects against overfitting to a particular partition.

5. **Linear Simplicity vs. Ensemble Capacity:**
   While Random Forest achieved higher recall on the single test split, Logistic Regression demonstrated superior average F1 stability across 5 cross-validation folds and offers direct, auditable feature coefficients—a critical consideration in healthcare informatics.

---

## 📜 License & Acknowledgments

- **Dataset Credit:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), courtesy of the UCI Machine Learning Repository.
- **License:** Distributed under the MIT License for educational and portfolio demonstration use.
