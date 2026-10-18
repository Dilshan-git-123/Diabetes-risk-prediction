"""
Train Logistic Regression Model
-------------------------------
This script trains and evaluates a baseline Logistic Regression model for the
Diabetes Risk Prediction project:
1. Reuses the preprocessing pipeline from src/prepare_data.py.
2. Performs a reproducible 80/20 stratified train/test split (random_state=42).
3. Fits the imputer, scaler, and Logistic Regression model on TRAINING data only.
4. Evaluates the model on the unseen held-out test set (154 patients) across:
   - Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report, and ROC-AUC.
5. Explains what each metric means in beginner-friendly terms.
6. Saves the trained pipeline to models/diabetes_logistic_regression.joblib.
7. Saves the evaluation metrics report to outputs/model_evaluation.txt.

Educational Disclaimer:
This model is built using the historical Pima Indians Diabetes educational dataset.
It is created strictly for learning machine learning concepts and is NOT clinically
validated for real-world medical diagnosis or patient risk assessment.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Ensure the 'src' directory is in Python's import search path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import modular preprocessing functions from prepare_data.py
try:
    from prepare_data import (
        load_data,
        split_data,
        create_preprocessed_copies,
        build_preprocessor,
        ZERO_IMPUTE_COLS,
    )
except ImportError as err:
    print(f"Error importing from prepare_data.py: {err}")
    sys.exit(1)


def train_logistic_regression():
    """Train the full preprocessing + Logistic Regression pipeline and evaluate it."""
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    models_dir = os.path.join(project_root, "models")
    outputs_dir = os.path.join(project_root, "outputs")
    
    # Ensure destination directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("=" * 75)
    print("STAGE 4: TRAIN LOGISTIC REGRESSION CLASSIFIER")
    print("=" * 75)
    
    # Step 1: Load Dataset
    print("\n[1] Loading dataset...")
    df = load_data()
    print(f"  Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Stratified Train/Test Split (80/20)
    print("\n[2] Performing stratified train/test split (80% train, 20% test, random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    print(f"  Training samples: {len(X_train)} patients")
    print(f"  Testing samples : {len(X_test)} patients (held-out for evaluation)")
    
    # Step 3: Documented zero-handling copy (treating medical zeros as NaN)
    print("\n[3] Creating documented feature copies with zero-to-NaN handling...")
    print(f"  Columns handled: {', '.join(ZERO_IMPUTE_COLS)}")
    print("  Pregnancies and Outcome are untouched.")
    X_train_clean, X_test_clean = create_preprocessed_copies(X_train, X_test)
    
    # Step 4: Construct Full Scikit-Learn Pipeline
    print("\n[4] Building full scikit-learn Pipeline (Preprocessor + Logistic Regression)...")
    preprocessor = build_preprocessor()
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Step 5: Fit on Training Data Only (Guarantees no data leakage)
    print("\n[5] Fitting pipeline ON TRAINING DATA ONLY...")
    model_pipeline.fit(X_train_clean, y_train)
    print("  Model training completed successfully!")
    
    # Inspect model feature coefficients
    feature_names = preprocessor.get_feature_names_out()
    coef_series = pd.Series(
        model_pipeline.named_steps["classifier"].coef_[0],
        index=feature_names
    ).sort_values(ascending=False)
    intercept = model_pipeline.named_steps["classifier"].intercept_[0]
    
    print("\n  Learned Logistic Regression Coefficients (Feature Impact):")
    for feat, coef in coef_series.items():
        direction = "Positive (+)" if coef > 0 else "Negative (-)"
        print(f"    - {feat:<26}: {coef:>7.4f}  [{direction}]")
    print(f"    - {'Model Intercept':<26}: {intercept:>7.4f}")
    
    # Step 6: Evaluate on Held-Out Test Set
    print("\n[6] Evaluating predictions on held-out test set (154 patients)...")
    y_pred = model_pipeline.predict(X_test_clean)
    y_pred_proba = model_pipeline.predict_proba(X_test_clean)[:, 1]
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["0 (Non-Diabetic)", "1 (Diabetic)"])
    
    # Format Confusion Matrix
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "-" * 75)
    print("TEST SET EVALUATION METRICS")
    print("-" * 75)
    print(f"  * Accuracy : {acc:.4f} ({acc * 100:.2f}% of all predictions are correct)")
    print(f"  * Precision: {prec:.4f} (When model predicts diabetes, it is correct {prec * 100:.2f}% of the time)")
    print(f"  * Recall   : {rec:.4f} (Model successfully catches {rec * 100:.2f}% of actual diabetes cases)")
    print(f"  * F1-Score : {f1:.4f} (Harmonic balance between Precision and Recall)")
    print(f"  * ROC-AUC  : {roc_auc:.4f} (Ability to distinguish between positive and negative patients)")
    
    print("\nConfusion Matrix Breakdown:")
    print(f"  * True Negatives  (TN): {tn:>3} (Correctly classified as Non-Diabetic)")
    print(f"  * False Positives (FP): {fp:>3} (Non-Diabetic wrongly classified as Diabetic)")
    print(f"  * False Negatives (FN): {fn:>3} (Diabetic missed by the model)")
    print(f"  * True Positives  (TP): {tp:>3} (Correctly classified as Diabetic)")
    
    print("\nDetailed Classification Report:")
    print(cr)
    
    # Step 7: Save Trained Model Pipeline
    model_save_path = os.path.join(models_dir, "diabetes_logistic_regression.joblib")
    print(f"\n[7] Saving trained pipeline to: {model_save_path}")
    joblib.dump(model_pipeline, model_save_path)
    print("  Model pipeline artifact saved successfully!")
    
    # Step 8: Save Evaluation Report to File
    eval_report_path = os.path.join(outputs_dir, "model_evaluation.txt")
    print(f"\n[8] Saving evaluation report to: {eval_report_path}")
    
    report_content = f"""===========================================================================
DIABETES RISK PREDICTION - BASELINE MODEL EVALUATION REPORT
Model: Logistic Regression (Scikit-Learn)
Trained on: 80% Stratified Training Set (614 patients)
Evaluated on: 20% Held-out Test Set (154 patients)
Pipeline: ColumnTransformer (Median Imputation + StandardScaler) -> LogisticRegression
===========================================================================

1. SUMMARY METRICS
---------------------------------------------------------------------------
Accuracy : {acc:.4f} ({acc * 100:.2f}%)
Precision: {prec:.4f} ({prec * 100:.2f}%)
Recall   : {rec:.4f} ({rec * 100:.2f}%)
F1-Score : {f1:.4f} ({f1 * 100:.2f}%)
ROC-AUC  : {roc_auc:.4f}

2. CONFUSION MATRIX
---------------------------------------------------------------------------
                      Predicted Non-Diabetic (0)   Predicted Diabetic (1)
Actual Non-Diabetic (0):        {tn:>3} (TN)                 {fp:>3} (FP)
Actual Diabetic (1)    :        {fn:>3} (FN)                 {tp:>3} (TP)

- True Negatives  (TN): {tn} (Correctly identified healthy / non-diabetic patients)
- False Positives (FP): {fp} (Type I error: healthy patient flagged as at-risk)
- False Negatives (FN): {fn} (Type II error: patient with diabetes was missed)
- True Positives  (TP): {tp} (Correctly identified patients with diabetes)

3. CLASSIFICATION REPORT
---------------------------------------------------------------------------
{cr}

4. LEARNED COEFFICIENTS (Feature Importance)
---------------------------------------------------------------------------
Intercept: {intercept:.4f}
{coef_series.to_string()}

Higher positive coefficients indicate features that increase the probability
of a diabetes diagnosis (e.g. Glucose and BMI).

5. EDUCATIONAL AND CLINICAL DISCLAIMER
---------------------------------------------------------------------------
This model is developed for educational and algorithmic demonstration purposes
using the historical Pima Indians Diabetes Database. It is NOT clinically
validated and MUST NOT be used for real-world patient diagnosis or healthcare decisions.
===========================================================================
"""
    with open(eval_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print("  Evaluation report saved successfully!")
    
    # Step 9: Verify Saved Model Reloading
    print("\n[9] Verifying saved pipeline can be reloaded and used...")
    loaded_pipe = joblib.load(model_save_path)
    sample_preds = loaded_pipe.predict(X_test_clean.head(3))
    print(f"  Reload verification successful! Test sample predictions: {sample_preds.tolist()}")
    
    print("\n" + "=" * 75)
    print("STAGE 4 TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    train_logistic_regression()
