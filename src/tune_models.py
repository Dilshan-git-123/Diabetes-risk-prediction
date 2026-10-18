"""
Cross-Validation and Hyperparameter Tuning Script - Stage 6
-----------------------------------------------------------
This script performs rigorous cross-validation and hyperparameter tuning for
three candidate machine learning models:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

Methodology & Data Leakage Prevention:
- Reuses dataset loading, splitting, and preprocessing from src/prepare_data.py.
- The 20% test set (154 patients) is strictly held out and NEVER touched during tuning.
- Hyperparameter search is conducted exclusively on the 80% training set (614 patients)
  using 5-Fold Stratified Cross-Validation (StratifiedKFold, shuffle=True, random_state=42).
- Each candidate model is evaluated within a scikit-learn Pipeline so that median imputation
  and feature scaling are re-computed separately inside each training fold.
- Primary selection metric: F1-Score (balances precision and recall for diabetes cases).
- Secondary tracking metric: ROC-AUC.
- The best overall model selected by cross-validation is refitted and exported to
  'models/best_tuned_pipeline.joblib'.
- All three tuned models are evaluated once on the unseen held-out test set.
- Results are saved to 'outputs/cross_validation_results.csv' and
  'outputs/tuned_model_test_results.csv'.

Educational Notice:
The Pima Indians Diabetes Database is a historical, population-specific dataset with
a modest sample size (768 patients). Cross-validation provides a more reliable estimate
of algorithmic stability than a single split, but it does NOT establish clinical
generalization to modern, diverse healthcare populations. None of these models are
clinically validated for patient diagnosis.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Ensure 'src' is in Python's import search path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import reusable preprocessing functions from prepare_data.py
try:
    from prepare_data import (
        load_data,
        split_data,
        create_preprocessed_copies,
        build_preprocessor,
    )
except ImportError as err:
    print(f"Error importing from prepare_data.py: {err}")
    sys.exit(1)


def tune_and_evaluate_models():
    """Perform 5-fold CV hyperparameter tuning, evaluate on held-out test set, and save results."""
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    outputs_dir = os.path.join(project_root, "outputs")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 6: CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print("=" * 80)

    # 1. Load data
    print("\n[1] Loading dataset...")
    df = load_data()
    print(f"  Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Stratified train/test split (80/20, random_state=42)
    print("\n[2] Splitting into 80% train (614 patients) and 20% held-out test (154 patients)...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set : {len(X_test)} samples (strictly reserved for final evaluation)")

    # 3. Documented zero-to-NaN handling copies
    print("\n[3] Handling medical zero values in documented working copies...")
    X_train_clean, X_test_clean = create_preprocessed_copies(X_train, X_test)

    # 4. Set up 5-Fold Stratified Cross-Validation
    print("\n[4] Configuring 5-Fold Stratified Cross-Validation on training data only...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("  StratifiedKFold ensures each fold preserves the ~65/35 class balance.")

    # 5. Define candidate models and parameter grids
    print("\n[5] Defining parameter search grids for tuning:")
    models_and_grids = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0]
            },
            "param_explanations": {
                "C": "Inverse regularization strength (smaller = stronger penalty, larger = weaker penalty)"
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5]
            },
            "param_explanations": {
                "n_estimators": "Number of decision trees in the ensemble",
                "max_depth": "Maximum depth/levels of each tree to prevent memorizing noise",
                "min_samples_split": "Minimum samples required to create a new branch"
            }
        },
        "Support Vector Machine": {
            "model": SVC(random_state=42),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto"]
            },
            "param_explanations": {
                "C": "Penalty parameter balancing margin width against misclassifications",
                "kernel": "Type of decision boundary: linear (flat plane) or rbf (non-linear curved)",
                "gamma": "Kernel coefficient defining the reach of individual training points"
            }
        }
    }

    # Data structures to collect results
    cv_summary_list = []
    test_eval_list = []
    fitted_grids = {}

    # 6. Execute GridSearchCV for each model on training data
    print("\n[6] Running GridSearchCV (Training data only, refit on full training set with best F1)...")
    for name, config in models_and_grids.items():
        print(f"\n  --> Tuning {name}...")
        
        # Build pipeline ensuring imputation and scaling happen inside each fold
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", config["model"])
        ])

        # Configure GridSearchCV with multiple metrics, refitting on F1
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config["param_grid"],
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            refit="f1",
            n_jobs=1
        )

        grid.fit(X_train_clean, y_train)
        fitted_grids[name] = grid

        # Extract best cross-validation metrics
        best_idx = grid.best_index_
        mean_cv_f1 = grid.cv_results_["mean_test_f1"][best_idx]
        std_cv_f1 = grid.cv_results_["std_test_f1"][best_idx]
        mean_cv_auc = grid.cv_results_["mean_test_roc_auc"][best_idx]
        std_cv_auc = grid.cv_results_["std_test_roc_auc"][best_idx]

        # Format parameter string without prefix for clean presentation
        clean_params = {k.replace("classifier__", ""): v for k, v in grid.best_params_.items()}
        clean_params_str = str(clean_params)

        print(f"      Best Parameters : {clean_params}")
        print(f"      5-Fold CV F1    : {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})")
        print(f"      5-Fold CV ROC-AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")

        cv_summary_list.append({
            "Model": name,
            "Best_Parameters": clean_params_str,
            "Mean_CV_F1": round(mean_cv_f1, 4),
            "Std_CV_F1": round(std_cv_f1, 4),
            "Mean_CV_ROC_AUC": round(mean_cv_auc, 4),
            "Std_CV_ROC_AUC": round(std_cv_auc, 4),
        })

        # 7. Evaluate each tuned model once on the held-out test set
        best_pipeline = grid.best_estimator_
        y_pred = best_pipeline.predict(X_test_clean)

        # Get probability or decision function for test set ROC-AUC
        if hasattr(best_pipeline, "predict_proba"):
            y_scores = best_pipeline.predict_proba(X_test_clean)[:, 1]
        elif hasattr(best_pipeline, "decision_function"):
            y_scores = best_pipeline.decision_function(X_test_clean)
        else:
            y_scores = y_pred

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_scores)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        test_eval_list.append({
            "Model": name,
            "Best_Parameters": clean_params_str,
            "Accuracy": round(test_acc, 4),
            "Precision": round(test_prec, 4),
            "Recall": round(test_rec, 4),
            "F1-Score": round(test_f1, 4),
            "ROC-AUC": round(test_auc, 4),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    # 8. Create and display summary tables
    cv_df = pd.DataFrame(cv_summary_list)
    test_df = pd.DataFrame(test_eval_list)

    print("\n" + "-" * 80)
    print("5-FOLD CROSS-VALIDATION SUMMARY TABLE (Training Set Only)")
    print("-" * 80)
    print(cv_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("HELD-OUT TEST SET EVALUATION TABLE (Unseen Test Data)")
    print("-" * 80)
    test_display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "TN", "FP", "FN", "TP"]
    print(test_df[test_display_cols].to_string(index=False))

    # 9. Save Cross-Validation Summary CSV
    cv_csv_path = os.path.join(outputs_dir, "cross_validation_results.csv")
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"\n[7] Cross-validation results saved to: {cv_csv_path}")

    # 10. Save Held-Out Test Evaluation CSV
    test_csv_path = os.path.join(outputs_dir, "tuned_model_test_results.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"[8] Tuned model test results saved to: {test_csv_path}")

    # 11. Determine and Save Best Model based on Training Cross-Validation F1-Score
    best_row = cv_df.sort_values(by="Mean_CV_F1", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_pipeline = fitted_grids[best_name].best_estimator_

    model_artifact_path = os.path.join(models_dir, "best_tuned_pipeline.joblib")
    joblib.dump(best_pipeline, model_artifact_path)
    print(f"\n[9] Selected Best Model: '{best_name}' (Highest 5-Fold CV F1: {best_row['Mean_CV_F1']:.4f})")
    print(f"    Saved artifact to: {model_artifact_path}")

    # 12. Verify reload of the best pipeline
    print("\n[10] Verifying saved pipeline can be reloaded...")
    reloaded_pipeline = joblib.load(model_artifact_path)
    sample_preds = reloaded_pipeline.predict(X_test_clean.head(3))
    print(f"    Reload verification successful! Test sample predictions: {sample_preds.tolist()}")

    # 13. Hyperparameter Explanations for Beginners
    print("\n" + "=" * 80)
    print("SELECTED HYPERPARAMETERS EXPLANATION (Beginner Guide)")
    print("=" * 80)
    for name, config in models_and_grids.items():
        best_p = {k.replace("classifier__", ""): v for k, v in fitted_grids[name].best_params_.items()}
        print(f"\n* {name}:")
        print(f"  Chosen settings: {best_p}")
        for param, val in best_p.items():
            desc = config["param_explanations"].get(param, "Parameter")
            print(f"  - '{param} = {val}': {desc}")

    print("\n" + "=" * 80)
    print("STAGE 6 TUNING & EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    tune_and_evaluate_models()
