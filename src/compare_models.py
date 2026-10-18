"""
Model Comparison Script - Stage 5
----------------------------------
This script compares four baseline machine learning algorithms on the
Diabetes Risk Prediction dataset:
1. Logistic Regression (linear boundary baseline)
2. Decision Tree (non-linear, rule-based classifier)
3. Random Forest (ensemble of decision trees with bagging)
4. Support Vector Machine / SVM (maximum margin hyperplane classifier)

Key Methodology:
- Reuses preprocessing logic from src/prepare_data.py.
- Fits all imputation and scaling strictly on the 80% training set (614 patients).
- Evaluates all models on the identical 20% held-out test set (154 patients).
- Compares models across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Exports results to 'outputs/model_comparison.csv'.
- Plots a comparative F1-Score bar chart saved to 'outputs/06_model_comparison.png'.
- Does not overwrite existing model artifacts in 'models/'.

Educational Notice:
These models are trained on the historical Pima Indians Diabetes educational dataset.
Results illustrate machine learning behavior and algorithm trade-offs. None of the
models are clinically validated for diagnosing real patients.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to save plots without popup windows
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Ensure 'src' is in Python's module search path
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


def compare_models():
    """Train and evaluate 4 baseline models, save metrics to CSV, and plot F1 comparison."""
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 75)
    print("STAGE 5: BASELINE MACHINE LEARNING ALGORITHM COMPARISON")
    print("=" * 75)

    # 1. Load data
    print("\n[1] Loading dataset...")
    df = load_data()
    print(f"  Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Stratified train/test split (80/20, random_state=42)
    print("\n[2] Splitting data into stratified 80% train / 20% test...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set : {len(X_test)} samples")

    # 3. Documented zero-to-NaN handling copies
    print("\n[3] Handling medical zero values in documented working copies...")
    X_train_clean, X_test_clean = create_preprocessed_copies(X_train, X_test)

    # 4. Define candidate baseline classifiers
    print("\n[4] Defining baseline classifiers for comparison:")
    candidate_models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42),
    }

    for name in candidate_models:
        print(f"  - {name}")

    # 5. Train and evaluate each model
    print("\n[5] Training models and evaluating on test set...")
    comparison_records = []

    for name, clf in candidate_models.items():
        print(f"  * Training {name}...")
        
        # Build individual pipeline ensuring median imputation and scaling
        # are fitted on training data only
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        
        # Fit on training data
        pipeline.fit(X_train_clean, y_train)

        # Predict on held-out test data
        y_pred = pipeline.predict(X_test_clean)

        # Retrieve probability or decision values for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            y_scores = pipeline.predict_proba(X_test_clean)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_scores = pipeline.decision_function(X_test_clean)
        else:
            y_scores = y_pred

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_scores)

        comparison_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(comparison_records)

    # 6. Display Comparison Table in Console
    print("\n" + "-" * 75)
    print("MODEL COMPARISON RESULTS (Evaluated on Held-Out Test Set)")
    print("-" * 75)
    print(results_df.to_string(index=False))

    # 7. Save Comparison Table to CSV
    csv_path = os.path.join(outputs_dir, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[6] Comparison table saved to: {csv_path}")

    # 8. Plot F1-Score Bar Chart
    print("\n[7] Generating F1-Score comparison bar chart...")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_theme(style="whitegrid")

    palette = ["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"]
    bars = sns.barplot(
        data=results_df,
        x="Model",
        y="F1-Score",
        palette=palette,
        hue="Model",
        legend=False,
        ax=ax,
        edgecolor="black",
        alpha=0.85
    )

    ax.set_title("Baseline Model Comparison: F1-Score (Diabetes Detection)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Machine Learning Algorithm", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1-Score (Test Set)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.8)

    # Annotate bar heights with numerical scores
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.4f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points"
        )

    chart_path = os.path.join(outputs_dir, "06_model_comparison.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"  -> Chart saved to: {chart_path}")

    print("\n" + "=" * 75)
    print("STAGE 5 COMPARISON COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    compare_models()
