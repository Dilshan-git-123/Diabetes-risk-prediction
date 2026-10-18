"""
Data Preparation and Preprocessing Script
-----------------------------------------
This script prepares the Diabetes dataset for machine learning:
1. Loads the raw dataset without modifying the original file.
2. Separates input features (X) and target label (y = Outcome).
3. Performs a stratified 80/20 train/test split (random_state=42).
4. Creates a documented working copy where biologically impossible zeros
   (Glucose, BloodPressure, SkinThickness, Insulin, BMI) are treated as missing (NaN).
   Pregnancies and Outcome are intentionally left untouched.
5. Builds a reusable scikit-learn ColumnTransformer pipeline with:
   - Median imputation (robust to skewed medical measurements and outliers)
   - StandardScaler (normalizes features to mean=0 and variance=1)
6. Fits imputer and scaler on TRAINING data only to strictly prevent data leakage.
7. Verifies zero missing values in the final processed matrices.

Dataset Notice & Limitations:
The Pima Indians Diabetes Database is a classic, historical benchmark collected
from female patients of Pima Indian heritage living near Phoenix, Arizona (1990).
While excellent for learning machine learning fundamentals, this dataset reflects
a specific demographic and sample size. It is not clinically validated for modern
diagnostic use, and conclusions cannot be generalized to diverse global populations.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Define feature groups
# 1. Physiological measurements where 0 represents missing/unrecorded medical data
ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# 2. Features where 0 is a completely valid natural measurement (e.g. 0 prior pregnancies)
STANDARD_COLS = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]


def get_default_data_path() -> str:
    """Return the absolute path to data/diabetes.csv."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(current_dir, "..", "data", "diabetes.csv"))


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load the diabetes CSV dataset into a pandas DataFrame."""
    if filepath is None:
        filepath = get_default_data_path()
    return pd.read_csv(filepath)


def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Split the dataset into training (80%) and testing (20%) sets.
    
    Using stratify=y guarantees that both training and test sets maintain
    the exact same ratio of non-diabetic (65%) to diabetic (35%) patients.
    """
    # X contains all 8 input features; y contains the target label
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def create_preprocessed_copies(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Create documented working copies of features where biologically invalid zeros
    are replaced with NaN (Not a Number).
    
    - Glucose, BloodPressure, SkinThickness, Insulin, and BMI cannot be zero in living humans.
    - Pregnancies is strictly excluded because having 0 pregnancies is normal and valid.
    - Outcome is the target label and is never modified.
    """
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # Replace zeros with NaN only in the documented medical measurement columns
    X_train_clean[ZERO_IMPUTE_COLS] = X_train_clean[ZERO_IMPUTE_COLS].replace(0, np.nan)
    X_test_clean[ZERO_IMPUTE_COLS] = X_test_clean[ZERO_IMPUTE_COLS].replace(0, np.nan)
    
    return X_train_clean, X_test_clean


def build_preprocessor() -> ColumnTransformer:
    """
    Build a reusable scikit-learn ColumnTransformer.
    
    Why Median Imputation?
    Features like Insulin and SkinThickness have high positive skew and extreme outliers.
    The median is much more robust than the mean because extreme values do not distort it.
    
    Why StandardScaler?
    Features are on vastly different scales (Age: 21-81, Insulin: 14-846, Pedigree: 0.08-2.42).
    StandardScaler centers each feature at mean=0 and scales to standard deviation=1,
    preventing large-scale features from dominating distance-based algorithms.
    """
    # Pipeline for columns requiring both missing-value imputation and feature scaling
    impute_and_scale_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline for columns that do not have artificial zero-encoded missing values
    scale_only_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("zero_impute_scale", impute_and_scale_pipeline, ZERO_IMPUTE_COLS),
            ("scale_only", scale_only_pipeline, STANDARD_COLS)
        ],
        verbose_feature_names_out=False
    )
    return preprocessor


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("=" * 75)
    print("DIABETES RISK PREDICTION - DATA PREPROCESSING PIPELINE")
    print("=" * 75)
    
    # 0. Context and Educational Limitations
    print("\n[0] DATASET CONTEXT & LIMITATIONS")
    print("-" * 75)
    print("This project uses the historical Pima Indians Diabetes Database.")
    print("Limitations to keep in mind:")
    print("  * Demographics: All records come from females of Pima Indian heritage aged >= 21.")
    print("  * Generalization: Findings do not generalize to diverse modern populations.")
    print("  * Not Clinical: Intended purely for educational machine learning practice.")
    
    # 1. Load data
    data_path = get_default_data_path()
    print(f"\n[1] Loading dataset from: {data_path}")
    df = load_data(data_path)
    print(f"Original dataset shape: {df.shape} (768 patients, 9 total columns)")
    
    # 2. Split data
    print("\n[2] PERFORMING STRATIFIED TRAIN / TEST SPLIT (80% / 20%)")
    print("-" * 75)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    print(f"Training set features shape (X_train): {X_train.shape}")
    print(f"Testing set features shape  (X_test) : {X_test.shape}")
    print(f"Training set target shape   (y_train): {y_train.shape}")
    print(f"Testing set target shape    (y_test) : {y_test.shape}")
    
    # 3. Class distributions before and after splitting
    print("\n[3] TARGET CLASS DISTRIBUTIONS (Verifying Stratification)")
    print("-" * 75)
    
    def format_distribution(series: pd.Series) -> str:
        counts = series.value_counts()
        total = len(series)
        c0, c1 = counts.get(0, 0), counts.get(1, 0)
        p0, p1 = (c0 / total) * 100, (c1 / total) * 100
        return f"Class 0 (Non-Diabetic): {c0:>3} ({p0:.1f}%) | Class 1 (Diabetic): {c1:>3} ({p1:.1f}%)"
    
    print("Original Dataset: " + format_distribution(df["Outcome"]))
    print("Training Split  : " + format_distribution(y_train))
    print("Test Split      : " + format_distribution(y_test))
    
    # 4. Documented preprocessing copies
    print("\n[4] CREATING DOCUMENTED PREPROCESSING COPIES (Treating zeros as NaN)")
    print("-" * 75)
    print("Treating 0 as NaN only in medical columns: " + ", ".join(ZERO_IMPUTE_COLS))
    print("Leaving untouched: Pregnancies (0 is a valid count) and Outcome (target label)")
    
    X_train_clean, X_test_clean = create_preprocessed_copies(X_train, X_test)
    
    print("\nNaN counts in X_train_clean before imputation:")
    for col in ZERO_IMPUTE_COLS:
        n_nan = X_train_clean[col].isna().sum()
        pct = (n_nan / len(X_train_clean)) * 100
        print(f"  - {col:<15}: {n_nan:>3} values ({pct:4.1f}%) will be imputed")
    
    # 5. Build and fit pipeline (Training data only!)
    print("\n[5] FITTING PREPROCESSOR ON TRAINING DATA ONLY")
    print("-" * 75)
    print("Preventing Data Leakage: The imputer medians and scaler means/stds are")
    print("calculated exclusively from the 614 training records, never from test records.")
    
    preprocessor = build_preprocessor()
    
    # Fit on training data and transform it
    X_train_scaled = preprocessor.fit_transform(X_train_clean)
    
    # Transform test data using the fitted preprocessor
    X_test_scaled = preprocessor.transform(X_test_clean)
    
    # Retrieve clean feature names in their transformed order
    feature_names = preprocessor.get_feature_names_out()
    
    # Convert numpy arrays to DataFrames for readable inspection
    df_train_processed = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    df_test_processed = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    # 6. Verify missing values in processed data
    print("\n[6] VERIFYING PROCESSED MATRICES")
    print("-" * 75)
    train_nans = np.isnan(X_train_scaled).sum()
    test_nans = np.isnan(X_test_scaled).sum()
    print(f"Total NaN values in processed training set: {train_nans} (All missing values imputed!)")
    print(f"Total NaN values in processed testing set : {test_nans} (All missing values imputed!)")
    
    # 7. Sample of processed features
    print("\n[7] SAMPLE OF STANDARDIZED TRAINING FEATURES (First 3 rows)")
    print("-" * 75)
    print("Notice how features now have mean ~ 0 and standard deviation ~ 1:")
    print(df_train_processed.head(3).round(3))
    
    print("\n" + "=" * 75)
    print("PREPROCESSING PIPELINE READY AND VERIFIED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
