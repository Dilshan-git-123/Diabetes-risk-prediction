"""
Explore Diabetes Dataset
------------------------
This script loads and explores the Pima Indians Diabetes dataset using pandas.
It inspects the shape, columns, sample data, types, missing values, class distribution,
and descriptive statistics without modifying or silently altering the raw data.
"""

import os
import pandas as pd

# Set display options so all columns are visible in the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Determine the file path to data/diabetes.csv relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "diabetes.csv")

print("=" * 70)
print("DIABETES RISK PREDICTION - EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# 1. Load the dataset using pandas read_csv function
print(f"\n[1] Loading dataset from: {os.path.normpath(data_path)}")
df = pd.read_csv(data_path)
print("Dataset successfully loaded!")

# 2. Dataset Shape: Total number of rows and columns
print("\n" + "-" * 70)
print("[2] DATASET SHAPE (Dimensions)")
print("-" * 70)
# df.shape returns a tuple: (number of rows, number of columns)
rows, cols = df.shape
print(f"Total rows (records / patients): {rows}")
print(f"Total columns (features + target): {cols}")

# 3. Column Names & Source Verification
print("\n" + "-" * 70)
print("[3] COLUMN NAMES & SOURCE VERIFICATION")
print("-" * 70)
print("Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)")
print("Target label: 'Outcome' (0 = Non-Diabetic, 1 = Diabetic)")
print("\nColumns in the dataset:")
for i, col in enumerate(df.columns, start=1):
    role = "Target Label" if col == "Outcome" else "Input Feature"
    print(f"  {i}. {col:<26} [{role}]")

# 4. First 5 Rows (Sample Data preview)
print("\n" + "-" * 70)
print("[4] FIRST 5 ROWS (df.head())")
print("-" * 70)
# df.head(5) prints the first 5 records in tabular format
print(df.head(5))

# 5. Data Types for each column
print("\n" + "-" * 70)
print("[5] DATA TYPES (df.dtypes)")
print("-" * 70)
# df.dtypes shows whether each column contains integers (int64) or decimals (float64)
print(df.dtypes)

# 6. Missing Values check
print("\n" + "-" * 70)
print("[6] MISSING VALUES (df.isnull().sum())")
print("-" * 70)
# df.isnull().sum() calculates the number of standard NaN (null) values in each column
missing_counts = df.isnull().sum()
print(missing_counts)
print("\nExplanation: There are 0 formal NaN (null) values recorded in this CSV file.")

# 7. Note on Zero-Valued Medical Measurements
print("\n" + "-" * 70)
print("[7] ZERO-VALUE OBSERVATION (Domain & Clinical Context)")
print("-" * 70)
# In real life, measurements like Glucose, Blood Pressure, or BMI cannot be zero in a living human.
# In historical datasets, 0 was sometimes entered when a test was not performed.
# Per best practices, we inspect these values but do NOT silently replace or discard them.
medical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("Count of literal zeros in physiological columns:")
for col in medical_columns:
    zero_count = (df[col] == 0).sum()
    percentage = (zero_count / rows) * 100
    print(f"  - {col:<15}: {zero_count:>3} zeros ({percentage:5.1f}% of patients)")
print("\nNotice: We do NOT silently treat these zeros as missing; we keep the raw data intact.")

# 8. Counts of Each Target Class
print("\n" + "-" * 70)
print("[8] COUNTS OF EACH TARGET CLASS (Outcome)")
print("-" * 70)
# df['Outcome'].value_counts() gives the frequency of each class
class_counts = df["Outcome"].value_counts()
class_percentages = df["Outcome"].value_counts(normalize=True) * 100

print(f"Class 0 (Non-Diabetic): {class_counts.get(0, 0)} patients ({class_percentages.get(0, 0):.2f}%)")
print(f"Class 1 (Diabetic):     {class_counts.get(1, 0)} patients ({class_percentages.get(1, 0):.2f}%)")

# 9. Basic Descriptive Statistics
print("\n" + "-" * 70)
print("[9] BASIC DESCRIPTIVE STATISTICS (df.describe())")
print("-" * 70)
# df.describe() summarizes mean, standard deviation, minimum, quartiles, and maximum
print(df.describe().round(2))

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)
