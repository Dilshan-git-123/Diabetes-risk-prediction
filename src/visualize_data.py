"""
Data Visualization Script
-------------------------
This script generates exploratory visualizations for the Diabetes Risk Prediction project:
1. Target class distribution (Count Plot of Outcome)
2. Feature distributions (Histograms for Glucose, Age, and BMI)
3. Comparisons across classes (Boxplots of Glucose and BMI vs Outcome)
4. Feature relationships (Correlation Heatmap)

All plots are automatically saved into the 'outputs/' directory.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to save plots cleanly without popups
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a clean visual theme for all plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11})

# Define paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
data_path = os.path.join(project_root, "data", "diabetes.csv")
output_dir = os.path.join(project_root, "outputs")

# Create outputs/ folder if it doesn't exist yet
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("DIABETES DATASET - EXPLORATORY VISUALIZATIONS")
print("=" * 70)

# Step 1: Load the dataset
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Data loaded successfully with shape: {df.shape}")

# Color palette for classes: Blue for Non-Diabetic (0), Coral-Red for Diabetic (1)
palette = {0: "#4A90E2", 1: "#E74C3C"}

# ----------------------------------------------------------------------
# 1. Count Plot: Outcome Distribution (0 = Non-Diabetic, 1 = Diabetic)
# ----------------------------------------------------------------------
print("\n[1/4] Generating Count Plot for Outcome...")
fig, ax = plt.subplots(figsize=(6, 5))

# Plot the counts of each outcome category
sns.countplot(data=df, x="Outcome", hue="Outcome", palette=palette, ax=ax, legend=False)

# Add custom labels and title for clarity
ax.set_title("Distribution of Outcome (Target Class)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Diagnosis Outcome", fontsize=12)
ax.set_ylabel("Number of Patients", fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(["0 (Non-Diabetic)", "1 (Diabetic)"])

# Annotate each bar with the exact count
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2.0, height),
                ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 4),
                textcoords="offset points")

outcome_plot_path = os.path.join(output_dir, "01_outcome_countplot.png")
plt.tight_layout()
plt.savefig(outcome_plot_path, dpi=300)
plt.close()
print(f"  -> Saved: {outcome_plot_path}")

# ----------------------------------------------------------------------
# 2. Histograms: Glucose, Age, and BMI Distributions
# ----------------------------------------------------------------------
print("\n[2/4] Generating Histograms for Glucose, Age, and BMI...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

features_to_plot = [
    ("Glucose", "#2ECC71", "Plasma Glucose Concentration (mg/dL)"),
    ("Age", "#9B59B6", "Age (years)"),
    ("BMI", "#F39C12", "Body Mass Index (BMI)")
]

for idx, (col, color, xlabel) in enumerate(features_to_plot):
    ax = axes[idx]
    # Plot histogram with Kernel Density Estimate (KDE) curve
    sns.histplot(df[col], kde=True, color=color, ax=ax, bins=25, edgecolor="black", alpha=0.6)
    ax.set_title(f"Distribution of {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)

hist_plot_path = os.path.join(output_dir, "02_histograms_glucose_age_bmi.png")
plt.tight_layout()
plt.savefig(hist_plot_path, dpi=300)
plt.close()
print(f"  -> Saved: {hist_plot_path}")

# ----------------------------------------------------------------------
# 3. Boxplots: Comparing Glucose and BMI across Outcome classes
# ----------------------------------------------------------------------
print("\n[3/4] Generating Boxplots comparing Glucose and BMI by Outcome...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: Glucose across Outcomes
sns.boxplot(
    data=df, 
    x="Outcome", 
    y="Glucose", 
    hue="Outcome", 
    palette=palette, 
    ax=axes[0], 
    legend=False, 
    width=0.4
)
axes[0].set_title("Glucose Level by Outcome", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Outcome", fontsize=11)
axes[0].set_ylabel("Glucose Concentration", fontsize=11)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["0 (Non-Diabetic)", "1 (Diabetic)"])

# Subplot 2: BMI across Outcomes
sns.boxplot(
    data=df, 
    x="Outcome", 
    y="BMI", 
    hue="Outcome", 
    palette=palette, 
    ax=axes[1], 
    legend=False, 
    width=0.4
)
axes[1].set_title("BMI by Outcome", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Outcome", fontsize=11)
axes[1].set_ylabel("Body Mass Index (BMI)", fontsize=11)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["0 (Non-Diabetic)", "1 (Diabetic)"])

box_plot_path = os.path.join(output_dir, "03_boxplots_glucose_bmi.png")
plt.tight_layout()
plt.savefig(box_plot_path, dpi=300)
plt.close()
print(f"  -> Saved: {box_plot_path}")

# ----------------------------------------------------------------------
# 4. Correlation Heatmap: Numerical Relationships
# ----------------------------------------------------------------------
print("\n[4/4] Generating Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate pairwise Pearson correlation coefficients
correlation_matrix = df.corr()

# Draw heatmap with numeric values in each cell
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax
)

ax.set_title("Correlation Heatmap of Diabetes Features", fontsize=14, fontweight="bold", pad=15)

heatmap_path = os.path.join(output_dir, "04_correlation_heatmap.png")
plt.tight_layout()
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"  -> Saved: {heatmap_path}")

print("\n" + "=" * 70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print(f"Check the figures inside the folder: {output_dir}")
print("=" * 70)
