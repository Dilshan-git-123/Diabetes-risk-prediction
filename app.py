"""
Diabetes Risk Prediction - Interactive Educational Web Application
------------------------------------------------------------------
This Streamlit application serves as an interactive educational tool
demonstrating the trained machine learning pipeline from Stage 6.

Features:
- Loads the pre-trained pipeline from models/best_tuned_pipeline.joblib.
- Accepts patient measurements for all 8 features in exact expected order.
- Applies consistent zero-to-NaN handling for physiological measurements.
- Outputs the predicted class (Class 0 or Class 1) and estimated probability.
- Clearly states dataset limitations and maintains strict educational disclaimers.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Diabetes Risk Prediction (Educational)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, clean, accessible styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #92400E;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .result-card-0 {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-card-1 {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Expected feature names in strict order
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Columns where 0 indicates missing/unrecorded medical measurements
ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@st.cache_resource
def load_trained_pipeline():
    """Load the pre-trained pipeline from disk without retraining."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_tuned_pipeline.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model artifact not found at: {model_path}. Please run src/tune_models.py first.")
        st.stop()
    return joblib.load(model_path)


def main():
    # Header
    st.markdown('<div class="main-header">🩺 Diabetes Risk Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An Educational Machine Learning Demonstration using Scikit-Learn and Streamlit</div>', unsafe_allow_html=True)

    # Prominent Educational Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>CRITICAL EDUCATIONAL & MEDICAL DISCLAIMER:</strong><br>
        This web application is developed solely for <strong>educational and algorithmic demonstration purposes</strong>.
        It is <strong>NOT clinically validated</strong>, has not received regulatory approval, and <strong>must never be used to diagnose,
        treat, manage, or evaluate real human patients</strong>. The underlying dataset reflects a historical, demographically specific
        sample that does not generalize to modern diverse clinical populations. Always consult qualified healthcare professionals
        for medical advice or screening.
    </div>
    """, unsafe_allow_html=True)

    # Load pre-trained model pipeline
    try:
        pipeline = load_trained_pipeline()
    except Exception as e:
        st.error(f"Failed to load machine learning pipeline: {e}")
        return

    # Sidebar: Project context and feature guide
    with st.sidebar:
        st.header("📖 About This Project")
        st.write("""
        This interactive tool demonstrates a **trained Logistic Regression pipeline** with built-in:
        - **Median Imputation** (fills unrecorded physiological measurements).
        - **StandardScaler** (centers and scales all inputs to unit variance).
        - **Regularized Classification** (tuned with 5-fold Stratified Cross-Validation).
        """)
        
        st.divider()
        st.subheader("📚 Dataset Limitations")
        st.write("""
        - **Source:** Historical Pima Indians Diabetes Database (1990).
        - **Cohort:** 768 adult female patients of Pima Indian heritage.
        - **Generalization:** Findings do **not** generalize across genders, ethnicities, or modern medical demographics.
        """)
        
        st.divider()
        st.subheader("⚙️ Zero-Value Handling")
        st.write("""
        In this dataset, literal zeros in *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin*, or *BMI* represent
        **missing tests**, not true zeros. The pipeline automatically imputes them with the training set median.
        """)

    # Main Input Form
    st.subheader("📋 Enter Patient Clinical Parameters")
    st.caption("Provide numerical measurements for the eight features below:")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "1. Pregnancies (Count)",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Number of times pregnant. 0 is a completely natural and valid medical value."
        )

        glucose = st.number_input(
            "2. Plasma Glucose Concentration (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            step=1.0,
            help="2-hour oral glucose tolerance test level in mg/dL. Normal fasting: 70–99; 2-hr post-meal: < 140. (Enter 0 if unrecorded)."
        )

        blood_pressure = st.number_input(
            "3. Diastolic Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=180.0,
            value=72.0,
            step=1.0,
            help="Diastolic blood pressure in mm Hg. Normal resting range: 60–80 mm Hg. (Enter 0 if unrecorded)."
        )

        skin_thickness = st.number_input(
            "4. Triceps Skinfold Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=23.0,
            step=1.0,
            help="Triceps skin fold thickness in millimeters, used as an indicator of body fat. (Enter 0 if unrecorded)."
        )

    with col2:
        insulin = st.number_input(
            "5. Two-Hour Serum Insulin (mu U/ml)",
            min_value=0.0,
            max_value=900.0,
            value=79.0,
            step=1.0,
            help="2-hour post-glucose serum insulin in mu U/ml. Normal fasting: 16–166 mu U/ml. (Enter 0 if unrecorded)."
        )

        bmi = st.number_input(
            "6. Body Mass Index - BMI (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=32.0,
            step=0.1,
            help="Weight in kg / (height in meters)². Normal: 18.5–24.9; Overweight: 25.0–29.9; Obese: ≥ 30.0. (Enter 0 if unrecorded)."
        )

        dpf = st.number_input(
            "7. Diabetes Pedigree Function",
            min_value=0.05,
            max_value=2.50,
            value=0.47,
            step=0.01,
            help="A genetic score synthesizing diabetes family history and pedigree relationships (higher values indicate stronger family history)."
        )

        age = st.number_input(
            "8. Age (Years)",
            min_value=21,
            max_value=100,
            value=33,
            step=1,
            help="Patient age in years (all individuals in the training cohort were 21 or older)."
        )

    # Expandable detail on zero handling
    with st.expander("🔍 Click to inspect how zeros and input boundaries are processed"):
        st.write("""
        - If `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, or `BMI` is entered as `0`, the application replaces it with `NaN`.
        - The pipeline's internal `SimpleImputer` replaces `NaN` with the median learned during model training.
        - `Pregnancies = 0` is left as `0` because zero prior pregnancies is completely valid.
        """)

    # Prediction Action
    st.write("")
    predict_clicked = st.button("🔮 Compute Model Prediction", type="primary", use_container_width=True)

    if predict_clicked:
        # Assemble DataFrame in exact expected order
        raw_input_data = pd.DataFrame([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]], columns=FEATURE_NAMES)

        # Apply zero-to-NaN handling for the 5 medical measurement columns
        clean_input_data = raw_input_data.copy()
        clean_input_data[ZERO_IMPUTE_COLS] = clean_input_data[ZERO_IMPUTE_COLS].replace(0, np.nan)

        # Generate model prediction and probability estimates
        try:
            predicted_class = int(pipeline.predict(clean_input_data)[0])
            
            # Predict probabilities if supported by the classifier
            if hasattr(pipeline, "predict_proba"):
                predicted_probas = pipeline.predict_proba(clean_input_data)[0]
                prob_class_0 = float(predicted_probas[0])
                prob_class_1 = float(predicted_probas[1])
            else:
                prob_class_0 = None
                prob_class_1 = None

            # Render Results Card (Notice: strictly using neutral, non-diagnostic wording)
            st.divider()
            st.subheader("📊 Model Output & Statistical Estimates")

            if predicted_class == 1:
                st.markdown(f"""
                <div class="result-card-1">
                    <h3 style="color: #991B1B; margin-top:0;">Model Classification: Class 1</h3>
                    <p style="font-size: 1.05rem; color: #7F1D1D;">
                        The model classified this profile as <strong>Class 1 (Elevated Risk Pattern)</strong> based on historical training patterns.
                    </p>
                    <p style="font-size: 0.95rem; color: #991B1B;">
                        <strong>Estimated Model Probability for Class 1:</strong> {prob_class_1 * 100:.2f}%<br>
                        <em>(This is a statistical model estimate from a Logistic Regression formula, NOT a clinical diagnosis or medical probability.)</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-0">
                    <h3 style="color: #065F46; margin-top:0;">Model Classification: Class 0</h3>
                    <p style="font-size: 1.05rem; color: #047857;">
                        The model classified this profile as <strong>Class 0 (Lower Risk Pattern)</strong> based on historical training patterns.
                    </p>
                    <p style="font-size: 0.95rem; color: #065F46;">
                        <strong>Estimated Model Probability for Class 1:</strong> {prob_class_1 * 100:.2f}% (Class 0: {prob_class_0 * 100:.2f}%)<br>
                        <em>(This is a statistical model estimate from a Logistic Regression formula, NOT a clinical diagnosis or medical probability.)</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Metric visual breakdown
            if prob_class_1 is not None:
                st.write("")
                st.markdown("**Probability Gauge for Class 1 (Model Output):**")
                st.progress(prob_class_1)
                
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Predicted Label", f"Class {predicted_class}")
                mcol2.metric("Class 1 Estimate", f"{prob_class_1 * 100:.1f}%")
                mcol3.metric("Class 0 Estimate", f"{prob_class_0 * 100:.1f}%")

            # Display the exact input feature payload
            with st.expander("📋 View Submitted Feature Values"):
                st.dataframe(raw_input_data, use_container_width=True)

        except Exception as err:
            st.error(f"Error during model inference: {err}")


if __name__ == "__main__":
    main()
