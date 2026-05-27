import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from inference.tabular.predict import predict, predict_mi_complication
from utils.nlp import parse_medical_report


# ---------------- Session State ----------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "patient" not in st.session_state:
    st.session_state.patient = {}
if "disease" not in st.session_state:
    st.session_state.disease = None


# ---------------- Analysis Helpers ----------------

def analyze_heart_inputs(data):

    issues = []

    if data.get("trestbps", 0) > 130:
        issues.append(("High Blood Pressure", "Reduce salt intake, exercise regularly."))

    if data.get("chol", 0) > 200:
        issues.append(("High Cholesterol", "Low fat diet and regular physical activity."))

    if data.get("oldpeak", 0) > 1:
        issues.append(("ST Depression", "Consult cardiologist for ECG evaluation."))

    return issues


def analyze_ckd_inputs(data):

    issues = []

    if data.get("SerumCreatinine", 0) > 1.3:
        issues.append(("High Creatinine", "Increase hydration and consult nephrologist."))

    if data.get("GFR", 999) < 60:
        issues.append(("Low GFR", "Strict BP and glucose control required."))

    if data.get("HemoglobinLevels", 99) < 12:
        issues.append(("Low Hemoglobin", "Iron supplements and anemia management."))

    return issues


st.title("Unified Disease Prediction System")


# =====================================================
# STEP 1 — Patient Details
# =====================================================

if st.session_state.step == 1:

    st.header("Step 1 — Patient Details")

    name = st.text_input("Patient Name")
    age = st.number_input("Age", 0, 120)
    gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Next"):

        st.session_state.patient = {
            "name": name,
            "age": age,
            "gender": gender
        }

        st.session_state.step = 2
        st.rerun()


# =====================================================
# STEP 2 — Disease Selection
# =====================================================

elif st.session_state.step == 2:

    st.header("Step 2 — Select Disease")

    disease = st.radio("Prediction Type", ["Heart", "CKD"])

    if st.button("Next"):

        st.session_state.disease = disease.lower()
        st.session_state.step = 3
        st.rerun()


# =====================================================
# STEP 3 — Upload + Auto Predict
# =====================================================

elif st.session_state.step == 3:

    st.header("Step 3 — Upload Medical Report (PDF)")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:

        # ================= EXTRACT DATA =================

        doc = parse_medical_report(uploaded)

        defaults = {

            # heart features
            "age": st.session_state.patient["age"],
            "trestbps": 120,
            "chol": 180,
            "oldpeak": 0,
            "cp": 0,
            "sex": 1 if st.session_state.patient["gender"] == "Male" else 0,
            "exang": 0,
            "ca": 0,

            # CKD features
            "SerumCreatinine": 1,
            "GFR": 90,
            "HemoglobinLevels": 13
        }

        final_input = {**defaults, **doc}

        st.success("Medical report processed successfully.")


        # ================= PATIENT DETAILS =================

        st.subheader("Patient Summary")

        st.write("Name:", st.session_state.patient["name"])
        st.write("Age:", st.session_state.patient["age"])
        st.write("Gender:", st.session_state.patient["gender"])


        # ================= EXTRACTED PARAMETERS =================

        st.subheader("Extracted Medical Parameters")

        if len(doc) == 0:
            st.warning("No parameters detected from the PDF.")

        st.json(doc)


        # ================= PARAMETERS USED BY MODEL =================

        st.subheader("Parameters Used for Prediction")

        df_features = pd.DataFrame([final_input])

        st.dataframe(df_features)


        # ================= PREDICTION =================

        result = predict(st.session_state.disease, final_input)

        label = result["label"]
        confidence = result["confidence"]
        pred_class = result["class"]

        st.subheader("Prediction Result")

        st.write("Predicted Class:", pred_class)
        st.info(f"Model Confidence: {round(confidence*100,2)}%")

        chart_data = pd.DataFrame({
            "Metric": ["Confidence"],
            "Value": [confidence]
        })

        st.bar_chart(chart_data.set_index("Metric"))


        # ================= FINAL SEVERITY OUTPUT =================

        st.markdown("---")

        st.markdown(
            f"""
            ## SEVERITY LEVEL: **{label}**
            """
        )


        # ================= HEART ATTACK COMPLICATION =================

        if st.session_state.disease == "heart":

            complication = predict_mi_complication(final_input)

            st.markdown("---")

            st.subheader("Post-Heart-Attack Complication")

            st.error(f"Predicted Complication: {complication}")


        # ================= PRECAUTIONS =================

        if st.session_state.disease == "heart":
            issues = analyze_heart_inputs(final_input)
        else:
            issues = analyze_ckd_inputs(final_input)

        if issues:

            st.subheader("Precautions")

            for issue in issues:
                st.warning(issue[0] + " → " + issue[1])

        else:

            st.success("All parameters within normal range.")