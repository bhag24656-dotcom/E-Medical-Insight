import joblib
import pandas as pd


# ================= LOAD MODELS =================

def load_model(disease):

    if disease == "heart":
        return joblib.load("models/tabular/heart/heart_model.pkl")

    elif disease == "ckd":
        return joblib.load("models/tabular/ckd/ckd_model.pkl")

    else:
        raise ValueError("Invalid disease type")


# Load MI complication model
mi_model = joblib.load("models/tabular/heart/mi_complication_model.pkl")


# ================= LABEL MAPPINGS =================

HEART_LABELS = {
    0: "Healthy",
    1: "Mild CAD",
    2: "Moderate CAD",
    3: "Severe CAD",
    4: "Critical CAD"
}

CKD_LABELS = {
    1: "Stage 1 – Normal kidney function",
    2: "Stage 2 – Mild kidney damage",
    3: "Stage 3 – Moderate kidney disease",
    4: "Stage 4 – Severe kidney disease",
    5: "Stage 5 – Kidney failure"
}


# MI COMPLICATION LABELS (Human readable)

MI_COMPLICATION_LABELS = {

    "FIBR_JELUD": "Ventricular Fibrillation (Severe heart rhythm disturbance)",

    "A_V_BLOK": "Atrioventricular Block (Electrical signal blockage in heart)",

    "OTEK_LANC": "Pulmonary Edema (Fluid accumulation in lungs)",

    "RAZRIV": "Heart Wall Rupture (Tear in heart muscle)",

    "DRESSLER": "Dressler Syndrome (Post-heart-attack inflammation)",

    "ZSN": "Congestive Heart Failure",

    "REC_IM": "Recurrent Myocardial Infarction (Second heart attack)",

    "P_IM_STEN": "Post-Infarction Angina (Chest pain after heart attack)",

    "LET_IS": "Lethal Outcome / Fatal Complication"
}


# ================= MAIN PREDICTION =================

def predict(disease, input_data: dict):

    model = load_model(disease)

    expected_cols = model.named_steps["prep"].feature_names_in_

    full_input = {}

    for col in expected_cols:
        if col in input_data:
            full_input[col] = input_data[col]
        else:
            full_input[col] = 0

    df = pd.DataFrame([full_input])

    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]
    confidence = float(max(probabilities))

    if disease == "heart":
        label = HEART_LABELS.get(int(prediction), str(prediction))

    elif disease == "ckd":
        label = CKD_LABELS.get(int(prediction), str(prediction))

    else:
        label = str(prediction)

    return {
        "class": int(prediction),
        "label": label,
        "confidence": confidence
    }


# ================= MI COMPLICATION PREDICTION =================

def predict_mi_complication(input_data: dict):

    expected_cols = mi_model.named_steps["preprocessor"].feature_names_in_

    full_input = {}

    for col in expected_cols:
        if col in input_data:
            full_input[col] = input_data[col]
        else:
            full_input[col] = 0

    df = pd.DataFrame([full_input])

    prediction = mi_model.predict(df)[0]

    # Convert code → readable label
    readable = MI_COMPLICATION_LABELS.get(prediction, prediction)

    return readable