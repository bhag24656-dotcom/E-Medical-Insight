import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "datasets/heart/mi/mi_complications.csv"

MODEL_PATH = "models/tabular/heart/mi_complication_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Complication columns
complication_cols = [
    "FIBR_JELUD",
    "A_V_BLOK",
    "OTEK_LANC",
    "RAZRIV",
    "DRESSLER",
    "ZSN",
    "REC_IM",
    "P_IM_STEN",
    "LET_IS"
]

# Create target
df["complication"] = df[complication_cols].idxmax(axis=1)

# Drop columns
df = df.drop(columns=complication_cols + ["ID"])

# Features / Target
X = df.drop(columns=["complication"])
y = df["complication"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing
num_cols = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols)
    ]
)

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nModel Training Complete\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)


MI_MODEL_PATH = "models/tabular/heart/mi_complication_model.pkl"

mi_model = joblib.load(MI_MODEL_PATH)


def predict_mi_complication(input_data: dict):
    
    df = pd.DataFrame([input_data])

    prediction = mi_model.predict(df)[0]

    return prediction