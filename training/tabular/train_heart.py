import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from preprocessing.tabular.preprocess import build_preprocessor, save_preprocessor


# ================= LOAD DATA =================
old_df = pd.read_csv("datasets/heart/tabular/heart_old.csv")
new_df = pd.read_csv("datasets/heart/tabular/heart_uci.csv")


# ---------------- FIX TARGET COLUMN ----------------

# rename target column if needed
if "num" in new_df.columns:
    new_df = new_df.rename(columns={"num": "target"})

if "target" not in old_df.columns:
    raise ValueError("Old heart dataset must contain 'target' column")


# ---------------- ALIGN SCHEMAS ----------------

common_cols = list(set(old_df.columns).intersection(set(new_df.columns)))

old_df = old_df[common_cols]
new_df = new_df[common_cols]


# ---------------- MERGE DATASETS ----------------

df = pd.concat([old_df, new_df], ignore_index=True)

df = df.drop_duplicates()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------- FIX MIXED DATA TYPES ----------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str)

# ensure target is numeric
df["target"] = pd.to_numeric(df["target"], errors="coerce")
df = df.dropna(subset=["target"])

target = "target"


# ================= PREPROCESS =================
X, y, preprocessor = build_preprocessor(df, target)


# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================= PIPELINE =================
pipeline = Pipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(random_state=42))
])


# ================= GRID SEARCH =================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}


grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)


# ================= TRAIN =================
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)
print("Best CV Macro F1:", grid.best_score_)


# ================= TEST EVALUATION =================
pred = best_model.predict(X_test)

print("\nTest Evaluation (Heart)")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, average="macro"))
print("Recall:", recall_score(y_test, pred, average="macro"))
print("Macro F1:", f1_score(y_test, pred, average="macro"))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# ================= CROSS VALIDATION =================
print("\nCross Validation (Full Dataset)")

print(
    "Mean CV Macro F1:",
    cross_val_score(best_model, X, y, cv=5, scoring="f1_macro").mean()
)

print(
    "Mean CV Accuracy:",
    cross_val_score(best_model, X, y, cv=5, scoring="accuracy").mean()
)


# ================= SAVE MODEL =================
joblib.dump(best_model, "models/tabular/heart/heart_model.pkl")

save_preprocessor(preprocessor, "models/tabular/heart/heart_preprocessor.pkl")

print("\nHeart model saved.")


# ================= FEATURE IMPORTANCE =================
feature_names = best_model.named_steps["prep"].get_feature_names_out()

importances = best_model.named_steps["model"].feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Heart Features:")
print(fi.head(10))

fi.to_csv("models/tabular/heart/heart_feature_importance.csv", index=False)