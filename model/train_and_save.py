from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from feature_engineering import build_features

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df, FEATURES = build_features(DATA_DIR)

# Forecasting-oriented chronological split.
# Earlier opportunities are used for training, then validation, then the
# latest opportunities are held out as the final test set.
df = df.sort_values("created_date").reset_index(drop=True)
n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

X_train = train_df[FEATURES]
y_train = train_df["target"]
X_val = val_df[FEATURES]
y_val = val_df["target"]
X_test = test_df[FEATURES]
y_test = test_df["target"]

# Explicit feature typing is safer than relying on pandas dtype inference.
# In particular, boolean CRM columns are converted to 0/1 and must not be
# accidentally routed through a numeric imputer with object-valued data.

categorical = [
    c for c in [
        "lead_source",
        "owner_role",
        "owner_department",
        "account_type",
        "industry",
        "billing_state",
        "billing_country",
        "lead_source_from_lead"
    ]
    if c in FEATURES
]

numeric = [
    c for c in [
        "amount",
        "amount_log",
        "created_year",
        "created_month",
        "created_quarter",
        "created_dow",
        "annual_revenue",
        "revenue_log",
        "employee_count",
        "emp_log",
        "account_active",
        "owner_active",
        "owner_tenure_days",
        "lead_age_days",
        "prior_contact_count",
        "prior_primary_contact_count",
        "prior_order_count",
        "prior_order_value",
        "prior_account_opps",
        "prior_account_win_rate",
        "prior_owner_opps",
        "prior_owner_win_rate",
        "prior_source_opps",
        "prior_source_win_rate",
        "amount_revenue_ratio",
        "amount_per_employee"
    ]
    if c in FEATURES
]

# Final defensive validation: no object/string column can enter the numeric
# pipeline. If the source data changes, the error identifies the exact field.
bad_numeric = [
    c for c in numeric
    if not pd.api.types.is_numeric_dtype(X_train[c])
]

if bad_numeric:
    raise TypeError(
        "The following columns are assigned as numeric but contain "
        f"non-numeric data: {bad_numeric}. "
        "Inspect feature_engineering.py or convert them explicitly."
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        )
    ]
)

models = {
    "Logistic Regression": LogisticRegression(
        C=0.1, max_iter=3000, class_weight="balanced", random_state=SEED
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=SEED
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=31, weights="distance"
    ),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1
    ),
    # Sixth model because the assignment text says "6 models" but lists only
    # five explicitly. Gradient Boosting is included as the sixth model.
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.04, max_depth=2,
        min_samples_leaf=10, subsample=0.85, random_state=SEED
    )
}

def make_pipeline(model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

def best_threshold(y_true, probabilities):
    # Select threshold only on validation data.
    thresholds = np.arange(0.20, 0.81, 0.01)
    scores = [
        (accuracy_score(y_true, probabilities >= t), t)
        for t in thresholds
    ]
    return max(scores, key=lambda x: x[0])[1]

results = []
saved_metadata = {
    "features": FEATURES,
    "categorical_features": categorical,
    "numeric_features": numeric,
    "train_rows": len(train_df),
    "validation_rows": len(val_df),
    "test_rows": len(test_df),
    "target_definition": "Closed Won=1, Closed Lost=0",
    "split": "chronological 60/20/20 by opportunity created_date",
    "seed": SEED
}

print("=" * 90)
print("CRM SALES FORECASTING - TRAINING")
print("=" * 90)
print(f"Closed opportunities: {len(df)}")
print(f"Features used: {len(FEATURES)}")
print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
print(f"Train win rate: {y_train.mean():.4f}")
print(f"Validation win rate: {y_val.mean():.4f}")
print(f"Test win rate: {y_test.mean():.4f}")

for name, model in models.items():
    print(f"\n{'=' * 25} {name} {'=' * 25}")

    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)

    val_prob = pipe.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val, val_prob)

    # Retrain using all information available before the final test period.
    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)

    final_pipe = make_pipeline(model)
    final_pipe.fit(X_fit, y_fit)

    test_prob = final_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, test_pred),
        "AUC": roc_auc_score(y_test, test_prob),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1": f1_score(y_test, test_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, test_pred),
        "Decision Threshold": threshold
    }
    results.append(metrics)

    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"AUC       : {metrics['AUC']:.4f}")
    print(f"Precision : {metrics['Precision']:.4f}")
    print(f"Recall    : {metrics['Recall']:.4f}")
    print(f"F1        : {metrics['F1']:.4f}")
    print(f"MCC       : {metrics['MCC']:.4f}")

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(final_pipe, MODEL_DIR / f"{safe_name}.joblib")

    # Save test predictions for Streamlit.
    if name == "Gradient Boosting":
        test_output = test_df[FEATURES + ["target"]].copy()
        test_output["Predicted_Class"] = test_pred
        test_output["Predicted_Label"] = np.where(
            test_pred == 1, "Closed Won", "Closed Lost"
        )
        test_output["Won_Probability"] = test_prob

        test_output.to_csv(ROOT / "test_data.csv", index=False)

        # Save confusion matrix data.
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, test_pred)
        np.savetxt(MODEL_DIR / "gradient_boosting_confusion_matrix.csv",
                   cm, delimiter=",", fmt="%d")

# Save comparison table and metadata.
comparison = pd.DataFrame(results).sort_values(
    ["Accuracy", "MCC", "AUC"], ascending=False
)
comparison.to_csv(ROOT / "model_comparison.csv", index=False)

saved_metadata["models"] = [r["ML Model Name"] for r in results]
saved_metadata["winner_by_accuracy"] = comparison.iloc[0]["ML Model Name"]
saved_metadata["results"] = comparison.to_dict("records")

with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(saved_metadata, f, indent=2, default=str)

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)
print(comparison.to_string(index=False))
print(f"\nWinner by test accuracy: {comparison.iloc[0]['ML Model Name']}")
print(f"Test data saved to: {ROOT / 'test_data.csv'}")
print("Training complete.")
