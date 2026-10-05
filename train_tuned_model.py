import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("credit_card_fraud_2026.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. TARGET COLUMN
# ==========================================

# Change this only if your target column has another name
target_column = "is_fraud"

X = df.drop(columns=[target_column])
y = df[target_column]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. IDENTIFY COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)


# ==========================================
# 6. BASELINE MODEL
# ==========================================

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)

baseline_model.fit(X_train, y_train)

baseline_prediction = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)

baseline_precision = precision_score(
    y_test,
    baseline_prediction,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_prediction,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_prediction,
    zero_division=0
)


print("\n==========================================")
print("BEFORE HYPERPARAMETER TUNING")
print("==========================================")

print(f"Accuracy  : {baseline_accuracy:.4f}")
print(f"Accuracy %: {baseline_accuracy * 100:.2f}%")
print(f"Precision : {baseline_precision:.4f}")
print(f"Recall    : {baseline_recall:.4f}")
print(f"F1 Score  : {baseline_f1:.4f}")


# ==========================================
# 7. HYPERPARAMETER GRID
# ==========================================

param_grid = {
    "classifier__criterion": [
        "gini",
        "entropy",
        "log_loss"
    ],

    "classifier__max_depth": [
        5,
        10,
        15,
        20,
        None
    ],

    "classifier__min_samples_split": [
        2,
        5,
        10
    ],

    "classifier__min_samples_leaf": [
        1,
        2,
        5
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# ==========================================
# 8. GRID SEARCH
# ==========================================

grid_search = GridSearchCV(
    estimator=baseline_model,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("\n==========================================")
print("STARTING HYPERPARAMETER TUNING")
print("==========================================")

grid_search.fit(X_train, y_train)


# ==========================================
# 9. BEST MODEL
# ==========================================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params__)


# ==========================================
# 10. NEW MODEL EVALUATION
# ==========================================

new_prediction = best_model.predict(X_test)

new_accuracy = accuracy_score(
    y_test,
    new_prediction
)

new_precision = precision_score(
    y_test,
    new_prediction,
    zero_division=0
)

new_recall = recall_score(
    y_test,
    new_prediction,
    zero_division=0
)

new_f1 = f1_score(
    y_test,
    new_prediction,
    zero_division=0
)


print("\n==========================================")
print("AFTER HYPERPARAMETER TUNING")
print("==========================================")

print(f"Accuracy  : {new_accuracy:.4f}")
print(f"Accuracy %: {new_accuracy * 100:.2f}%")
print(f"Precision : {new_precision:.4f}")
print(f"Recall    : {new_recall:.4f}")
print(f"F1 Score  : {new_f1:.4f}")


# ==========================================
# 11. COMPARISON
# ==========================================

print("\n==========================================")
print("BEFORE vs AFTER")
print("==========================================")

print(f"Accuracy : {baseline_accuracy * 100:.2f}% -> {new_accuracy * 100:.2f}%")
print(f"Precision: {baseline_precision:.4f} -> {new_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f} -> {new_recall:.4f}")
print(f"F1 Score : {baseline_f1:.4f} -> {new_f1:.4f}")


accuracy_change = (
    new_accuracy - baseline_accuracy
) * 100

print(f"\nAccuracy change: {accuracy_change:+.2f} percentage points")


# ==========================================
# 12. SAVE MODEL
# ==========================================

# Extract the fitted preprocessor and classifier
fitted_preprocessor = best_model.named_steps["preprocessor"]
fitted_classifier = best_model.named_steps["classifier"]

with open("fraud_detection_model.pkl", "wb") as file:
    pickle.dump(fitted_classifier, file)

with open("preprocessor.pkl", "wb") as file:
    pickle.dump(fitted_preprocessor, file)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print("fraud_detection_model.pkl")
print("preprocessor.pkl")