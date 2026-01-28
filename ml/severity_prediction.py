"""
Train and evaluate KNN and Random Forest classifiers
using YOLO-derived acne features (density, confidence, detections).

This script follows the methodology described in paper
Maitimua, G. F., Gunawan, P. H., & Ilyas, M. (2025).
*Classification of Acne Severity Using K-Nearest Neighbor (KNN) and Random Forest Methods*.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

FEATURE_COLS = ["density", "avg_conf", "n_detections"]
LABEL_COL = "severity"
CLASS_NAMES = ["level0", "level1", "level2", "level3"]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)


def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def train_knn(X_train, y_train, k=3):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def predict_knn(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    return model.predict(X_test_scaled)


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {model_name} ===")
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix:\n", cm)
    print("Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return acc


def compare_models(acc_knn, acc_rf):
    print("\n=== Comparison (short) ===")
    if acc_rf > acc_knn:
        print(f"Random Forest performed better than KNN by {acc_rf - acc_knn:.3f} accuracy.")
    elif acc_knn > acc_rf:
        print(f"KNN performed better than Random Forest by {acc_knn - acc_rf:.3f} accuracy.")
    else:
        print("KNN and Random Forest achieved the same accuracy.")


def main():
    csv_path = "../data/features/features2.csv"

    X, y = load_dataset(csv_path)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # KNN
    knn_model, knn_scaler = train_knn(X_train_sm, y_train_sm, k=3)
    y_pred_knn = predict_knn(knn_model, knn_scaler, X_test)
    acc_knn = evaluate(y_test, y_pred_knn, model_name="KNN (k=3)")

    # Random Forest
    rf_model = train_random_forest(X_train_sm, y_train_sm, n_estimators=100, max_depth=10)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = evaluate(y_test, y_pred_rf, model_name="Random Forest (100 trees, depth 10)")

    compare_models(acc_knn, acc_rf)


if __name__ == "__main__":
    main()
