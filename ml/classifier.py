import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


FEATURE_COLS = ["density", "avg_conf", "n_detections"]
LABEL_COL = "severity"
FILENAME_COL = "image"


def label_from_filename(name: str) -> int:
    try:
        return int(str(name)[5])
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract level from filename: {name}")


def load_dataset(csv_path: str, merge_level3_into_level2: bool = True):
    df = pd.read_csv(csv_path)

    if FILENAME_COL not in df.columns:
        raise ValueError(f"CSV must contain a '{FILENAME_COL}' column. Found: {list(df.columns)}")

    df[LABEL_COL] = df[FILENAME_COL].apply(label_from_filename)

    if merge_level3_into_level2:
        df[LABEL_COL] = df[LABEL_COL].replace({3: 2})

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing feature columns: {missing}. Found: {list(df.columns)}")

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    return X, y


def compute_auc_macro_ovr(y_true, y_proba, n_classes: int):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")


def eval_and_collect(y_true, y_pred, model_name: str, class_names):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    print(f"\n=== {model_name} ===")
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix:\n", cm)
    print("Report:\n", classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Build row
    row = {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }

    # Per-class metrics
    for cname in class_names:
        row[f"{cname}_precision"] = report[cname]["precision"]
        row[f"{cname}_recall"] = report[cname]["recall"]
        row[f"{cname}_f1"] = report[cname]["f1-score"]
        row[f"{cname}_support"] = report[cname]["support"]

    # Confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            row[f"cm_{i}{j}"] = int(cm[i, j])

    return row


def run_cv_model_collect(X, y, model_name, pipeline, class_names, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_pred = cross_val_predict(pipeline, X, y, cv=skf, method="predict")

    row = eval_and_collect(y, y_pred, f"{model_name} ({n_splits}-fold CV)", class_names)

    try:
        y_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")
        auc = compute_auc_macro_ovr(y, y_proba, n_classes=len(class_names))
        row["auc_macro_ovr"] = float(auc)
    except Exception:
        row["auc_macro_ovr"] = None

    return row


def main():
    csv_path = "../data/features/features_yolov8m_85.csv"

    merge_to_3_classes = True
    X, y = load_dataset(csv_path, merge_level3_into_level2=merge_to_3_classes)

    class_names = ["level0", "level1", "level2"] if merge_to_3_classes else ["level0", "level1", "level2", "level3"]

    models = {
        "KNN (k=3, distance)": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3, weights="distance")),
        ]),

        "SVM (RBF)": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]),

        "Logistic Regression": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ]),

        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "HistGradientBoosting": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ))
        ]),
    }

    rows = []
    for name, pipe in models.items():
        row = run_cv_model_collect(X, y, name, pipe, class_names, n_splits=5)
        rows.append(row)

    df_metrics = pd.DataFrame(rows).sort_values(by="accuracy", ascending=False)

    out_dir = Path("../results/severity")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(csv_path).stem
    out_csv = out_dir / f"metrics_{input_stem}.csv"

    df_metrics.to_csv(out_csv, index=False)

    print("\n== Summary ==")
    print(df_metrics[["model", "accuracy", "f1_macro", "auc_macro_ovr"]].to_string(index=False))
    print(f"\nSaved metrics CSV: {out_csv}")


if __name__ == "__main__":
    main()
