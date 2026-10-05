"""End-to-end training pipeline: load -> preprocess -> train -> evaluate -> serialize."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)

RANDOM_STATE = 42

def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    return df, list(data.feature_names), list(data.target_names)

def main():
    df, features, target_names = load_data()
    print("Dataset shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Class balance:\n", df["target"].value_counts())

    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    }
    results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        results[name] = {"cv_mean": float(cv.mean()), "cv_std": float(cv.std())}
        print(f"{name:20s} CV acc = {cv.mean():.4f} +/- {cv.std():.4f}")

    # Hyper-parameter tuning on the best family (Logistic Regression is strong & interpretable here)
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    print("Best CV model:", best_name)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", candidates[best_name])])
    grids = {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "Random Forest": {"clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 2, 4]},
        "SVM (RBF)": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.001]},
    }
    gs = GridSearchCV(pipe, grids[best_name], cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    print("Best params:", gs.best_params_)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": best_name,
        "best_params": {k: (v if v is not None else "None") for k, v in gs.best_params_.items()},
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cv_results": results,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(json.dumps({k: v for k, v in metrics.items() if k in ["accuracy","precision","recall","f1","roc_auc"]}, indent=2))

    # ---- Serialization (Joblib + Pickle) ----
    joblib.dump(model, "models/model.joblib")
    import pickle
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    meta = {"features": features, "target_names": target_names,
            "feature_ranges": {f: [float(X[f].min()), float(X[f].max()), float(X[f].mean())] for f in features},
            "metrics": metrics}
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)

    # ---- Figures for the report ----
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=target_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)"); fig.tight_layout(); fig.savefig("reports/confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(results); vals = [results[n]["cv_mean"] for n in names]; errs = [results[n]["cv_std"] for n in names]
    ax.bar(names, vals, yerr=errs, color=["#2563eb", "#16a34a", "#d97706"], capsize=4)
    ax.set_ylim(0.9, 1.0); ax.set_ylabel("5-fold CV accuracy"); ax.set_title("Model Comparison")
    fig.tight_layout(); fig.savefig("reports/model_comparison.png", dpi=150); plt.close(fig)

    coefs = model.named_steps["clf"]
    if hasattr(coefs, "coef_"):
        imp = pd.Series(np.abs(coefs.coef_[0]), index=features).sort_values().tail(10)
    else:
        imp = pd.Series(coefs.feature_importances_, index=features).sort_values().tail(10) if hasattr(coefs, "feature_importances_") else None
    if imp is not None:
        fig, ax = plt.subplots(figsize=(6, 4)); imp.plot.barh(ax=ax, color="#2563eb")
        ax.set_title("Top 10 Most Influential Features"); fig.tight_layout(); fig.savefig("reports/feature_importance.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    main()
