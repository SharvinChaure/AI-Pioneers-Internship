"""
Week 2 Task: Supervised Machine Learning Models (Scikit-learn)
Regression  : Linear Regression, Decision Tree, Random Forest, KNN  (Diabetes dataset)
Classification: Logistic Regression, Decision Tree, Random Forest, KNN (Breast Cancer dataset)
Run:  python supervised_models.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

SEED = 42
log = {}
NAVY, LIGHT = "#1F4E79", "#9DB7D5"

# =====================================================================
# PART A: REGRESSION  (predict diabetes disease progression)
# =====================================================================
diab = load_diabetes(as_frame=True)
Xr, yr = diab.data, diab.target
log["reg_shape"] = list(Xr.shape)
log["reg_features"] = list(Xr.columns)
log["reg_target_stats"] = {"mean": round(float(yr.mean()), 1), "min": float(yr.min()),
                           "max": float(yr.max())}

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=SEED)
log["reg_split"] = [len(Xr_train), len(Xr_test)]

reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6,
                                           random_state=SEED),
    "KNN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
}
reg_results, reg_preds = {}, {}
for name, m in reg_models.items():
    m.fit(Xr_train, yr_train)
    pred = m.predict(Xr_test)
    reg_preds[name] = pred
    mse = mean_squared_error(yr_test, pred)
    cv = cross_val_score(m, Xr, yr, cv=5, scoring="r2")
    reg_results[name] = {
        "MAE": round(mean_absolute_error(yr_test, pred), 2),
        "MSE": round(mse, 2),
        "RMSE": round(float(np.sqrt(mse)), 2),
        "R2": round(r2_score(yr_test, pred), 3),
        "CV_R2": round(float(cv.mean()), 3),
    }
log["reg_results"] = reg_results
best_reg = max(reg_results, key=lambda k: reg_results[k]["R2"])
log["best_reg"] = best_reg

lr = reg_models["Linear Regression"]
log["lin_coefs"] = {c: round(float(v), 1) for c, v in zip(Xr.columns, lr.coef_)}
log["lin_intercept"] = round(float(lr.intercept_), 2)

# chart: R2 comparison
fig, ax = plt.subplots(figsize=(6, 3.2))
names = list(reg_results)
ax.bar(names, [reg_results[n]["R2"] for n in names],
       color=[NAVY if n == best_reg else LIGHT for n in names])
ax.set_ylabel("R² on test set"); ax.set_title("Regression Models: R² Comparison")
plt.xticks(rotation=15)
for i, n in enumerate(names):
    ax.text(i, reg_results[n]["R2"] + 0.01, f'{reg_results[n]["R2"]:.2f}', ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("chart_reg_r2.png", dpi=150); plt.close()

# chart: actual vs predicted for best model
plt.figure(figsize=(4.6, 4))
plt.scatter(yr_test, reg_preds[best_reg], color=NAVY, alpha=0.7)
lims = [yr_test.min(), yr_test.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_reg})"); plt.legend()
plt.tight_layout(); plt.savefig("chart_reg_actual_pred.png", dpi=150); plt.close()

# =====================================================================
# PART B: CLASSIFICATION  (predict malignant / benign tumour)
# =====================================================================
cancer = load_breast_cancer(as_frame=True)
Xc, yc = cancer.data, cancer.target       # 0 = malignant, 1 = benign
log["clf_shape"] = list(Xc.shape)
log["clf_classes"] = {"malignant": int((yc == 0).sum()), "benign": int((yc == 1).sum())}

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=SEED, stratify=yc)
log["clf_split"] = [len(Xc_train), len(Xc_test)]

clf_models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}
clf_results, clf_cm = {}, {}
for name, m in clf_models.items():
    m.fit(Xc_train, yc_train)
    pred = m.predict(Xc_test)
    proba = m.predict_proba(Xc_test)[:, 1]
    cv = cross_val_score(m, Xc, yc, cv=5, scoring="accuracy")
    clf_results[name] = {
        "Accuracy": round(accuracy_score(yc_test, pred), 3),
        "Precision": round(precision_score(yc_test, pred), 3),
        "Recall": round(recall_score(yc_test, pred), 3),
        "F1": round(f1_score(yc_test, pred), 3),
        "ROC_AUC": round(roc_auc_score(yc_test, proba), 3),
        "CV_Acc": round(float(cv.mean()), 3),
    }
    clf_cm[name] = confusion_matrix(yc_test, pred).tolist()
log["clf_results"] = clf_results
log["clf_cm"] = clf_cm
best_clf = max(clf_results, key=lambda k: (clf_results[k]["F1"], clf_results[k]["Accuracy"]))
log["best_clf"] = best_clf

# chart: grouped metrics
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics)); wd = 0.2
fig, ax = plt.subplots(figsize=(7, 3.6))
cols = [NAVY, "#2E75B6", LIGHT, "#BFBFBF"]
for i, n in enumerate(clf_results):
    ax.bar(x + i * wd, [clf_results[n][m] for m in metrics], wd, label=n, color=cols[i])
ax.set_xticks(x + 1.5 * wd); ax.set_xticklabels(metrics)
ax.set_ylim(0.85, 1.0); ax.set_title("Classification Models: Metric Comparison")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("chart_clf_metrics.png", dpi=150); plt.close()

# chart: confusion matrix of best model
cm = np.array(clf_cm[best_clf])
plt.figure(figsize=(3.8, 3.4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["Malignant", "Benign"]); plt.yticks([0, 1], ["Malignant", "Benign"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix\n({best_clf})")
plt.tight_layout(); plt.savefig("chart_clf_cm.png", dpi=150); plt.close()

# KNN: effect of k
ks = range(1, 26)
k_acc = []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    m.fit(Xc_train, yc_train)
    k_acc.append(accuracy_score(yc_test, m.predict(Xc_test)))
log["knn_best_k"] = int(list(ks)[int(np.argmax(k_acc))])
log["knn_best_acc"] = round(max(k_acc), 3)
plt.figure(figsize=(5.6, 3.2))
plt.plot(list(ks), k_acc, marker="o", color=NAVY)
plt.xlabel("Number of neighbours (k)"); plt.ylabel("Test accuracy")
plt.title("KNN: Effect of k on Accuracy"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("chart_knn_k.png", dpi=150); plt.close()

# Decision tree depth (overfitting demo)
depths = range(1, 11)
tr_acc, te_acc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xc_train, yc_train)
    tr_acc.append(t.score(Xc_train, yc_train)); te_acc.append(t.score(Xc_test, yc_test))
log["dt_depth"] = {"train_d10": round(tr_acc[-1], 3), "test_d10": round(te_acc[-1], 3),
                   "best_depth": int(list(depths)[int(np.argmax(te_acc))]),
                   "best_test": round(max(te_acc), 3)}
plt.figure(figsize=(5.6, 3.2))
plt.plot(list(depths), tr_acc, marker="o", label="Train", color=NAVY)
plt.plot(list(depths), te_acc, marker="s", label="Test", color="#E36C0A")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Decision Tree: Depth vs Accuracy")
plt.tight_layout(); plt.savefig("chart_dt_depth.png", dpi=150); plt.close()

# Random Forest feature importance
rf = clf_models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=Xc.columns).sort_values(ascending=False)
log["rf_top"] = {k: round(float(v), 3) for k, v in imp.head(5).items()}
plt.figure(figsize=(5.8, 3.4))
imp.head(8)[::-1].plot(kind="barh", color=NAVY)
plt.title("Random Forest: Top 8 Feature Importances")
plt.tight_layout(); plt.savefig("chart_rf_imp.png", dpi=150); plt.close()

# Logistic Regression: also report the regression-style 'Linear Regression'
# is NOT used for classification; shown only in Part A.

with open("log2.json", "w") as f:
    json.dump(log, f, indent=2)
print(json.dumps({k: log[k] for k in ["reg_results", "best_reg", "clf_results", "best_clf",
                                       "knn_best_k", "knn_best_acc", "dt_depth", "rf_top",
                                       "lin_coefs"]}, indent=1))
