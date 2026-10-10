"""
Group 9 - Machine Learning Mini Project
Topic: Support Vector Machine (SVM)
Question: Can breast tumors be classified as benign or malignant?
Run: python Group9_SVM_Project.py
Requirements: pandas, numpy, matplotlib, scikit-learn
Place data.csv in the same folder as this script.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)

# 1. Load and inspect data
df = pd.read_csv("data.csv")
print("Shape:", df.shape)
print("\\nColumns:\\n", df.columns.tolist())
print("\\nData types:\\n", df.dtypes)
print("\\nFirst records:\\n", df.head())
print("\\nLast records:\\n", df.tail())
print("\\nInfo:")
df.info()
print("\\nMissing values:\\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\\nDiagnosis counts:\\n", df["diagnosis"].value_counts())

# 2. Preprocess
# Remove identifier and completely empty Kaggle-export column.
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
df = df.drop_duplicates()
# Diagnosis: B=0 (benign), M=1 (malignant)
y = df["diagnosis"].map({"B": 0, "M": 1})
X = df.drop(columns=["diagnosis"])
if y.isna().any():
    raise ValueError("Unexpected diagnosis labels found.")
# Stratified split preserves the class ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Statistical analysis
numeric = X.select_dtypes(include=np.number)
stats = numeric.agg(["min", "max", "mean", "median", "std", "var"]).T
stats["range"] = numeric.max() - numeric.min()
stats["IQR"] = numeric.quantile(.75) - numeric.quantile(.25)
stats["MAD"] = numeric.apply(lambda col: np.mean(np.abs(col - col.mean())))
print("\\nDescriptive statistics (first 10 features):\\n", stats.head(10))
q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
iqr = q3 - q1
outlier_count = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
print("\\nIQR outlier counts (largest 10):\\n", outlier_count.sort_values(ascending=False).head(10))
print("\\nPearson correlation matrix:\\n", numeric.corr())

# 4. EDA plots
df["diagnosis"].value_counts().reindex(["B", "M"]).plot(kind="bar")
plt.title("Diagnosis Class Counts"); plt.xlabel("B=Benign, M=Malignant"); plt.ylabel("Count")
plt.tight_layout(); plt.show()
for label, name in [(0, "Benign"), (1, "Malignant")]:
    plt.hist(X.loc[y == label, "radius_mean"], bins=20, alpha=.6, label=name)
plt.title("Mean Radius by Diagnosis"); plt.xlabel("radius_mean"); plt.ylabel("Frequency")
plt.legend(); plt.tight_layout(); plt.show()
X.assign(diagnosis=y).boxplot(column="area_mean", by="diagnosis")
plt.title("Mean Area by Diagnosis"); plt.suptitle(""); plt.tight_layout(); plt.show()
corr = numeric.corr()
plt.figure(figsize=(12, 9))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=5)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=5)
plt.title("Correlation Heatmap"); plt.tight_layout(); plt.show()

# 5. SVM pipeline and hyperparameter tuning
# Scaling is fitted inside each CV fold to avoid data leakage.
pipeline = Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(probability=True, random_state=42))])
param_grid = {
    "svc__kernel": ["rbf", "linear"],
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", "auto"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)
model = search.best_estimator_
print("\\nBest parameters:", search.best_params_)
print("Best mean CV F1:", search.best_score_)

# 6. Evaluation
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
print("\\nAccuracy:", accuracy_score(y_test, pred))
print("Precision (malignant positive class):", precision_score(y_test, pred))
print("Recall / sensitivity (malignant):", recall_score(y_test, pred))
print("F1-score:", f1_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("\\nClassification report:\\n", classification_report(
    y_test, pred, target_names=["Benign", "Malignant"]))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Benign", "Malignant"])
plt.title("SVM Confusion Matrix"); plt.tight_layout(); plt.show()
RocCurveDisplay.from_predictions(y_test, prob)
plt.title("ROC Curve"); plt.tight_layout(); plt.show()

# 7. Learning curve (appropriate for SVM; not epoch-based)
sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=cv, train_sizes=np.linspace(.2, 1, 5),
    scoring="accuracy", n_jobs=-1
)
plt.plot(sizes, train_scores.mean(axis=1), marker="o", label="Training")
plt.plot(sizes, val_scores.mean(axis=1), marker="o", label="Cross-validation")
plt.xlabel("Training examples"); plt.ylabel("Accuracy")
plt.title("SVM Learning Curve"); plt.legend(); plt.tight_layout(); plt.show()

# 8. Pearson correlations with target, for exploratory association only
target_corr = numeric.apply(lambda col: col.corr(y)).sort_values(
    key=lambda s: s.abs(), ascending=False
)
print("\\nFeatures most correlated with diagnosis (absolute Pearson correlation):\\n",
      target_corr.head(10))
print("\\nReminder: educational project only; not a clinical diagnostic tool.")
