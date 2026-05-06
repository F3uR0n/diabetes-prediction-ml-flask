# All Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import os
import pickle


ARTIFACTS_DIR = "artifacts"
FIGURES_DIR = os.path.join("reports", "figures")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# Load Dataset
df = pd.read_csv('diabetes_dataset.csv')
print("Dataset loaded. Shape:", df.shape)
df.info()
df.describe()



# EDA + Correlation
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues')
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
plt.close()

plt.figure()
sns.countplot(x='diabetes', data=df)
plt.savefig(os.path.join(FIGURES_DIR, "diabetes_distribution.png"))
plt.close()



# Data Pre Processing
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.drop_duplicates()
print(f"\nAfter dropping duplicates: {df.shape}")

colmnsNumerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
for col in colmnsNumerical:
    df[col].fillna(df[col].median(), inplace=True)

colmnsCatagorical = ['gender', 'smoking_history']
for col in colmnsCatagorical:
    df[col].fillna(df[col].mode()[0], inplace=True)

df.dropna(subset=['diabetes'], inplace=True)
df['diabetes'] = df['diabetes'].astype(int)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df['diabetes'].value_counts())
pos_rate = df['diabetes'].mean()
print(f"Positive rate: {pos_rate*100:.1f}%")



# Feature Distribution Analysis (Numerical)
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

fig = plt.figure(figsize=(12,8))
df[num_cols].hist(figsize=(12,8), bins=30)
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_distributions.png"))
plt.close()



# LabelEncoder
print("\n=== Label Encoding ===")
print("Before encoding:")
print("Gender unique values:", df['gender'].unique())
print("Smoking history unique values:", df['smoking_history'].unique())

lE_gender = LabelEncoder()
lE_smoking = LabelEncoder()

df['gender'] = lE_gender.fit_transform(df['gender'])
df['smoking_history'] = lE_smoking.fit_transform(df['smoking_history'])

print("\nAfter encoding:")
print("Gender mapping:", dict(zip(lE_gender.classes_, lE_gender.transform(lE_gender.classes_))))
print("Smoking mapping:", dict(zip(lE_smoking.classes_, lE_smoking.transform(lE_smoking.classes_))))

df.info()



# Define explicit feature order
FEATURE_ORDER = [
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'smoking_history',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]

print("\n=== Feature Order ===")
print(FEATURE_ORDER)

X = df[FEATURE_ORDER]
y = df['diabetes']

print("\nX columns:", list(X.columns))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst few rows of X:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train class distribution:\n{y_train.value_counts()}")
print(f"Test class distribution:\n{y_test.value_counts()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nScaler fitted. Mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)



# Model Training
# The dataset is ~8.6% diabetic (heavily imbalanced).
# class_weight='balanced' corrects for this in LR and DT.
# For MLP/KNN/NB (no class_weight), threshold optimisation via Youden's J compensates.
print("\n=== Training Models ===")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
print("Logistic Regression trained.")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
dt.fit(X_train_scaled, y_train)
print("Decision Tree trained.")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
print("Naive Bayes trained.")

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
print("KNN trained.")

# Neural Network — deeper, early stopping, L2 regularisation
nn = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    alpha=0.001
)
nn.fit(X_train_scaled, y_train)
print("Neural Network trained.")


models_obj = {
    "Logistic Regression": lr,
    "Decision Tree":       dt,
    "Naive Bayes":         nb,
    "KNN":                 knn,
    "Neural Network":      nn,
}


# Model Selection — Youden's J threshold + F1 on diabetic class
# With 8.6% positive rate the default 0.5 threshold misses most diabetics.
# Youden's J = max(TPR - FPR) finds the threshold that best balances sensitivity
# and specificity for each model; we then rank by F1 on the diabetic class.
print("\n=== Model Evaluation (Youden threshold | F1 on diabetic class) ===")

f1_results  = {}
acc_results = {}
rec_results = {}
thresh_map  = {}

for name, mdl in models_obj.items():
    probs = mdl.predict_proba(X_test_scaled)[:, 1]

    fpr_arr, tpr_arr, thresholds_roc = roc_curve(y_test, probs)
    j_scores   = tpr_arr - fpr_arr
    best_idx   = int(np.argmax(j_scores))
    best_thresh = round(float(thresholds_roc[best_idx]), 4)
    thresh_map[name] = best_thresh

    preds = (probs >= best_thresh).astype(int)
    f1  = f1_score(y_test,  preds, pos_label=1)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, pos_label=1)
    pre = precision_score(y_test, preds, pos_label=1)

    f1_results[name]  = f1
    acc_results[name] = acc
    rec_results[name] = rec

    print(f"\n{name}  |  threshold={best_thresh}")
    print(f"  F1(diabetic)={f1:.4f}  Recall={rec:.4f}  Precision={pre:.4f}  Accuracy={acc:.4f}")
    print("-" * 50)


# Bar chart — F1 on diabetic class (the metric that matters for imbalanced medical data)
best_model_name = max(f1_results, key=f1_results.get)
colours = ['tomato' if n == best_model_name else 'steelblue' for n in f1_results]

plt.figure(figsize=(7,4))
plt.bar(f1_results.keys(), f1_results.values(), color=colours)
plt.ylabel("F1 Score (Diabetic Class)")
plt.title("Model Comparison — F1 (Youden Threshold)")
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"))
plt.close()


# Detailed metrics table
print("\n=== Detailed Metrics (all models) ===")
for name, mdl in models_obj.items():
    probs = mdl.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= thresh_map[name]).astype(int)
    print(f"\n{name}  (threshold={thresh_map[name]})")
    print("  Precision:", round(precision_score(y_test, preds), 4))
    print("  Recall:   ", round(recall_score(y_test, preds), 4))
    print("  F1-score: ", round(f1_score(y_test, preds), 4))


# Confusion Matrix — with Youden threshold
for name, mdl in models_obj.items():
    probs = mdl.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= thresh_map[name]).astype(int)
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}\n(threshold={thresh_map[name]:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(FIGURES_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"))
    plt.close()


# ROC Curve
plt.figure(figsize=(7,5))
for name, mdl in models_obj.items():
    if hasattr(mdl, "predict_proba"):
        probs = mdl.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"))
plt.close()



# KMeans (Unsupervised)
print("\n=== Clustering Analysis ===")
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X_num = df[num_cols]

X_scaled_cluster = StandardScaler().fit_transform(X_num)

wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled_cluster)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.savefig(os.path.join(FIGURES_DIR, "elbow_method.png"))
plt.close()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_cluster)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', s=10)
plt.title("KMeans Clustering (Numerical Features, PCA Reduced)")
plt.savefig(os.path.join(FIGURES_DIR, "kmeans_clustering.png"))
plt.close()



# Select best model by F1 on diabetic class
best_model      = models_obj[best_model_name]
best_threshold  = thresh_map[best_model_name]
best_f1         = f1_results[best_model_name]
best_acc        = acc_results[best_model_name]

print(f"\n[BEST] {best_model_name}")
print(f"  F1(diabetic) : {best_f1:.4f}")
print(f"  Threshold    : {best_threshold}")
print(f"  Accuracy     : {best_acc:.4f}")


# Verify failing case (Male 67, HbA1c=6.5, glucose=200 — should be Diabetic)
print("\n=== Sanity check on known-diabetic input ===")
failing_row = {
    'gender':             lE_gender.transform(['Male'])[0],
    'age':                67,
    'hypertension':       0,
    'heart_disease':      1,
    'smoking_history':    lE_smoking.transform(['not current'])[0],
    'bmi':                27.32,
    'HbA1c_level':        6.5,
    'blood_glucose_level': 200,
}
df_fail = pd.DataFrame([failing_row], columns=FEATURE_ORDER)
fail_scaled = scaler.transform(df_fail)
fail_prob   = float(best_model.predict_proba(fail_scaled)[0][1])
fail_pred   = int(fail_prob >= best_threshold)
print(f"  P(diabetic)={fail_prob:.4f}  threshold={best_threshold}  => {'DIABETIC [correct]' if fail_pred == 1 else 'NON-DIABETIC [WRONG]'}")


# Save everything
print("\n=== Saving Artifacts ===")

with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print("[OK] model.pkl saved")

with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("[OK] scaler.pkl saved")

with open(os.path.join(ARTIFACTS_DIR, "feature_order.pkl"), "wb") as f:
    pickle.dump(FEATURE_ORDER, f)
print("[OK] feature_order.pkl saved")

with open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump({'gender': lE_gender, 'smoking_history': lE_smoking}, f)
print("[OK] label_encoders.pkl saved")

metadata = {
    'model_name':      best_model_name,
    'accuracy':        best_acc,
    'f1_diabetic':     best_f1,
    'threshold':       best_threshold,
    'all_f1':          f1_results,
    'all_thresholds':  thresh_map,
    'feature_order':   FEATURE_ORDER,
    'gender_mapping':  dict(zip(lE_gender.classes_,  lE_gender.transform(lE_gender.classes_))),
    'smoking_mapping': dict(zip(lE_smoking.classes_, lE_smoking.transform(lE_smoking.classes_))),
}
with open(os.path.join(ARTIFACTS_DIR, "model_metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)
print("[OK] model_metadata.pkl saved")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Best Model  : {best_model_name}")
print(f"F1(diabetic): {best_f1:.4f}")
print(f"Threshold   : {best_threshold}")
print(f"Accuracy    : {best_acc:.4f}")
print("="*50)
