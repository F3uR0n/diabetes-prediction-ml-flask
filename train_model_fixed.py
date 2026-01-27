# All Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle



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
plt.savefig('correlation_heatmap.png')
plt.close()

plt.figure()
sns.countplot(x='diabetes', data=df)
plt.savefig('diabetes_distribution.png')
plt.close()



# Data Pre Processing
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.drop_duplicates() # Duplicates
print(f"\nAfter dropping duplicates: {df.shape}")

colmnsNumerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease'] # Missing Values
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



# Feature Distribution Analysis (Numerical)
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

fig = plt.figure(figsize=(12,8))
df[num_cols].hist(figsize=(12,8), bins=30)
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()



# LabelEncoder - SAVE THE ENCODERS!
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



# 🔥 DEFINE EXPLICIT FEATURE ORDER
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

# Split with EXPLICIT column order
X = df[FEATURE_ORDER]
y = df['diabetes']

print("\nX columns:", list(X.columns))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst few rows of X:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train class distribution:\n{y_train.value_counts()}")
print(f"Test class distribution:\n{y_test.value_counts()}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaler fitted. Mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)



# Model Training
print("\n=== Training Models ===")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print(f"Logistic Regression trained. Sample predictions: {lr_pred[:10]}")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)
print(f"Decision Tree trained. Sample predictions: {dt_pred[:10]}")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_pred = nb.predict(X_test_scaled)
print(f"Naive Bayes trained. Sample predictions: {nb_pred[:10]}")

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
print(f"KNN trained. Sample predictions: {knn_pred[:10]}")

# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
nn.fit(X_train_scaled, y_train)
nn_pred = nn.predict(X_test_scaled)
print(f"Neural Network trained. Sample predictions: {nn_pred[:10]}")



# Model Selection & Comparison Analysis
models_obj = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "KNN": knn,
    "Naive Bayes": nb,
    "Neural Network": nn
}
results = {}

print("\n=== Model Evaluation ===")
for name, model in models_obj.items():
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

plt.figure(figsize=(7,4))
plt.bar(results.keys(), results.values())
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Precision & Recall Comparison
print("\n=== Detailed Metrics ===")
for name, model in models_obj.items():
    pred = model.predict(X_test_scaled)
    print(f"\n{name}")
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1-score:", f1_score(y_test, pred))
    print("-"*40)

# Confusion Matrix
for name, model in models_obj.items():
    pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

# ROC Curve + AUC Score
plt.figure(figsize=(7,5))

for name, model in models_obj.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_scaled)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig('roc_curves.png')
plt.close()



# KMeans (Unsupervised)
print("\n=== Clustering Analysis ===")
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X_num = df[num_cols]

# Scale
X_scaled_cluster = StandardScaler().fit_transform(X_num)

# Elbow
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
plt.savefig('elbow_method.png')
plt.close()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_cluster)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', s=10)
plt.title("KMeans Clustering (Numerical Features, PCA Reduced)")
plt.savefig('kmeans_clustering.png')
plt.close()

# Select best model based on accuracy
best_model_name = max(results, key=results.get)
best_model = models_obj[best_model_name]

print(f"\n✅ Best Model Selected: {best_model_name}")
print(f"✅ Accuracy: {results[best_model_name]:.4f}")

# Test the best model with a sample prediction
print("\n=== Testing Best Model with Sample Data ===")
sample_diabetic = X_test_scaled[y_test == 1][0:1]  # Get a diabetic sample
sample_non_diabetic = X_test_scaled[y_test == 0][0:1]  # Get a non-diabetic sample

print("Sample Diabetic prediction:", best_model.predict(sample_diabetic))
print("Sample Non-Diabetic prediction:", best_model.predict(sample_non_diabetic))

if hasattr(best_model, "predict_proba"):
    print("Sample Diabetic probability:", best_model.predict_proba(sample_diabetic))
    print("Sample Non-Diabetic probability:", best_model.predict_proba(sample_non_diabetic))

# Save everything
print("\n=== Saving Model and Artifacts ===")

# Save best model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("✅ model.pkl saved")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl saved")

# Save feature order
with open("feature_order.pkl", "wb") as f:
    pickle.dump(FEATURE_ORDER, f)
print("✅ feature_order.pkl saved")

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump({
        'gender': lE_gender,
        'smoking_history': lE_smoking
    }, f)
print("✅ label_encoders.pkl saved")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': results[best_model_name],
    'feature_order': FEATURE_ORDER,
    'gender_mapping': dict(zip(lE_gender.classes_, lE_gender.transform(lE_gender.classes_))),
    'smoking_mapping': dict(zip(lE_smoking.classes_, lE_smoking.transform(lE_smoking.classes_))),
    'all_results': results
}

with open("model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print("✅ model_metadata.pkl saved")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]:.4f}")
print("\nFiles saved:")
print("  - model.pkl")
print("  - scaler.pkl")
print("  - feature_order.pkl")
print("  - label_encoders.pkl")
print("  - model_metadata.pkl")
print("="*50)