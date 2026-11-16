from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# ===============================
# 1. AMBIL DATASET
# ===============================
dataset = fetch_ucirepo(id=697)

X = dataset.data.features
y = dataset.data.targets

# Ambil hanya 1 kolom target
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# ===============================
# 2. ENCODING DATA KATEGORIK
# ===============================
X_encoded = X.copy()

for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        le_col = LabelEncoder()
        X_encoded[col] = le_col.fit_transform(X_encoded[col].astype(str))

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print("Kelas Target:", le_target.classes_)

# ===============================
# 3. SPLIT DATA TRAIN & TEST
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ===============================
# 4. MODEL DECISION TREE
# ===============================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 5. PREDIKSI
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 6. EVALUASI MODEL
# ===============================
accuracy = accuracy_score(y_test, y_pred)
print("\nAkurasi Model :", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# ===============================
# 7. VISUALISASI DISTRIBUSI KELAS AKTUAL (dengan warna per kelas)
# ===============================
aktual_counts = np.bincount(y_encoded)

# Tentukan warna untuk setiap kelas
colors = ['pink', 'purple', 'deepskyblue']  # bisa diganti sesuai jumlah kelas

plt.figure(figsize=(6,4))
plt.bar(le_target.classes_, aktual_counts, color=colors)
plt.xlabel("Kategori Prestasi Akademik")
plt.ylabel("Jumlah Mahasiswa")
plt.title("Distribusi Data Aktual")
plt.show()

# ===============================
# 8. PERBANDINGAN AKTUAL VS PREDIKSI
# ===============================
pred_counts = np.bincount(y_pred)

x = np.arange(len(le_target.classes_))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, np.bincount(y_test), width, label="Aktual")
plt.bar(x + width/2, pred_counts, width, label="Prediksi")

plt.xticks(x, le_target.classes_)
plt.xlabel("Kelas")
plt.ylabel("Jumlah Data")
plt.title("Perbandingan Aktual vs Prediksi")
plt.legend()
plt.show()

# ===============================
# 9. CONFUSION MATRIX (HEATMAP)
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_,
            cmap='Blues')

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
