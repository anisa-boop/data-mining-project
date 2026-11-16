import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 1. DOWNLOAD DATASET
# ============================================
path = kagglehub.dataset_download("hasibalmuzdadid/global-air-pollution-dataset")

print("Path:", path)
print("Files:", os.listdir(path))

file_name = "global air pollution dataset.csv"
data_path = os.path.join(path, file_name)

# ============================================
# 2. LOAD DATASET
# ============================================
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()

print("\n===== Kolom Dalam Dataset =====")
print(list(data.columns))

# ============================================
# 3. FITUR & TARGET SESUAI DATASET
# ============================================
features = [
    'CO AQI Value',
    'Ozone AQI Value',
    'NO2 AQI Value',
    'PM2.5 AQI Value'
]

target = 'AQI Value'

X = data[features]
y = data[target]

# ============================================
# 4. SPLIT DATA
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 5. NORMALISASI
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 6. TRAIN MLP NEURAL NETWORK
# ============================================
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ============================================
# 7. EVALUASI MODEL
# ============================================
y_pred = model.predict(X_test_scaled)

print("\n===== HASIL EVALUASI MODEL =====")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# ============================================
# 8. CONTOH PREDIKSI
# ============================================
sample = X_test.iloc[0:1]
sample_pred = model.predict(scaler.transform(sample))

print("\nPrediksi AQI:", sample_pred[0])
print("Nilai Asli:", y_test.iloc[0])

result = X_test.copy()
result['AQI_Actual'] = y_test
result['AQI_Prediksi'] = y_pred
result.to_csv("hasil_prediksi.csv", index=False)