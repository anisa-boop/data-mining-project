# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================
# 2. DOWNLOAD DATASET
# ============================================
path = kagglehub.dataset_download("tawfikelmetwally/advertising-dataset")
file_name = "Advertising.csv"
data_path = os.path.join(path, file_name)

data = pd.read_csv(data_path)

print("===== 5 Data Teratas =====")
print(data.head())


# ============================================
# 3. SEPARATE FEATURES & TARGET
# ============================================
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


# ============================================
# 4. ESTIMATE LINEAR REGRESSION PARAMETERS
# ============================================
model = LinearRegression()
model.fit(X, y)

print("\n===== HASIL ESTIMASI PARAMETER =====")
print("Intercept:", model.intercept_)
print("Koef TV:", model.coef_[0])
print("Koef Radio:", model.coef_[1])
print("Koef Newspaper:", model.coef_[2])


# ============================================
# 5. HITUNG ERROR MENGGUNAKAN DATA ASLI
# ============================================
y_estimated = model.predict(X)

mse = mean_squared_error(y, y_estimated)
mae = mean_absolute_error(y, y_estimated)

print("\n===== PERHITUNGAN ERROR (Estimasi vs Data Asli) =====")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)


# ============================================
# 6. VISUALISASI ESTIMASI & ERROR
# ============================================

# Residual (Error)
residuals = y - y_estimated

# --- Plot 1: Actual vs Estimated + Garis y=x ---
plt.figure(figsize=(8,6))
plt.scatter(y, y_estimated, alpha=0.7, label="Estimated")
plt.plot(y, y, linestyle='--', label="Actual (y=x)")

plt.xlabel("Actual Sales")
plt.ylabel("Estimated Sales")
plt.title("Actual vs Estimated Sales")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot 2: Residual Plot ---
plt.figure(figsize=(8,6))
plt.scatter(y, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Actual Sales")
plt.ylabel("Residual (Error)")
plt.title("Residual Plot (Error dari Estimasi)")
plt.grid(True)
plt.show()

# --- Plot 3: Distribusi Error ---
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)

plt.xlabel("Residual (Error)")
plt.ylabel("Frekuensi")
plt.title("Distribusi Error Model Linear Regression")
plt.grid(True)
plt.show()



# ============================================
# 7. VISUALISASI 3D: TV - Radio - Sales
# (Menggunakan 2 variabel agar bisa divisualisasikan 3D)
# ============================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data["TV"], data["Radio"], y, c='blue', label="Actual Sales")

# Buat grid untuk plane regresi
tv_range = np.linspace(data["TV"].min(), data["TV"].max(), 30)
radio_range = np.linspace(data["Radio"].min(), data["Radio"].max(), 30)
tv_grid, radio_grid = np.meshgrid(tv_range, radio_range)

# Prediksi plane
sales_plane = (
    model.intercept_ +
    model.coef_[0] * tv_grid +
    model.coef_[1] * radio_grid +
    model.coef_[2] * data["Newspaper"].mean()  # newspaper dibuat rata-rata
)

ax.plot_surface(tv_grid, radio_grid, sales_plane, alpha=0.4, color='orange')

ax.set_xlabel("TV")
ax.set_ylabel("Radio")
ax.set_zlabel("Sales")
ax.set_title("3D Visualization of Linear Regression Estimation")

plt.show()

# ============================================
# 7B. VISUALISASI 3D: TV - Newspaper - Sales
# ============================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data["TV"], data["Newspaper"], y, c='green', label="Actual Sales")

# Grid untuk plane
tv_range = np.linspace(data["TV"].min(), data["TV"].max(), 30)
news_range = np.linspace(data["Newspaper"].min(), data["Newspaper"].max(), 30)
tv_grid, news_grid = np.meshgrid(tv_range, news_range)

sales_plane_2 = (
    model.intercept_ +
    model.coef_[0] * tv_grid +
    model.coef_[1] * data["Radio"].mean() +  # radio jadi rata-rata
    model.coef_[2] * news_grid
)

ax.plot_surface(tv_grid, news_grid, sales_plane_2, alpha=0.4, color='yellow')
ax.set_xlabel("TV")
ax.set_ylabel("Newspaper")
ax.set_zlabel("Sales")
ax.set_title("3D Visualization: TV - Newspaper - Sales")
plt.show()


# ============================================
# 7C. VISUALISASI 3D: Radio - Newspaper - Sales
# ============================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data["Radio"], data["Newspaper"], y, c='red', label="Actual Sales")

# Grid untuk plane
radio_range = np.linspace(data["Radio"].min(), data["Radio"].max(), 30)
news_range = np.linspace(data["Newspaper"].min(), data["Newspaper"].max(), 30)
radio_grid, news_grid = np.meshgrid(radio_range, news_range)

sales_plane_3 = (
    model.intercept_ +
    model.coef_[0] * data["TV"].mean() +  # TV jadi rata-rata
    model.coef_[1] * radio_grid +
    model.coef_[2] * news_grid
)

ax.plot_surface(radio_grid, news_grid, sales_plane_3, alpha=0.4, color='orange')
ax.set_xlabel("Radio")
ax.set_ylabel("Newspaper")
ax.set_zlabel("Sales")
ax.set_title("3D Visualization: Radio - Newspaper - Sales")
plt.show()



print("\n===== SELESAI =====")
