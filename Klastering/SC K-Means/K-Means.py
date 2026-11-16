import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# ======================================================
# 1. Load dataset
# ======================================================
path = kagglehub.dataset_download("zahidmughal2343/amazon-sales-2025")
print("Path to dataset files:", path)

# Cari file CSV
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_path = os.path.join(path, file)
        print("CSV file found:", csv_path)

df = pd.read_csv(csv_path)

# ======================================================
# 2. Hitung fitur untuk segmentasi produk
# ======================================================
product_freq = df.groupby("Product")["Order ID"].count().rename("Purchase_Frequency")
avg_price = df.groupby("Product")["Price"].mean().rename("Average_Price")
total_sales = df.groupby("Product")["Total Sales"].sum().rename("Total_Sales")
category = df.groupby("Product")["Category"].first().rename("Category")

product_df = pd.concat([product_freq, avg_price, total_sales, category], axis=1).reset_index()

# ======================================================
# 3. One-Hot Encoding kategori
# ======================================================

ohe = OneHotEncoder(sparse_output=False)

category_encoded = ohe.fit_transform(product_df[["Category"]])
category_cols = [f"Category_{cat}" for cat in ohe.categories_[0]]
category_df = pd.DataFrame(category_encoded, columns=category_cols)

# Gabungkan ke product_df
product_df = pd.concat([product_df, category_df], axis=1)

# Simpan mapping kategori → kolom one-hot
category_mapping = pd.DataFrame({
    "Category": ohe.categories_[0],
    "Column": category_cols
})
category_mapping.to_csv("category_mapping_onehot.csv", index=False)

# ======================================================
# 4. Pilih fitur & normalisasi
# ======================================================
features = ["Purchase_Frequency", "Average_Price", "Total_Sales"] + category_cols
scaler = StandardScaler()
scaled_features = scaler.fit_transform(product_df[features])

# ======================================================
# 5. Tentukan jumlah cluster optimal (Elbow & Silhouette)
# ======================================================
inertias = []
sil_scores = []
K_RANGE = range(2, 8)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(scaled_features)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_features, labels))

# Plot Elbow
plt.figure(figsize=(8,4))
plt.plot(K_RANGE, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.show()

# Plot Silhouette
plt.figure(figsize=(8,4))
plt.plot(K_RANGE, sil_scores, marker='o', color='orange')
plt.title("Silhouette Score per Jumlah Cluster")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Silhouette Score")
plt.show()

# ======================================================
# 6. Pilih jumlah cluster terbaik berdasarkan silhouette
# ======================================================
best_k = K_RANGE[np.argmax(sil_scores)]
print(f"Jumlah cluster terbaik menurut Silhouette Score = {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
product_df["Cluster"] = kmeans.fit_predict(scaled_features)

# ======================================================
# 7. Tambah deskripsi cluster (opsional, bisa disesuaikan)
# ======================================================
cluster_labels = {
    0: "Produk harga menengah, frekuensi tinggi",
    1: "Produk premium, total sales tinggi",
    2: "Produk terjangkau, frekuensi tinggi",
    3: "Produk murah, frekuensi rendah",
    4: "Produk sangat murah, frekuensi rendah"
}
product_df["Cluster_Description"] = product_df["Cluster"].map(cluster_labels).fillna("Cluster lain")

# ======================================================
# 8. Summary cluster
# ======================================================
numeric_cols = ["Purchase_Frequency", "Average_Price", "Total_Sales"]
print("\n=== CLUSTER SUMMARY (NUMERIK) ===")
print(product_df.groupby("Cluster")[numeric_cols].mean())

# ======================================================
# 9. Visualisasi PCA 2D
# ======================================================
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
product_df["PC1"] = reduced_features[:,0]
product_df["PC2"] = reduced_features[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=product_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=60)
plt.title("Product Clustering PCA 2D")
plt.show()

# ======================================================
# 10. Visualisasi 3D
# ======================================================
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    product_df["Purchase_Frequency"],
    product_df["Average_Price"],
    product_df["Total_Sales"],
    c=product_df["Cluster"],
    cmap="tab10",
    s=60
)
ax.set_xlabel("Purchase Frequency")
ax.set_ylabel("Average Price")
ax.set_zlabel("Total Sales")
ax.set_title("3D Product Clustering")
plt.show()

# ======================================================
# 11. Evaluasi Silhouette
# ======================================================
sil_score = silhouette_score(scaled_features, product_df["Cluster"])
print(f"\nSilhouette Score akhir = {sil_score:.3f}")

if sil_score < 0.25:
    print("→ Clustering buruk / kurang terpisah")
elif sil_score < 0.5:
    print("→ Clustering cukup, masih overlap")
elif sil_score < 0.7:
    print("→ Clustering bagus!")
else:
    print("→ Clustering sangat bagus dan terpisah jelas!")

# ======================================================
# 12. Simpan hasil
# ======================================================
product_df.to_csv("hasil_segmentasi_produk.csv", index=False)
df_with_cluster = df.merge(
    product_df[["Product", "Cluster", "Cluster_Description"]],
    on="Product",
    how="left"
)
df_with_cluster.to_csv("data_transaksi_cluster.csv", index=False)

print("\nFile hasil segmentasi disimpan.")
