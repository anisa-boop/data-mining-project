import kagglehub
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# =============================================
# 1. DOWNLOAD DATASET 
# =============================================
path = kagglehub.dataset_download("heeraldedhia/groceries-dataset")
print("Path dataset:", path)

# =============================================
# 2. LOAD DATASET
# =============================================
print(os.listdir(path))

file_path = os.path.join(path, "Groceries_dataset.csv")

data = pd.read_csv(file_path)
print("\n5 Data Teratas:")
print(data.head())
print("\nJumlah transaksi:", len(data))

# =============================================
# 3. KONVERSI DATA MENJADI TRANSAKSI
# =============================================
transactions = data.groupby("Member_number")["itemDescription"].apply(list).tolist()

print("\nContoh 5 transaksi:")
for t in transactions[:5]:
    print(t)

# =============================================
# 4. ONE HOT ENCODER
# =============================================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print("\nMatriks One Hot Encoding:")
print(df.head())

# =============================================
# 5. FP-GROWTH
# =============================================
frequent_itemsets = fpgrowth(df, min_support=0.02, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

# =============================================
# 6. ASSOCIATION RULES
# =============================================
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
print("\nAssociation Rules:")
print(rules)

# =============================================
# 7. RULE TERKUAT
# =============================================
strong_rules = rules.sort_values(by="lift", ascending=False)
print("\n10 Rule Terkuat Berdasarkan Lift:")
print(strong_rules.head(10))