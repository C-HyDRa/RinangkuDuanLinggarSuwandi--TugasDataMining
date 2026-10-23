from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = Path(__file__).resolve().parent / "dataset_tambak_ikan.csv"
data = pd.read_csv(DATASET_PATH)

print("=== Informasi dataset ===")
print(f"Nama file: {DATASET_PATH.name}")
print(f"Jumlah baris: {data.shape[0]}")
print(f"Jumlah kolom: {data.shape[1]}")
print("\nTipe data dan jumlah nilai terisi:")
data.info()

categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

print("\n=== Kolom kategorikal: nilai urut dan jumlahnya ===")
for column in categorical_columns:
	counts = data[column].value_counts(dropna=False).sort_index()
	print(f"\n{column} ({data[column].nunique(dropna=True)} kategori unik):")
	if len(counts) > 20:
		print(f"Menampilkan 20 dari {len(counts)} nilai, diurutkan berdasarkan label:")
		counts = counts.head(20)
	print(counts.to_string())

print("\n=== Kolom numerikal: statistik dan 10 nilai terkecil ===")
for column in numeric_columns:
	sorted_values = data[column].dropna().sort_values()
	print(f"\n{column} (jumlah nilai terisi: {sorted_values.count()}):")
	print(data[column].describe().to_string())
	print("10 nilai terkecil:")
	print(sorted_values.head(10).to_list())

print("\n=== Data terurut ===")
sorted_categorical_data = data.sort_values(by=categorical_columns, na_position="last")
sorted_numeric_data = data.sort_values(by=numeric_columns, na_position="last")
print("\n10 baris pertama setelah diurutkan berdasarkan kolom kategorikal:")
print(sorted_categorical_data.head(10).to_string(index=False))
print("\n10 baris pertama setelah diurutkan berdasarkan kolom numerikal:")
print(sorted_numeric_data.head(10).to_string(index=False))

duplicate_count = int(data.duplicated().sum())
null_counts = data.isna().sum()
zero_counts = pd.Series(
	{column: int(np.count_nonzero(data[column].dropna().to_numpy() == 0)) for column in numeric_columns}
)

print("\n=== Pemeriksaan kualitas data ===")
print(f"Jumlah baris duplikat: {duplicate_count}")
print(f"Jumlah seluruh nilai null: {int(null_counts.sum())}")
print("Jumlah nilai null per kolom:")
print(null_counts.to_string())
print(f"Jumlah seluruh nilai zero pada kolom numerikal: {int(zero_counts.sum())}")
print("Jumlah nilai zero per kolom numerikal:")
print(zero_counts.to_string())

encoder = OneHotEncoder(handle_unknown="ignore")
encoded_categories = encoder.fit_transform(data[categorical_columns])
print("\n=== Persiapan data scikit-learn ===")
print(f"Ukuran matriks hasil OneHotEncoder: {encoded_categories.shape}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
data["jenis_ikan"].value_counts().sort_values(ascending=False).plot(
	kind="bar", ax=axes[0], color="steelblue"
)
axes[0].set_title("Jumlah Data per Jenis Ikan")
axes[0].set_xlabel("Jenis ikan")
axes[0].set_ylabel("Jumlah")
axes[0].tick_params(axis="x", rotation=45)

axes[1].hist(data["estimasi_panen_kg"].dropna().to_numpy(), bins=20, color="seagreen", edgecolor="white")
axes[1].set_title("Distribusi Estimasi Panen")
axes[1].set_xlabel("Estimasi panen (kg)")
axes[1].set_ylabel("Frekuensi")

fig.tight_layout()
plt.show()