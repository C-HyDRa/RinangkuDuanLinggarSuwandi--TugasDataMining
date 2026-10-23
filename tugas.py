import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("Iris.csv")
print("LIMA DATA PERTAMA")
print(df.head())

print("\nUKURAN DATA")
print(df.shape)

print("\nINFORMASI DATA")
df.info()
 
print("\nDATA KOSONG")
print(df.isnull().sum())

print("\nDATA DUPLIKAT")
print(df.duplicated().sum())

print("\nSTATISTIK DESKRIPTIF")
print(df.describe())

print("\nJUMLAH SPESIES")
print(df["Species"].value_counts())

rata_rata = np.mean(df["PetalLengthCm"])
print("\nRATA-RATA PANJANG MAHKOTA")
print(rata_rata)

df["Species"].value_counts().plot(
    kind="bar",
    color="skyblue"
)
plt.title("Jumlah Data Setiap Spesies Iris")
plt.xlabel("Spesies")
plt.ylabel("Jumlah")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

for species, data in df.groupby("Species"):
    plt.scatter(
        data["PetalLengthCm"],
        data["PetalWidthCm"],
        label=species
    )

plt.title("Hubungan Panjang dan Lebar Mahkota")
plt.xlabel("Panjang Mahkota (cm)")
plt.ylabel("Lebar Mahkota (cm)")
plt.legend()
plt.tight_layout()
plt.show()