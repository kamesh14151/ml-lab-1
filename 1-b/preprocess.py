# ==========================================================
# DATA COLLECTION AND PREPROCESSING TECHNIQUES
# ==========================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_excel(
    "public datasets/Waste_Segregation_Dataset_1000_With_Duplicates (1).xlsx",
    engine="openpyxl"
)

# ==========================================================
# Display Dataset
# ==========================================================

print("\n========== ORIGINAL DATASET ==========")
print(df.head())

# ==========================================================
# Dataset Information
# ==========================================================

print("\n========== DATASET INFORMATION ==========")
df.info()

# ==========================================================
# Missing Values
# ==========================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ==========================================================
# Handle Missing Values
# ==========================================================

# Numerical Columns
df["Weight_kg"] = df["Weight_kg"].fillna(df["Weight_kg"].mean())
df["Moisture"] = df["Moisture"].fillna(df["Moisture"].mean())
df["Size_cm"] = df["Size_cm"].fillna(df["Size_cm"].mean())

# Categorical Columns
df["Material"] = df["Material"].fillna(df["Material"].mode()[0])

# ==========================================================
# Remove Duplicates
# ==========================================================

print("\nDuplicates Before :", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates After  :", df.duplicated().sum())

# ==========================================================
# Data Types
# ==========================================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# ==========================================================
# Mean
# ==========================================================

print("\n========== MEAN ==========")
print(df[["Weight_kg", "Moisture", "Size_cm"]].mean())

# ==========================================================
# Variance
# ==========================================================

print("\n========== VARIANCE ==========")
print(df[["Weight_kg", "Moisture", "Size_cm"]].var())

# ==========================================================
# Standard Deviation
# ==========================================================

print("\n========== STANDARD DEVIATION ==========")
print(df[["Weight_kg", "Moisture", "Size_cm"]].std())

# ==========================================================
# Label Encoding
# ==========================================================

encoder = LabelEncoder()

columns = [
    "Waste_Item",
    "Material",
    "Biodegradable",
    "Recyclable",
    "Color",
    "Shape",
    "Texture",
    "Odor",
    "Collection_Source",
    "Bin_Colour",
    "Category"
]

for col in columns:
    df[col] = encoder.fit_transform(df[col])

print("\n========== ENCODED DATASET ==========")
print(df.head())

# ==========================================================
# Standard Scaling
# ==========================================================

standard_df = df.copy()

scaler = StandardScaler()

standard_df[["Weight_kg", "Moisture", "Size_cm"]] = scaler.fit_transform(
    standard_df[["Weight_kg", "Moisture", "Size_cm"]]
)

print("\n========== DATASET AFTER STANDARD SCALER ==========")
print(standard_df.head())

# ==========================================================
# Min-Max Scaling
# ==========================================================

minmax_df = df.copy()

minmax = MinMaxScaler()

minmax_df[["Weight_kg", "Moisture", "Size_cm"]] = minmax.fit_transform(
    minmax_df[["Weight_kg", "Moisture", "Size_cm"]]
)

print("\n========== DATASET AFTER MIN-MAX SCALER ==========")
print(minmax_df.head())

# ==========================================================
# Save Processed Dataset
# ==========================================================

standard_df.to_excel(
    "Waste_Standard_Scaled.xlsx",
    index=False,
    engine="openpyxl"
)

minmax_df.to_excel(
    "Waste_MinMax_Scaled.xlsx",
    index=False,
    engine="openpyxl"
)

print("\n========================================")
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("========================================")
print("Files Saved:")
print("1. Waste_Standard_Scaled.xlsx")
print("2. Waste_MinMax_Scaled.xlsx")
