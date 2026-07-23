# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Load Dataset
df = pd.read_excel("Waste_Segregation_Final_Dataset.xlsx")

# Display Dataset
print("Original Dataset")
print(df)

# Dataset Information
print("\nDataset Information")
print(df.info())

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Values (Numerical)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fill Missing Values (Categorical)
df["Material"].fillna(df["Material"].mode()[0], inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Convert Data Types
df["Weight_kg"] = df["Weight_kg"].astype(float)
df["Moisture"] = df["Moisture"].astype(float)

# Mean
print("\nMean")
print(np.mean(df[["Weight_kg", "Moisture"]]))

# Standard Deviation
print("\nStandard Deviation")
print(np.std(df[["Weight_kg", "Moisture"]]))

# --------------------------------
# Encode Categorical Data
# --------------------------------
le = LabelEncoder()


df["Biodegradable"] = le.fit_transform(df["Biodegradable"])
df["Recyclable"] = le.fit_transform(df["Recyclable"])
df["Bin_Colour"] = le.fit_transform(df["Bin_Colour"])
df["Category"] = le.fit_transform(df["Category"])

print("\nEncoded Dataset")
print(df)

# --------------------------------
# Standard Scaling
# --------------------------------
df_standard = df.copy()

standard = StandardScaler()

df_standard[["Weight_kg", "Moisture"]] = standard.fit_transform(
    df_standard[["Weight_kg", "Moisture"]]
)

print("\nDataset after StandardScaler")
print(df_standard)

# --------------------------------
# Min-Max Scaling
# --------------------------------
df_minmax = df.copy()

minmax = MinMaxScaler()

df_minmax[["Weight_kg", "Moisture"]] = minmax.fit_transform(
    df_minmax[["Weight_kg", "Moisture"]]
)

print("\nDataset after MinMaxScaler")
print(df_minmax)

# Save Processed Datasets
df_standard.to_excel("our team/Waste_Standard_Scaled.xlsx", index=False)
df_minmax.to_excel("Waste_MinMax_Scaled.xlsx", index=False)

print("\nDatasets Saved Successfully")
