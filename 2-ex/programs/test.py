# ===============================================
# SIMPLE LINEAR REGRESSION
# Laptop Battery Remaining Prediction
# ===============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===============================================
# Load Dataset
# ===============================================

df = pd.read_excel("lab-2/Laptop_Battery_Dataset_13250_Noisy.xlsx")

print("========== ORIGINAL DATASET ==========")
print(df.head())
print("\nDataset Shape:", df.shape)

# ===============================================
# Data Preprocessing
# ===============================================

# Dataset Information
print("\n========== DATASET INFO ==========")
print(df.info())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Convert Battery Column to Numeric
df["Battery_Remaining_Percent"] = pd.to_numeric(
    df["Battery_Remaining_Percent"],
    errors="coerce"
)

# Remove Missing Values
df = df.dropna()

# Remove Duplicate Rows
df = df.drop_duplicates()

# Remove Invalid Usage Hours
df = df[
    (df["Usage_Hours"] >= 0) &
    (df["Usage_Hours"] <= 12)
]

# Remove Invalid Battery Percentage
df = df[
    (df["Battery_Remaining_Percent"] >= 0) &
    (df["Battery_Remaining_Percent"] <= 100)
]

print("\n========== CLEANED DATASET ==========")
print(df.head())
print("\nCleaned Dataset Shape:", df.shape)

# Save Clean Dataset
df.to_excel("Laptop_Battery_Dataset_Cleaned.xlsx", index=False)

print("\nClean Dataset Saved Successfully!")

# ===============================================
# Prepare Data
# ===============================================

X = df[["Usage_Hours"]]
Y = df["Battery_Remaining_Percent"]

# ===============================================
# Split Dataset
# ===============================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42
)

# ===============================================
# Create Model
# ===============================================

model = LinearRegression()

# ===============================================
# Train Model
# ===============================================

model.fit(X_train, Y_train)

# ===============================================
# Prediction
# ===============================================

Y_pred = model.predict(X_test)

# ===============================================
# Model Evaluation
# ===============================================

r2 = r2_score(Y_test, Y_pred)

print("\n========== MODEL DETAILS ==========")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nR² Score:", r2)
print("Accuracy: {:.2f}%".format(r2 * 100))

# ===============================================
# Show Sample Predictions
# ===============================================

results = pd.DataFrame({
    "Usage Hours": X_test["Usage_Hours"].values,
    "Actual Battery (%)": Y_test.values,
    "Predicted Battery (%)": Y_pred.round(2)
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(results.head(10))

# ===============================================
# Visualization
# ===============================================

plt.figure(figsize=(10,6))

plt.scatter(
    X_train,
    Y_train,
    color="blue",
    label="Training Data",
    alpha=0.6
)

plt.scatter(
    X_test,
    Y_test,
    color="green",
    label="Testing Data",
    alpha=0.6
)

plt.plot(
    X_train,
    model.predict(X_train),
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.title("Laptop Battery Remaining Prediction")
plt.xlabel("Usage Time (Hours)")
plt.ylabel("Battery Remaining (%)")
plt.legend()
plt.grid(True)

plt.show()
