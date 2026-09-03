# ============================================
# KNN - HEALTH RISK CLASSIFICATION (Interactive)
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA
df = pd.read_csv("health_risk_knn_1000.csv")

features = ["Age", "BMI", "Systolic_BP", "Cholesterol", "Heart_Rate",
            "Glucose", "Smoking", "Physical_Activity_Hours", "Family_History"]

X = df[features]
y = df["Health_Risk"]

# 2. SPLIT + SCALE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. ASK USER FOR K
k = int(input("Enter value of k (number of neighbors): "))

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# 4. ACCURACY
y_pred = knn.predict(X_test_scaled)
print(f"\nAccuracy with k={k}: {round(accuracy_score(y_test, y_pred)*100, 2)}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. ASK USER FOR NEW PERSON'S DATA (single pass, with range check)
bounds = {f: (X[f].min(), X[f].max()) for f in features}

print("\nEnter details of the new person:")
user_data = []
for f in features:
    val = float(input(f"{f} (typical range {bounds[f][0]}-{bounds[f][1]}): "))
    if not (bounds[f][0] <= val <= bounds[f][1]):
        print(f"  ⚠ Warning: {f}={val} is outside the training data range — prediction may be unreliable.")
    user_data.append(val)

# 6. PREDICT
new_df = pd.DataFrame([user_data], columns=features)
new_scaled = scaler.transform(new_df)
prediction = knn.predict(new_scaled)

print("\nPredicted Health Risk:", prediction[0])
