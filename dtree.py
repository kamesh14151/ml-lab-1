# ============================================
# DECISION TREE - LOAN APPROVAL (Interactive)
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA
df = pd.read_csv("loan_approval_decision_tree_1000.csv").dropna()

features = ["Age", "Income", "Credit_Score", "Loan_Amount", "Employment_Years",
            "Existing_Loan", "Loan_Term", "Debt_to_Income"]

X = df[features]
encoder = LabelEncoder()
y = encoder.fit_transform(df["Loan_Approved"])

# 2. ENTROPY & INFORMATION GAIN
def entropy(data):
    _, counts = np.unique(data, return_counts=True)
    p = counts / len(data)
    return -np.sum(p * np.log2(p))

def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted = sum((c/len(y)) * entropy(y[X[feature] == v]) for v, c in zip(values, counts))
    return total_entropy - weighted

print("\nDataset Entropy:", round(entropy(y), 4))

print("\nInformation Gain per feature:")
gains = {}
for f in features:
    temp = X[f].copy()
    if temp.nunique() > 10:
        temp = pd.qcut(temp, q=4, duplicates="drop")
    gains[f] = information_gain(pd.DataFrame({f: temp}), y, f)
    print(f, ":", round(gains[f], 4))

best_feature = max(gains, key=gains.get)
print("\nBest Feature:", best_feature, "| Gain:", round(gains[best_feature], 4))

# 3. TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tree = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(f"\nAccuracy: {round(accuracy_score(y_test, y_pred)*100, 2)}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

print("\nFeature Importance:")
for f, imp in zip(features, tree.feature_importances_):
    print(f, ":", round(imp, 4))

# 4. ASK USER FOR NEW APPLICANT
print("\nEnter new applicant details:")
user_data = [float(input(f"{f}: ")) for f in features]

new_applicant = pd.DataFrame([user_data], columns=features)
pred = tree.predict(new_applicant)
label = encoder.inverse_transform(pred)[0]
confidence = np.max(tree.predict_proba(new_applicant)) * 100

print("\nPredicted Loan Status:", label)
print("Prediction Confidence:", round(confidence, 2), "%")
