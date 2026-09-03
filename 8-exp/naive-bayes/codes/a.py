import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_excel("naive_bayes_1000_samples.xlsx")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# --------------------------------------------------
# 2. CHECK DATA
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

# --------------------------------------------------
# 3. PREPROCESSING
# --------------------------------------------------

# Remove Sample_ID because it is not useful for prediction
df = df.drop("Sample_ID", axis=1)

# Create label encoders
weather_encoder = LabelEncoder()
temperature_encoder = LabelEncoder()
humidity_encoder = LabelEncoder()
wind_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# Encode categorical columns
df["Weather"] = weather_encoder.fit_transform(df["Weather"])
df["Temperature"] = temperature_encoder.fit_transform(df["Temperature"])
df["Humidity"] = humidity_encoder.fit_transform(df["Humidity"])
df["Wind"] = wind_encoder.fit_transform(df["Wind"])

# Encode target column
df["Play_Sports"] = target_encoder.fit_transform(df["Play_Sports"])

print("\nAfter Encoding:")
print(df.head())

# --------------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# --------------------------------------------------

X = df[["Weather", "Temperature", "Humidity", "Wind"]]

y = df["Play_Sports"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# --------------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# --------------------------------------------------
# 6. TRAIN NAIVE BAYES MODEL
# --------------------------------------------------

model = CategoricalNB()

model.fit(X_train, y_train)

print("\nNaive Bayes model training completed.")

# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values[:20])

print("\nPredicted Values:")
print(y_pred[:20])

# --------------------------------------------------
# 8. ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# --------------------------------------------------
# 9. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_encoder.classes_
)

disp.plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# --------------------------------------------------
# 10. CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# --------------------------------------------------
# 11. PREDICT A NEW SAMPLE
# --------------------------------------------------

# Example:
# Weather = Sunny
# Temperature = Cool
# Humidity = Normal
# Wind = Weak

new_sample = pd.DataFrame({
    "Weather": [
        weather_encoder.transform(["Sunny"])[0]
    ],
    "Temperature": [
        temperature_encoder.transform(["Cool"])[0]
    ],
    "Humidity": [
        humidity_encoder.transform(["Normal"])[0]
    ],
    "Wind": [
        wind_encoder.transform(["Weak"])[0]
    ]
})

prediction = model.predict(new_sample)

predicted_class = target_encoder.inverse_transform(prediction)

print("\nNew Sample:")
print("Weather = Sunny")
print("Temperature = Cool")
print("Humidity = Normal")
print("Wind = Weak")

print("\nPrediction:")
print(predicted_class[0])
