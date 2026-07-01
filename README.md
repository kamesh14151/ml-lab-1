# Experiment 1: Data Collection and Preprocessing Techniques

## Objective

To understand and implement various data collection and preprocessing techniques using Python and machine learning libraries.

---

## 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

### Library Purpose

| Library        | Purpose                                           |
| -------------- | ------------------------------------------------- |
| Pandas         | Data manipulation and analysis                    |
| NumPy          | Numerical operations and handling missing values  |
| LabelEncoder   | Converts categorical values into numerical values |
| StandardScaler | Performs feature scaling and normalization        |

---

# Method 1: Creating Dataset Inside Python

## Step 1: Create Dataset

```python
data = {
    "Student_ID": [101, 102, 103, 104, 105, 105],
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha", "Esha"],
    "Age": [20, 21, np.nan, 22, 20, 20],
    "Gender": ["Male", "Male", "Male", "Female", "Female", "Female"],
    "Department": ["CSE", "ECE", "CSE", "IT", "AIML", "AIML"],
    "Marks": [85, 90, 88, np.nan, 91, 91]
}

df = pd.DataFrame(data)
```

---

## Step 2: Display Original Dataset

```python
print("Original Dataset")
print(df)
```

### Sample Output

| Student_ID | Name   | Age | Gender | Department | Marks |
| ---------- | ------ | --- | ------ | ---------- | ----- |
| 101        | Arun   | 20  | Male   | CSE        | 85    |
| 102        | Bala   | 21  | Male   | ECE        | 90    |
| 103        | Charan | NaN | Male   | CSE        | 88    |
| 104        | Divya  | 22  | Female | IT         | NaN   |
| 105        | Esha   | 20  | Female | AIML       | 91    |
| 105        | Esha   | 20  | Female | AIML       | 91    |

---

## Step 3: Display Dataset Information

```python
df.info()
```

This provides:

* Number of rows and columns
* Data types of each column
* Non-null values count
* Memory usage

---

## Step 4: Statistical Summary

```python
df.describe()
```

The `describe()` function generates:

* Mean
* Standard Deviation
* Minimum Value
* Maximum Value
* Quartiles (25%, 50%, 75%)

---

## Step 5: Check Missing Values

```python
print(df.isnull().sum())
```

### Output

```text
Student_ID    0
Name          0
Age           1
Gender        0
Department    0
Marks         1
dtype: int64
```

---

## Step 6: Handle Missing Values

Replace missing values using the mean of the respective column.

```python
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
```

---

## Step 7: Remove Duplicate Records

Check and remove duplicate rows.

```python
print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()
```

---

## Step 8: Convert Data Types

Convert the `Age` column from float to integer.

```python
df["Age"] = df["Age"].astype(int)
```

---

## Step 9: Encode Categorical Data

Convert categorical values into numerical format.

```python
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Department"] = le.fit_transform(df["Department"])
```

### Example Encoding

#### Gender

| Original | Encoded |
| -------- | ------- |
| Female   | 0       |
| Male     | 1       |

#### Department

| Original | Encoded |
| -------- | ------- |
| AIML     | 0       |
| CSE      | 1       |
| ECE      | 2       |
| IT       | 3       |

---

## Step 10: Feature Scaling

Normalize numerical values using Standard Scaling.

```python
scaler = StandardScaler()

df[["Age", "Marks"]] = scaler.fit_transform(
    df[["Age", "Marks"]]
)
```

### Formula Used

```text
Z = (X - Mean) / Standard Deviation
```

This ensures all numerical features have:

* Mean = 0
* Standard Deviation = 1

---

## Step 11: Display Processed Dataset

```python
print("Processed Dataset")
print(df)
```

---

## Step 12: Save Dataset

Save the processed dataset into a CSV file.

```python
df.to_csv("Processed_Dataset.csv", index=False)

print("Dataset Saved Successfully")
```

---

# Preprocessing Operations Performed

✅ Missing Value Handling
✅ Duplicate Removal
✅ Data Type Conversion
✅ Label Encoding
✅ Feature Scaling
✅ Dataset Export

---

# Learning Outcomes

After completing this experiment, students will be able to:

* Create datasets using Python dictionaries.
* Identify and handle missing values.
* Detect and remove duplicate records.
* Convert data types appropriately.
* Encode categorical attributes into numerical form.
* Perform feature scaling using standardization techniques.
* Save processed datasets for machine learning applications.

---

# Result

The dataset was successfully preprocessed by handling missing values, removing duplicates, encoding categorical variables, and scaling numerical features. The cleaned dataset was then exported as a CSV file for further machine learning tasks.
