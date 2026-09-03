# ============================================================
# ICE CREAM SALES - TIME SERIES TREND ANALYSIS
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("ice_cream_trend_analysis_dataset.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1]) 


# ============================================================
# 3. UNDERSTAND THE DATASET
# ============================================================

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# Remove rows containing missing values if any exist
df = df.dropna()


# ============================================================
# 5. CONVERT DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

# Sort the data chronologically
df = df.sort_values("Date")

print("\n========== DATA AFTER DATE CONVERSION ==========")
print(df.head())


# ============================================================
# 6. CHECK DUPLICATE VALUES
# ============================================================

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# Remove duplicates
df = df.drop_duplicates()


# ============================================================
# 7. BASIC INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")
print(df.info())


# ============================================================
# 8. SALES TREND OVER TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Ice_Cream_Sales"],
    label="Daily Ice Cream Sales"
)

plt.xlabel("Date")
plt.ylabel("Ice Cream Sales")
plt.title("Ice Cream Sales Trend Over Time")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. 30-DAY MOVING AVERAGE
# ============================================================

df["30_Day_Moving_Average"] = (
    df["Ice_Cream_Sales"]
    .rolling(window=30)
    .mean()
)

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Ice_Cream_Sales"],
    label="Daily Sales",
    alpha=0.4
)

plt.plot(
    df["Date"],
    df["30_Day_Moving_Average"],
    label="30-Day Moving Average"
)

plt.xlabel("Date")
plt.ylabel("Ice Cream Sales")
plt.title("Ice Cream Sales Trend with 30-Day Moving Average")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. TEMPERATURE VS ICE CREAM SALES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Temperature_C"],
    df["Ice_Cream_Sales"],
    alpha=0.6
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales")
plt.title("Temperature vs Ice Cream Sales")

plt.tight_layout()
plt.show()


# ============================================================
# 11. CALCULATE CORRELATION
# ============================================================

correlation = df["Temperature_C"].corr(
    df["Ice_Cream_Sales"]
)

print("\n========== TEMPERATURE CORRELATION ==========")
print("Correlation:", round(correlation, 3))

if correlation > 0.7:
    print("Strong positive relationship")

elif correlation > 0.3:
    print("Moderate positive relationship")

elif correlation > -0.3:
    print("Weak or no linear relationship")

elif correlation > -0.7:
    print("Moderate negative relationship")

else:
    print("Strong negative relationship")


# ============================================================
# 12. SALES BY SEASON
# ============================================================

season_sales = (
    df.groupby("Season")["Ice_Cream_Sales"]
    .mean()
)

print("\n========== AVERAGE SALES BY SEASON ==========")
print(season_sales)


plt.figure(figsize=(8, 6))

season_sales.plot(kind="bar")

plt.xlabel("Season")
plt.ylabel("Average Ice Cream Sales")
plt.title("Average Ice Cream Sales by Season")

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 13. SALES BY DAY OF WEEK
# ============================================================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_sales = (
    df.groupby("Day")["Ice_Cream_Sales"]
    .mean()
    .reindex(day_order)
)

print("\n========== AVERAGE SALES BY DAY ==========")
print(day_sales)


plt.figure(figsize=(10, 6))

day_sales.plot(kind="bar")

plt.xlabel("Day")
plt.ylabel("Average Ice Cream Sales")
plt.title("Average Ice Cream Sales by Day")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 14. MONTHLY SALES TREND
# ============================================================

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

monthly_sales = (
    df.groupby("Month")["Ice_Cream_Sales"]
    .mean()
    .reindex(month_order)
)

print("\n========== AVERAGE SALES BY MONTH ==========")
print(monthly_sales)


plt.figure(figsize=(12, 6))

monthly_sales.plot(
    kind="line",
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Average Ice Cream Sales")
plt.title("Monthly Ice Cream Sales Trend")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 15. FIND HIGHEST SALES
# ============================================================

highest_sales = df.loc[
    df["Ice_Cream_Sales"].idxmax()
]

print("\n========== HIGHEST SALES ==========")
print("Date:", highest_sales["Date"])
print("Day:", highest_sales["Day"])
print("Season:", highest_sales["Season"])
print("Temperature:", highest_sales["Temperature_C"], "°C")
print("Sales:", highest_sales["Ice_Cream_Sales"])


# ============================================================
# 16. FIND LOWEST SALES
# ============================================================

lowest_sales = df.loc[
    df["Ice_Cream_Sales"].idxmin()
]

print("\n========== LOWEST SALES ==========")
print("Date:", lowest_sales["Date"])
print("Day:", lowest_sales["Day"])
print("Season:", lowest_sales["Season"])
print("Temperature:", lowest_sales["Temperature_C"], "°C")
print("Sales:", lowest_sales["Ice_Cream_Sales"])


# ============================================================
# 17. CREATE DAY NUMBER FOR TREND MODEL
# ============================================================

df["Day_Number"] = np.arange(len(df))


# ============================================================
# 18. LINEAR REGRESSION TREND ANALYSIS
# ============================================================

X = df[["Day_Number"]]
y = df["Ice_Cream_Sales"]

model = LinearRegression()

model.fit(X, y)

# Predict sales based on the trend
trend_prediction = model.predict(X)


# ============================================================
# 19. CALCULATE TREND SLOPE
# ============================================================

slope = model.coef_[0]
intercept = model.intercept_

print("\n========== LINEAR TREND ANALYSIS ==========")

print("Slope:", round(slope, 3))
print("Intercept:", round(intercept, 3))

if slope > 0:
    print("Overall trend: INCREASING")

elif slope < 0:
    print("Overall trend: DECREASING")

else:
    print("Overall trend: STABLE")


# ============================================================
# 20. VISUALIZE LINEAR TREND
# ============================================================

plt.figure(figsize=(12, 6))

plt.scatter(
    df["Date"],
    df["Ice_Cream_Sales"],
    alpha=0.4,
    label="Actual Sales"
)

plt.plot(
    df["Date"],
    trend_prediction,
    label="Trend Line"
)

plt.xlabel("Date")
plt.ylabel("Ice Cream Sales")
plt.title("Overall Ice Cream Sales Trend")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 21. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y,
    trend_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        trend_prediction
    )
)

print("\n========== MODEL EVALUATION ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


# ============================================================
# 22. TEMPERATURE → SALES LINEAR REGRESSION
# ============================================================

X_temperature = df[["Temperature_C"]]
y_sales = df["Ice_Cream_Sales"]

temperature_model = LinearRegression()

temperature_model.fit(
    X_temperature,
    y_sales
)

temperature_prediction = temperature_model.predict(
    X_temperature
)

temperature_slope = temperature_model.coef_[0]

print("\n========== TEMPERATURE-SALES MODEL ==========")

print(
    "Sales increase/decrease per 1°C:",
    round(temperature_slope, 2)
)


# ============================================================
# 23. VISUALIZE TEMPERATURE-SALES TREND
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Temperature_C"],
    df["Ice_Cream_Sales"],
    alpha=0.5,
    label="Actual Data"
)

# Sort values so the regression line is displayed correctly
sorted_df = df.sort_values("Temperature_C")

sorted_prediction = temperature_model.predict(
    sorted_df[["Temperature_C"]]
)

plt.plot(
    sorted_df["Temperature_C"],
    sorted_prediction,
    label="Regression Trend"
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales")
plt.title("Temperature vs Ice Cream Sales Trend")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 24. FINAL TREND SUMMARY
# ============================================================

print("\n")
print("================================================")
print("             FINAL TREND ANALYSIS")
print("================================================")

print(
    "\nOverall Sales Trend Slope:",
    round(slope, 3)
)

print(
    "Temperature-Sales Correlation:",
    round(correlation, 3)
)

print(
    "Highest Sales:",
    highest_sales["Ice_Cream_Sales"]
)

print(
    "Lowest Sales:",
    lowest_sales["Ice_Cream_Sales"]
)

print(
    "Highest Sales Season:",
    season_sales.idxmax()
)

print(
    "Highest Sales Day:",
    day_sales.idxmax()
)

print(
    "Highest Sales Month:",
    monthly_sales.idxmax()
)

print("\n================================================")
print("                 CONCLUSION")
print("================================================")

if slope > 0:
    print(
        "The overall ice cream sales trend is increasing over time."
    )
elif slope < 0:
    print(
        "The overall ice cream sales trend is decreasing over time."
    )
else:
    print(
        "The overall ice cream sales trend is relatively stable."
    )

if correlation > 0:
    print(
        "Temperature has a positive relationship with ice cream sales."
    )
elif correlation < 0:
    print(
        "Temperature has a negative relationship with ice cream sales."
    )
else:
    print(
        "Temperature has little linear relationship with ice cream sales."
    )

print(
    "The analysis also compares sales across seasons, days, and months."
)

print("\n================================================")
print("             ANALYSIS COMPLETED")
print("================================================")
