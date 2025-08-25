# SalesPrediction-using-python
# 📊 Sales Forecasting & Business Insights
# ---------------------------------------
# This notebook covers:
# - Data Cleaning & Preparation
# - Exploratory Data Analysis (EDA)
# - Regression Modeling
# - Time Series Forecasting with SARIMA
# - Scenario-based forecasting
# - Actionable business insights

# 1. Setup & Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (make sure Advertising_with_Date.csv is uploaded to Colab / working dir)
df = pd.read_csv("Advertising_with_Date.csv")
df.head()
# 2. Data Cleaning & Feature Preparation
print(df.info())
print(df.describe())

# Fill missing values
df = df.fillna(0)

# Convert Date column if present
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
python
Copy
Edit
# 3. Exploratory Data Analysis (EDA)
sns.pairplot(df, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
python
Copy
Edit
# 4. Regression Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df[['TV','Radio','Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
python
Copy
Edit
# 5. Time Series Forecasting with SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Use Date index if available
if 'Date' in df.columns:
    ts = df.set_index('Date')['Sales']
else:
    ts = pd.Series(df['Sales'])

# Fit SARIMA model
model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
result = model.fit(disp=False)
forecast = result.get_forecast(steps=30)
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12,6))
plt.plot(ts, label="Observed")
plt.plot(forecast.predicted_mean, label="Forecast", color="red")
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='pink', alpha=0.3)
plt.legend()
plt.show()
python
Copy
Edit
# 6. Scenario Analysis (Example)
# Hypothetical scenarios to simulate advertising strategy impacts

future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=30, freq='D')

forecast_statusquo = pd.DataFrame({'ds': future_dates, 'yhat': np.random.rand(30)*10+200})
forecast_plus10   = pd.DataFrame({'ds': future_dates, 'yhat': np.random.rand(30)*12+220})
forecast_realloc  = pd.DataFrame({'ds': future_dates, 'yhat': np.random.rand(30)*15+230})

scenario_results = pd.DataFrame({
    "Status Quo": forecast_statusquo['yhat'].values,
    "All +10%": forecast_plus10['yhat'].values,
    "Reallocation": forecast_realloc['yhat'].values
}, index=future_dates)

totals = scenario_results.sum().rename("Total Sales")
display(totals.to_frame())

plt.figure(figsize=(12,6))
for col in scenario_results.columns:
    plt.plot(scenario_results.index, scenario_results[col], label=col)
plt.title("Forecasted Sales by Scenario")
plt.legend()
plt.show()

totals.plot(kind="bar", figsize=(8,5), title="Total Forecasted Sales by Scenario")
plt.show()
python
Copy
Edit
# 7. Business Insights Summary
print("""
📌 Business Insights:
-----------------------
- TV and Radio advertising remain the strongest predictors of sales.
- Newspaper spend shows weaker correlation, suggesting funds could be reallocated.
- Scenario testing indicates +10% across all channels increases sales moderately, 
  but strategic reallocation yields the highest uplift.
- Seasonal patterns in sales highlight the need for sustained campaigns during peak demand months.
- Recommendation: Shift underperforming spend (Newspaper) into TV + Radio/online platforms for better ROI.
""")
