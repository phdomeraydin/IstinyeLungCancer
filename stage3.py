# =========================================
# FULL ADVANCED ANALYSIS PIPELINE (CLEANED)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel("dataset.xlsx")
df = df.sort_values("Year").reset_index(drop=True)

# Remove rows with missing critical values
df = df.dropna(subset=["HospitalBasedPrevalence", "NormalizedDiagnosisRate"])

# =========================================
# 1. ANNUAL % CHANGE (PREVALENCE)
# =========================================

df["Prevalence_Annual_%_Change"] = df["HospitalBasedPrevalence"].pct_change() * 100

plt.figure()
plt.plot(df["Year"], df["Prevalence_Annual_%_Change"])
plt.title("Annual Percentage Change in Hospital-Based Prevalence")
plt.xlabel("Year")
plt.ylabel("Annual % Change")
plt.xticks(rotation=45)
plt.show()

annual_change_summary = df["Prevalence_Annual_%_Change"].describe()

# =========================================
# 2. 2021 SPIKE SENSITIVITY ANALYSIS
# =========================================

model_full = LinearRegression()
model_full.fit(df[["Year"]], df["NormalizedDiagnosisRate"])
full_slope = model_full.coef_[0]

df_no_2021 = df[df["Year"] != 2021]

model_no_2021 = LinearRegression()
model_no_2021.fit(df_no_2021[["Year"]], df_no_2021["NormalizedDiagnosisRate"])
no_2021_slope = model_no_2021.coef_[0]

sensitivity_results = pd.DataFrame({
    "Model": ["Full Model", "Excluding 2021"],
    "Slope": [full_slope, no_2021_slope]
})

# =========================================
# 3. SURVIVAL TREND ANALYSIS
# =========================================

if "MeanSurvivalMonth" in df.columns:
    df_surv = df.dropna(subset=["MeanSurvivalMonth"])
    plt.figure()
    plt.plot(df_surv["Year"], df_surv["MeanSurvivalMonth"])
    plt.title("Mean Survival Months Over Time")
    plt.xlabel("Year")
    plt.ylabel("Mean Survival (Months)")
    plt.xticks(rotation=45)
    plt.show()
    survival_summary = df_surv["MeanSurvivalMonth"].describe()
else:
    survival_summary = "MeanSurvivalMonth column not found."

# =========================================
# 4. MACHINE LEARNING FORECAST (2026–2030)
# =========================================

model_forecast = LinearRegression()
model_forecast.fit(df[["Year"]], df["HospitalBasedPrevalence"])

future_years = pd.DataFrame({"Year": np.arange(2026, 2031)})
future_predictions = model_forecast.predict(future_years)

forecast_df = pd.DataFrame({
    "Year": future_years["Year"],
    "ForecastedPrevalence": future_predictions
})

plt.figure()
plt.plot(df["Year"], df["HospitalBasedPrevalence"])
plt.plot(forecast_df["Year"], forecast_df["ForecastedPrevalence"])
plt.title("Forecasted Hospital-Based Prevalence (2026–2030)")
plt.xlabel("Year")
plt.ylabel("Hospital-Based Prevalence")
plt.xticks(rotation=45)
plt.show()

annual_change_summary, sensitivity_results, survival_summary, forecast_df
