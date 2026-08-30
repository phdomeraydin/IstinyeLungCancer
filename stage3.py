# =========================================
# ADVANCED TIME-SERIES FORECASTING PIPELINE
# WITH PROPER VALIDATION AND UNCERTAINTY QUANTIFICATION
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# >>> NEW: Import time-series forecasting libraries <<<
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ARIMA_AVAILABLE = True
except ImportError:
    print("WARNING: statsmodels not installed. Install with: pip install statsmodels")
    ARIMA_AVAILABLE = False



# Load dataset
df = pd.read_excel("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/dataset.xlsx")
df = df.sort_values("Year").reset_index(drop=True)

# Keep all years with valid prevalence values for forecasting
df = df.dropna(subset=["HospitalBasedPrevalence"]).copy()

# Separate dataset for analyses requiring the normalized diagnosis rate
df_normalized = df.dropna(subset=["NormalizedDiagnosisRate"]).copy()

print("="*70)
print("ADVANCED TIME-SERIES FORECASTING ANALYSIS")
print("="*70)
print(f"\nDataset: {len(df)} years from {df['Year'].min()} to {df['Year'].max()}")

# =========================================
# 1. ANNUAL % CHANGE (PREVALENCE)
# =========================================

print("\n" + "="*70)
print("1. ANNUAL PERCENTAGE CHANGE ANALYSIS")
print("="*70)

previous_prevalence = df["HospitalBasedPrevalence"].shift(1)

df["Prevalence_Annual_%_Change"] = np.where(
    previous_prevalence > 0,
    (
        (df["HospitalBasedPrevalence"] - previous_prevalence)
        / previous_prevalence
    ) * 100,
    np.nan
)
annual_change_summary = (
    df["Prevalence_Annual_%_Change"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .describe()
)

annual_change_plot = df.dropna(
    subset=["Prevalence_Annual_%_Change"]
)

plt.figure(figsize=(12, 6))
plt.plot(
    annual_change_plot["Year"],
    annual_change_plot["Prevalence_Annual_%_Change"],
    marker='o',
    linewidth=2
)

plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title("Annual Percentage Change in Hospital-Based Prevalence", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Annual % Change", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/analysis_annual_change.png", 
            dpi=300, bbox_inches='tight')
plt.close()

annual_change_summary = df["Prevalence_Annual_%_Change"].describe()
print("\nAnnual % Change Statistics:")
print(annual_change_summary)

# =========================================
# >>> NEW: STATISTICAL TREND TEST <<<
# =========================================

print("\n" + "="*70)
print("2. STATISTICAL TREND SIGNIFICANCE TEST")
print("="*70)

from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(
    df["Year"], df["HospitalBasedPrevalence"]
)

print(f"\nLinear Regression Results:")
print(f"  Slope (annual increase): {slope:.2f} cases/year")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Standard error: {std_err:.2f}")

if p_value < 0.001:
    print(f"  → Trend is HIGHLY SIGNIFICANT (p < 0.001)")
elif p_value < 0.05:
    print(f"  → Trend is SIGNIFICANT (p < 0.05)")
else:
    print(f"  → Trend is NOT SIGNIFICANT (p >= 0.05)")

# =========================================
# 3. COVID-19 SENSITIVITY ANALYSIS
# =========================================

print("\n" + "="*70)
print("3. COVID-19 PERIOD SENSITIVITY ANALYSIS")
print("="*70)

# Model 1: Full normalized-rate dataset
model_full = LinearRegression()
model_full.fit(
    df_normalized[["Year"]],
    df_normalized["NormalizedDiagnosisRate"]
)
full_slope = model_full.coef_[0]

# Model 2: Exclude 2021
df_no_2021 = df_normalized[df_normalized["Year"] != 2021].copy()
model_no_2021 = LinearRegression()
model_no_2021.fit(
    df_no_2021[["Year"]],
    df_no_2021["NormalizedDiagnosisRate"]
)
no_2021_slope = model_no_2021.coef_[0]

# Model 3: Pre-COVID period
df_pre_covid = df_normalized[df_normalized["Year"] < 2020].copy()
model_pre_covid = LinearRegression()
model_pre_covid.fit(
    df_pre_covid[["Year"]],
    df_pre_covid["NormalizedDiagnosisRate"]
)
pre_covid_slope = model_pre_covid.coef_[0]

full_label = (
    f"Full Model ({df_normalized['Year'].min()}-"
    f"{df_normalized['Year'].max()})"
)

pre_covid_label = (
    f"Pre-COVID ({df_pre_covid['Year'].min()}-"
    f"{df_pre_covid['Year'].max()})"
)

sensitivity_results = pd.DataFrame({
    "Model": [
        full_label,
        "Excluding 2021",
        pre_covid_label
    ],
    "Slope": [
        full_slope,
        no_2021_slope,
        pre_covid_slope
    ],
    "N_Years": [
        len(df_normalized),
        len(df_no_2021),
        len(df_pre_covid)
    ]
})

print("\nSensitivity Analysis Results:")
print(sensitivity_results)

sensitivity_results.to_excel(
    "d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/sensitivity_analysis.xlsx",
    index=False
)


# =========================================
# >>> NEW: TRAIN-TEST SPLIT FOR VALIDATION <<<
# =========================================

print("\n" + "="*70)
print("5. MODEL VALIDATION WITH TRAIN-TEST SPLIT")
print("="*70)

# Split: Train on 2002-2020, Test on 2021-2025
train = df[df["Year"] <= 2020].copy()
test = df[df["Year"] > 2020].copy()

print(f"\nTrain set: {train['Year'].min()}-{train['Year'].max()} ({len(train)} years)")
print(f"Test set: {test['Year'].min()}-{test['Year'].max()} ({len(test)} years)")

# =========================================
# >>> NEW: MULTIPLE FORECASTING MODELS <<<
# =========================================

print("\n" + "="*70)
print("6. FORECASTING WITH MULTIPLE MODELS (2026-2030)")
print("="*70)

forecast_years = np.arange(2026, 2031)
future_df = pd.DataFrame({"Year": forecast_years})

all_forecasts = {}
model_performance = []

# -----------------------------------
# MODEL 1: Simple Linear Regression
# -----------------------------------
print("\n--- Model 1: Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(train[["Year"]], train["HospitalBasedPrevalence"])

# Predictions on test set
lr_test_pred = lr_model.predict(test[["Year"]])

# Calculate metrics
lr_rmse = np.sqrt(mean_squared_error(test["HospitalBasedPrevalence"], lr_test_pred))
lr_mae = mean_absolute_error(test["HospitalBasedPrevalence"], lr_test_pred)
lr_mape = mean_absolute_percentage_error(test["HospitalBasedPrevalence"], lr_test_pred) * 100

print(f"Test Set Performance:")
print(f"  RMSE: {lr_rmse:.2f}")
print(f"  MAE: {lr_mae:.2f}")
print(f"  MAPE: {lr_mape:.2f}%")


# Refit Linear Regression on the full observed series for final forecasting
lr_full_model = LinearRegression()
lr_full_model.fit(
    df[["Year"]],
    df["HospitalBasedPrevalence"]
)

# Forecast directly for 2026-2030
lr_forecast = lr_full_model.predict(
    future_df[["Year"]]
)

# Approximate 95% prediction interval based on full-series residuals
lr_full_residuals = (
    df["HospitalBasedPrevalence"]
    - lr_full_model.predict(df[["Year"]])
)

lr_std = np.std(lr_full_residuals)
lr_pi_95 = 1.96 * lr_std

all_forecasts["Linear_Regression"] = {
    "predictions": lr_forecast,
    "lower_pi": lr_forecast - lr_pi_95,
    "upper_pi": lr_forecast + lr_pi_95
}

model_performance.append({
    "Model": "Linear Regression",
    "RMSE": lr_rmse,
    "MAE": lr_mae,
    "MAPE": lr_mape
})
# =========================================
# ARIMA ORDER SELECTION
# =========================================

print("\n" + "="*70)
print("ARIMA ORDER SELECTION BASED ON AIC")
print("="*70)

selected_order = (2, 1, 2)  # fallback only
arima_order_results = []

if ARIMA_AVAILABLE:
    best_aic = np.inf
    best_bic = np.inf
    best_order = None

    # d=1 is fixed based on the ADF stationarity results.
    # p and q are evaluated over a small grid because the annual
    # training series contains a limited number of observations.
    for p in range(0, 4):
        for q in range(0, 4):

            # Avoid overly complex models for the small annual dataset
            if p + q > 4:
                continue

            order = (p, 1, q)

            try:
                candidate_model = ARIMA(
                    train["HospitalBasedPrevalence"],
                    order=order
                )

                candidate_fit = candidate_model.fit()

                arima_order_results.append({
                    "p": p,
                    "d": 1,
                    "q": q,
                    "AIC": candidate_fit.aic,
                    "BIC": candidate_fit.bic
                })

                if candidate_fit.aic < best_aic:
                    best_aic = candidate_fit.aic
                    best_bic = candidate_fit.bic
                    best_order = order

            except Exception as e:
                print(f"ARIMA{order} failed: {e}")

    if best_order is not None:
        selected_order = best_order

    print(f"\nSelected ARIMA order: {selected_order}")
    print(f"Best AIC: {best_aic:.2f}")
    print(f"Corresponding BIC: {best_bic:.2f}")

    # Save all candidate models
    arima_order_df = pd.DataFrame(arima_order_results)

    if not arima_order_df.empty:
        arima_order_df = arima_order_df.sort_values("AIC")

        print("\nTop ARIMA specifications:")
        print(arima_order_df.head(10).to_string(index=False))

        arima_order_df.to_excel(
            "d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/arima_order_selection.xlsx",
            index=False
        )
# -----------------------------------
# MODEL 2: ARIMA
# -----------------------------------
if ARIMA_AVAILABLE:
    print("\n--- Model 2: ARIMA ---")
    try:
        # Fit ARIMA on training data
        arima_model = ARIMA(
            train["HospitalBasedPrevalence"],
            order=selected_order
        )

        arima_fitted = arima_model.fit()

        print(f"ARIMA Model Summary:")
        print(f"  Selected order: {selected_order}")
        print(f"  AIC: {arima_fitted.aic:.2f}")
        print(f"  BIC: {arima_fitted.bic:.2f}")
        # -----------------------------------
        # Ljung-Box residual diagnostic
        # -----------------------------------

        arima_residuals = arima_fitted.resid.dropna()

        # Lag 4 is used because the annual training series is short.
        ljung_box_result = acorr_ljungbox(
            arima_residuals,
            lags=[4],
            return_df=True
        )

        lb_stat = ljung_box_result["lb_stat"].iloc[0]
        lb_pvalue = ljung_box_result["lb_pvalue"].iloc[0]

        print("\nLjung-Box Residual Diagnostic:")
        print(f"  Lag: 4")
        print(f"  Ljung-Box statistic: {lb_stat:.4f}")
        print(f"  p-value: {lb_pvalue:.4f}")

        if lb_pvalue > 0.05:
            print("  → No significant residual autocorrelation detected.")
        else:
            print("  → Significant residual autocorrelation remains.") 
            
        # Predict on test set
        arima_test_pred = arima_fitted.forecast(steps=len(test))
        
        arima_rmse = np.sqrt(mean_squared_error(test["HospitalBasedPrevalence"], arima_test_pred))
        arima_mae = mean_absolute_error(test["HospitalBasedPrevalence"], arima_test_pred)
        arima_mape = mean_absolute_percentage_error(test["HospitalBasedPrevalence"], arima_test_pred) * 100
        
        print(f"Test Set Performance:")
        print(f"  RMSE: {arima_rmse:.2f}")
        print(f"  MAE: {arima_mae:.2f}")
        print(f"  MAPE: {arima_mape:.2f}%")
        
        # Refit on full data for final forecast
        arima_full = ARIMA(
            df["HospitalBasedPrevalence"],
            order=selected_order
        )

        arima_full_fitted = arima_full.fit()

        # The observed prevalence series ends in 2024.
        # Forecast six steps (2025-2030), then report 2026-2030.
        arima_forecast_obj_all = arima_full_fitted.get_forecast(steps=6)

        arima_forecast_all = arima_forecast_obj_all.predicted_mean
        arima_pi_all = arima_forecast_obj_all.conf_int()

        # Exclude 2025 and retain 2026-2030
        arima_forecast = arima_forecast_all.iloc[1:]
        arima_pi = arima_pi_all.iloc[1:]

        all_forecasts["ARIMA"] = {
            "predictions": arima_forecast.values,
            "lower_pi": arima_pi.iloc[:, 0].values,
            "upper_pi": arima_pi.iloc[:, 1].values
        }

        model_performance.append({
            "Model": f"ARIMA{selected_order}",
            "RMSE": arima_rmse,
            "MAE": arima_mae,
            "MAPE": arima_mape
        })       
    except Exception as e:
        print(f"ARIMA failed: {e}")

# -----------------------------------
# MODEL 3: Exponential Smoothing
# -----------------------------------
if ARIMA_AVAILABLE:
    print("\n--- Model 3: Exponential Smoothing ---")
    try:
        es_model = ExponentialSmoothing(
            train["HospitalBasedPrevalence"], 
            trend='add',
            seasonal=None
        )
        es_fitted = es_model.fit()
        
        # Predict on test
        es_test_pred = es_fitted.forecast(steps=len(test))
        
        es_rmse = np.sqrt(mean_squared_error(test["HospitalBasedPrevalence"], es_test_pred))
        es_mae = mean_absolute_error(test["HospitalBasedPrevalence"], es_test_pred)
        es_mape = mean_absolute_percentage_error(test["HospitalBasedPrevalence"], es_test_pred) * 100
        
        print(f"Test Set Performance:")
        print(f"  RMSE: {es_rmse:.2f}")
        print(f"  MAE: {es_mae:.2f}")
        print(f"  MAPE: {es_mape:.2f}%")
        
        # Refit on full data
        es_full = ExponentialSmoothing(df["HospitalBasedPrevalence"], trend='add', seasonal=None)
        es_full_fitted = es_full.fit()
        # Forecast 2025-2030 and retain 2026-2030
        es_forecast_all = es_full_fitted.forecast(steps=6)
        es_forecast = es_forecast_all.iloc[1:]
        
        # Estimate CI (ES doesn't provide built-in CI)
        es_residuals = (
            df["HospitalBasedPrevalence"]
            - es_full_fitted.fittedvalues
        )

        es_std = np.std(es_residuals)
        es_pi_95 = 1.96 * es_std        
        
        all_forecasts["Exponential_Smoothing"] = {
            "predictions": es_forecast.values,
            "lower_pi": es_forecast.values - es_pi_95,
            "upper_pi": es_forecast.values + es_pi_95
        }
        
        model_performance.append({
            "Model": "Exponential Smoothing",
            "RMSE": es_rmse,
            "MAE": es_mae,
            "MAPE": es_mape
        })
        
    except Exception as e:
        print(f"Exponential Smoothing failed: {e}")


# =========================================
# >>> NEW: MODEL COMPARISON TABLE <<<
# =========================================

print("\n" + "="*70)
print("7. MODEL PERFORMANCE COMPARISON")
print("="*70)

performance_df = pd.DataFrame(model_performance)
performance_df = performance_df.sort_values("RMSE")
print("\nModel Performance (sorted by RMSE):")
print(performance_df.to_string(index=False))

performance_df.to_excel(
    "d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/model_comparison.xlsx",
    index=False
)

# Select best model
best_model_name = performance_df.iloc[0]["Model"]
print(f"\n✓ BEST MODEL: {best_model_name} (Lowest RMSE)")

# =========================================
# >>> NEW: COMPREHENSIVE FORECAST VISUALIZATION <<<
# =========================================

print("\n" + "="*70)
print("8. GENERATING FORECAST VISUALIZATIONS")
print("="*70)

# Create comprehensive forecast plot
fig, axes = plt.subplots(len(all_forecasts), 1, figsize=(14, 5*len(all_forecasts)))

if len(all_forecasts) == 1:
    axes = [axes]

for idx, (model_name, forecast_data) in enumerate(all_forecasts.items()):
    ax = axes[idx]
    
    # Historical data
    ax.plot(df["Year"], df["HospitalBasedPrevalence"], 
            marker='o', linewidth=2, label="Historical Data", color='blue')
    
    # Forecast
    ax.plot(forecast_years, forecast_data["predictions"], 
            marker='s', linewidth=2, linestyle='--', label="Forecast", color='red')
    
    # Confidence interval
    ax.fill_between(forecast_years, 
                     forecast_data["lower_pi"], 
                     forecast_data["upper_pi"],
                     alpha=0.3, color='red', label="95% PI")
    
    # Vertical line at forecast start
    ax.axvline(x=2025.5, color='green', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_title(f"{model_name} - Forecast 2026-2030", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Hospital-Based Prevalence", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/forecast_all_models.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Forecast visualization saved")

# =========================================
# >>> NEW: SAVE ALL FORECASTS TO EXCEL <<<
# =========================================

# Create comprehensive forecast table
forecast_table = pd.DataFrame({"Year": forecast_years})

for model_name, forecast_data in all_forecasts.items():
    forecast_table[f"{model_name}_Forecast"] = forecast_data["predictions"]
    forecast_table[f"{model_name}_lower_pi"] = forecast_data["lower_pi"]
    forecast_table[f"{model_name}_upper_pi"] = forecast_data["upper_pi"]

forecast_table.to_excel(
    "d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/forecast_2026_2030.xlsx",
    index=False
)

print("\n✓ Forecast table saved to forecast_2026_2030.xlsx")
print("\nForecast Summary (2026-2030):")
print(forecast_table.to_string(index=False))

# =========================================
# FINAL SUMMARY
# =========================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. analysis_annual_change.png - Annual % change in prevalence")
print("  2. sensitivity_analysis.xlsx - COVID-19 sensitivity results")
print("  3. model_comparison.xlsx - Performance comparison of all models")
print("  4. forecast_all_models.png - Visual forecast from all models")
print("  5. forecast_2026_2030.xlsx - Detailed forecast with confidence intervals")
print("\n" + "="*70)
