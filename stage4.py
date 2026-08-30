import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_excel(
    "dataset.xlsx"
)

df = df.sort_values("Year").reset_index(drop=True)

# =========================================
# 2. FEATURES
# =========================================

xgb_features = [
    "HospitalBasedPrevalence_lag1",
    "HospitalBasedPrevalence_lag2"
]

xgb_df = df.dropna(
    subset=xgb_features + ["HospitalBasedPrevalence"]
).copy()

# Final untouched test period
xgb_train = xgb_df[
    xgb_df["Year"] <= 2020
].copy()

xgb_test = xgb_df[
    xgb_df["Year"] > 2020
].copy()

print("=" * 70)
print("XGBOOST FORECASTING ANALYSIS")
print("=" * 70)

print(
    f"\nTraining period: "
    f"{xgb_train['Year'].min()}-"
    f"{xgb_train['Year'].max()} "
    f"({len(xgb_train)} years)"
)

print(
    f"Test period: "
    f"{xgb_test['Year'].min()}-"
    f"{xgb_test['Year'].max()} "
    f"({len(xgb_test)} years)"
)

# =========================================
# 3. SMALL PARAMETER GRID
# =========================================

parameter_grid = [
    {
        "n_estimators": 20,
        "max_depth": 1,
        "learning_rate": 0.05
    },
    {
        "n_estimators": 50,
        "max_depth": 1,
        "learning_rate": 0.05
    },
    {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.03
    },
    {
        "n_estimators": 100,
        "max_depth": 1,
        "learning_rate": 0.03
    },
    {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.03
    }
]

# =========================================
# 4. TIME-AWARE INTERNAL VALIDATION
# =========================================

# Use 2017-2020 only for internal validation.
# Earlier observations remain training data.
validation_years = [2017, 2018, 2019, 2020]

grid_results = []

for params in parameter_grid:

    fold_errors = []

    for val_year in validation_years:

        fold_train = xgb_train[
            xgb_train["Year"] < val_year
        ]

        fold_val = xgb_train[
            xgb_train["Year"] == val_year
        ]

        if len(fold_train) < 5 or fold_val.empty:
            continue

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.8,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            random_state=42
        )

        model.fit(
            fold_train[xgb_features],
            fold_train["HospitalBasedPrevalence"]
        )

        pred = model.predict(
            fold_val[xgb_features]
        )

        error = mean_absolute_error(
            fold_val["HospitalBasedPrevalence"],
            pred
        )

        fold_errors.append(error)

    mean_validation_mae = np.mean(fold_errors)

    grid_results.append({
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "Validation_MAE": mean_validation_mae
    })

# Results table
grid_df = pd.DataFrame(grid_results)
grid_df = grid_df.sort_values(
    "Validation_MAE"
).reset_index(drop=True)

print("\nParameter Search Results:")
print(grid_df.to_string(index=False))

# =========================================
# 5. SELECT BEST PARAMETERS
# =========================================

best = grid_df.iloc[0]

best_params = {
    "n_estimators": int(best["n_estimators"]),
    "max_depth": int(best["max_depth"]),
    "learning_rate": float(best["learning_rate"])
}

print("\nBest XGBoost Parameters:")
print(best_params)

# =========================================
# 6. FIT FINAL MODEL ON TRAINING SET
# =========================================

xgb_model = XGBRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=0.8,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(
    xgb_train[xgb_features],
    xgb_train["HospitalBasedPrevalence"]
)

# =========================================
# 7. FINAL TEST EVALUATION
# =========================================

xgb_test_pred = xgb_model.predict(
    xgb_test[xgb_features]
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        xgb_test["HospitalBasedPrevalence"],
        xgb_test_pred
    )
)

xgb_mae = mean_absolute_error(
    xgb_test["HospitalBasedPrevalence"],
    xgb_test_pred
)

xgb_mape = (
    mean_absolute_percentage_error(
        xgb_test["HospitalBasedPrevalence"],
        xgb_test_pred
    ) * 100
)
# =========================================
#
# =========================================
print("\n" + "=" * 70)
print("FINAL XGBOOST TEST PERFORMANCE")
print("=" * 70)

print(f"RMSE: {xgb_rmse:.2f}")
print(f"MAE: {xgb_mae:.2f}")
print(f"MAPE: {xgb_mape:.2f}%")



print("\n" + "=" * 70)
print("XGBOOST DIAGNOSTIC CHECK")
print("=" * 70)

print("\nTraining feature ranges:")
for feature in xgb_features:
    print(
        f"{feature}: "
        f"min={xgb_train[feature].min():.2f}, "
        f"max={xgb_train[feature].max():.2f}"
    )

print("\nTest feature values:")
print(
    xgb_test[
        ["Year"] + xgb_features + ["HospitalBasedPrevalence"]
    ].to_string(index=False)
)

print("\nTest predictions:")
diagnostic_test = xgb_test[
    ["Year"] + xgb_features + ["HospitalBasedPrevalence"]
].copy()

diagnostic_test["Prediction"] = xgb_test_pred

print(diagnostic_test.to_string(index=False))

print("\nFeature importances:")
for feature, importance in zip(
    xgb_features,
    xgb_model.feature_importances_
):
    print(f"{feature}: {importance:.6f}")

print("\nNumber of unique test predictions:")
print(len(np.unique(np.round(xgb_test_pred, 6))))
leaf_indices = xgb_model.apply(
    xgb_test[xgb_features]
)

print("\nLeaf indices for test observations:")
print(leaf_indices)
# =========================================
# 8. TEST-YEAR DETAILS
# =========================================

comparison = pd.DataFrame({
    "Year": xgb_test["Year"].values,
    "Actual": xgb_test["HospitalBasedPrevalence"].values,
    "Predicted": xgb_test_pred
})

comparison["Absolute_Error"] = (
    comparison["Actual"] -
    comparison["Predicted"]
).abs()

print("\nTest-Year Predictions:")
print(comparison.to_string(index=False))

# =========================================
# 9. SAVE RESULTS
# =========================================

grid_df.to_excel(
    "xgboost_parameter_search.xlsx",
    index=False
)

comparison.to_excel(
    "xgboost_test_predictions.xlsx",
    index=False
)

print("\nResults saved successfully.")


# =========================================
# 10. REFIT XGBOOST ON ALL AVAILABLE DATA
# =========================================

xgb_full = xgb_df.copy()

xgb_final_model = XGBRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=0.8,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    random_state=42
)

xgb_final_model.fit(
    xgb_full[xgb_features],
    xgb_full["HospitalBasedPrevalence"]
)

# =========================================
# 11. RECURSIVE FORECAST: 2025-2030
# =========================================

last_observed_year = int(df["Year"].max())

last_prev = float(
    df.loc[
        df["Year"] == last_observed_year,
        "HospitalBasedPrevalence"
    ].iloc[0]
)

second_last_prev = float(
    df.loc[
        df["Year"] == last_observed_year - 1,
        "HospitalBasedPrevalence"
    ].iloc[0]
)

recursive_forecasts = []

lag1 = last_prev
lag2 = second_last_prev

for year in range(2025, 2031):

    X_future = pd.DataFrame({
        "HospitalBasedPrevalence_lag1": [lag1],
        "HospitalBasedPrevalence_lag2": [lag2]
    })

    pred = float(
        xgb_final_model.predict(X_future)[0]
    )

    recursive_forecasts.append({
        "Year": year,
        "XGBoost_Forecast": pred
    })

    # Recursive update
    lag2 = lag1
    lag1 = pred

xgb_forecast_df = pd.DataFrame(
    recursive_forecasts
)

print("\nRecursive XGBoost Forecasts:")
print(xgb_forecast_df.to_string(index=False))

# =========================================
# 12. BOOTSTRAP APPROXIMATE 95% PI
# =========================================

# Residuals from full fitted XGBoost model
full_fitted = xgb_final_model.predict(
    xgb_full[xgb_features]
)

residuals = (
    xgb_full["HospitalBasedPrevalence"].values
    - full_fitted
)

n_bootstrap = 1000
rng = np.random.default_rng(42)

bootstrap_forecasts = []

for b in range(n_bootstrap):

    boot_lag1 = last_prev
    boot_lag2 = second_last_prev

    boot_path = []

    for year in range(2025, 2031):

        X_future = pd.DataFrame({
            "HospitalBasedPrevalence_lag1": [boot_lag1],
            "HospitalBasedPrevalence_lag2": [boot_lag2]
        })

        base_pred = float(
            xgb_final_model.predict(X_future)[0]
        )

        sampled_residual = rng.choice(
            residuals
        )

        boot_pred = base_pred + sampled_residual

        # Prevent impossible negative prevalence
        boot_pred = max(0, boot_pred)

        boot_path.append(
            boot_pred
        )

        boot_lag2 = boot_lag1
        boot_lag1 = boot_pred

    bootstrap_forecasts.append(
        boot_path
    )

bootstrap_forecasts = np.array(
    bootstrap_forecasts
)

lower_pi = np.percentile(
    bootstrap_forecasts,
    2.5,
    axis=0
)

upper_pi = np.percentile(
    bootstrap_forecasts,
    97.5,
    axis=0
)

# =========================================
# 13. COMBINE FORECAST + PI
# =========================================

xgb_forecast_df["Lower_95_PI"] = lower_pi
xgb_forecast_df["Upper_95_PI"] = upper_pi

print("\nXGBoost Forecasts with Approximate 95% PI:")
print(
    xgb_forecast_df.to_string(
        index=False
    )
)

# =========================================
# 14. REPORT ONLY 2026-2030
# =========================================

xgb_report_df = xgb_forecast_df[
    xgb_forecast_df["Year"] >= 2026
].copy()

print("\nXGBoost Forecasts for Manuscript (2026-2030):")
print(
    xgb_report_df.to_string(
        index=False
    )
)

# =========================================
# 15. SAVE FORECAST RESULTS
# =========================================

xgb_report_df.to_excel(
    "d:/"
    "xgboost_forecast_2026_2030.xlsx",
    index=False
)

print(
    "\nXGBoost forecast results saved successfully."
)
