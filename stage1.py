import pandas as pd
import numpy as np

# ================================
# 1. READ DATA
# ================================
input_file = "cancer.xlsx"
output_file = "dataset.xlsx"

df = pd.read_excel(input_file)
df.columns = df.columns.str.strip()

# ================================
# 2. STANDARDIZE BINARY VARIABLES
# ================================

# Automatically capture Hospital Application columns
hospital_cols = [col for col in df.columns if "Hospital Application" in col]

binary_cols = hospital_cols + [
    "Exitus Status",
    "Is There Heart Failure?",
    "Is There Heart Failure After Diagnosis?",
    "Is There Ischemic Heart Disease After Diagnosis?",
    "Is There Chronic Kidney Failure?",
    "Is There Chronic Kidney Failure After Diagnosis?",
    "Is There Cerebral Infarction?",
    "Is There Cerebral Infarction After Diagnosis?",
    "Is There Diabetes?",
    "Is There Blood Pressure?",
    "Is There COPD?",
    "Is There Asthma?",
    "Is There a Stroke After Diagnosis?",
    "Is There a Stroke or Cerebral Event After Diagnosis?"
]

for col in binary_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
            .fillna(0)
            .astype(int)
        )

# ================================
# 3. DATES AND DEFINITIONS
# =================================

df["Diagnosis month-year"] = pd.to_datetime(
    df["Diagnosis month-year"], errors="coerce"
)

df["Year_of_Diagnosis"] = df["Year of Diagnosis"].astype(int)

# ================================
# 4. INCIDENT CASE (DIAGNOSIS DENSITY)
# ================================

diagnosis_density = (
    df.groupby("Year_of_Diagnosis")
      .size()
      .rename("DiagnosisCount")
      .reset_index()
      .rename(columns={"Year_of_Diagnosis": "Year"})
)

# ================================
# 5. HOSPITAL-BASED PREVALENCE + TOTAL APPLICATION
# ================================

years = list(range(2000, 2025))
records = []

for year in years:
    app_col = f"{year} Hospital Application"
    
    if app_col not in df.columns:
        continue
    
    active = df[df[app_col] == 1]
    
    prevalence = active[
        active["Year_of_Diagnosis"] <= year
    ].shape[0]
    
    total_apps = df[app_col].sum()
    
    records.append({
        "Year": year,
        "HospitalBasedPrevalence": prevalence,
        "TotalApplications": total_apps
    })

yearly_base = pd.DataFrame(records)

# ================================
# 6. DIAGNOSIS DENSITY COMBINE
# =================================

yearly = yearly_base.merge(
    diagnosis_density,
    on="Year",
    how="left"
)

yearly["DiagnosisCount"] = yearly["DiagnosisCount"].fillna(0).astype(int)

# ================================
# 7. NORMALIZED DIAGNOSIS RATE
# ================================

yearly["NormalizedDiagnosisRate"] = (
    yearly["DiagnosisCount"] /
    yearly["TotalApplications"].replace(0, np.nan)
)

# ================================
# 8. DEMOGRAPHY + COMORBIDITY AGGREGATES
# ================================

agg_records = []

for year in years:
    app_col = f"{year} Hospital Application"
    if app_col not in df.columns:
        continue
    
    cohort = df[
        (df[app_col] == 1) &
        (df["Year_of_Diagnosis"] <= year)
    ]
    
    if cohort.empty:
        continue
    
    agg_records.append({
        "Year": year,
        "MeanAge": cohort["Age"].mean(),
        "FemaleRatio": (cohort["Gender"].astype(str).str.lower() == "female").mean(),
        "COPD_Rate": cohort["Is There COPD?"].mean() if "Is There COPD?" in df.columns else np.nan,
        "HT_Rate": cohort["Is There Blood Pressure?"].mean() if "Is There Blood Pressure?" in df.columns else np.nan,
        "DM_Rate": cohort["Is There Diabetes?"].mean() if "Is There Diabetes?" in df.columns else np.nan,
        "CHF_Rate": cohort["Is There Heart Failure?"].mean() if "Is There Heart Failure?" in df.columns else np.nan,
        "CKF_Rate": cohort["Is There Chronic Kidney Failure?"].mean() if "Is There Chronic Kidney Failure?" in df.columns else np.nan,
        "MeanSurvivalMonth": cohort["Survival Month"].mean() if "Survival Month" in df.columns else np.nan,
        "AnnualDeaths": cohort["Exitus Status"].sum() if "Exitus Status" in df.columns else np.nan
    })

yearly_agg = pd.DataFrame(agg_records)

# ================================
# 9. CREATE TIME-SERIES DATASET
# ================================

ts_data = yearly.merge(
    yearly_agg,
    on="Year",
    how="left"
).sort_values("Year").reset_index(drop=True)

# ================================
# 10. LAG FEATURES
# ================================

lag_features = [
    "HospitalBasedPrevalence",
    "DiagnosisCount",
    "TotalApplications",
    "NormalizedDiagnosisRate"
]

for col in lag_features:
    ts_data[f"{col}_lag1"] = ts_data[col].shift(1)
    ts_data[f"{col}_lag2"] = ts_data[col].shift(2)

# Exclude the first two years due to lag.
ts_data_final = ts_data[ts_data["Year"] >= 2002].reset_index(drop=True)

# ================================
# 11. SAVE AS EXCEL FILE
# ================================

ts_data_final.to_excel(output_file, index=False)

print("Successfully created dataset.xlsx!")
