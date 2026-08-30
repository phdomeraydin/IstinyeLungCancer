import pandas as pd
import numpy as np

# ================================
# 1. READ DATA
# ================================
input_file = "d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/cancer.csv"
output_file = "d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/dataset.xlsx"

# Read CSV correctly
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

# ================================
# >>> NEW: DATA QUALITY CHECKS <<<
# ================================
print("="*50)
print("DATA QUALITY REPORT")
print("="*50)
print(f"Total patients: {len(df)}")
print(f"\nColumn names found: {len(df.columns)} columns")
print(f"\nFirst 5 column names:\n{list(df.columns[:5])}")
print(f"\nDiagnosis year range: {df['Year of Diagnosis'].min()} - {df['Year of Diagnosis'].max()}")
print(f"\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("="*50)

# ================================
# 2. STANDARDIZE BINARY VARIABLES
# ================================

# Automatically capture Hospital Application columns
hospital_cols = [col for col in df.columns if "Hospital Application" in col]

# >>> NEW: Print detected hospital columns <<<
print(f"\nDetected {len(hospital_cols)} hospital application columns:")
print(hospital_cols)

binary_cols = hospital_cols + [
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

# >>> CHANGED: Auto-detect years from hospital columns instead of hardcoding <<<
years = sorted([int(col.split()[0]) for col in hospital_cols])

print(f"\nYears to process: {years}")
print(f"Year range: {min(years)} to {max(years)}")

records = []

for year in years:
    app_col = f"{year} Hospital Application"
    
    if app_col not in df.columns:
        print(f"WARNING: Missing column {app_col}")
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
        "CKF_Rate": cohort["Is There Chronic Kidney Failure?"].mean() if "Is There Chronic Kidney Failure?" in df.columns else np.nan
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

# >>> CHANGED: Auto-calculate first valid year (min_year + 2) instead of hardcoding 2002 <<<
#min_year = ts_data["Year"].min()
#first_valid_year = min_year + 2

#print(f"\nExcluding first 2 years ({min_year}-{min_year+1}) due to lag features")
#print(f"Final dataset years: {first_valid_year} to {ts_data['Year'].max()}")

#ts_data_final = ts_data[ts_data["Year"] >= first_valid_year].reset_index(drop=True)
ts_data_final = ts_data.copy().reset_index(drop=True)

print("\nKeeping all available calendar years in the final dataset.")
print(f"Final dataset years: {ts_data_final['Year'].min()} to {ts_data_final['Year'].max()}")
print("Lag variables are retained with missing values in the first one or two years,")
print("but these lag variables are not used as predictors in the final forecasting models.")

# >>> NEW: Final dataset summary <<<
print("\n" + "="*50)
print("FINAL DATASET SUMMARY")
print("="*50)
print(f"Total years in final dataset: {len(ts_data_final)}")
print(f"Year range: {ts_data_final['Year'].min()} - {ts_data_final['Year'].max()}")
print(f"\nColumns in final dataset: {len(ts_data_final.columns)}")
print(f"\nFirst 5 rows:")
print(ts_data_final.head())
print("\nDescriptive statistics:")
print(ts_data_final.describe())
print("="*50)

# ================================
# 11. SAVE AS EXCEL FILE
# ================================

ts_data_final.to_excel(output_file, index=False)

print(f"\n✓ Successfully created dataset.xlsx!")
print(f"✓ File saved to: {output_file}")
