# ================================
# 1. READ THE DATA AND INITIAL CHECK
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# >>> NEW: Import for correlation heatmap <<<
import seaborn as sns

# >>> NEW: Import for stationarity test <<<
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr

file_path = "d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/dataset.xlsx"
df = pd.read_excel(file_path)

print("First 5 rows of dataset:")
print(df.head())

# ================================
# 2. BASIC DEFINITIVE TABLE
# ================================

summary_table = df.describe()
print("\nDescriptive Statistics:")
print(summary_table)

# >>> NEW: Save summary table to Excel <<<
summary_table.to_excel("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/summary_statistics.xlsx")
print("\n✓ Summary statistics saved to summary_statistics.xlsx")

# ================================
# >>> NEW: CORRELATION ANALYSIS <<<
# ================================

print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

key_vars = [
    'HospitalBasedPrevalence',
    'DiagnosisCount',
    'TotalApplications',
    'NormalizedDiagnosisRate',
    'MeanAge',
    'FemaleRatio',
    'COPD_Rate',
    'HT_Rate',
    'DM_Rate',
    'CHF_Rate',
    'CKF_Rate'
]

# Filter only existing columns
existing_vars = [var for var in key_vars if var in df.columns]

correlation_matrix = df[existing_vars].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Save correlation matrix
correlation_matrix.to_excel("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/correlation_matrix.xlsx")

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title("Correlation Matrix of Key Variables", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/correlation_heatmap.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved")

print("\nKEY CORRELATION RESULTS")
print("="*50)

target = "HospitalBasedPrevalence"

for var in [
    "DiagnosisCount",
    "TotalApplications",
    "NormalizedDiagnosisRate"
]:
    if target in df.columns and var in df.columns:
        valid = df[[target, var]].dropna()

        r, p = pearsonr(
            valid[target],
            valid[var]
        )

        print(
            f"{target} vs {var}: "
            f"r = {r:.4f}, p = {p:.6f}"
        )
# ================================
# >>> NEW: STATIONARITY TEST <<<
# ================================

print("\n" + "="*50)
print("STATIONARITY TEST (Augmented Dickey-Fuller)")
print("="*50)

def test_stationarity(series, name):
    """Perform ADF test and print results"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'\n{name}:')
    print(f'  ADF Statistic: {result[0]:.4f}')
    print(f'  p-value: {result[1]:.4f}')
    print(f'  Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print(f'  → Result: STATIONARY (reject null hypothesis)')
    else:
        print(f'  → Result: NON-STATIONARY (fail to reject null hypothesis)')
    
    return result

# Test key time series
stationarity_results = {}
test_vars = [
    'HospitalBasedPrevalence',
    'DiagnosisCount',
    'NormalizedDiagnosisRate'
]

for var in test_vars:
    if var in df.columns:
        stationarity_results[var] = test_stationarity(df[var], var)

# ================================
# 3. GRAPHS
# Each graph will be separate and simple.
# ================================

# >>> CHANGED: Save figures instead of just showing them <<<
# >>> CHANGED: Add grid and improve styling <<<

# 3.1 Hospital-Based Prevalence Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["HospitalBasedPrevalence"], marker='o', linewidth=2, markersize=6)
plt.title(
    f"Hospital-Based Lung Cancer Prevalence "
    f"({df['Year'].min()}–{df['Year'].max()})",
    fontsize=14,
    fontweight='bold'
)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Hospital-Based Prevalence", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
# >>> NEW: Save figure <<<
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/fig1_prevalence_trend.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Figure 1 saved: Prevalence trend")

# 3.2 Diagnosis Count Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["DiagnosisCount"], marker='s', linewidth=2, markersize=6, color='green')
plt.title("Annual Lung Cancer Diagnosis Count", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Diagnosis Count", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/fig2_diagnosis_count.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved: Diagnosis count")

# 3.3 Total Hospital Applications Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["TotalApplications"], marker='^', linewidth=2, markersize=6, color='orange')
plt.title("Total Hospital Applications per Year", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Applications", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/fig3_total_applications.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved: Total applications")

# 3.4 Normalized Diagnosis Rate Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["NormalizedDiagnosisRate"], marker='D', linewidth=2, markersize=6, color='red')
plt.title("Normalized Diagnosis Rate Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Normalized Diagnosis Rate", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/fig4_normalized_rate.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved: Normalized diagnosis rate")

# ================================
# >>> NEW: COMBINED TREND VISUALIZATION <<<
# ================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Prevalence
axes[0, 0].plot(df["Year"], df["HospitalBasedPrevalence"], marker='o', linewidth=2)
axes[0, 0].set_title("Hospital-Based Prevalence", fontweight='bold')
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Prevalence")
axes[0, 0].grid(True, alpha=0.3)

# Diagnosis Count
axes[0, 1].plot(df["Year"], df["DiagnosisCount"], marker='s', linewidth=2, color='green')
axes[0, 1].set_title("Diagnosis Count", fontweight='bold')
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Count")
axes[0, 1].grid(True, alpha=0.3)

# Total Applications
axes[1, 0].plot(df["Year"], df["TotalApplications"], marker='^', linewidth=2, color='orange')
axes[1, 0].set_title("Total Applications", fontweight='bold')
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("Applications")
axes[1, 0].grid(True, alpha=0.3)

# Normalized Rate
axes[1, 1].plot(df["Year"], df["NormalizedDiagnosisRate"], marker='D', linewidth=2, color='red')
axes[1, 1].set_title("Normalized Diagnosis Rate", fontweight='bold')
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Rate")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/fig5_combined_trends.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved: Combined trends")

# ================================
# >>> NEW: COMORBIDITY TRENDS <<<
# ================================

comorbidity_vars = ['COPD_Rate', 'HT_Rate', 'DM_Rate', 'CHF_Rate', 'CKF_Rate']
existing_comorbidities = [var for var in comorbidity_vars if var in df.columns and df[var].notna().any()]

if existing_comorbidities:
    plt.figure(figsize=(14, 7))
    for var in existing_comorbidities:
        plt.plot(df["Year"], df[var], marker='o', linewidth=2, label=var.replace('_Rate', ''))
    
    plt.title("Comorbidity Trends Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Rate (Proportion)", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("d:/Academics/01-YAYIN/Z-2026-ISTINYE-LUNG-CANCER/fig6_comorbidity_trends.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6 saved: Comorbidity trends")

print("\n" + "="*50)
print("Stage 2 Analysis Completed Successfully!")
print("="*50)
print("\nGenerated files:")
print("  - summary_statistics.xlsx")
print("  - correlation_matrix.xlsx")
print("  - correlation_heatmap.png")
print("  - fig1_prevalence_trend.png")
print("  - fig2_diagnosis_count.png")
print("  - fig3_total_applications.png")
print("  - fig4_normalized_rate.png")
print("  - fig5_combined_trends.png")
print("  - fig6_comorbidity_trends.png")
