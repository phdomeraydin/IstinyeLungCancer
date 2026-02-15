# ================================
# 1. READ THE DATA AND INITIAL CHECK
# ================================

import pandas as pd
import matplotlib.pyplot as plt

file_path = "dataset.xlsx"
df = pd.read_excel(file_path)

df.head()
# ================================
# 2. BASIC DEFINITIVE TABLE
# ================================

summary_table = df.describe()
summary_table

# ================================
# 3. GRAPHS
# Each graph will be separate and simple.
# ================================

# 3.1 Hospital-Based Prevalence Trend
plt.figure()
plt.plot(df["Year"], df["HospitalBasedPrevalence"])
plt.title("Hospital-Based Lung Cancer Prevalence (2002–2024)")
plt.xlabel("Year")
plt.ylabel("Hospital-Based Prevalence")
plt.xticks(rotation=45)
plt.show()

# 3.2 Diagnosis Count Trend
plt.figure()
plt.plot(df["Year"], df["DiagnosisCount"])
plt.title("Annual Lung Cancer Diagnosis Count")
plt.xlabel("Year")
plt.ylabel("Diagnosis Count")
plt.xticks(rotation=45)
plt.show()

# 3.3 Total Hospital Applications Trend
plt.figure()
plt.plot(df["Year"], df["TotalApplications"])
plt.title("Total Hospital Applications per Year")
plt.xlabel("Year")
plt.ylabel("Total Applications")
plt.xticks(rotation=45)
plt.show()

# 3.4 Normalized Diagnosis Rate Trend
plt.figure()
plt.plot(df["Year"], df["NormalizedDiagnosisRate"])
plt.title("Normalized Diagnosis Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Normalized Diagnosis Rate")
plt.xticks(rotation=45)
plt.show()
