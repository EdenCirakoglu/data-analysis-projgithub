import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, ttest_ind
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# --- Load data ---
INPUT = Path("cleaned_sales_data.xlsx")
df = pd.read_excel(INPUT)

# --- 1. ANOVA: Compare Total_Spend across regions ---
regions = df['Region'].dropna().unique()
region_groups = [df[df['Region'] == reg]['Total_Spend'].dropna() for reg in regions]
anova_result = f_oneway(*region_groups)
print("\n[ANOVA] Total_Spend across Regions")
print(f"F-statistic: {anova_result.statistic:.3f}, p-value: {anova_result.pvalue:.4f}")

# --- 2. Hypothesis Testing: Impact of Churn on Total_Spend ---
churn_yes = df[df['Churned'] == 'Yes']['Total_Spend'].dropna()
churn_no = df[df['Churned'] == 'No']['Total_Spend'].dropna()
t_test_result = ttest_ind(churn_yes, churn_no, equal_var=False)
print("\n[T-Test] Total_Spend: Churned vs Non-Churned")
print(f"T-statistic: {t_test_result.statistic:.3f}, p-value: {t_test_result.pvalue:.4f}")

# --- 3. Factor Analysis: Key drivers for purchase decisions ---
# Select relevant numeric columns (customize as needed)
features = ["Total_Spend", "Purchase_Frequency", "Marketing_Spend", "Seasonality_Index"]
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

fa = FactorAnalysis(n_components=2, random_state=42)
fa.fit(X_scaled)
print("\n[Factor Analysis] Loadings (top 2 factors):")
loadings = pd.DataFrame(fa.components_.T, index=features, columns=["Factor1", "Factor2"])
print(loadings.round(3))