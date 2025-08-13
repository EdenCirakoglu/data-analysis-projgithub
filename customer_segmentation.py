import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# --- Load data ---
INPUT = Path("cleaned_sales_data.xlsx")
df = pd.read_excel(INPUT)

# --- 1. Decision Tree: Segment customers based on purchasing behavior ---
features = ["Total_Spend", "Purchase_Frequency", "Marketing_Spend", "Seasonality_Index"]
target = "Churned"  # Example target for segmentation

if target in df.columns:
    X = df[features].dropna()
    y = df.loc[X.index, target].map({"Yes": 1, "No": 0})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(Xtr, ytr)
    print("\n[Decision Tree] Classification Report:")
    print(classification_report(yte, dt.predict(Xte), digits=3))
    plt.figure(figsize=(8,4))
    plot_tree(dt, feature_names=features, class_names=["No", "Yes"], filled=True)
    plt.title("Decision Tree for Customer Segmentation")
    plt.show()

# --- 2. K-Means Clustering: Group customers ---
scaler = StandardScaler()
X_kmeans = scaler.fit_transform(df[["Total_Spend", "Purchase_Frequency"]].dropna())
kmeans = KMeans(n_clusters=3, random_state=42)
df.loc[df[["Total_Spend", "Purchase_Frequency"]].dropna().index, "Customer_Segment"] = kmeans.fit_predict(X_kmeans)
print("\n[K-Means] Customer segments assigned (first 5 rows):")
print(df[["Total_Spend", "Purchase_Frequency", "Customer_Segment"]].head())

# --- 3. Ensemble Learning: Random Forest & XGBoost ---
if target in df.columns:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(Xtr, ytr)
    print("\n[Random Forest] Classification Report:")
    print(classification_report(yte, rf.predict(Xte), digits=3))
    
# Save segmented data
df.to_excel("segmented_customers.xlsx", index=False)