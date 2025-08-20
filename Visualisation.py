import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset_week4.csv")

# --- 1. Anomaly Detection: Smoothed Line Chart with Visible Outlier Dots ---
df = df.sort_values("Customer_ID")
window_size = 5
df['Sales_MA'] = df['Sales'].rolling(window=window_size, center=True).mean()

q1 = df['Sales'].quantile(0.25)
q3 = df['Sales'].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.0 * iqr
high = q3 + 1.5 * iqr
anomalies = df[(df['Sales'] < low) | (df['Sales'] > high)]

plt.figure(figsize=(12, 6))
plt.plot(df['Customer_ID'], df['Sales_MA'], label='Sales (Moving Average)', color='blue')
plt.scatter(anomalies['Customer_ID'], anomalies['Sales'], color='red', label='Anomaly', s=80, edgecolor='black', zorder=5)
plt.title("Sales Trend Over Customer_ID with Anomaly Detection")
plt.xlabel("Customer_ID")
plt.ylabel("Sales")
plt.legend()
plt.show()

# --- 2. Key Influencers: Feature Importance for Customer Churn ---
features = ['Age', 'Income', 'Spending_Score', 'Credit_Score', 'Loan_Amount',
            'Previous_Defaults', 'Marketing_Spend', 'Purchase_Frequency', 'Sales']
X = df[features]
y = df['Customer_Churn']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print("Key Influencers for Customer Churn:")
print(importances)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance for Customer Churn")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- 3. AI-Powered Visual Insights: Simplified Correlation Heatmap ---
corr = df[features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5, square=True, cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- 4. Alternative Method: Train a Regression Model for Sales Forecasting ---
X_sales = df[features].drop('Sales', axis=1)
y_sales = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("Sales Forecasting Regression Model")
print("R^2 Score on Test Set:", reg.score(X_test, y_test))

# --- 5. Predict Loan Default Risk with Python (Random Forest Classifier) ---
features_default = [
    'Age', 'Income', 'Spending_Score', 'Credit_Score', 'Loan_Amount',
    'Previous_Defaults', 'Marketing_Spend', 'Purchase_Frequency', 'Sales'
]
X_default = df[features_default]
y_default = df['Defaulted']  # Target variable

X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_default, y_default, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_def, y_train_def)

y_pred_def = clf.predict(X_test_def)
print("Classification Report for Loan Default Prediction:")
print(classification_report(y_test_def, y_pred_def))

# Feature importance visualization (Key Influencers)
importances_def = pd.Series(clf.feature_importances_, index=features_default).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_def.values, y=importances_def.index, palette="mako")
plt.title("Key Influencers for Loan Default Risk")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Optional: Confusion matrix visualization
from sklearn.metrics import confusion_matrix
cm_def = confusion_matrix(y_test_def, y_pred_def)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_def, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Loan Default Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Optional: Plot actual vs predicted sales
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.8, color='dodgerblue', edgecolor='black', s=80, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()