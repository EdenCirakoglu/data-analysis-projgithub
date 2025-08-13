import pandas as pd
import numpy as np
from pathlib import Path

# --- ML ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, classification_report, roc_auc_score

# ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------- config ----------
INPUT = Path("cleaned_sales_data.xlsx")   # <- this is your uploaded file name
FORECAST_STEPS = 6
# ---------------------------

def load_any(p: Path) -> pd.DataFrame:
    return pd.read_excel(p) if p.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(p)

def save_any(df: pd.DataFrame, p: Path):
    if p.suffix.lower() in {".xls", ".xlsx"}:
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False)


# --------------- tasks -----------------
def do_linear_regression(df: pd.DataFrame):
    req = ["Total_Spend", "Marketing_Spend", "Seasonality_Index"]
    if not all(c in df.columns for c in req):
        print("[Linear] Skipped: missing columns", req);  return df

    X = df[["Marketing_Spend", "Seasonality_Index"]].astype(float)
    y = df["Total_Spend"].astype(float)

    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print("\n[Linear Regression] Total_Spend ~ Marketing_Spend + Seasonality_Index")
    print(f"R^2: {r2_score(yte, pred):.3f}    RMSE: {mean_squared_error(yte, pred, squared=False):.2f}")


    # (optional) coefficients in the *scaled feature space*
    lr = pipe.named_steps["lr"]
    print("Coefficients (scaled features):", dict(zip(X.columns, lr.coef_)))
    print("Intercept:", lr.intercept_)


    # fit on all & add predictions
    pipe.fit(X, y)
    df["_Total_Spend_Pred"] = pipe.predict(X)
    return df


def do_logistic_regression(df: pd.DataFrame):
    if "Churned" not in df.columns:
        print("[Logistic] Skipped: missing 'Churned' column");  return df

    y = df["Churned"].map({"Yes": 1, "No": 0})
    num = [c for c in ["Total_Spend", "Purchase_Frequency", "Marketing_Spend", "Seasonality_Index"] if c in df.columns]
    cat = [c for c in ["Region"] if c in df.columns]
    X = df[num + cat].copy()

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ], remainder="drop")

    clf = Pipeline([("prep", pre), ("lr", LogisticRegression(max_iter=1000, random_state=42))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    yproba = clf.predict_proba(Xte)[:, 1]

    print("\n[Logistic Regression] Churned (Yes/No)")
    print(classification_report(yte, ypred, digits=3))
    try:
        print(f"ROC AUC: {roc_auc_score(yte, yproba):.3f}")
    except Exception:
        pass

    # add scores on full data
    df["_Churn_Prob"] = clf.predict_proba(X)[:, 1]
    df["_Churn_Pred"] = (df["_Churn_Prob"] >= 0.5).astype(int)
    return df


def find_date_col(df: pd.DataFrame):
    """Find a likely date column from common names."""
    for col in ["Order_Date", "Date", "Month"]:
        if col in df.columns:
            return col
    # Try to find any column with 'date' or 'month' in its name
    for col in df.columns:
        if "date" in col.lower() or "month" in col.lower():
            return col
    return None

def do_time_series_forecast(df: pd.DataFrame, steps=6):
    date_col = find_date_col(df)
    if date_col is None or "Total_Spend" not in df.columns:
        print("[ARIMA] Skipped: missing date or Total_Spend column")
        return None

    s = df[[date_col, "Total_Spend"]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.set_index(date_col)["Total_Spend"].sort_index()
    s = s.asfreq("MS")  # monthly start frequency

    model = SARIMAX(s, order=(1,1,1), seasonal_order=(0,0,0,0),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps)
    out = fc.predicted_mean.to_frame("Forecast")
    ci = fc.conf_int()
    out["Lower"], out["Upper"] = ci.iloc[:, 0].values, ci.iloc[:, 1].values

    print(f"\n[ARIMA] {steps}-month forecast (Total_Spend):")
    print(out.round(2))
    return out


# --------------- main ---------------
if __name__ == "__main__":
    INPUT = Path("cleaned_sales_data.xlsx")  # <- change to your cleaned file
    df = load_any(INPUT)

    # 1) Linear
    df = do_linear_regression(df)

    # 2) Logistic
    df = do_logistic_regression(df)

    # Save enriched file (pred columns)
    out_pred = INPUT.with_name(INPUT.stem + "_with_preds" + INPUT.suffix)
    save_any(df, out_pred)
    print(f"\nSaved predictions to: {out_pred.resolve()}")

    # 3) Time series
    fc = do_time_series_forecast(df, steps=6)
    if fc is not None:
        out_fc = INPUT.with_name(INPUT.stem + "_forecast.csv")
        fc.to_csv(out_fc)
        print(f"Saved forecast to: {out_fc.resolve()}")
