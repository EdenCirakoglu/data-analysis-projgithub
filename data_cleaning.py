import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="in_path",  required=True, help="input CSV/XLSX path")
    parser.add_argument("--out", dest="out_path", default="cleaned_sales_data.xlsx", help="output CSV/XLSX path")
    parser.add_argument("--z",   dest="z_thr", type=float, default=3.0, help="Z-score threshold (default 3.0)")
    args = parser.parse_args()

    in_p  = Path(args.in_path)
    out_p = Path(args.out_path)

     # ---- load (auto-detect csv/xlsx)
    if in_p.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(in_p)
    else:
        df = pd.read_csv(in_p)

    #  Standardize categoricals
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    if "Region" in df.columns:
        df["Region"] = (
            df["Region"].str.title()
            .replace({"N":"North","S":"South","E":"East","W":"West",
                      "Na":"Unknown","Nan":"Unknown","None":"Unknown"})
        )

    if "Churned" in df.columns:
        df["Churned"] = (df["Churned"].str.lower()
                         .replace({"y":"yes","n":"no","true":"yes","false":"no"})
                         .map({"yes":"Yes","no":"No"}))
        
    # Ensure numeric dtypes for the known numeric columns
    num_cols = [c for c in ["Total_Spend","Purchase_Frequency","Marketing_Spend","Seasonality_Index"] if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Impute missing values
    #    - numeric -> median
    #    - categorical -> mode
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=["object","category"]).columns:
        mode_val = df[c].mode(dropna=True)
        df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    #  Detect & REMOVE outliers with Z-score on numeric columns
    #    Drop rows where ANY numeric column has |z| >= threshold
    z = df[num_cols].apply(zscore, nan_policy="omit")
    mask_keep = (z.abs() < args.z_thr).all(axis=1) | z.isna().all(axis=1)  # keep rows with all NaN z (rare) or within threshold
    df_clean = df.loc[mask_keep].reset_index(drop=True)

    #  Save
    if out_p.suffix.lower() in {".xls", ".xlsx"}:
        df_clean.to_excel(out_p, index=False)
    else:
        df_clean.to_csv(out_p, index=False)

    # Quick summary
    print(f"Input rows:  {len(df)}")
    print(f"Kept rows:   {len(df_clean)}  (removed {len(df) - len(df_clean)} outliers)")
    print(f"Saved to:    {out_p.resolve()}")
    print("\nPreview:")
    print(df_clean.head().to_string(index=False))

if __name__ == "__main__":
    main()