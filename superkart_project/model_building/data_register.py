"""Data registration: validate the dataset stored in the repository and print a summary."""
import hashlib
import sys

import pandas as pd

DATA_PATH = "/content/drive/MyDrive/AIML_UTA/AIML_MLOps/SuperKart_Assignment/superkart_project/data/SuperKart.csv"

EXPECTED_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total",
]
NUMERIC_COLUMNS = [
    "Product_Weight", "Product_Allocated_Area", "Product_MRP",
    "Store_Establishment_Year", "Product_Store_Sales_Total",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # 1) the expected columns must all be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        sys.exit(f"[register] FAILED - missing columns: {missing}")
    if extra:
        print(f"[register] note - unexpected extra columns: {extra}")

    # 2) the file must contain data and the numeric columns must really be numeric
    if df.empty:
        sys.exit("[register] FAILED - dataset is empty")
    bad_types = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_types:
        sys.exit(f"[register] FAILED - non-numeric columns: {bad_types}")

    # 3) summary
    with open(DATA_PATH, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    print("[register] dataset registered successfully")
    print(f"  path            : {DATA_PATH}")
    print(f"  sha256 (short)  : {digest}")
    print(f"  rows x columns  : {df.shape[0]} x {df.shape[1]}")
    print(f"  missing values  : {int(df.isna().sum().sum())}")
    print(f"  duplicate rows  : {int(df.duplicated().sum())}")
    print(f"  stores / products: {df['Store_Id'].nunique()} / {df['Product_Id'].nunique()}")
    print("  target summary  :")
    print(df["Product_Store_Sales_Total"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
