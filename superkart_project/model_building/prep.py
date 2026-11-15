"""Data preparation: load -> clean -> engineer -> drop unnecessary columns -> train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "/content/drive/MyDrive/AIML_UTA/AIML_MLOps/SuperKart_Assignment/superkart_project/data/SuperKart.csv"
TARGET = "Product_Store_Sales_Total"
PREFIX_TO_CATEGORY = {"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # strip stray whitespace in text columns
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip()

    # standardise inconsistent labels: 'reg' is the same category as 'Regular'
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    # remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # defensive imputation (the current file has no missing values, but new data might)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0])

    # feature engineering: product category from the Product_Id prefix (FD / DR / NC)
    df["Product_Category"] = df["Product_Id"].str[:2].map(PREFIX_TO_CATEGORY).fillna("Other")

    # drop unnecessary columns: Product_Id is unique per row, Store_Id is redundant with store attributes
    return df.drop(columns=["Product_Id", "Store_Id"])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"[prep] loaded {df.shape[0]} rows x {df.shape[1]} columns from {DATA_PATH}")

    df = clean(df)
    print(f"[prep] cleaned data: {df.shape[0]} rows x {df.shape[1]} columns")
    print("[prep] columns:", list(df.columns))

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    print(f"[prep] saved Xtrain{Xtrain.shape} Xtest{Xtest.shape} ytrain{ytrain.shape} ytest{ytest.shape}")


if __name__ == "__main__":
    main()
