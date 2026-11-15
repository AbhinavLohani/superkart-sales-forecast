"""SuperKart sales forecast - Streamlit app.

Loads the model committed to the repository by the CI/CD pipeline and predicts
Product_Store_Sales_Total for a product / store combination.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "superkart_model.joblib"

# the four SuperKart stores (store attributes are fixed per store in the data)
STORES = {
    "OUT001": {"Store_Establishment_Year": 1987, "Store_Size": "High",   "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type1"},
    "OUT002": {"Store_Establishment_Year": 1998, "Store_Size": "Small",  "Store_Location_City_Type": "Tier 3", "Store_Type": "Food Mart"},
    "OUT003": {"Store_Establishment_Year": 1999, "Store_Size": "Medium", "Store_Location_City_Type": "Tier 1", "Store_Type": "Departmental Store"},
    "OUT004": {"Store_Establishment_Year": 2009, "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2"},
}
PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables",
                 "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods",
                 "Soft Drinks", "Starchy Foods"]
FEATURE_ORDER = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type", "Product_MRP",
                 "Store_Establishment_Year", "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Category"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Forecast")
st.write("Predict the total revenue a product will generate in a SuperKart store.")

st.subheader("Product details")
c1, c2 = st.columns(2)
with c1:
    product_type = st.selectbox("Product type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Dairy"))
    product_category = st.selectbox("Product category", ["Food", "Drinks", "Non-Consumable"])
    sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
with c2:
    weight = st.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.65, step=0.1)
    mrp = st.number_input("Product MRP", min_value=1.0, max_value=400.0, value=147.0, step=1.0)
    area = st.number_input("Allocated display area (ratio)", min_value=0.0, max_value=0.5, value=0.069, step=0.001, format="%.3f")

st.subheader("Store details")
store_choice = st.selectbox("Store", list(STORES) + ["Custom"])
if store_choice == "Custom":
    s1, s2 = st.columns(2)
    with s1:
        year = st.number_input("Store establishment year", min_value=1950, max_value=2025, value=2000, step=1)
        size = st.selectbox("Store size", ["Small", "Medium", "High"])
    with s2:
        city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])
    store = {"Store_Establishment_Year": int(year), "Store_Size": size,
             "Store_Location_City_Type": city, "Store_Type": store_type}
else:
    store = STORES[store_choice]
    st.caption(f"{store['Store_Type']} | {store['Store_Size']} size | {store['Store_Location_City_Type']} | "
               f"established {store['Store_Establishment_Year']}")

# collect the inputs into a dataframe (same columns / order as training)
input_df = pd.DataFrame([{
    "Product_Weight": weight,
    "Product_Sugar_Content": sugar,
    "Product_Allocated_Area": area,
    "Product_Type": product_type,
    "Product_MRP": mrp,
    **store,
    "Product_Category": product_category,
}])[FEATURE_ORDER]

with st.expander("Model input (dataframe)"):
    st.dataframe(input_df)

if st.button("Predict sales", type="primary"):
    prediction = float(load_model().predict(input_df)[0])
    st.success(f"Forecast sales total: **{prediction:,.2f}**")
