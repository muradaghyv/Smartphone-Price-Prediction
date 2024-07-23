import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.title("Welcome to the Data Preprocessing Page")

if "data" in st.session_state:
    df = st.session_state["data"]

    if st.button("Begin the process!"):
        df.drop(df[df["price"]==650000].index, axis=0, inplace=True)
        df.drop(["model", "extended_upto"], axis=1, inplace=True)
        df["brand_name"] = df["brand_name"].replace({"oneplus":"oppo",
                                                "realme":"xiaomi", 
                                                "redmi":"xiaomi",
                                                "poco":"xiaomi"})
        df_copy = df.copy()
        temp = df_copy.groupby(["brand_name"])["price"].mean()
        df_copy = df_copy.merge(temp.reset_index(), how="left", on="brand_name")
        dct = {}
        for i in range (df_copy["price_y"].nunique()): 
            if df_copy["brand_name"].unique()[i] not in dct:
                dct[df_copy["brand_name"].unique()[i]] = df_copy["price_y"].unique()[i]

        bins = [0, 25000, 50000, 130000]
        label = ["budget_friendly", "middle_range", "expensive"]

        df["category"] = pd.cut(df_copy["price_y"], bins, right=False, labels=label)
        df["fast_charging_available"] = df["fast_charging_available"].astype("bool")
        df["extended_memory_available"] = df["extended_memory_available"].astype("bool")
        df["category"] = df["category"].astype("object")

        num_col = df.select_dtypes(exclude=["object", "bool"]).columns
        cat_col = df.select_dtypes(include=["object", "bool"]).columns

        imputer_mean = SimpleImputer(strategy="mean")
        imputer_median = SimpleImputer(strategy="median")
        imputer_mode = SimpleImputer(strategy="most_frequent")

        df["rating"] = imputer_mean.fit_transform(df["rating"].values.reshape(-1, 1))
        df["processor_brand"] = imputer_mode.fit_transform(df["processor_brand"].values.reshape(-1, 1)).reshape(-1)
        df["num_cores"] = imputer_median.fit_transform(df["num_cores"].values.reshape(-1, 1))
        df["processor_speed"] = imputer_mean.fit_transform(df["processor_speed"].values.reshape(-1, 1))
        df["battery_capacity"] = imputer_mean.fit_transform(df["battery_capacity"].values.reshape(-1, 1))
        df["fast_charging"] = imputer_mean.fit_transform(df["fast_charging"].values.reshape(-1, 1))
        df["num_front_cameras"] = imputer_median.fit_transform(df["num_front_cameras"].values.reshape(-1, 1))
        df["os"] = imputer_mode.fit_transform(df["os"].values.reshape(-1, 1)).reshape(-1)
        df["primary_camera_front"] = imputer_median.fit_transform(df["primary_camera_front"].values.reshape(-1, 1))
        
        st.session_state["full_data"] = df

        df_encoded = pd.get_dummies(df, drop_first=True, dtype="int32")

        data = df_encoded[['rating', 'processor_speed', 'fast_charging', 'ram_capacity', 'internal_memory',
                    'screen_size', 'primary_camera_front', 'resolution_height', 'has_nfc', 'os_ios', "price"]]
        
        data["has_nfc"] = data["has_nfc"].astype("int64")
        st.session_state["final_data"] = data
        st.write(data.head())
        st.success("Data Preprocessing is successfull!")

else:
    st.error("Upload the data first!")