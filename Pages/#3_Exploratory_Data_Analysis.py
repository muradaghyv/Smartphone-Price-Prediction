import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

st.title("Welcome to the Exploratory Data Analysis!")

if ("data" in st.session_state) and ("full_data" in st.session_state):

    df = st.session_state["full_data"]
    num_col = df.select_dtypes(exclude=["object", "bool"]).columns
    cat_col = df.select_dtypes(include=["object", "bool"]).columns

    option = st.selectbox("Observe the plots and calculations: ",
                          ("--", "Outlier Distribution Graph", "Distribution According to Brands", 
                           "Distribution of price Column", "Correlation matrix Display", "Chi Square Test Results"))
    if option == "Outlier Distribution Graph":
        q1 = np.quantile(df["price"], 0.25)
        q3 = np.quantile(df["price"], 0.75)
        iqr = q3 - q1
        limit = 1.5 * iqr
        lower, upper = q1-limit, q3+limit

        df_wth = df[(df["price"]>lower) & (df["price"]<upper)]
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.boxplot(data=df_wth["price"], orient="v")
        ax.set_title("Outliers of Price Column")
        st.pyplot(fig)

    elif option == "Distribution According to Brands":
        plt.figure(figsize=(6, 6))
        ax = df["brand_name"].value_counts().plot(kind="bar", stacked=True)
        plt.title("Distribution according to brand name")
        st.pyplot(plt)

    elif option == "Distribution of price Column":
        plt.figure(figsize=(6, 6))
        plt.title("Price Distribution Plot")
        sns.distplot(df["price"])
        st.pyplot(plt)
    
    elif option == "Correlation matrix Display":
        correlation_matrix = df[num_col].corr()
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(correlation_matrix, annot=True)
        ax.set_title("Correlation Plot")
        st.pyplot(fig)

    elif option == "Chi Square Test Results":
        for i in range (len(cat_col)):
            for j in range (i+1, len(cat_col)):
                cont_table = pd.crosstab(df[cat_col[i]], df[cat_col[j]])
                chi2, p, dof, ex = chi2_contingency(cont_table)
                if p < 0.05:
                    st.write(f"There is a significant relationship between {cat_col[i]} and {cat_col[j]}.")
                else:
                    st.write(f"There is not a significant relationship between {cat_col[i]} and {cat_col[j]}.")

    st.info("Results of EDA:")
    st.write("""
             According to the correlation matrices, it can be easily observed that these columns have important relationships with the price column:
            * rating
            * processor_speed
            * fast_charging_available
            * fast_charging
            * ram_capacity
            * internal_memory
            * screen_size
            * primary_camera_front
            * resolution_height
             
             From boxplots, we can observe that the following categorical columns hava important relation with the price columns:
            * brand_name
            * has_5g
            * has_nfc
            * processor_brand
            * os
            * extended_memory_available """)
    
else:
    st.error("Upload the dataset before the visualization and analysis!")