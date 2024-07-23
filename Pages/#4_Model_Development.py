import streamlit as st
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

st.title("Welcome to the Training Process")

if "data" not in st.session_state:
    st.error("Upload and preprocess the data first!")
elif "final_data" not in st.session_state:
    st.error("Preprocess the data first!")
elif ("data" in st.session_state) and ("final_data" in st.session_state):
    df = st.session_state["data"]
    data = st.session_state["final_data"]

    test = data.iloc[np.arange(0, df.shape[0], 150)]
    data = data.drop(np.arange(0, df.shape[0], 150), axis=0)

    test_y = test["price"]
    test_x = test.drop("price", axis=1)

    y = data["price"]
    X = data.drop("price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    option = st.selectbox("Select the model for prediction:",
                          ("---", "Random Forest", "KNN model"))
    if option == "Random Forest":
        if st.button("Train the Random Forest Regressor Model:"):

            progress_text = "Model in the training process . . . "
            my_bar = st.sidebar.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete+1, text=progress_text)
            my_bar.empty()

            rfc = RandomForestRegressor()
            rfc.fit(X_train, y_train)
            pred_rfc = rfc.predict(X_test)
            score_rfc = r2_score(y_test, pred_rfc)
            score_rfc = round(score_rfc*100, 2)
            st.write(f"The result of the Random Forest Regressor is: {score_rfc}%")
            st.session_state["rfc"] = rfc
            st.session_state["rfc_score"] = score_rfc
            
    elif option == "KNN model":
        if st.button("Train the KNN Model:"):

            progress_text = "Model in the training process . . . "
            my_bar = st.sidebar.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete+1, text=progress_text)
            my_bar.empty()

            knn = KNeighborsRegressor(n_neighbors=10, weights="distance", leaf_size=40, algorithm="ball_tree")
            knn.fit(X_train, y_train)
            pred_knn = knn.predict(X_test)
            score_knn = r2_score(y_test, pred_knn)
            score_knn = round(score_knn*100, 2)
            st.write(f"The result of the KNN Model is: {score_knn}%")
            st.session_state["knn"] = knn
            st.session_state["knn_score"] = score_knn

else:
    st.error("You cannot train the model!")
