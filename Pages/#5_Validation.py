import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

st.title('Welcome to the testing procedure!')

if ("knn" in st.session_state) and ("rfc" in st.session_state):
    if st.session_state["rfc_score"] > st.session_state["knn_score"]:
        model = st.session_state["rfc"]
    else:
        model = st.session_state["knn"]

    # model = st.session_state['model']

    # brand_name = st.text_input('Brand Name', '-1')
    # brand_name = float(brand_name)

    # model_name = st.text_input('Model Name', '-1')
    # model_name = float(model_name)

    st.info(f"The selected model is {model}")

    rating = st.text_input('Rating', '-1')
    rating = float(rating)

    processor_speed = st.text_input('Processor speed', '-1')
    processor_speed = float(processor_speed)

    fast_charging = st.text_input('Fast Charging', '-1')
    fast_charging = float(fast_charging)

    ram_capacity = st.text_input('Ram Capacity', '-1')
    ram_capacity = float(ram_capacity)

    internal_memory = st.text_input('Internal Memory', '-1')
    internal_memory = float(internal_memory)

    screen_size = st.text_input('Screen Size', '-1')
    screen_size = float(screen_size)

    primary_camera_front = st.text_input('Quality of front cameras', '-1')
    primary_camera_front = float(primary_camera_front)

    resolution_height = st.text_input('Resolution Height', '-1')
    resolution_height = float(resolution_height)

    has_nfc = st.text_input('Has NFC', '-1')
    has_nfc = int(has_nfc)

    os_ios = st.text_input('OS IOS', '-1')
    os_ios = int(os_ios)
    
    test = np.array([rating, processor_speed, fast_charging, ram_capacity, internal_memory,
                     screen_size, primary_camera_front, resolution_height, has_nfc, os_ios]).reshape(1, -1)
    
    single_prediction = model.predict(test)
    
    st.write(single_prediction)
        
else:
    st.error('You should train the model before evaluating it!')