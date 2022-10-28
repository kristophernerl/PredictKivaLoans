import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
label_encoder_sector = data["label_encoder_sector"]
label_encoder_gender = data["label_encoder_gender"]

def show_predict_page():
    st.title("Kiva Loans Prediction")

    st.write("""### We need some information to predict your recomended loan""")

