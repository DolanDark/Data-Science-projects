import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Penguin Prediction app

This app predicts the **Palmer Penguin** species

Data is subject to change over difference in csv files
''')

st.sidebar.header("User Input Features")

st.sidebar.markdown('''
Please check you dataset before uploading
''')

upload = st.sidebar.file_uploader("Upload the file in CSV format", type=["csv"])

if upload is not None:
    input_dataframe = pd.read_csv(upload)
else:
    def user_input():
        island = st.sidebar.selectbox("Island",("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex",("male", "female"))            #be smart about this
        bill_length = st.sidebar.slider("Bill Length (mm)",32.1,59.6,44.0)
        bill_depth = st.sidebar.slider("Bill Depth (mm)",13.1,21.5,17.0)
        flipper_length = st.sidebar.slider("Flipper Length (mm)",172.0, 231.0, 201.0)
        body_mass = st.sidebar.slider("Body Mass (g)",2700.0, 6300.0, 4207.0)
        data = {"island":island,
                "bill_length":bill_length,
                "bill_depth":bill_depth,
                "flipper_length":flipper_length,
                "body_mass":body_mass,
                "sex":sex}

        features = pd.DataFrame(data, index=[0])
        return features

    input_dataframe = user_input()

penguins_raw = pd.read_csv("penguins_cleaned.csv")
penguin = penguins_raw.drop(columns=["species"])
init_df = pd.concat([input_dataframe,penguin], axis = 0)

encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(init_df[col], prefix = col)
    init_df = pd.concat([init_df,dummy], axis=1)
    del init_df[col]

init_df = init_df[:1]       #selects first row of user input

st.subheader("User Input Features")

if upload is not None:
    st.write(init_df)
else:
    st.write("Awaiting CSV File to be Uploaded")
    st.write(init_df)

load_clf = pickle.load(open("penguins_clf.pkl", "rb"))  #classification model

prediction = load_clf.predict(init_df)
prediction_proba = load_clf.predict_proba(init_df)

st.subheader("Prediction")
penguin_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.write(penguin_species[prediction])

st.subheader("Probability")
st.write(prediction_proba)

