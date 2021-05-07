import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from PIL import Image
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# from rdkit import Chem
# from rdkit.Chem import Descriptors

img  = Image.open("pogchamp.png")
st.image(img, use_column_width=True)

st.write('''
# Molecular Solubility Prediction App

This app predicts the **solubility (LogS)** values of a molecule

''')

dataset = pd.read_csv("delaney_solubility_with_descriptors.csv")

X = dataset.drop(['logS'], axis = 1)
Y = dataset.iloc[:,-1]

model = linear_model.LinearRegression()
model.fit(X, Y)

Y_predict = model.predict(X)

st.write(Y_predict)
st.write("Co-efficients: ", model.coef_ )
st.write("Intercepts: ", model.intercept_)
st.write("Mean Square Error: ", mean_squared_error(Y, Y_predict))
st.write("Co-efficient of Determination :", r2_score(Y, Y_predict))

st.write(f"LogS = {model.intercept_} LogP {model.coef_[0]} MW {model.coef_[1]} RB {model.coef_[2]} AP {model.coef_[3]}")

plt.figure(figsize=(5,5))
plt.scatter(x=Y, y=Y_predict, c="#7CAE00", alpha=0.3)

z = np.polyfit(Y, Y_predict, 1)
p = np.poly1d(z)

plt.plot(Y,p(Y),"#F8766D")
plt.ylabel("Predicted LogS")
plt.xlabel("Experimental LogS")
plt.show()

pickle.dump(model, open("sol_model.pkl", "wb"))

