import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston Housing Prediction App
This app predicts the **Boston House Price**
""")

st.write("---")

boston = datasets.load_boston()

#print(boston.isnull().sum())   #to check for null value in dataset

X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

st.sidebar.header("Specify input parameters")

def userinput():

    CRIM = st.sidebar.slider("CRIM", X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider("ZN", X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider("INDUS", X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider("CHAS", X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX = st.sidebar.slider("NOX", X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider("RM", X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider("AGE", X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider("DIS", X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider("RAD", X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX = st.sidebar.slider("TAX", X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO = st.sidebar.slider("PTRATIO", X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    B = st.sidebar.slider("B", X.B.min(), X.B.max(), X.B.mean())
    LSTAT = st.sidebar.slider("LSTAT", X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())

    data = {"CRIM": CRIM,
            "ZN" : ZN,
            "INDUS" :INDUS,
            "CHAS" : CHAS,
            "NOX" : NOX,
            "RM" : RM,
            "AGE" : AGE,
            "DIS" : DIS,
            "RAD" : RAD,
            "TAX" : TAX,
            "PTRATIO" : PTRATIO,
            "B" : B,
            "LSTAT" : LSTAT

    }

    values = pd.DataFrame(data, index=[0])
    return values

dataframe = userinput()

st.header("Specified Input Values")
st.write(dataframe)
st.write("---")

model = RandomForestRegressor()
model.fit(X,Y)

prediction = model.predict(dataframe)

st.header("Prediction of MEDV (MEdian Value)")
st.write(prediction)
st.write("---")

treechart = shap.TreeExplainer(model)
shapval = treechart.shap_values(X)

st.set_option('deprecation.showPyplotGlobalUse', False) #disables warning

st.header("Value Importance")

plt.title("Feature Importance with SHAP values in mind")
shap.summary_plot(shapval, X)
st.pyplot(bbox_inches="tight")
st.write("---")

plt.title("Feature Importance base on SHAP (Bar)")
shap.summary_plot(shapval, X, plot_type="bar")
st.pyplot(bbox_inxhes="tight")



