import streamlit
import pandas
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

streamlit.write('''
# Sipmle iris prediction App

This app predicts simple **iris flower** type
''')
streamlit.sidebar.subheader("User input Parameters")

def user_input():
    sepal_length = streamlit.sidebar.slider("Sepal Length", 4.3, 7.9, 5.3)
    sepal_width = streamlit.sidebar.slider("Sepal Width",2.0, 4.4, 3.4)
    petal_length = streamlit.sidebar.slider("Petal Length",1.0, 6.9, 1.3 )
    petal_width = streamlit.sidebar.slider("Petal Width, ", 0.1, 2.5, 0.2)
    data = {
        "sepal_length" : sepal_length,
        "sepal_width" : sepal_width,
        "petal_length" : petal_length,
        "petal width" : petal_width
    }
    features = pandas.DataFrame(data, index=[0])
    return features

DF = user_input()

streamlit.subheader("User Input Parameters")

streamlit.write(DF)

iris = datasets.load_iris()
x = iris.data
y = iris.target

CLF = RandomForestClassifier()
CLF.fit(x,y)

prediction = CLF.predict(DF)
prediction_proba = CLF.predict_proba(DF)

streamlit.subheader("Class labels and their corresponding Index Numbers ")
streamlit.write(iris.target_names)

streamlit.subheader("Prediction")
streamlit.write(iris.target_names[prediction])

streamlit.subheader("Prediction Probability")
streamlit.write(prediction_proba)


