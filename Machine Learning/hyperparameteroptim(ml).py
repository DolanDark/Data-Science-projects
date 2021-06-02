import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes, load_boston

st.set_page_config(page_title="Hyperparameter Optimization Machine Learning App", layout="wide")

st.write('''
# Hyperparameter Machine Learning Optimization app
''')

st.sidebar.header("Upload your CSV data")
file_upload = st.sidebar.file_uploader("Upload your CSV input file", type=["csv"])

st.sidebar.markdown('''
[Example CSV input file] (https://raw.githubusercontent.com/DolanDark/Data-Science-projects/main/BioInfomatics/delaney_solubility_with_descriptors.csv)''')

st.sidebar.header("Set Parameters")
split_size = st.sidebar.slider("Data split ratio (% for training set)", 10, 90, 80, 5)

st.sidebar.subheader("Learning Parameters")
param_n_estimators = st.sidebar.slider("Number of estimators (n estimators)", 0, 500, (10, 50), 5)
param_n_estimators_step = st.sidebar.number_input("Step size of n estimators", 10)
st.sidebar.write("---")
param_max_features = st.sidebar.slider("Max features (max_features)", 1, 50, (1, 3), 1)

st.sidebar.number_input("Step size for max_features", 1)
st.sidebar.write("---")

param_min_samples_split = st.sidebar.slider("Minimum number of samples required to split internal node (min_samples_split)",1, 10, 2, 1)
param_min_samples_leaf = st.sidebar.slider("Minimum number of samples required to be at leaf node (min_samples_leaf)",1, 10, 2, 1)

st.sidebar.subheader("General Parameters")

param_random_state = st.sidebar.slider("Seed Number (random state)",0, 1000, 41, 1)
param_criterion = st.sidebar.select_slider("Performance measure (criterion)", options=["mse", "mae"])
param_bootstrap = st.sidebar.select_slider("Bootstrap Samples (bootstrap)", options=["True", "False"])
param_oob_score = st.sidebar.select_slider("To use out-of-bag samples to estimate R^2 (oob_score)", options=["False", "True"])
param_n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1, -1])

n_estimators_range = np.arange(param_n_estimators[0],param_n_estimators[1]+param_n_estimators_step, param_n_estimators_step)
max_features_range = np.arange(param_max_features[0], param_max_features[1]+1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

st.subheader("Datasets")

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64, {b64}" download="model_performance.csv"> Download File </a>'
    return href

def build_model(df):
    X = df.iloc[:,:-1]  #using all columns except the last one
    Y = df.iloc[:, -1]  #using only the last column

    st.markdown("A model is being build to predict the Y variable")
    st.info(Y.name)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    #st.write(X_train.shape, Y_train.shape)
    #st.write(X_test.shape, Y_test.shape)

    ranfo = RandomForestRegressor(n_estimators=param_n_estimators,
                                  random_state=param_random_state,
                                  max_features=param_max_features,
                                  criterion=param_criterion,
                                  min_samples_split=param_min_samples_split,
                                  min_samples_leaf=param_min_samples_leaf,
                                  bootstrap=param_bootstrap,
                                  oob_score=param_oob_score,
                                  n_jobs=param_n_jobs)

    grid = GridSearchCV(estimator=ranfo, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)

    st.subheader("Model Performance")

    Y_pred_test = grid.predict(X_test)
    st.write("Coefficient of determination (R^2)")
    st.info(r2_score(Y_test, Y_pred_test))

    st.write("Error (MSE or MAE):")
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.write("The best parameters are %s with a score of %0.2f" %(grid.best_params_, grid.best_score_))

    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])], axis=1)
    grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    X_grid = grid_pivot.columns.levels[1].values
    Y_grid = grid_pivot.index.values
    Z_grid = grid_pivot.values

    layout = go.Layout(
        xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = 'n_estimators')),
        yaxis = go.layout.YAxis(title= go.layout.yaxis.Title(text = 'max_features'))
        )

    fig = go.Figure(data= [go.Surface(z = Z_grid, y = Y_grid, x = X_grid)], layout = layout)
    fig.update_layout(title = "Hyperparameter Tuning",
                      scene = dict(xaxis_title = "n_estimators",
                                   yaxis_title = "max_features",
                                   zaxis_title = "R2"),
                      autosize = False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))

    st.plotly_chart(fig)

    X_df = pd.DataFrame(X_grid)
    Y_df = pd.DataFrame(Y_grid)
    Z_df = pd.DataFrame(Z_grid)

    df = pd.concat([X_df, Y_df, Z_df], axis=1)
    st.markdown(file_download(grid_results), unsafe_allow_html=True)

if file_upload is not None:
    df = pd.read_csv(file_upload)
    st.write(df)
    build_model(df)

else:
    st.info("Awaiting for the CSV to be uploaded")
    if st.button("Press to use the Example Dataset : "):
        boston = load_boston()
        X_bos = pd.DataFrame(boston.data, columns = boston.feature_names)
        Y_bos = pd.Series(boston.target, name="response")
        df = pd.concat([X_bos, Y_bos], axis=1)
        st.markdown("The Boston housing dataset is being used")
        st.write(df.head(5))
        build_model(df)


