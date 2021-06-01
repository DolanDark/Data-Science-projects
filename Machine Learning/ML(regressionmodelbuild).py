import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston, load_diabetes

st.set_page_config(page_title = "Machine Learning", layout="wide")

def model_build(dataframe):
    X = dataframe.iloc[:,:-1]   #eveything except the last column
    Y = dataframe.iloc[:,-1]   #using the last column

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size)/100)

    st.markdown("Data Splits")
    st.write("Training Set")
    st.info(X_train.shape)
    st.write("Test Set")
    st.info(X_test.shape)

    st.markdown("Variable Details")
    st.write("X variable")
    st.info(list(X.columns))
    st.write("Y Variable")
    st.info(Y.name)

    forest = RandomForestRegressor(n_estimators=para_n_estimators,
                                   random_state=para_random_state,
                                   max_features=para_max_features,
                                   min_samples_split=para_min_sample_split,
                                   min_samples_leaf=para_min_sample_leaf,
                                   criterion=para_criterion,
                                   bootstrap=para_bootstrap,
                                   oob_score=para_oob_score,
                                   n_jobs=para_n_jobs
                                   )

    forest.fit(X_train, Y_train)

    st.subheader("Model Performance")

    st.markdown("Training Set")
    Y_pred_train = forest.predict(X_train)
    st.write("Coefficient of determination (R^2) :")
    st.info(r2_score(Y_train, Y_pred_train))

    st.write("Error (MSE or MAE)")
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown("Test Set")
    Y_pred_test = forest.predict(X_test)
    st.write("Coefficient of determination (R^2)")
    st.info(r2_score(Y_test, Y_pred_test))

    st.write("Error (MSE or MAE)")
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader("Model Parameters")
    st.write(forest.get_params())



st.write("""
This is a machine learning app utilizing the RandomForestRegressor to build a regession model using the RandomForest algo
Please adjust the hyperparameters for required result""")

with st.sidebar.header("Upload your CSV data"):
    csv_upload = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    st.sidebar.markdown('''
    [Here's an example input file](https://raw.githubusercontent.com/DolanDark/Data-Science-projects/main/BioInfomatics/delaney_solubility_with_descriptors.csv)''')

with st.sidebar.header("Set Parameters"):
    split_size = st.sidebar.slider("Data split ratio (for training set)", 10, 90, 80, 5)

with st.sidebar.subheader("Learning Parameters"):
    para_n_estimators = st.sidebar.slider("Number of Estimators (n_estimators)", 1, 1000, 100, 100)
    para_max_features = st.sidebar.select_slider("Max features (max_features)",options=["auto", "sqrt", "log2"])
    para_min_sample_split = st.sidebar.slider("Minimum number of samples required to split an internal node (min_sample_split)", 1, 10, 2, 1 )
    para_min_sample_leaf = st.sidebar.slider("Minimum number of samples required for leaf node (min_sample_leaf)",1, 10, 2, 1)

with st.sidebar.subheader("General Parameters"):
    para_random_state = st.sidebar.slider("Seed number (random_state)",0, 1000, 42, 1 )
    para_criterion = st.sidebar.select_slider("Performance measure (criterion)", options=["mse", "mae"])
    para_bootstrap = st.sidebar.select_slider("Bootstrap Samples (bootstrap)", options=["True", "False"])
    para_oob_score = st.sidebar.select_slider("Whether to use out-of-bag samples to estimate R^2 (oob_score)", options=["False", "True"])
    para_n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1, -1])

st.subheader("Dataset")

if csv_upload is not None:
    df = pd.read_csv(csv_upload)
    st.markdown("Glimpse of Dataset")
    st.write(df)
    model_build(df)

else:
    st.info("Please upload CSV file")
    if st.button("Press here to use examples data set"):
        #uncomment this for diabetes dataset
        # diabetes = load_diabetes()
        # X_dia = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        # Y_dia = pd.Series(diabetes.target, name="response")
        # df = pd.concat([X_dia, Y_dia], axis=1)
        # st.markdown("The Diabetes dataset is being used")
        # st.write(df.head(5))
        # model_build(df)

        boston = load_boston()
        X_bos = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y_bos = pd.Series(boston.target, name='response')
        df = pd.concat([X_bos, Y_bos], axis=1)
        st.markdown("The boston housing dataset is being used")
        st.write(df.head(5))
        model_build(df)

