import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier



penguins = pd.read_csv("penguins_cleaned.csv")

dataframe = penguins.copy()

predict = "species"
encode = ["sex","island"]

for col in encode:
    dummy_var = pd.get_dummies(dataframe[col], prefix = col)
    dataframe = pd.concat([dataframe, dummy_var], axis =1)
    del dataframe[col]

predict_map = {"Adelie":0, "Chinstrap":1, "Gentoo":2}

def predict_encode(val):
    return predict_map[val]

dataframe["species"] = dataframe["species"].apply(predict_encode)

X = dataframe.drop("species", axis =1)
Y = dataframe["species"]

CLF = RandomForestClassifier()
CLF.fit(X,Y)

pickle.dump(CLF, open("penguins_clf.pkl", "wb"))

