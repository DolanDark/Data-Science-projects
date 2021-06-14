import pandas as pd

from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

X_pre,Y = load_iris(return_X_y=True)

# custom_dataset = pd.read_csv("delaney.csv")
# X = custom_dataset.drop("Activity", axis = 1)
# Y = custom_dataset["Activity"].copy()

select = VarianceThreshold(threshold=(0.1))     #low variance removal
X = select.fit_transform(X_pre)
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,stratify=Y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

knn = KNeighborsClassifier(3)       #K nearest neighbor
knn.fit(X_train, Y_train)

Y_train_pred = knn.predict(X_train)
Y_test_pred = knn.predict(X_test)

knn_train_accuracy = accuracy_score(Y_train, Y_train_pred)
knn_train_mcc = matthews_corrcoef(Y_train, Y_train_pred)
knn_train_f1 = f1_score(Y_train, Y_train_pred, average="weighted")

knn_test_accuracy = accuracy_score(Y_test, Y_test_pred)
knn_test_mcc = matthews_corrcoef(Y_test, Y_test_pred)
knn_test_f1 = f1_score(Y_test, Y_test_pred, average="weighted")


print(" Model Performance with K Neighbors :")
print("Accuracy :", knn_train_accuracy)
print("MCC : ", knn_train_mcc)
print("F1 score :", knn_train_f1)
print("-------------------------------")
print("Accuracy :  ", knn_test_accuracy)
print("MCC : ", knn_test_mcc)
print("F1 score : ", knn_test_f1)
print("-------------------------------")

svm_rbf = SVC(gamma=2, C=1)     #support vector machine
svm_rbf.fit(X_train, Y_train)

Y_train_pred = svm_rbf.predict(X_train)
Y_test_pred = svm_rbf.predict(X_test)

svm_rbf_train_accuracy = accuracy_score(Y_train, Y_train_pred)
svm_rbf_train_mcc = matthews_corrcoef(Y_train, Y_train_pred)
svm_rbf_train_f1 = f1_score(Y_train, Y_train_pred, average="weighted")

svm_rbf_test_accuracy = accuracy_score(Y_test, Y_test_pred)
svm_rbf_test_mcc = matthews_corrcoef(Y_test, Y_test_pred)
svm_rbf_test_f1 = f1_score(Y_test, Y_test_pred, average="weighted")

print("Model Performance with Support Vector Machine :")
print("  Accuracy :", svm_rbf_train_accuracy)
print("  MCC : ", svm_rbf_train_mcc)
print("  F1 score :", svm_rbf_train_f1)
print("-------------------------------")
print("  Accuracy :  ", svm_rbf_test_accuracy)
print("  MCC : ", svm_rbf_test_mcc)
print("  F1 score : ", svm_rbf_test_f1)
print("-------------------------------")


dec_tree = DecisionTreeClassifier(max_depth=5)      #DecisionTree
dec_tree.fit(X_train, Y_train)

Y_train_pred = dec_tree.predict(X_train)
Y_test_pred = dec_tree.predict(X_test)

dec_tree_train_accuracy = accuracy_score(Y_train, Y_train_pred)
dec_tree_train_mcc = matthews_corrcoef(Y_train, Y_train_pred)
dec_tree_train_f1 = f1_score(Y_train, Y_train_pred, average="weighted")

dec_tree_test_accuracy = accuracy_score(Y_test, Y_test_pred)
dec_tree_test_mcc = matthews_corrcoef(Y_test, Y_test_pred)
dec_tree_test_f1 = f1_score(Y_test, Y_test_pred, average="weighted")

print("Model Performance with Decision Tree :")
print("  Accuracy :", dec_tree_train_accuracy)
print("  MCC : ", dec_tree_train_mcc)
print("  F1 score :", dec_tree_train_f1)
print("-------------------------------")
print("  Accuracy :  ", dec_tree_test_accuracy)
print("  MCC : ", dec_tree_test_mcc)
print("  F1 score : ", dec_tree_test_f1)
print("-------------------------------")

mlp = MLPClassifier(alpha=1, max_iter=1000)     #Neural Network
mlp.fit(X_train, Y_train)

Y_train_pred = mlp.predict(X_train)
Y_test_pred = mlp.predict(X_test)

mlp_train_accuracy = accuracy_score(Y_train, Y_train_pred)
mlp_train_mcc = matthews_corrcoef(Y_train, Y_train_pred)
mlp_train_f1 = f1_score(Y_train, Y_train_pred, average="weighted")

mlp_test_accuracy = accuracy_score(Y_test, Y_test_pred)
mlp_test_mcc = matthews_corrcoef(Y_test, Y_test_pred)
mlp_test_f1 = f1_score(Y_test, Y_test_pred, average="weighted")

print("Model Performance with Neural Network :")
print("  Accuracy :", mlp_train_accuracy)
print("  MCC : ", mlp_train_mcc)
print("  F1 score :", mlp_train_f1)
print("-------------------------------")
print("  Accuracy :  ", mlp_test_accuracy)
print("  MCC : ", mlp_test_mcc)
print("  F1 score : ", mlp_test_f1)
print("-------------------------------")

rf = RandomForestClassifier(n_estimators=10)        #Random Forest
rf.fit(X_train, Y_train)

Y_train_pred = rf.predict(X_train)
Y_test_pred = rf.predict(X_test)

rf_train_accuracy = accuracy_score(Y_train, Y_train_pred)
rf_train_mcc = matthews_corrcoef(Y_train, Y_train_pred)
rf_train_f1 = f1_score(Y_train, Y_train_pred, average="weighted")

rf_test_accuracy = accuracy_score(Y_test, Y_test_pred)
rf_test_mcc = matthews_corrcoef(Y_test, Y_test_pred)
rf_test_f1 = f1_score(Y_test, Y_test_pred, average="weighted")

print("Model Performance with RandomForest :")
print("  Accuracy :", rf_train_accuracy)
print("  MCC : ", rf_train_mcc)
print("  F1 score :", rf_train_f1)
print("-------------------------------")
print("  Accuracy :  ", rf_test_accuracy)
print("  MCC : ", rf_test_mcc)
print("  F1 score : ", rf_test_f1)
print("-------------------------------")

estim_list = [("knn", knn),
              ("svm_rbf", svm_rbf),
              ("dec_tree", dec_tree),
              ("mlp", mlp),
              ("rf", rf)]

stacking_model = StackingClassifier(estimators=estim_list, final_estimator=LogisticRegression())

stacking_model.fit(X_train, Y_train)

Y_train_pred = stacking_model.predict(X_train)
Y_test_pred = stacking_model.predict(X_test)

stacking_model_train_accuracy = accuracy_score(Y_train, Y_train_pred)
stacking_model_train_mcc = matthews_corrcoef(Y_train, Y_train_pred)
stacking_model_train_f1 = f1_score(Y_train, Y_train_pred, average="weighted")

stacking_model_test_accuracy = accuracy_score(Y_test, Y_test_pred)
stacking_model_test_mcc = matthews_corrcoef(Y_test, Y_test_pred)
stacking_model_test_f1 = f1_score(Y_test, Y_test_pred, average="weighted")

print("Model Performance with Neural Network :")
print("  Accuracy :", stacking_model_train_accuracy)
print("  MCC : ", stacking_model_train_mcc)
print("  F1 score :", stacking_model_train_f1)
print("-------------------------------")
print("  Accuracy :  ", stacking_model_test_accuracy)
print("  MCC : ", stacking_model_test_mcc)
print("  F1 score : ", stacking_model_test_f1)
print("-------------------------------")

accu_train_list = {
    "knn": knn_train_accuracy,
    "svm_rbf": svm_rbf_train_accuracy,
    "dec_tree": dec_tree_train_accuracy,
    "mlp": mlp_train_accuracy,
    "rf": rf_train_accuracy,
    "stacking_model": stacking_model_train_accuracy
}

mcc_train_list = {
    "knn": knn_train_mcc,
    "svm_rbf": svm_rbf_train_mcc,
    "dec_tree": dec_tree_train_mcc,
    "mlp": mlp_train_mcc,
    "rf": rf_train_mcc,
    "stacking_model": stacking_model_train_mcc
}

f1_train_list = {
    "knn": knn_train_f1,
    "svm_rbf": svm_rbf_train_f1,
    "dec_tree": dec_tree_train_f1,
    "mlp": mlp_train_f1,
    "rf": rf_train_f1,
    "stacking_model": stacking_model_train_f1
}

#print(mcc_train_list)

acc_data_frame = pd.DataFrame.from_dict(accu_train_list,orient="index", columns=["Accuracy"])
mcc_data_frame = pd.DataFrame.from_dict(mcc_train_list, orient="index", columns=["MCC"])
f1_data_frame = pd.DataFrame.from_dict(f1_train_list, orient="index", columns=["F1"])

main_data_frame = pd.concat([acc_data_frame, mcc_data_frame, f1_data_frame], axis=1)
print(main_data_frame)
main_data_frame.to_csv("meta_stacking_model.csv")

