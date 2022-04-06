import pandas as pd
import numpy as np


def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data_X = data.iloc[:, :-1]  # All features exepct the last item - X values
    data_Y = data.iloc[:, -1]  # Last feature ( that is y value) - Y values

    return data_X, data_Y

def linear_regression(X_train, Y_train):
    W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
    return W

#linear regression
X_train, Y_train = read_data("../dataset/A2Q2Data_train.csv")
w_ML = np.matrix(linear_regression(X_train, Y_train))

print(w_ML)