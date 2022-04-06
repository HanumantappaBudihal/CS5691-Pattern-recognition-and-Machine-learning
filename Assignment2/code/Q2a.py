import pandas as pd
import numpy as np


def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1]  # All features exepct the last item - X values
    Y = data.iloc[:, -1]  # Last feature ( that is y value) - Y values

    return X, Y

def linear_regression(X, Y):
    W = np.linalg.inv(X.T @ X) @ X.T @ Y
    return W

X, Y = read_data("../dataset/A2Q2Data_train.csv")
w_ML = np.matrix(linear_regression(X, Y))

print(w_ML)