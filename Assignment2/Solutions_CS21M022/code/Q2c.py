import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1]  # All features exepct the last item - X values
    Y = data.iloc[:, -1]  # Last feature ( that is y value) - Y values

    return X, Y

def linear_regression(X, Y):
    W = np.linalg.inv(X.T @ X) @ X.T @ Y
    return W

def stochastic_graient_descent(ite, X, Y, bs=50, n=0.0001):
    nb = int(len(X)/bs)
    ite = int(ite/nb)

    errors = []
    W = np.matrix(np.ones((100, 1)))
    W_ml = np.matrix(linear_regression(X, Y))
    X_b = [X[(i*bs):(i+1)*bs] for i in range(nb)]
    Y_b = [Y[(i*bs):(i+1)*bs] for i in range(nb)]

    for i in range(ite):
        for j in range(nb):
            X = np.matrix(X_b[j])
            Y = np.matrix(Y_b[j])
            W = W - n*(X.T @ X @ W - X.T @ Y)
            errors.append(
                np.power(np.sum(np.power((np.subtract(W, W_ml)), 2)), 1/2))

    return W, errors

X, Y = read_data('../dataset/A2Q2Data_train.csv')
X = np.matrix(X)
Y = np.matrix(Y).T
W_sgd, error_sgd = stochastic_graient_descent(10000, X, Y, 100, 0.0001)

plt.plot(error_sgd)
plt.plot(error_sgd, "r")
plt.xlabel("no of batches (t) (batch size=100)")
plt.ylabel("least square error (|w_t - w_ml|)")
plt.savefig("../plots/Q2c.png", dpi=300)
