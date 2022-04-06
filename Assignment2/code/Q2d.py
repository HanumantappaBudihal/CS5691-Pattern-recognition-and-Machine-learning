import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data_X = data.iloc[:, :-1]  # All features exepct the last item - X values
    data_Y = data.iloc[:, -1]  # Last feature ( that is y value) - Y values

    return data_X, data_Y

X_train, Y_train = read_data('../dataset/A2Q2Data_train.csv')
X_test, Y_test = read_data('../dataset/A2Q2Data_test.csv')

X_train = np.matrix(X_train)
Y_train = np.matrix(Y_train).T

X_test = np.matrix(X_test)
Y_test = np.matrix(Y_test).T

def linear_regression(X_train, Y_train):
    W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
    return W

W_ml= linear_regression(X_train,Y_train)

def ridge_regression(iterations, l_reg, n=0.000001):
    W = np.matrix(np.ones((100, 1)))

    px_train = X_train[:7000][:]
    py_train = Y_train[:7000]

    px_test = X_train[3000:][:]
    py_test = Y_train[3000:]

    C = px_train.T @ px_train
    D = px_train.T @ py_train

    for i in range(iterations):
        W = W - n * (C @ W - D + l_reg * np.identity(100) @ W)

    errorr = np.sum(
        np.power((np.subtract(py_test, px_test @ W)), 2))/len(py_test)
    return W, errorr


lambda_reg=[1 * i/10 for i in range(42,47)]
error_l = []
W_temp = []

for i in lambda_reg:
    t, e = ridge_regression(6000, i)
    error_l.append(e)
    W_temp.append(t)

print(lambda_reg,error_l)
plt.plot(lambda_reg,error_l)
plt.xlabel("lambda")
plt.ylabel("MSEt")
plt.savefig("../plots/Q2d.png", dpi=300)

test_error_ml = np.sum(np.power((np.subtract(Y_test, X_test @ W_ml)), 2))/len(Y_test)
W_t_reg = W_temp[lambda_reg.index(4.2)]
test_error_reg = np.sum(np.power((np.subtract(Y_test, X_test @ W_t_reg)), 2))/len(Y_test)

print(test_error_ml, test_error_reg)
print("percentage difference in error", (test_error_ml-test_error_reg)*100)
