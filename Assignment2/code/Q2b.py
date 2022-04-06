
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt

def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data_X = data.iloc[:, :-1]  # All features exepct the last item - X values
    data_Y = data.iloc[:, -1]  # Last feature ( that is y value) - Y values

    return data_X, data_Y

def linear_regression(X_train,Y_train):
  W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
  return W

def gradient_descent(iterations,learning_rate,X_train,Y_train):
  W= np.matrix(np.ones((100,1)))
  
  A = X_train.T @ X_train
  B = X_train.T @ Y_train
  W_ml = np.matrix(linear_regression(X_train,Y_train))
  
  error = []
  for i in range(iterations):
    W = W - learning_rate * (A @ W - B)
    error.append( np.power(np.sum(np.power((np.subtract(W , W_ml)),2)),1/2) )

  return W,error

X_train,Y_train=read_data('../dataset/A2Q2Data_train.csv')
X_train = np.matrix(X_train)
Y_train = np.matrix(Y_train).T

W_gd , error = gradient_descent(5000,0.000007,X_train,Y_train)

plt.plot(error)
plt.xlabel("no of iterations (t)")
plt.ylabel("least square error (|w_t - w_ml|)")
plt.savefig("../plots/Q2b.png",dpi=400)


