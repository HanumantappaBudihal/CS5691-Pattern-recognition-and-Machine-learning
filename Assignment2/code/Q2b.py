
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt

def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1]  # All features exepct the last item - X values
    Y = data.iloc[:, -1]  # Last feature ( that is y value) - Y values

    return X, Y

def linear_regression(X,Y):
  W = np.linalg.inv(X.T @ X) @ X.T @ Y
  return W

def gradient_descent(iterations,learning_rate,X,Y):
  W= np.matrix(np.ones((100,1)))
  
  A = X.T @ X
  B = X.T @ Y
  W_ml = np.matrix(linear_regression(X,Y))
  
  error = []
  for i in range(iterations):
    W = W - learning_rate * (A @ W - B)
    error.append( np.power(np.sum(np.power((np.subtract(W , W_ml)),2)),1/2) )

  return W,error

X,Y=read_data('../dataset/A2Q2Data_train.csv')
X = np.matrix(X)
Y = np.matrix(Y).T

W_gd , error = gradient_descent(5000,0.000007,X,Y)

plt.plot(error)
plt.xlabel("no of iterations (t)")
plt.ylabel("least square error (|w_t - w_ml|)")
plt.savefig("../plots/Q2b.png",dpi=400)


