import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data_X = data.iloc[:, :-1]  # All features exepct the last item - X values
    data_Y = data.iloc[:, -1]  # Last feature ( that is y value) - Y values

    return data_X, data_Y

def linear_regression(X_train,Y_train):
  W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
  return W

def stochastic_graient_descent(iterations,X_train,Y_train,batch_size=50,n=0.0001):
  no_of_batch = int(len(X_train)/batch_size)
  iterations = int(iterations/no_of_batch)

  errors=[]
  W = np.matrix(np.ones((100,1)))
  W_ml = np.matrix(linear_regression(X_train,Y_train))
  batch_X = [ X_train[(i*batch_size):(i+1)*batch_size] for i in range(no_of_batch)]
  batch_Y = [Y_train[(i*batch_size):(i+1)*batch_size] for i in range(no_of_batch)]

  for i in range(iterations):
    for j in range(no_of_batch):
      X = np.matrix(batch_X[j])
      Y = np.matrix(batch_Y[j])
      W = W - n*( X.T @ X @ W - X.T @ Y)
      errors.append( np.power(np.sum(np.power((np.subtract(W , W_ml)),2)),1/2) )

  return W,errors

X_train,Y_train=read_data('../dataset/A2Q2Data_train.csv')
X_train = np.matrix(X_train)
Y_train = np.matrix(Y_train).T
W_sgd,error_sgd = stochastic_graient_descent(10000,X_train,Y_train,100,0.0001)

plt.plot(error_sgd)
plt.plot(error_sgd,"r")
plt.xlabel("no of batches (t) (batch size=100)")
plt.ylabel("least square error (|w_t - w_ml|)")
plt.savefig("../plots/Q2c.png",dpi=300)

