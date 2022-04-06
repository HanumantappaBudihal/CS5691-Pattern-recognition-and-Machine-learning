# import numpy as np
# import csv
# import matplotlib.pyplot as plt
# x = []

# with open('A2Q1Data.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter = ',')
#     for row in plots:
#         x.append(int(row[0]))

# X = np.array(x)
# X = X.reshape((1000,1))
# N = 1000
# # plt.hist(X, bins = 100)
# # plt.title('Original data')
# # plt.xlabel('X (data points)')
# # plt.ylabel('PMF(X)')
# # plt.show()

# No_Iterations = 30
# K  = 4

# np.random.seed(0)
# Lambda = np.zeros((N,K))
# denom=np.array(0)
# logLL = np.zeros((30,1))
# for I in range(0, 100):
#     np.random.seed(0) 
#     P = np.random.default_rng().uniform(0,1,K)
#     np.random.seed(0) 
#     Pi = np.random.default_rng().uniform(0,1,K)
#     Pi /= Pi.sum()
#     for T in range(No_Iterations):
#         Temp_P = np.zeros((K,1))
#         for i in range(N):
#             denom = 0
#             for l in range(K):
#                 denom += (pow(P[l],X[i]) * pow(1-P[l], 1-X[i])) * Pi[l]
#             for k in range(K):
#                 Lambda[i][k] = ((pow(P[k],X[i]) * pow(1-P[k], 1-X[i])) * Pi[k]) / denom
#                 logLL[T] +=  Lambda[i][k] * (np.log(Pi[k]) + np.log(pow(P[l],X[i]) * pow(1-P[l], 1-X[i])) - np.log(Lambda[i][k]))
#                 Temp_P[k] += Lambda[i][k] * X[i]
#         Pi = Lambda.sum(axis = 0)
#         for k in range(K):
#             P[k] = Temp_P[k]/Pi[k]
#         Pi = Pi/N
# logLL = logLL / 100
# for i in range(30):
#     plt.scatter(i,logLL[i])
# plt.title('BMM')
# plt.xlabel('No of Iterations')
# plt.ylabel('Log likelihood')
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("../dataset/A2Q1.csv")

plt.hist(data)
plt.show()