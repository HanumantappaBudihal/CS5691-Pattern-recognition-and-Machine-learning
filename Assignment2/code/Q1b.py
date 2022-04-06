import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


def maximize_ll(x, lamda_values):
    mu = np.zeros(lamda_values.shape[1])
    simga = np.zeros(lamda_values.shape[1])
    pi = np.zeros(lamda_values.shape[1])

    for k in range(lamda_values.shape[1]):
        total = np.sum(lamda_values[:, k])

        for i in range(lamda_values.shape[0]):
            mu[k] += (lamda_values[i][k]*x[i])
        mu[k] /= total

        for i in range(lamda_values.shape[0]):
            simga[k] += (lamda_values[i][k]*((x[i]-mu[k])**2))
        simga[k] /= total
        pi[k] = total / lamda_values.shape[0]

    return mu, simga, pi

def gaussian(x, mu, simga):
    return (np.exp(-((x-mu)**2)/(2*simga)) / np.sqrt(2*np.pi*simga))

def lambda_mml(x, mu, simga, pi):
    n = x.shape[0]
    k = len(mu)
    lamda_values = np.zeros((n, k), dtype=float)
    for i in range(n):
        for j in range(k):
            lamda_values[i][j] = pi[j]*gaussian(x[i], mu[j], simga[j])
        lamda_values[i] /= np.sum(lamda_values[i])

    return lamda_values

def calculate_ll(x, mu, simga, pi):
    logl = 0
    for i in range(x.shape[0]):
        inside = 0
        for k in range(len(mu)):
            inside += pi[k]*gaussian(x[i], mu[k], simga[k])
        logl += np.log(inside)
    return logl


def init(data, k):
    mu = random.sample(list(data), k)
    simga = np.random.rand(k)*(data.shape[0] / k)
    pi = np.random.rand(k)
    pi = pi/pi.sum()
    return mu, simga, pi


# load the dataset
data = data = np.array(pd.read_csv("../dataset/A2Q1.csv"))
n = data.shape[0]
epsilon = 0.0001
overall_likelihood = []

for itr in range(1):  # iterations
    mu, simga, pi = init(data, 4)  # k=4
    ll = []

    while(True):  # until convergence
        lamda_values = lambda_mml(data, mu, simga, pi)
        mu, simga, pi = maximize_ll(data, lamda_values)
        ll.append(calculate_ll(data, mu, simga, pi))

        if len(ll) > 1 and np.abs(np.abs(ll[-1])-np.abs(ll[-2])) < epsilon:
            break

    overall_likelihood.append(ll)

mlen = len(max(overall_likelihood, key=len))
for i in range(len(overall_likelihood)):
    if len(overall_likelihood[i]) < mlen:
        overall_likelihood[i] = overall_likelihood[i] + \
            (mlen-len(overall_likelihood[i]))*[overall_likelihood[i][-1]]

avg_likelihood = np.mean(overall_likelihood, axis=0)

plt.plot(avg_likelihood)
plt.xlabel('No. of iterations')
plt.ylabel('log likelihood')
plt.savefig("../plots/Q1b.png", dpi=300)
