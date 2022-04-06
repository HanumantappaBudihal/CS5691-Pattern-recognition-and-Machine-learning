import numpy as np
import matplotlib.pyplot as plt
import random

# calculate gaussian pdf


def gauss(x, m, s):
    return (np.exp(-((x-m)**2)/(2*s)) / np.sqrt(2*np.pi*s))

# expectation - calculate lambda to maximize log likelihood


def lMML(x, m, s, p):
    n = x.shape[0]
    k = len(m)
    l = np.zeros((n, k), dtype=float)
    for i in range(n):
        for j in range(k):
            l[i][j] = p[j]*gauss(x[i], m[j], s[j])

        l[i] /= np.sum(l[i])

    return l

# maximization - calculate parameters to maximize log likelihood


def pMML(x, l):

    m = np.zeros(l.shape[1])
    s = np.zeros(l.shape[1])
    p = np.zeros(l.shape[1])

    for k in range(l.shape[1]):
        total = np.sum(l[:, k])

        # calculate means
        for i in range(l.shape[0]):
            m[k] += (l[i][k]*x[i])
        m[k] /= total

        # calculate variances
        for i in range(l.shape[0]):
            s[k] += (l[i][k]*((x[i]-m[k])**2))
        s[k] /= total

        # calculate probabilities of each distribution in mixture
        p[k] = total / l.shape[0]

    return m, s, p

# calculate log likelihood


def calcLL(x, m, s, p):
    logl = 0
    for i in range(x.shape[0]):
        inside = 0
        for k in range(len(m)):
            inside += p[k]*gauss(x[i], m[k], s[k])
        logl += np.log(inside)
    return logl

# initialize parameters


def init_params(data, k):
    # random initialization of parameters
    m = random.sample(list(data), k)
    s = np.random.rand(k)*(data.shape[0] / k)
    p = np.random.rand(k)
    p = p/p.sum()
    return m, s, p


# load the dataset
file = open("../dataset/A2Q1.csv")
data = np.loadtxt(file, delimiter=",")

# no of datapoints
n = data.shape[0]
# no of distribution in mixture
k = 4
# no of random initializations
no_of_iterations = 100
# tolerence value
epsilon = 0.0001
# to store log likelihoods of all initializations
allLL = []

# EM Algorithm
for itr in range(no_of_iterations):

    m, s, p = init_params(data, k)

    # to store log likelihoods
    ll = []

    convergence = True
    while(convergence):

        # estimation step
        l = lMML(data, m, s, p)

        # maximization step
        m, s, p = pMML(data, l)

        # calculate log likelihood
        ll.append(calcLL(data, m, s, p))

        # check convergence
        if len(ll) > 1 and np.abs(np.abs(ll[-1])-np.abs(ll[-2])) < epsilon:
            convergence = False

    allLL.append(ll)

# to make the length of all rows same
mlen = len(max(allLL, key=len))
for i in range(len(allLL)):
    if len(allLL[i]) < mlen:
        allLL[i] = allLL[i] + (mlen-len(allLL[i]))*[allLL[i][-1]]

# calculate average log likelihood over random initializations
avgLL = np.mean(allLL, axis=0)

# plot the average log likelihood vs iterations graph
plt.plot(avgLL)
plt.xlabel('No. of iterations')
plt.ylabel('log likelihood')
plt.savefig("../dataset/Q1b.png", dpi=300)
