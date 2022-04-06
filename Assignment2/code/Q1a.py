import numpy as np
import matplotlib.pyplot as plt
import random 

# calculate exponential pdf
def expo(x,s):
    return (s * np.exp(-(s*x)))

# expectation - calculate lambda to maximize log likelihood
def lMML(x,s,p):
    n = x.shape[0]
    k = len(s)
    l = np.zeros((n,k), dtype= float)
    for i in range(n):
        for j in range(k):
            l[i][j] = p[j]*expo(x[i],s[j])
            
        l[i] /= np.sum(l[i]) 

    return l

# maximization - calculate parameters to maximize log likelihood
def pMML(x,l):
    s = np.zeros(l.shape[1])
    p = np.zeros(l.shape[1])
    for k in range(l.shape[1]):
        total = np.sum(l[:,k])
        for i in range(l.shape[0]):
            s[k] += (l[i][k]*(x[i]))
        s[k] = total / s[k]
        p[k] = total / l.shape[0]

    return s,p

# calculate log likelihood
def calcLL(x,s,p):
    logl = 0
    for i in range(x.shape[0]):
        inside = 0
        for k in range(len(s)):
            inside += p[k]*expo(x[i],s[k])
        logl += np.log(inside)    
    return logl

# initialize parameters
def init_params(data, k):
    # random initialization of parameters
    s = np.random.rand(k)
    p = np.random.rand(k)
    p = p/p.sum()
    return s,p

# load the dataset
file = open("../dataset/A2Q1.csv")
data = np.loadtxt(file, delimiter=",")

# no of datapoints
n = data.shape[0]
# no of distribution in mixture
k = 4
# no of random initializations
no_of_iterations = 30
# tolerence value
epsilon = 0.0001
# to store log likelihoods of all initializations
allLL = []

# EM Algorithm
for itr in range(no_of_iterations):
    print(itr)
    s,p = init_params(data,k)

    # to store log likelihoods
    ll = []
    
    convergence = True
    while(convergence):

        # estimation step
        l = lMML(data,s,p)
        
        # maximization step
        s,p = pMML(data,l)

        # calculate log likelihood
        ll.append(calcLL(data,s,p))

        # check convergence
        if len(ll)>1 and np.abs(np.abs(ll[-1])-np.abs(ll[-2])) < epsilon:
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
plt.title('log-likelihood (averaged over 100 random initializations) as a function of iterations')
plt.xlabel('No. of iterations')
plt.ylabel('log likelihood')
plt.show()