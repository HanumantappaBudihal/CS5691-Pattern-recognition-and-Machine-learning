import numpy as np
import matplotlib.pyplot as plt
import random 

file = open("../dataset/A2Q1.csv")
data = np.loadtxt(file, delimiter=",")

n = data.shape[0]
k = 4
ite = 100
epsilon = 0.0001
allLL = []

def exponential1(x,s):
    return (s * np.exp(-(s*x)))

def lamda_mml(x,s,p):
    n = x.shape[0]
    k = len(s)
    l = np.zeros((n,k), dtype= float)
    for i in range(n):
        for j in range(k):
            l[i][j] = p[j]*exponential1(x[i],s[j])
    
        l[i] /= np.sum(l[i]) 

    return l

def maximize_log_likelihood(x,l):
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
def likelihood(x,s,p):
    logl = 0
    for i in range(x.shape[0]):
        inside = 0
        for k in range(len(s)):
            inside += p[k]*exponential1(x[i],s[k])
        logl += np.log(inside)    
    return logl

# initialize parameters
def init(data, k):
    # random initialization of parameters
    s = np.random.rand(k)
    p = np.random.rand(k)
    p = p/p.sum()
    return s,p

# EM algo
for itr in range(ite):
    s,p = init(data,k)
    ll = []

    convergence = True
    while(convergence):        
        l = lamda_mml(data,s,p)      
        s,p = maximize_log_likelihood(data,l)
        ll.append(likelihood(data,s,p))

        if len(ll)>1 and np.abs(np.abs(ll[-1])-np.abs(ll[-2])) < epsilon:
            convergence = False
    
    allLL.append(ll)

# mean of  rows 
mlen = len(max(allLL, key=len))
for i in range(len(allLL)):
    if len(allLL[i]) < mlen:
        allLL[i] = allLL[i] + (mlen-len(allLL[i]))*[allLL[i][-1]]


avgLL = np.mean(allLL, axis=0)
#original data
plt.hist(data)
plt.xlabel("x")
plt.ylabel("no of points")
plt.savefig("../plots/Q1a_1.png")

plt.close()
plt.cla()
plt.clf()

#graph-> log likelihood vs iterations 
plt.plot(avgLL)
plt.xlabel('Ierations')
plt.ylabel('Log likelihood')
plt.savefig("../plots/Q1a_2.png")