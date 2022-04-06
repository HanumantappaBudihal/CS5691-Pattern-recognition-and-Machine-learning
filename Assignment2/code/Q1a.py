import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =np.array(pd.read_csv("../dataset/A2Q1.csv"))
n = data.shape[0]
epsilon = 0.0001
overall_likelihood = []

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

def likelihood(x,s,p):
    logl = 0
    for i in range(x.shape[0]):
        inside = 0
        for k in range(len(s)):
            inside += p[k]*exponential1(x[i],s[k])
        logl += np.log(inside)    
    return logl

def init(data, k):
    s = np.random.rand(k)
    p = np.random.rand(k)
    p = p/p.sum()
    return s,p

# EM algo
for itr in range(100): #iteration
    s,p = init(data,4) # k=4
    ll = []

    while(True):        
        l = lamda_mml(data,s,p)      
        s,p = maximize_log_likelihood(data,l)
        ll.append(likelihood(data,s,p))

        if len(ll)>1 and np.abs(np.abs(ll[-1])-np.abs(ll[-2])) < epsilon:
            break;
    
    overall_likelihood.append(ll)

# mean of  rows 
mlen = len(max(overall_likelihood, key=len))
for i in range(len(overall_likelihood)):
    if len(overall_likelihood[i]) < mlen:
        overall_likelihood[i] = overall_likelihood[i] + (mlen-len(overall_likelihood[i]))*[overall_likelihood[i][-1]]


avg_likelihood = np.mean(overall_likelihood, axis=0)
#original data
plt.hist(data)
plt.xlabel("x")
plt.ylabel("no of points")
plt.savefig("../plots/Q1a_1.png")

plt.close()
plt.cla()
plt.clf()

#graph-> log likelihood vs iterations 
plt.plot(avg_likelihood)
plt.xlabel('Ierations')
plt.ylabel('Log likelihood')
plt.savefig("../plots/Q1a_2.png")