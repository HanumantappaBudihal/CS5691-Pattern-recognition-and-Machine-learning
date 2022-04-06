
import pandas as pd
import random
import matplotlib.pyplot as plt

def read_data(path):
  d= pd.read_csv(path,header=None)
  return d

data=read_data('../dataset/A2Q1.csv')
data=list(data[0])
K=4

def k_means(data):
  n=len(data)
  z=[ random.randint(0,K-1) for i in range(n)]
  flag=1
  objective=[]

  while flag:
    flag=0
    p=[0 for i in range(K)]
    l=[0 for i in range(K)]

    for i in range(n):
      p[z[i]]+=data[i]
      l[z[i]]+=1

    u=[ (p[i])/(l[i]+1) for i in range(K)]

    tt3=4
    for i in range(n):
      dist=[(data[i]-j)**2 for j in u ]
      t=dist.index(min(dist))
      if t != z[i]:
        flag=1

      z[i]=t

    objective.append(sum([ (data[i]-u[z[i]])**2 for i in range(n)]))
  
  return objective,z,u

c,ass,mean=k_means(data)
plt.hist(ass,orientation="horizontal")
plt.xlabel("x")
plt.ylabel("class asssigned")
plt.savefig("../plots/Q1c_1.png",dpi=300)

plt.close()
plt.cla()
plt.clf()

plt.plot(c)
plt.xlabel("no of iterations")
plt.ylabel("objective")
plt.savefig("../plots/Q1c_2.png",dpi=300)


