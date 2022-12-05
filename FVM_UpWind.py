#Solve FVM UpWind convection-diffusion 1D equation

import numpy as np
import matplotlib.pyplot as plt

n=6
a=np.zeros((n,n))
F=0.1
D=0.5
phiA=1
phiB=0

for i in range(n):
    for j in range(n):
        if i==j:
            a[i,j]=F+2*D
        if i==j-1:
            a[i,j]=-D
        if i==j+1:
            a[i,j]=-F-D
a[0,0]=F+(3*D)
a[-1,-1]=F+(3*D)
b=np.zeros((n,1)).reshape(n,1)
b[0]=phiA*(F+ 2*D)
b[-1]=2*D*phiB
print(b)
print(a)

#Gauss Sidel iteration method
#LU decomposition

U=a.copy()
L=a.copy()


for i in range (0,n):
    for j in range (0,n):        
        if i>=j:
            U[i,j]=0
for i in range (0,n):
    for j in range (0,n): 
        if j>i:
            L[i,j]=0

Linv=np.linalg.inv(L)
T=np.dot(-Linv,U)
C=np.dot(Linv,b)

####Solution####
x_init=np.zeros(n).reshape(n,1)
epsilon=0.000000001
conv_criteria=np.array([epsilon]*n).reshape(n,1)

###Iteration process
iteration_number=7

x=[None]*n
for i in range(1,iteration_number):
    x[0]=np.dot(T,x_init) + C
    x[i]=np.dot(T,x[i-1]) + C
    x.append(x[i])
    check_conv=np.less(x[i]-x[i-1],conv_criteria)
    if check_conv.any()==True:
        print('CONVERGED!')
        break

print(x[-1])
#postProcessing
#plt.plot(x[-n+1])
plt.plot(x[-1])

#plt.show()