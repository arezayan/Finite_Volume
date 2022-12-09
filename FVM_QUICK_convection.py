'''
Problem is solution a convection-diffusion 1D equation by finite volume method. 
For approximation of convection terms QUICK scheme is used.


'''



#Solve FVM QUICK convection-diffusion 1D equation

import numpy as np
import matplotlib.pyplot as plt

#Domain data
n=15    #number of control volumes
a=np.zeros((n,n))   #initial matriX domain
ro=1    #Density
u0=0.2  #velocity
l=1    #length of Domain
dx=l/n  #distance between nodes
sigma = 0.1

#Diffusin and convection term
F=ro*u0
D=sigma / dx

#values of phi at boundaries of the Domain
phiA=1
phiB=0

#first Loop for arranging nodes values
for i in range(n):
    for j in range(n):
        if i==j:
            a[i,j]=3/8*F + 2*D  #1.075
        if i==j-1:
            a[i,j]=3/8*F - D  #-0.675
        if i==j+2:
            a[i,j]=F/8  #0.025
        if i==j+1:
            a[i,j]=(-7*F/8) - D  #-0.675
a[0,0]=7/8*F+4*D
a[0,1] = 3/8*F - 4/3*D
a[1,0] = -F-D
a[n-1,-2] = -6/8*F - 4/3 * D
a[-1,-1]=-3/8*F + 4*D

#answer matrix
b=np.zeros((n,1)).reshape(n,1)
b[0]=2/8*F + phiA*F + 8/3 * D * phiA
b[1]=-2*(F/8) * phiA
b[-1]=-F*phiB + D/3*8*phiB
#print OUt values of created Matrix
print(b)
print(a)

#Gauss-Sidel iteration method
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


x_init=np.zeros(n).reshape(n,1)
epsilon=0.0001
conv_criteria=np.array([epsilon]*n).reshape(n,1)

###Iteration process
iteration_number=10000

x=[None]*n
for i in range(1,iteration_number):
    x[0]=np.dot(T,x_init) + C
    x[i]=np.dot(T,x[i-1]) + C
    x.append(x[i])
    
    check_conv=np.less(x[i]-x[i-1],conv_criteria)
    if check_conv.any()==True:
        print('CONVERGED!')
        break
        

#calculating Analytical solution
x_exact = np.linspace(0.1,1,n)
phi_exact = 1-(((np.exp(ro*u0*x_exact/sigma))-1)/((np.exp(ro*u0*l/sigma))-1))
print(x[-1])

#postProcessing
plt.plot(x_exact,phi_exact,'k',ls='',marker='+',label = 'exact solution')
plt.plot(x_exact,x[-1],marker='^',label='Quick Scheme')

plt.legend()
plt.show()