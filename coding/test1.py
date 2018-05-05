import numpy as np
# import scipy as sp
import matplotlib
from functools import reduce

#########################################################
# testing python to generate probability distributions
#########################################################

# Joint probability table
# Order of dimensions X,Y,Z,U (4x4x2x2)

Ptable = np.zeros((4,4,2,2))
Ptable[0,0,0,0] = Ptable[1,1,0,0] = Ptable[0,1,1,0] = Ptable[1,0,1,0] = 1.0/8.0
Ptable[2,2,0,1] = Ptable[3,3,1,1] = 1.0/4.0

PP = Ptable[:,:,0,0] + Ptable[:,:,0,1] + Ptable[:,:,1,1] + Ptable[:,:,1,0]

# Note: different behaviour using
# math.log2(p) and np.log(p)/np.log(2)
h = lambda p: -(p*np.log2(p) + (1-p)*np.log2(1-p))

def I(p,q):
    r = p/(p+q)
    foo = 0.5*(1-h(r))
    bar = 0.5*h(r)
    res = 1.0 + foo + bar
    return res

alpha,stepsize = np.linspace(0,1,num=2000, retstep=True)
def step_linear(Ptable):
    sh = np.shape(Ptable)
    norm = 1.0/reduce((lambda x,y: x*y), sh)
    normal = np.ones_like(Ptable) * norm
    PPs = alpha[1]*normal + (1.0-alpha[1])*Ptable

    return PPs


# Renormalize the joint probability distribution
def normalize(Ptable):
    sh = np.shape(Ptable)
    n = reduce((lambda x, y: x * y), sh)

    Ptable = 1/n * Ptable

# range of all [0,1]
ps = np.linspace(0,1,num=1000)
qs = 1-ps

#mock values
p = q = 0.5
# Z channel vector
# Pzz = np.array([[p,p],[q,q]]);
Pzz = np.array([p,p,q,q])
# print(I(ps,qs))
print(Ptable[0,0,0,0])
normalize(Ptable)
print(Ptable[0,0,0,0])
print("*********")
print(PP)
normalize(PP)
print("--")
print(PP)
print("*********")
PPs = step_linear(PP)
print(np.shape(PPs))
print(PPs)