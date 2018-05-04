import numpy as np
# import scipy as sp
import matplotlib

#########################################################
# testing python to generate probability distributions
#########################################################

# Note: different behaviour using
# math.log2(p) and np.log(p)/np.log(2)
h = lambda p: -(p*np.log2(p) + (1-p)*np.log2(1-p));

def I(p,q):
    r = p/(p+q)
    foo = 0.5*(1-h(r))
    bar = 0.5*h(r)
    res = 1.0 + foo + bar
    return res

# range of all [0,1]
ps =  np.linspace(0,1,num=1000);
qs = 1- ps;

#mock values
p = q = 0.5
# Z channel vector
# Pzz = np.array([[p,p],[q,q]]);
Pzz = np.array([p,p,q,q]);
print(I(ps,qs));