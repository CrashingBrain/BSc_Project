import numpy as np
# import scipy as sp #TODO doesn't work
import matplotlib
from functools import reduce

# Constants
epsilon = np.finfo(float).eps

#########################################################
# Joint probability table
# Order of dimensions X,Y,Z,U (4x4x2x2)

Ptable = np.zeros((4,4,2,2))
Ptable[0,0,0,0] = Ptable[1,1,0,0] = Ptable[0,1,1,0] = Ptable[1,0,1,0] = 1.0/8.0
Ptable[2,2,0,1] = Ptable[3,3,1,1] = 1.0/4.0
# Table for P_XY
PP = np.sum(Ptable, axis=(2,3))
#########################################################


#########################################################
# Utilities on probabilities
def marginals(Ptable):
    res = []
    sh = np.shape(Ptable)
    numDim = np.shape(sh)[0] # number of dimensions of the table

    # generate numDim tuples for sum
    # (1,2,3), (2,3,0), (3,0,1), (0,1,2)
    ll = list(range(0,numDim))
    tuples = []
    for i in range(numDim):
        tuples.append(tuple(ll[1:]))
        ll = ll[1:] + [ll[0]]
    
    # reduce Ptable with `sum` and get the marginals
    for tup in tuples:
        res.append(np.sum(Ptable, axis=tup))

    return res
#########################################################

#########################################################
# Utilies on Mutual Information measure

# Note: different behaviour using
# math.log2(p) and np.log(p)/np.log(2)
h = lambda p: -(p*np.log2(p) + (1-p)*np.log2(1-p))

def entropy(px):
    """Entropy of random variable x

        Input:
        px:     list containing probabilities of random variable X

        Output:
        res:    entropy of random varaible
    """
    res = 0.0
    for p in px:
        res += p*np.log2(p)
    return -res

def I(Ptable):
    """Mutual information of r.v. X and Y
    
        Input:
        Ptable:     array-like table of joint probability for X and Y

        Output:
        res:        mutual information I(X;Y)
    """
    m = marginals(Ptable)
    px = m[0]
    py = m[1]
    res = 0.0
    z = 0
    for ix,x in enumerate(px):
        for iy,y in enumerate(py):
            temp = Ptable[ix,iy]
            if temp > epsilon:
                res += temp*np.log2(temp/(x*y))

    return res

#########################################################

#########################################################
# Utilities  on interpolation towards uniform distribution
#
alpha,stepsize = np.linspace(0,1,num=10, retstep=True)
def step_linear(Ptable):
    """Linear stepping

        Input:
        Ptable: table of joint probability

        Output:
        PPs: list of tables, interpolated from Ptable
                towards uniform ditribution
    """
    sh = np.shape(Ptable)
    norm = 1.0/reduce((lambda x,y: x*y), sh)
    normal = np.ones_like(Ptable) * norm
    alphash = np.shape(alpha)

    PPs = np.empty(alphash + sh)
    
    for idx,a in enumerate(alpha):
        PPs[idx] = a*normal + (1.0-a)*Ptable

    return PPs


# Renormalize the joint probability distribution
def normalize(Ptable):
    sh = np.shape(Ptable)
    n = reduce((lambda x, y: x * y), sh)

    Ptable = 1/n * Ptable
#########################################################



# range of all [0,1]
# ps = np.linspace(0,1,num=1000)
# qs = np.linspace(0,1,num=1000)

#mock values
# p = q = 0.5
# Z channel vector
# Pzz = np.array([[p,p],[q,q]]);
# Pzz = np.array([p,p,q,q])
# print(I(ps,qs))

print("*********")
PPs = step_linear(PP)
results = []
for table in PPs:
    results.append( I(table) )

print(results)
print("*********")
a = marginals(PP)
print(a)
print(epsilon)