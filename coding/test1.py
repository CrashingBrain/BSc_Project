import numpy as np
# import scipy as sp #TODO doesn't work
import matplotlib
from functools import reduce

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
    res = 0
    for p in px:
        res += p*np.log2(p)
    return -res

def I(r):
    """Mutual inforfomation

        Input:
        p:      probability of channel P_ZZ(0,0), P_ZZ(1,0)
        q:      probability of channel P_ZZ(0,1), P_ZZ(1,1)

        Output:
        res:    mutal information of 
    """
    # r = p/(p+q)
    foo = 0.5*(1-h(r))
    bar = 0.5*h(r)
    res = 1.0 + foo + bar
    return res
#########################################################

#########################################################
# Utilities  on interpolation towards uniform distribution
#
alpha,stepsize = np.linspace(0,1,num=20, retstep=True)
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
    # take the p/(p+q) value from the first element of the table
    #TODO adjust for arbitrarely dimension
    foop = 4.0*table[0,0]
    results.append( I(foop) )

print(results)
print("*********")
a = marginals(Ptable)
print(a)