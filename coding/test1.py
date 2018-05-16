import numpy as np
# import scipy as sp #TODO doesn't work
from matplotlib.pyplot import *
import itertools as itools
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

def create_table(n=4):
    """ Create a n-dimensions-joint probability table for X,Y
        X and Y have range n

        Input:
        n:      range of varaibles X and Y

        Output:
        Ptable: Table of joint probability for X,Y. Dimension is nxn
    """
    Ptable = np.zeros((n,n))
    h = n//2
    Ptable[:h,:h] = 1.0/(2*h*h)
    Ptable[h:,h:] = np.eye(h)*1.0/(2*h)

    return Ptable

#########################################################

#########################################################
# Utilities on probabilities

def marginals(Ptable):
    """Compute the value of marginals for a joint probability

        Input:
        Ptable:     table of joint probability

        Output:
        res:        list of probabilities of the marginals, in the same order
                    as listed in np.shape(Ptable)

    """
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

def marginal_Z(Ptable):
    """Values of the marginal Z, derived from P(X,Y)

        Input:
        Ptable: Table of joint probability P(X,Y)

        Output:
        res:    array of probabilities for marginal Z
    """
    sh = np.shape(Ptable)
    n = sh[0]//2
    res = np.zeros(n)

    # get all combinations of indices for Ptable
    combinations = [(x,y) for x in range(sh[0]) for y in range(sh[1]) if x<n and y<n]
    # combinations = list(filter(lambda t: (t[0]<n) and (t[1]<n), combinations))
    # get marginal of X
    m = marginals(Ptable)
    px = m[0]
    for z in range(n):
        # create a filter for this value of Z
        f = lambda t: (t[0] + t[1])%n == z
        # get only the indices that satisfy X+Y mod n = z
        # where z is the value of Z (also the index in res)
        ls = list(filter(f, combinations))
        # P(Z=z) = sum_{X+Ymod2=z} P(X,Y) + P(X=n+z)
        for idx in ls:
            res[z]+= Ptable[idx]
        res[z]+= px[n+z]

    return res

def evalZ(x,y):
    """value of Z based on values of X and Y

        Input:
        x,y:    values of x and y

        Output:
        res:    Z

    """
    if (x >= 2):
        return x%2
    else:
        return (x+y)%2
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
    
    #TODO
    # mask = Ptable != 0
    # Ptable[mask] = Ptable[mask]*np.log2(Ptable[mask]/(x*y))
    # A[A!=0] += np.log2(A[A!=0]/(x*y))

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
def step_linear(Ptable, n=10):
    """Linear stepping

        Input:
        Ptable: table of joint probability
        n:      number of steps to interpolate. Default n=10

        Output:
        PPs: list of tables, interpolated from Ptable
                towards uniform ditribution
    """
    alpha,stepsize = np.linspace(0,1,num=n, retstep=True)
    sh = np.shape(Ptable)
    norm = 1.0/np.product(sh)
    normal = np.ones_like(Ptable) * norm
    alphash = np.shape(alpha)

    PPs = np.empty(alphash + sh)
    
    for idx,a in enumerate(alpha):
        PPs[idx] = a*normal + (1.0-a)*Ptable

    return PPs


# Renormalize the joint probability distribution
def normalize(Ptable):
    sh = np.shape(Ptable)
    n = np.product(sh)

    Ptable = 1/n * Ptable
    return Ptable
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

print("*********\nSIMPLE TEST\n")
PPs = step_linear(PP)
results = []
for table in PPs:
    results.append( I(table) )

print(results)
print("*********\nSIMPLE MARGINALS TEST\n")
a = marginals(Ptable)
print(a)
print("*********\nBIG TEST\n")
# pz = marginal_Z(PP)
# print("---")
# print(pz)
dim = 8192
big = create_table(dim)
# print(big)
n = 20
Bigs = step_linear(big, n)
results2 = []
for table in Bigs:
    results2.append( I(table) )
    # pz = marginal_Z(table)
    # print(pz)
    # print(np.sum(pz))
    print("---")

print(results2)
print("*********\n")

#########################################################
# PRINT RESULTS

t = np.arange(n)
figure()
plot(t, results2, label=r'$I(X;Y)$')
xlabel("steps")
# ylabel(r'$V \; [V]$')
grid(True)
# ylim(-Vo - 1, Vo + 1)
legend(loc='upper right')
savefig('I_XY-'+str(dim)+'-'+str(n)+'.png')

#########################################################