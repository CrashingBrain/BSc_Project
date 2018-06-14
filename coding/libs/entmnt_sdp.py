import cvxopt as co
import numpy as np

# Utilities:
def tensorNflatten( b1,b2,b3):
    B = np.tensordot( b1, np.tensordot( b2, b3, 0), 0)
    return B.flatten()

# The goal is the alternative formulation of the SDP -> appendix A

# Compute the number of variables
def sizeX( dim):
    return ((dim[0]**4)*(dim[1]**2)- (dim[0]**2)*(dim[1]**2))//2 + 1

# compute the objective function min t (t... auxiliary var)
def c( dim):
    c = co.matrix(0., (sizeX(dim), 1))
    c[0] = 1
    return c

def HSnorm( P,S):
    return np.trace( np.matmul( np.transpose(P), S))

def HSnormalize(P, val=1.):
    return (val/np.sqrt(HSnorm(P,P)))*P

# Get trace zero bases from su(n) + identity
def basisH( dim):
    # the normalized identity as a first element to satisfy the second part of eq 15
    # the norm of id/n is 1/n -> have to normalize all other matrices to 1/n
    bases = [ 1./dim*np.identity(dim) ] 
    # fill off-diagonal elements: normalized to 
    for i in range(0, dim):
        for j in range(i+1, dim):
            T = np.zeros( (dim, dim))
            T[i,j] = T[j,i] = 1.0
            bases += [ np.multiply( 1./(dim*np.sqrt(2)), T) ]
    # n-1 remaining diagonal, trace-zero basis elements
    for i in range(1, dim):
        T = np.zeros( (dim, dim))
        T[i, i] = -i
        for j in range(1,i):
            T[j,j] = 1
        bases += [ HSnormalize( T, 1./dim) ]
    return bases

# Compute the function F (G in cvxopt manual)

# compute the length of the flattened three partite system... last party is a copy of the first
def rowLen( dim):
    return ((dim[0]**2)*dim[1])**2

def G0( rho, dim):
    G = co.matrix(0, (1,rowLen(dim)))
    basisA = basisH( dim[0])
    basisB = basisH( dim[1])
    for j in range(0,dim[1]):
        Gp =  rho[1,j] * tensorNflatten( basisA[0], basisB[j], basisA[0])
        G += Gp 
    return G

def Gt( dim):
    G = np.zeros( (1, rowLen(dim)) )
    G[0] = 1.
    return G

def Giji( dim):
    G = np.zeros( ((dim[0]-1)*dim[1], rowLen(dim)) )
    basisA = basisH( dim[0])
    basisB = basisH( dim[1])
    for i in range(1,dim[0]):
        for j in range(0,dim[1]):
            G[ (i-1) + (dim[0]-1) * j, :] = tensorNflatten( basisA[i], basisB[j], basisA[i])
    return G

def Gijk( dim):
    G = np.zeros((((dim[0]-1)*dim[1]*dim[0])//2, rowLen(dim)) )
    basisA = basisH( dim[0])
    basisB = basisH( dim[1])
    ctr = 0
    for i in range(1,dim[0]):
        for j in range(0,dim[1]):
            for k in range(i+1,dim[0]):
                G[ctr, :] =  tensorNflatten( basisA[i], basisB[j], basisA[k])
                G[ctr, :] += tensorNflatten( basisA[k], basisB[j], basisA[i])
                ctr += 1
    return G

def assembleGx( dim):
    Gx = co.matrix( np.vstack( (Gt(dim), Giji(dim), Gijk(dim))) )
    return Gx

def PPTsymmExt( rho, dim):
    sol = co.solvers.sdp( c( dim), Gs = assembleGx( dim), hs = -G0( rho, dim))
    return sol
