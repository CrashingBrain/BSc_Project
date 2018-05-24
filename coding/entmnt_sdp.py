import cvxopt as co
import numpy as np

# Utilities:
def tensorNflatten( b1,b2,b3):
    B = np.tensordot( b1, np.tensordot( b2, b3, 0), 0)
    return B.flatten()


# The goal is the alternative formulation of the SDP -> appendix A

# compute the objective function min t (t... auxiliary var)
def c( dim):
    c = co.matrix(0, (1, np.prod(dim)))
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
def G0( rho, dim):
    G = co.matrix(0, (((dim[0]**2)*dim[1])**2, )
    basisA = basisH( dim[0])
    basisB = basisH( dim[1])
    for j in range(0,dim[1]):
        Gp =  rho[1,j] * tensorNflatten( basisA[0], basisB[j], basisA[0])
        G += Gp 
    return G

def Gt ( dim, ctr):
    G = [co.matrix(0, (1, np.power(dim[0],2)*dim[1]))]
    ctr += 1
    return G

def Giji( dim, ctr):
    for i in range(1,dim[1]):
        for j in range(0,dim[1]):
            G += [ co.matrix(tensorNflatten( basesA[i], basesB[j], basesA[i])) ] 
            ctr += 1
    return G

def Gijk( dim, ctr):
    for i in range(1,dim[1]):
        for j in range(0,dim[1]):
            for k in range(i+1,dim[2]):
                G += [ co.matrix(tensorNflatten( basesA[i], basesB[j], basesA[i])) ]
                ctr += 1
    return G

def assembleGx( dim):
    Gx = Giji + Gijk
    # check if counter is correct
    if ( ctr != (np.power(dim[0],4)*np.power(dim[1],2)- np.power(dim[0],2)*np.power(dim[1],2))/2 + 1):
        print("assembeGx: counter does not match no arguments")
    return Gx

def PPTsymmExt( rho, dim):
    sol = co.solvers.sdp( c( dim), h = -G0( rho, dim), Gl = assembeGx( dim) )
    return sol
