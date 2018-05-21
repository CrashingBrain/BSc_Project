import cvxopt as co
import numpy as np

# Utilities:
def tensorNflatten( b1,b2,b3):
    B = np.tensordot( b1, np.tensordot( b2, b3, 0), 0)
    return B.flatten()


# Goal is the alternative formulation of the SDP -> appendix A

# compute the objective function min t (t... auxiliary var)
def c( dim):
    c = co.matrix(0, (1, np.prod(dim)))
    c[0] = 1
    return c

def HSnorm( P,S):
    return np.trace( np.matmul( np.adjoint(P), T))

def HSnormalize(P):
    return np.multiply( 1./sqrt(HSnorm(P,P)), P )

# Get trace zero bases from su(n) + diag(1,0,...)
def basesH( dim):
    bases = [ np.multiply( 1./dim, np.identity(dim)) ] 
    bases[0][0,0] = 1.0
    # fill off-diagonal elements
    for i in range(0, dim):
        for j in range(j+1, dim):
            T = np.zeros( (dim, dim))
            T[i,j] = T[j,i] = 1.0
            bases += [ np.multiply( 1./np.sqrt(2), T) ]
    # n-1 remaining diagonal, trace-zero basis elements
    for i in range(1, dim):
        T = np.zeros( (dim, dim))
        T[i, i] = -i
        for j in range(1,i):
            T[j,j] = 1
        bases += [ HSnormalize( T) ]
    return bases

# Compute the function F (G in cvxopt manual)
def G0( rho, dim):
    G = co.matrix(0, (1, np.power(dim[0],2)*dim[1]))
    basesA = basis( dim[0])
    basesB = basis( dim[1])
    for j in range(0,dim[1]):
        G += np.multiply( rho[1,j], tensorNflatten( basesA[0], basesB[j], basesA[0]))
    return G

def Giji( rho, dim):
    G = co.matrix(0, (1, np.power(dim[0],2)*dim[1]))

    return G
