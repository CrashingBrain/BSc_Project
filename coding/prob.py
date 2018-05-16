import numpy as np

def normalize(P):
    factor = 1./np.sum(P)
    return np.multiply(factor, P)

def marginal( P, overdim):
    return np.sum( P, axis=overdim)

def FourDimToTwo(P):
    P_res = np.zeros( (P.shape[0]*P.shape[1], P.shape[2]*P.shape[3]))
    for x in range(0, P.shape[0]):
        for y in range(0, P.shape[1]):
            for z in range(0, P.shape[2]):
                for u in range(0, P.shape[3]):
                    P_res[ x*P.shape[1] + y, z*P.shape[3] + u ] = P[x][y][z][u]
    return P_res

def OneDimToTwo( P, dim=(4,4)):
    return np.reshape( P, dim)

