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
                    P_res[ x*P.shape[1] + y, z*P.shape[3] + u ] = P[x,y,z,u]
    return P_res

# WANTED: Proper formatting a la fprintf
def PrintFourPDstrb(P):
    s = str()
    s += '( x, y) '
    for z in range(0, P.shape[2]):
        for u in range(0, P.shape[3]):
            s += '( '
            s += str(z)
            s += ', '
            s += str(u)
            s += ') '
            #s += '(' + str(z) + ', ' + str(u) ')'
    s += '   (mrg)'
    print(s)
    for x in range(0, P.shape[0]):
        for y in range(0, P.shape[1]):
            s = str()
            s += '( '
            s += str(x)
            s += ', '
            s += str(y)
            s += ') '
            #s += ('(' + str(x) + ', ' + str(y) ') ')
            for z in range(0, P.shape[2]):
                for u in range(0, P.shape[3]):
                    if P[x,y,z,u] > 0:
                        s += str(P[x,y,z,u])
                    else:
                        s += '        '
            s += '   '
            s += str(np.sum(P, (2,3))[x,y])
            print(s)
    s = str()
    s += '(mrg) '
    for z in range(0, P.shape[2]):
        for u in range(0, P.shape[3]):
            s += str(np.sum(P, (0,1))[z,u])
            s += ' ' 
    print(s)
    pass

def OneDimToTwo( P, dim=(4,4)):
    return np.reshape( P, dim)

