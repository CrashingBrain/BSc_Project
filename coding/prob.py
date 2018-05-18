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

def PrintFourPDstrb(P):
    s = str()
    s += '(   x,   y) '
    for z in range(0, P.shape[2]):
        for u in range(0, P.shape[3]):
            s += '( '
            s += str('%3d' %z)
            s += ', '
            s += str('%3d' %u)
            s += ') '
            #s += '(' + str(z) + ', ' + str(u) ')'
    s += '   (mrg)'
    print(s)
    for x in range(0, P.shape[0]):
        for y in range(0, P.shape[1]):
            s = str()
            s += '( '
            s += str('%3d' %x)
            s += ', '
            s += str('%3d' %y)
            s += ') '
            #s += ('(' + str(x) + ', ' + str(y) ') ')
            for z in range(0, P.shape[2]):
                for u in range(0, P.shape[3]):
                    if P[x,y,z,u] > 0:
                        s += str('     %0.4f ' %P[x,y,z,u])
                    else:
                        s += '           :'
            s += '   '
            s += str(' %0.4f ' %np.sum(P, (2,3))[x,y])
            print(s)
    s = str()
    s += '( marginal) '
    for z in range(0, P.shape[2]):
        for u in range(0, P.shape[3]):
            s += str('     %0.3f ' %np.sum(P, (0,1))[z,u])
            s += ' ' 
    print(s)
    pass

def PrintThreePDstrb(P):
    s = str()
    s += '(   x,   y) '
    for z in range(0, P.shape[2]):
            s += ' ( '
            s += str('%3d' %z)
            s += ')'
    s += '   (mrg)'
    print(s)
    for x in range(0, P.shape[0]):
        for y in range(0, P.shape[1]):
            s = str()
            s += '( '
            s += str('%3d' %x)
            s += ', '
            s += str('%3d' %y)
            s += ') '
            #s += ('(' + str(x) + ', ' + str(y) ') ')
            for z in range(0, P.shape[2]):
                if P[x,y,z] > 0:
                    s += str(' %0.4f' %P[x,y,z])
                else:
                    s += '      :'
            s += '   '
            s += str(' %0.4f' %np.sum(P, (2))[x,y])
            print(s)
    s = str()
    s += '( marginal) '
    for z in range(0, P.shape[2]):
        s += str(' %0.3f' %np.sum(P, (0,1))[z])
        s += ' ' 
    s += str(' %0.3f' %np.sum(P))
    print(s)
    pass

def OneDimToTwo( P, dim=(4,4)):
    return np.reshape( P, dim)

def mixBhvs( P1, P2, alpha=.5):
    return np.add( np.multiply( alpha, P1), np.multiply( 1-alpha, P2))
