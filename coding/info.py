import numpy as np
import prob as pr

def coeffOfNo( no, mixedBasis):
    coeffs = ()
    for k in range(0, len(mixedBasis)):
        coeffs = coeffs + ( no%mixedBasis[k],)
        no = no//mixedBasis[k]
    return coeffs
        
def randChannel( dim_out, dim_in):
    PC = np.random.rand(dim_out, dim_in)
    for i in range(0,dim_in):
        factor = 1./np.sum(PC[:,i])
        PC[:,i] = np.multiply( factor, PC[:,i])
    return PC

# Assume now that there are multiple parties for inputs and outputs
# -> dim_out, dim_in are lists
# (No easy way of fixing the last (dim_in) indeces to perform sum etc on the resulting array...)
def randChannelMultipart( dim_out, dim_in):
    #print( dim_out)
    #print( dim_in)
    eps = np.finfo(float).eps
    PC = np.random.random_sample( dim_out+dim_in);
    for k in range(0, np.prod(dim_in)):
        factor = 0.
        for l in range(0, np.prod(dim_out)):
            #print(coeffOfNo( l, dim_out))
            #print(coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in))
            factor += PC[coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in)]
        
        if factor > eps:
            for l in range(0, np.prod(dim_out)):
                PC[coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in)] *= 1./factor
    return PC

# Seems to swap "channelled" party always to the very end
def applyChannel( P, PC, toParty):
    return np.tensordot(P,PC,(toParty,1))

def entropy( P):
    res = 0.0
    for p in P:
        if p != 0:
            res += p*np.log2(p)
    return -res

def entropy_(P):
    """
        works fo rany dimension of joint P distr
        since the mask flatten the array
    """
    res = 0.0

    mask = P != 0.0 # avoid 0 in log
    f = lambda x: x*np.log2(x)
    temp = list(map(f, P[mask]))
    res = -np.sum(temp, dtype=float)
    return res

def mutInf(P):
    Pprod = np.tensordot( np.sum(P,0), np.sum(P,1), 0)
    I = 0.
    for x in range(0,P.shape[0]):
        for y in range(0,P.shape[1]):
            if P[x,y] > 0.:
                I += P[x,y]*np.log2( P[x,y]/Pprod[x,y] )
    return I

def condMutInf(P):
    Pz = np.sum(P, (0,1))
    I = 0.
    for z in range(0, P.shape[2]):
        I += Pz[z] * mutInf( np.multiply(1./Pz[z], P[:,:,z]))
    return I

def condMutInf_(P, dimX, dimY, dimZ):
    res = 0.0
    Pxz = pr.marginal(P,(dimY))
    Pyz = pr.marginal(P,(dimX))
    Pz = pr.marginal(P,(dimX,dimY))
    res = entropy_(Pxz) + entropy_(Pyz) - entropy_(P) - entropy_(Pz)
    return res

# Monte Carlo way of computing an upper bound on the intrinsic information
def MCupperBoundIntrinInf(P, noIter):
    minVal = 0.
    for i in range(0, noIter):
        PC = randChannel( P.shape[2], P.shape[2])
        Pprime = applyChannel( P, PC, 2)
        val = condMutInf( Pprime)
        if i == 0:
            minVal = val
        elif val < minVal:
            minVal = val
    return minVal

def MCupperBoundIntrinInfMultipart(P, noIter):
    """ Takes a joint probability of any size. 
        Encode the extra dimensions and flattens them in the third component.
        Then estimate a bound on the intrinsic information.
    """
    minVal = np.finfo(float).max
    sh = P.shape
    for i in range(noIter):
        PC_UZ = randChannelMultipart( (sh[0], sh[3]), (sh[0], sh[3]))

        # apply channel
        Pprime = np.tensordot( P, PC_UZ, ( (0,3), (0,1)))
        # Pprime has form P_XYUZ because of reordering of tensordot
        # P_UZ = np.sum( Pprime, (0,1))
        # val = 0.
        # for u in range(P_UZ.shape[0]):
        #     for z in range(P_UZ.shape[1]):
        #         val += P_UZ[u,z] * mutInf(np.multiply(1./P_UZ[u,z], Pprime[:,:,u,z]))
        val = condMutInf_(Pprime, 0,1,(3,2))
        val -= entropy(np.sum(Pprime, (0,1,2)))
        if val < minVal:
            minVal = val
    return minVal
            
# Monte Carlo way of computing an upper bound on the reduced intrinsic information
# -> need to choose 2 channels at random
# -> choose size of U as product of 3 dimensions X,Y,Z
def MCupperBoundRedIntrinInf( P, noIterOuter, noIterInner):
    minVal = 0.
    for i in range(0, noIterOuter):
        # Setup random channel XYZ->U and compute P_UXYZ
        #print('----')
        #print(P.shape)
        PC_UXYZ = randChannelMultipart( (np.prod(P.shape),), P.shape)
        P_UXYZ = np.zeros_like(PC_UXYZ)
        # print(PC_UXYZ.shape)
        for u in range(0,PC_UXYZ.shape[0]):
            P_UXYZ[u,:,:,:] = np.multiply( PC_UXYZ[u,:,:,:], P)
        # print(P_UXYZ.shape)
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        for k in range(0, noIterInner):
            PC_UZ = randChannelMultipart( (P_UXYZ.shape[0], P_UXYZ.shape[3]), (P_UXYZ.shape[0], P_UXYZ.shape[3]))
            #print('----')
            # print(PC_UZ.shape)
            # print(P_UXYZ.shape)
            # Pprime has form P_XYUZ because of reordering of tensordot
            Pprime = np.tensordot( P_UXYZ, PC_UZ, ( (0,3), (0,1)))
            # print(Pprime.shape)
            P_UZ = np.sum( Pprime, (0,1))
            # print(P_UZ.shape)
            I = 0.
            for u in range(0,Pprime.shape[2]):
                for z in range(0,Pprime.shape[3]):
                    I += P_UZ[u,z] * mutInf( np.multiply(1./P_UZ[u,z], Pprime[:,:,u,z]))
            Pu = pr.marginal(P_UXYZ, (1,2,3))
            # print(Pu)
            # print("Temp_I: %.3f\t Entropy: %.3f" % (I, entropy(Pu)))
            I -= entropy( np.sum( P_UZ, (0)))
            if (i == 0 and k == 0):
                minVal = I
            elif I < minVal:
                minVal = I
    return minVal
                
def MCupperBoundRedIntrinInf_( P, noIterOuter, noIterInner):
    minVal = np.finfo(float).max
    for i in range(0, noIterOuter):
        # Setup random channel XYZ->U and compute P_UXYZ
        PC_UXYZ = randChannelMultipart( (np.prod(P.shape),), P.shape)
        P_UXYZ = np.zeros_like(PC_UXYZ)
        for u in range(0,PC_UXYZ.shape[0]):
            P_UXYZ[u,:,:,:] = np.multiply( PC_UXYZ[u,:,:,:], P)
        
        # call MCupperBoundIntrinInfMultipart to get intrInf
        val = MCupperBoundIntrinInfMultipart(P_UXYZ, noIterInner)
        # check for min
        if val<minVal:
            minVal = val
            
    return minVal
