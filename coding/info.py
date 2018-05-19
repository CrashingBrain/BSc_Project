import numpy as np

def coeffOfNo( no, mixedBasis):
    coeffs = ()
    for k in len(mixedBasis):
        coeffs = coeffs + ( no//np.prod( mixedBasis[k:]))
        no = no%np.prod( mixedBasis[k:])
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
    PC = np.random.random_sample( dim_out+dim_in);
    for k in range(0, np.prod(dim_in)):
        factor = 0.
        for l in range(0, np.prod(dim_out)):
            factor += PC[coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in)]
        for l in range(0, np.prod(dim_out)):
            PC[coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in)] *= 1./factor
    return PC

# Seems to swap "channelled" party always to the very end
def applyChannel( P, PC, toParty):
    return np.tensordot(P,PC,(toParty,1))

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
            
# Monte Carlo way of computing an upper bound on the reduced intrinsic information
# -> need to choose 2 channels at random
# -> choose size of U as product of 3 dimensions X,Y,Z
def MCupperBoundRedIntrinInf( P, noIterOuter, noIterInner):
    minVal = 0.
    for i in range(0, noIterOuter):
        # Setup random channel XYZ->U and compute P_UXYZ
        PC_UXYZ = randChannelMultipart( (np.prod(P.shape),), P.shape)
        P_UXYZ = np.zeros_like(P)
        for u in range(0,PC_UXYZ.shape[0]):
            P_UXYZ[u,:,:,:] = np.multiply( PC_UXYZ[u,:,:,:], P)
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        for k in range(0, noIterInner):
            PC_UZ = randChannelMultipart( (P.shape[0], P.shape[3]), (P.shape[0], P.shape[3]))
            Pprime = np.tensordot( P, PC_UZ, ( (0,2), (0,1)))
            P_UZ = np.sum( Pprime, (1,2))
            I = 0.
            for u in range(0,Pprime.shpae[0]):
                for z in range(0,Pprime.shape[2]):
                    I += P_UZ[u][z] * mutInf( np.multiply(1./P_UZ[u][z], Pprime[u,:,:,z]))
            if (i == 0 & k == 0):
                minVal = I
            elif I < minVal:
                minVal = I
    return minVal
                

