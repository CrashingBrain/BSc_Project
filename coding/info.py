import numpy as np
import prob as pr

def normalizeChannel(PC):
    ax = tuple([i+1 for i in range(len(PC.shape))])
    factor = np.sum(PC, axis=ax)
    for i in range(len(factor)):
        PC[:,i] = factor[i] * PC[:,i]
    
    return PC

def coeffOfNo( no, mixedBasis):
    if no > np.prod(mixedBasis):
        print("[inf.coeffOfNo] No is out of range of mixed basis")
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

# Map the input dimensions to an encoding in an additional party
def detChannel( dim_in):
    PC = np.zeros( (np.prod(dim_in))+dim_in)
    for k in range(0, np.prod(dim_in)):
        PC[ (k)+coeffOfNo(k, dim_in)] = 1.
    return PC

# More general deterministic channel
# Expects dim_out as an integer, and dim_in as a tuple
# k is the number of the deterministic channel
def detChannel( dim_out, dim_in, k):
    PC = np.zeros( (dim_out,)+ dim_in)
    # Get k-th deterministic channel: in a coefficient term
    coeffs = coeffOfNo( k, tuple( [dim_out] * np.prod(dim_in)))
    #print( coeffs)
    for k in range(0, len(coeffs)):
        PC[ (coeffs[k],)+coeffOfNo(k,dim_in)] = 1
    return PC

def identityChannel( dims):
    PC = np.zeros( dims, dims)
    for k in range(0, np.prod(dims)):
        coeffs = coeffOfNo( k, dims)
        PC[ coeffs, coeffs] = 1.
    return PC
        
def noisyChannel( dim_out, dim_in):
    PC = np.ones( dim_out, dim_in)
    PC *= 1./np.prod(dim_out)
    return PC

def epsilonNoiseChannel( epsilon, dims):
    return epsilon*noisyChannel(dims, dims)+(1.-epsilon)*identityChannel(dims)

# Assume now that there are multiple parties for inputs and outputs
# -> dim_out, dim_in are lists
# (No easy way of fixing the last (dim_in) indeces to perform sum etc on the resulting array...)
def randChannelMP( dim_out, dim_in):
    eps = np.finfo(float).eps
    PC = np.random.random_sample( dim_out+dim_in);
    for k in range(0, np.prod(dim_in)):
        factor = 0.
        inCoeffs = coeffOfNo( k, dim_in)
        
        for l in range(0, np.prod(dim_out)):
            factor += PC[coeffOfNo( l, dim_out) + inCoeffs]

        if factor > eps:
            norm = 1./factor

            for l in range(0, np.prod(dim_out)):
                PC[coeffOfNo( l, dim_out) + inCoeffs] *= norm
    return PC

# Seems to swap "channelled" party always to the very end
def applyChannel( P, PC, toParty):
    return np.tensordot(P,PC,(toParty,list(range( len(PC.shape)//2, len(PC.shape)))))

def entropy( P):
    P = P.flatten()    
    E = 0
    for x in range(0,len(P)):
        if P[x] > 0:
            E += -P[x] * np.log2(P[x])
    return E

def entropy_(P):
    """
        works fo rany dimension of joint P distr
        since the mask flatten the array
    """
    res = 0.0

    mask = P != 0.0 # avoid 0 in log
    f = lambda x: x*np.log2(x)
    # map-reduce strategy (likely to be more optimized than loops)
    temp = list(map(f, P[mask]))
    res = -np.sum(temp, dtype=float)
    return res

def mutInf(P):
    Pprod = np.tensordot( np.sum(P,0), np.sum(P,1), 0)
    I = 0.
    for x in range(0,P.shape[0]):
        for y in range(0,P.shape[1]):
            if P[x,y] > 1e-15 and Pprod[x,y] > 1e-15:
                I += P[x,y]*np.log2( P[x,y]/Pprod[x,y] )
    return I

def condMutInf(P):
    Pz = np.sum(P, (0,1))
    I = 0.
    for z in range(0, P.shape[2]):
        if Pz[z] > 0.:
            I += Pz[z] * mutInf( np.multiply(1./Pz[z], P[:,:,z]))
    return I

def condMutInf_(P, dimX, dimY, dimZ):
    """ Evaluates the I(X;Y|Z) for joint probability P
        Utilizes the definition of condMutInf
            I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        Works for any dimensions of P,
        where the first two dimensions are interpreted as X,Y and
        all the remainig are grouped as Z

        Input:
            P       : array-like probability distribution
            dimX    : index of the X in the array
            dimY    : index of the Y in the array
            dimZ    : index(ices) of the remaining component(s) for the Z (can be a tuple)
        
    """
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
        Pprime = applyChannel( P, PC, (2))
        val = condMutInf( Pprime)
        if i == 0:
            minVal = val
        elif val < minVal:
            minVal = val
    return minVal

def MCupperBoundIntrinInfMP_(P, noIter):
    # P has shape UXYZ
    minVal = np.finfo(float).max
    sh = P.shape
    # print(sh)
    Hu = entropy(np.sum(P, (1,2,3)))
    for i in range(noIter):
        # considering now a binarization channel ZU -> {0,1}
        PC_UZ = randChannelMP( (sh[3],), (sh[0], sh[3]))

        # apply channel
        # Pprime = applyChannel(P, PC_UZ, (0,3))
        Pprime = np.zeros((sh[1],sh[2],PC_UZ.shape[0]))

        # print("input shape:\t" + str(P.shape))
        # print("Channel shape:\t" + str(PC_UZ.shape))
        # print("Pprime shape:\t" + str(Pprime.shape))

        for u in range(sh[0]):
            for x in range(sh[1]):
                for y in range(sh[2]):
                    for z in range(sh[3]):
                        Pprime[x,y,0] += P[u,x,y,z] * PC_UZ[0,u,z]
                        Pprime[x,y,1] += P[u,x,y,z] * PC_UZ[1,u,z]
        val = condMutInf( Pprime)
        # print("val: %.3f\tHu: %.3f" % (val, Hu))
        val += Hu
        if val < minVal:
            minVal = val
    return minVal
            
# Computes an upper bound on the BoundIntrinInf taking all but the first two parties to be in Eve's possesion
def MCupperBoundIntrinInfMP(P, dimBZ, noIter, verbose=False):
    minVal = 0.
    # If dimBZ is zero: set dim to prod of dimensions Eve's systems
    if dimBZ ==0:
        dimBZ = np.prod(P.shape[2:])
    for k in range(0, noIter):
        PC = randChannelMP( (dimBZ,), P.shape[2:])
        Pprime = applyChannel( P, PC, tuple(range(2,len(P.shape))))
        val = condMutInf( Pprime)
        if k == 0:
            minVal = val
        elif val < minVal:
            minVal = val
            if verbose:
                print( "[MCupperBoundIntrinInfMP] k = %d, val = %f" % (k,val))
    return minVal

def MCupperBoundIntrinInfMPDet(P, dimBZ, verbose=False):
    minVal = 0.
    # If dimBZ is zero: set dim to prod of dimensions Eve's systems
    if dimBZ ==0:
        dimBZ = np.prod(P.shape[2:])
    for k in range(0, dimBZ**(np.prod(P.shape[2:]))):
        # get deterministic channel and compute primed behavior
        PC = detChannel( dimBZ, P.shape[2:], k)
        Pprime = applyChannel( P, PC, tuple(range(2,len(P.shape))))
        val = condMutInf( Pprime)
        if k == 0:
            minVal = val
        elif val < minVal:
            minVal = val
            if verbose:
                print( "[MCupperBoundIntrinInfMP] k = %d, val = %f" % (k,val))
    return minVal

# Monte Carlo way of computing an upper bound on the reduced intrinsic information
# -> need to choose 2 channels at random
# the channel is chosen as XY->U
def MCupperBoundRedIntrinInf_( P, noIterOuter, noIterInner):
    minVal = np.finfo(float).max
    for i in range(0, noIterOuter):
        # Setup random channel XYZ->U and compute P_UXYZ
        PC_UXYZ = randChannelMP( (np.prod(P.shape),), P.shape)
        P_UXYZ = np.zeros_like(PC_UXYZ)
        for u in range(0,PC_UXYZ.shape[0]):
            P_UXYZ[u,:,:,:] = np.multiply( PC_UXYZ[u,:,:,:], P)
        
        # call MCupperBoundIntrinInfMultipart to get intrInf
        val = MCupperBoundIntrinInfMP_(P_UXYZ, noIterInner)
        # check for min
        if val<minVal:
            minVal = val
            
    return minVal

def MCupperBoundRedIntrinInfDet_(P, dimU, dimBZU, noIterOuter, noIterInner, verbose=False):
    minVal = np.finfo(float).max
    for i in range(noIterOuter):
        #setup random channel XYZ->U and compute P_UXYZ
        PC_UXYZ = randChannelMP( (dimU,), P.shape)
        shpc = PC_UXYZ.shape
        P_XYZU = np.zeros((shpc[1:] + (shpc[0],)))
        for u in range(PC_UXYZ.shape[-1]):
            P_XYZU[:,:,:,u] = np.multiply( PC_UXYZ[u,:,:,:], P)
        
        #get entropy
        Hu = entropy_( pr.marginal(P_XYZU, tuple(range(len(P_XYZU.shape)))[:-1]))
        # get the bound on intrInf of X;Y|ZU with detChannel
        val = MCupperBoundIntrinInfMPDet(P_XYZU, dimBZU)
        #check for min
        if val < minVal:
            minVal = val
            if verbose:
                print( "[MCupperBoundRedIntrinInfDet_] I = %f, E_U = %f" % (val, Hu))
    
    return minVal


def MCupperBoundRedIntrinInfXY( P, dimU, dimBZU, noIterOuter, noIterInner, verbose=False):
    minVal = 0.
    for k in range(0, noIterOuter):
        # Setup random channel XY->U and compute P_UXYZ
        PC_U_XY = randChannelMP( (dimU,), P.shape[0:2])
        P_XYZU = np.zeros( P.shape+(dimU,))
        for u in range(0,PC_U_XY.shape[0]):
            for z in range(0, P.shape[2]):
                P_XYZU[ :, :, z, u] = np.multiply( P[:,:,z], PC_U_XY[ u,:,:])
        E_U = entropy( np.sum( P_XYZU, (0,1,2)))
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        I = MCupperBoundIntrinInfMP( P_XYZU, dimBZU, noIterInner) + E_U
        if k == 0:
            minVal = I
        elif I < minVal:
            minVal = I
            if verbose:
                print( "[MCupperBoundRedIntrinInfXY] I = %f, E_U = %f" % (I, E_U))
    return minVal
 
# the channel is chosen as X->U
def MCupperBoundRedIntrinInfX( P, dimU, dimBZU, noIterOuter, noIterInner, verbose=False):
    minVal = 0.
    for k in range(0, noIterOuter):
        # Setup random channel XY->U and compute P_UXYZ
        PC_U_XY = randChannelMP( (dimU,), P.shape[0:1])
        P_XYZU = np.zeros( P.shape+(dimU,))
        for u in range(0,PC_U_XY.shape[0]):
            for y in range(0, P.shape[1]):
                for z in range(0, P.shape[2]):
                    P_XYZU[ :, y, z, u] = np.multiply( P[:,y,z], PC_U_XY[ u,:])
        E_U = entropy( np.sum( P_XYZU, (0,1,2)))
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        I = MCupperBoundIntrinInfMP( P_XYZU, dimBZU, noIterInner) + E_U
        if k == 0:
            minVal = I
        elif I < minVal:
            minVal = I
            if verbose:
                print( "[MCupperBoundRedIntrinInfXY] I = %f, E_U = %f" % (I, E_U))
    return minVal
 
def MCupperBoundRedIntrinInfXYDet( P, dimU, dimBZU, noIterInner, verbose=False):
    minVal = 0.
    for k in range(0, dimU**(np.prod(P.shape[0:2]))):
        # Setup deterministic channel XY->U and compute P_UXYZ
        PC_U_XY = detChannel( dimU, P.shape[0:2], k)
        P_XYZU = np.zeros( P.shape+(dimU,))
        for u in range(0,PC_U_XY.shape[0]):
            for z in range(0, P.shape[2]):
                P_XYZU[ :, :, z, u] = np.multiply( P[:,:,z], PC_U_XY[ u,:,:])
        E_U = entropy( np.sum( P_XYZU, (0,1,2)))
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        I = MCupperBoundIntrinInfMP( P_XYZU, dimBZU, noIterInner) + E_U
        if k == 0:
            minVal = I
        elif I < minVal:
            minVal = I
            if verbose:
                print( "[MCupperBoundRedIntrinInfXY] I = %f, E_U = %f" % (I, E_U))
    return minVal

def MCupperBoundRedIntrinInfXDet( P, dimU, dimBZU, noIterInner, verbose=False):
    minVal = 0.
    for k in range(0, dimU**(np.prod(P.shape[0:1]))):
        # Setup deterministic channel XY->U and compute P_UXYZ
        PC_U_XY = detChannel( dimU, P.shape[0:1], k)
        P_XYZU = np.zeros( P.shape+(dimU,))
        for u in range(0,PC_U_XY.shape[0]):
            for y in range(0, P.shape[1]):
                for z in range(0, P.shape[2]):
                    P_XYZU[ :, y, z, u] = np.multiply( P[:,y,z], PC_U_XY[ u,:])
        E_U = entropy( np.sum( P_XYZU, (0,1,2)))
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        I = MCupperBoundIntrinInfMP( P_XYZU, dimBZU, noIterInner) + E_U
        if k == 0:
            minVal = I
        elif I < minVal:
            minVal = I
            if verbose:
                print( "[MCupperBoundRedIntrinInfXY] I = %f, E_U = %f" % (I, E_U))
    return minVal

# Loop over the deterministic channels in both, the outer *and* the inner loop
def MCupperBoundRedIntrinInfXYDD( P, dimU, dimBZU, verbose=False):
    minVal = 0.
    for k in range(0, dimU**(np.prod(P.shape[0:2]))):
        # Setup deterministic channel XY->U and compute P_UXYZ
        PC_U_XY = detChannel( dimU, P.shape[0:2], k)
        P_XYZU = np.zeros( P.shape+(dimU,))
        for u in range(0,PC_U_XY.shape[0]):
            for z in range(0, P.shape[2]):
                P_XYZU[ :, :, z, u] = np.multiply( P[:,:,z], PC_U_XY[ u,:,:])
        E_U = entropy( np.sum( P_XYZU, (0,1,2)))
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        I = MCupperBoundIntrinInfMPDet( P_XYZU, dimBZU) + E_U
        if k == 0:
            minVal = I
        elif I < minVal:
            minVal = I
            if verbose:
                print( "[MCupperBoundRedIntrinInfXY] I = %f, E_U = %f" % (I, E_U))
    return minVal

# Loop over the deterministic channels in both, the outer *and* the inner loop
def MCupperBoundRedIntrinInfXDD( P, dimU, dimBZU, verbose=False):
    minVal = 0.
    for k in range(0, dimU**(np.prod(P.shape[0:1]))):
        # Setup deterministic channel XY->U and compute P_UXYZ
        PC_U_XY = detChannel( dimU, P.shape[0:1], k)
        P_XYZU = np.zeros( P.shape+(dimU,))
        for u in range(0,PC_U_XY.shape[0]):
            for y in range(0, P.shape[1]):
                for z in range(0, P.shape[2]):
                    P_XYZU[ :, y, z, u] = np.multiply( P[:,y,z], PC_U_XY[ u,:])
        E_U = entropy( np.sum( P_XYZU, (0,1,2)))
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        I = MCupperBoundIntrinInfMPDet( P_XYZU, dimBZU) + E_U
        if k == 0:
            minVal = I
        elif I < minVal:
            minVal = I
            if verbose:
                print( "[MCupperBoundRedIntrinInfXY] I = %f, E_U = %f" % (I, E_U))
    return minVal

# Channel from the proof of Lemma7
def zuChannel():
    PC_zu = np.zeros( (2,2,2,2))
    PC_zu[0,0,0,0] = 1.
    PC_zu[0,0,1,0] = 1.
    PC_zu[0,1,0,1] = 1.
    PC_zu[1,0,1,1] = 1.
    return PC_zu

def zuChannel2():
    PC_zu = np.zeros( (4,2,2))
    PC_zu[0,0,0] = 1.
    PC_zu[0,1,0] = 1.
    PC_zu[1,0,1] = 1.
    PC_zu[2,1,1] = 1.
    return PC_zu
