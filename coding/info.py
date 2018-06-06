import numpy as np

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

# Map the input dimensions to an encoding in an additional party
def detChannel( dim_in):
    PC = np.zeros( (np.prod(dim_in))+dim_in)
    for k in range(0, np.prod(dim_in)):
        PC[ (k)+coeffOfNo(k, dim_in)] = 1.
    return PC

def noisyChannel( dim_out, dim_in):
    PC = np.ones( dim_out, dim_in)
    PC *= 1./np.prod(dim_out)
    return PC

# Assume now that there are multiple parties for inputs and outputs
# -> dim_out, dim_in are lists
# (No easy way of fixing the last (dim_in) indeces to perform sum etc on the resulting array...)
def randChannelMultipart( dim_out, dim_in):
    PC = np.random.random_sample( dim_out+dim_in);
    for k in range(0, np.prod(dim_in)):
        factor = 0.
        for l in range(0, np.prod(dim_out)):
            #print(coeffOfNo( l, dim_out))
            #print(coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in))
            factor += PC[coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in)]
        
        if factor > 1e-15:
            for l in range(0, np.prod(dim_out)):
                PC[coeffOfNo( l, dim_out) + coeffOfNo( k, dim_in)] *= 1./factor
    return PC

# Seems to swap "channelled" party always to the very end
def applyChannel( P, PC, toParty):
    #print("List with conditioning parts in channel")
    #print( PC.shape)
    #print(list(range( len(PC.shape)//2, len(PC.shape))))
    return np.tensordot(P,PC,(toParty,list(range( len(PC.shape)//2, len(PC.shape)))))

def entropy( P):
    P = P.flatten()    
    E = 0
    for x in range(0,len(P)):
        if P[x] > 0:
            E += -P[x] * np.log2(P[x])
    return E

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
        if Pz[z] > 0.:
            I += Pz[z] * mutInf( np.multiply(1./Pz[z], P[:,:,z]))
    return I

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
        #print(PC_UXYZ.shape)
        for u in range(0,PC_UXYZ.shape[0]):
            P_UXYZ[u,:,:,:] = np.multiply( PC_UXYZ[u,:,:,:], P)
        print(P_UXYZ.shape)
        E_U = entropy( np.sum( P_UZ, (1,2,3)))
        # Inner Loop: get random channel UZ->bar(UZ) and compute the cond mutual information
        for k in range(0, noIterInner):
            PC_UZ = randChannelMultipart( (P_UXYZ.shape[0], P_UXYZ.shape[3]), (P_UXYZ.shape[0], P_UXYZ.shape[3]))
            #print('----')
            #print(PC_UZ.shape)
            #print(P_UXYZ.shape)
            # Pprime has form P_XYUZ because of reordering of tensordot
            Pprime = np.tensordot( P_UXYZ, PC_UZ, ( (0,3), (2,3)))
            Ppp = applyChannel( P_UXYZ, PC_UZ, (0,3))
            if ( np.amax(np.absolute( Pprime - Ppp)) > 10e-10):
                print("MCupperBoundRedIntrinInf: Diff between the Pprime and Ppp %f" % np.amax(np.absolute( Pprime - Ppp)))
            if ( np.absolute(np.sum( Pprime) - 1.0) > 10e-10):
                print("MCupperBoundRedIntrinInf: check normalization after applying the channel: %f" % np.sum(Pprime))
            #print(Pprime.shape)
            P_UZ = np.sum( Pprime, (0,1))
            I = 0.
            for u in range(0,Pprime.shape[2]):
                for z in range(0,Pprime.shape[3]):
                    I += P_UZ[u,z] * mutInf( np.multiply(1./P_UZ[u,z], Pprime[:,:,u,z]))
            print("MCupperBoundRedIntrinInf: cond mut inf I(X;Y|ZU) = %f" % I)
            I += E_U
            if (i == 0 and k == 0):
                minVal = I
            elif I < minVal:
                minVal = I
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
