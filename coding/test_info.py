import bhvs as bv
import info as inf
import numpy as np
import prob as pr
import matplotlib.pyplot as plt

chnTsts = False
if chnTsts:
    PC = inf.randChannel(3,2)
    print(PC.shape)
    print(PC)

    P = bv.FourPDstrb()
    print(P.shape)
    print(inf.applyChannel(P, PC, (2)).shape)
    print(inf.applyChannel(P, PC, (3)).shape)
    print(inf.applyChannel(P, PC, (2)))
    print(inf.applyChannel(P, PC, (3)))
    print(inf.mutInf(pr.marginal(P, (2,3))))

# Check the reduced intrinsic information upper bound
redIntrInf1 = False
if redIntrInf1:
    P = pr.marginal(P,(3))
    PC_UXYZ = inf.randChannelMultipart( (np.prod(P.shape),), P.shape)
    print( np.sum( PC_UXYZ, axis=(0)))
    print( PC_UXYZ.shape)
    P_UXYZ = np.zeros_like(PC_UXYZ)
    P_UXYZ_prime = np.zeros_like(PC_UXYZ)
    print(P_UXYZ.shape)
    print(P_UXYZ_prime.shape)
    for u in range(0,PC_UXYZ.shape[0]):
        P_UXYZ[u,:,:,:] = np.multiply( PC_UXYZ[u,:,:,:], P)
    for u in range(0,np.prod(P.shape)):
        for x in range(0, P.shape[0]):
            for y in range(0, P.shape[1]):
                for z in range(0, P.shape[2]):
                    P_UXYZ_prime[u,x,y,z] = PC_UXYZ[u,x,y,z]*P[x,y,z]
    print("Diff between P_UXYZ_prime and P_UXYZ: %f" % np.amax(np.absolute( P_UXYZ- P_UXYZ_prime)))
    print("Diff: marginal(PC_UXYZ*P_XYZ,U) - P_XYZ %f" % np.amax(np.absolute( pr.marginal( P_UXYZ, (0)) - P)))
    print("Diff: marginal(PC_UXYZ*P_XYZ,U) - P_XYZ %f" % np.amax(np.absolute( pr.marginal( P_UXYZ_prime, (0)) - P)))
    P_UZ = np.sum( P_UXYZ, (1,2))
    print("Diff: P_Z from P_UZ and from P_XYZ %f" % np.amax(np.absolute( pr.marginal( P_UZ, (0)) - pr.marginal( P, (0,1)))))
    # Compute the intrinsic information I(X;Y\d UZ)
    P = bv.FourPDstrb()
    I_rd = 100.
    no_iter = 1 
    for k in range(0, no_iter):
        PC_UZ = inf.randChannelMultipart( (P.shape[2:]), (P.shape[2:]))
        P_XYZU_p = inf.applyChannel( P, PC_UZ, (2,3))
        P_ZU = np.sum( P_XYZU_p, (0,1))
        I = 0.
        for z in range(0, P_XYZU_p.shape[2]):
            for u in range(0,P_XYZU_p.shape[3]):
                I += P_ZU[z,u] * inf.mutInf( np.multiply( 1./P_ZU[z,u], P_XYZU_p[:,:, z,u]))
        if (I_rd > I):
            I_rd = I
    print("Intrinsic information I(X;Y\d ZU) = %f (should go down to zero)" % I_rd)
    # Compute the intrinsic information I(X;Y\d UZ)
    # Replace the channel by one that goes to joint variable
    P = bv.FourPDstrb()
    I_rd = 100.
    for k in range(0, no_iter):
        PC_UZ = inf.randChannelMultipart( (np.prod(P.shape[2:]),), (P.shape[2:]))
        P_XYZU_p = inf.applyChannel( P, PC_UZ, (2,3))
        P_ZU = np.sum( P_XYZU_p, (0,1))
        I = 0.
        for z in range(0, P_XYZU_p.shape[2]):
                I += P_ZU[z] * inf.mutInf( np.multiply( 1./P_ZU[z], P_XYZU_p[:,:, z]))
        if (I_rd > I):
            I_rd = I
    print("Intrinsic information I(X;Y\d ZU) = %f (should go down to zero)" % I_rd)
    # Alternatively: join the parties ZU to a new one and apply MCupperBoundIntrinInf directly
    P_prime = np.zeros( (P.shape[0], P.shape[1], P.shape[2]*P.shape[3]))
    for x in range(0,P.shape[0]):
        for y in range(0,P.shape[1]):
            for zu in range(0, P.shape[2]*P.shape[3]):
                P_prime[x,y,zu] = P[ (x,y)+inf.coeffOfNo(zu,(P.shape[2],P.shape[3]))]
    print("Intrinsic information I(X;Y\d ZU) after joining Z and U = %f" % inf.MCupperBoundIntrinInf(P_prime, no_iter))
    # Use the channel from the paper
    P3 = inf.applyChannel( P, inf.zuChannel2(), (2,3))
    print("Conditional mutual information I(X;Y|bar{UZ}) %f" % inf.condMutInf( P3))
    P4 = inf.applyChannel( P, inf.zuChannel(), (2,3))
    I = 0.
    P4_ZU = pr.marginal( P4, (0,1))
    for z in range(0, P4.shape[2]):
        for u in range(0, P4.shape[3]):
            if P4_ZU[z,u] > 0: 
                I += P4_ZU[z,u] * inf.mutInf( np.multiply( 1./P4_ZU[z,u], P4[:,:, z,u]))
    print("Conditional mutual information I(X;Y|bar{UZ}) %f" % I)
    print("Entropy of P_U %f" % inf.entropy( pr.marginal(P, (0,1,2))))

redIntrInf2 = True
if redIntrInf2:
    P = bv.FourPDstrb()
    P = pr.marginal(P, 3)
    print( "Test MCupperBoundIntrinInfMP with FourPDstrb()")
    for dimBZU in range(2,5):
        print( inf.MCupperBoundIntrinInfMP( P, dimBZU, 200))
 
redIntrInf3 = True
if redIntrInf3:
    P = bv.FourPDstrb()
    print( "Test MCupperBoundIntrinInfMP with FourPDstrb()")
    for dimBZU in range(2,5):
        print( inf.MCupperBoundIntrinInfMP( P, dimBZU, 200))
 
redIntrInf4 = True
if redIntrInf4:
    P = bv.FourPDstrb()
    P = pr.marginal(P, 3)
    print( "Test MCupperBoundRedIntrinInfX(Y) with FourPDstrb()")
    for dimU in range(2,5):
        for dimBZU in range(2,5):
            print( "dimBZU = ", dimBZU, ", dimU = ", dimU)
            print( inf.MCupperBoundRedIntrinInfXY( P, dimU, dimBZU, 200, 200))
            print( inf.MCupperBoundRedIntrinInfX ( P, dimU, dimBZU, 200, 200))
        
# Loop over different random channels
loops = False
if loops:
    for k in range(0, 10):
        PC = inf.randChannel(2,2)
        print(PC)
        # Print P_Z after channel.
        # NB: last parties are swapped after applying the channel
        print( pr.marginal( inf.applyChannel( P, PC, 3), (0,1,2)))
        print( inf.mutInf( pr.marginal( inf.applyChannel(P, PC, 3), (2,3))))
        print( inf.MCupperBoundIntrinInf( pr.marginal(P, 3), 100))
        print( inf.MCupperBoundRedIntrinInf( pr.marginal( P, 3), 10, 10))
        pass

# Test random bipartite channel
cCMulti = False
if cCMulti:
    CMulti = inf.randChannelMultipart( (4,2), (2,2))
    print( CMulti.shape )
    print( CMulti.min()) 
    print( np.sum( CMulti , axis=(0,1)))

    # Test deterministic and general uniform behaviors and then the respective entropy
    print( bv.determBhv( (2,2), 3 ) )
    print( bv.determBhv( (2,2), 2 ) )
    print( bv.determBhv( (4,), 2 ) )
    print( inf.entropy(bv.determBhv( (4,), 2 ) ))

    print( bv.unifBhv( (2,2) ))
    print( bv.unifBhv( (4,2) ))
    print( bv.unifBhv( (2,) ))
    print( inf.entropy(bv.unifBhv( (2,4)) ))
    print( inf.entropy(bv.unifBhv( (2,2)) ))
    print( inf.entropy(bv.unifBhv( (2,)) ))

# Test the entropy
cEntr = False
if cEntr:
    values = []
    for p in np.arange(0, 1, 0.01):
        values.append( inf.entropy( bv.coin( p )))

    plt.plot(values)
    plt.savefig("binEntropy.pdf")
    plt.gcf().clear()
    
    values1 = []
    values2 = []
    for i in range(0,100):
        bhv = bv.randBhv( (2,))
        values1.append( bhv[0])
        values2.append( inf.entropy( bhv)) 
    
    plt.scatter(values1, values2)
    plt.savefig("randomlySampledBinEntropy.pdf")

# Test the application of a channel
cApplChannel = False
if cApplChannel:
    dimsChn = (4,5)
    bhv = bv.randBhv( (2,2,2,2) )
    rChn = inf.randChannelMultipart( dimsChn, (2,2))
    # Apply the channel to the first two parties
    bhvAfterChn1 = np.zeros( (2,2)+dimsChn)
    for x in range(0,dimsChn[0]):
        for y in range(0, dimsChn[1]):
            for z in range(0,2):
                for u in range(0,2):
                    for xp in range(0,2):
                        for yp in range(0,2):
                            bhvAfterChn1[ z,u,x,y ] += bhv[xp,yp,z,u]*rChn[x,y,xp, yp] 
    bhvAfterChn = inf.applyChannel( bhv, rChn, (0,1))
    print( np.amax(np.absolute(bhvAfterChn-bhvAfterChn1)))
    # Apply the channel to the first and the third party
    bhvAfterChn1 = np.zeros( (2,2)+dimsChn)
    for x in range(0,dimsChn[0]):
        for z in range(0, dimsChn[1]):
            for y in range(0,2):
                for u in range(0,2):
                    for xp in range(0,2):
                        for zp in range(0,2):
                            bhvAfterChn1[ y,u,x,z ] += bhv[xp,y,zp,u]*rChn[x, z, xp, zp] 
    bhvAfterChn = inf.applyChannel( bhv, rChn, (0,2))
    print( np.amax(np.absolute(bhvAfterChn-bhvAfterChn1)))
    # Apply the channel to the second and the third party
    bhvAfterChn1 = np.zeros( (2,2)+dimsChn)
    for y in range(0,dimsChn[0]):
        for z in range(0, dimsChn[1]):
            for x in range(0,2):
                for u in range(0,2):
                    for yp in range(0,2):
                        for zp in range(0,2):
                            bhvAfterChn1[ x,u,y,z ] += bhv[x,yp,zp,u]*rChn[y, z, yp, zp] 
    bhvAfterChn = inf.applyChannel( bhv, rChn, (1,2))
    print( np.amax(np.absolute(bhvAfterChn-bhvAfterChn1)))
    # Apply the channel to the first and the fourth party
    bhvAfterChn1 = np.zeros( (2,2)+dimsChn)
    for x in range(0,dimsChn[0]):
        for u in range(0, dimsChn[1]):
            for y in range(0,2):
                for z in range(0,2):
                    for xp in range(0,2):
                        for up in range(0,2):
                            bhvAfterChn1[ y,z,x,u ] += bhv[xp,y,z,up]*rChn[x, u, xp, up] 
    bhvAfterChn = inf.applyChannel( bhv, rChn, (0,3))
    print( np.amax(np.absolute(bhvAfterChn-bhvAfterChn1)))
