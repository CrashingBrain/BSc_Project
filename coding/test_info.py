import bhvs as bv
import info as inf
import numpy as np
import prob as pr
import time
import matplotlib.pyplot as plt

# Test conditiona MutInf
if False:
    P = bv.FourPDstrb()    
    print("Test CondMutInfo")
    start = time.time()
    print(inf.condMutInf(pr.marginal(P,3)))
    end = time.time()
    print("time Arne: %.8f" % (end - start))
    start = time.time()
    print(inf.condMutInf_(pr.marginal(P,3),0,1,2))
    end = time.time()
    print("time Mio: %.8f" % (end - start))
    print("---")

# Test channels
if False:
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

# Test deterministic channels
if False:
    dim_in = (2,2)
    dim_out = 3
    for l in range(0, dim_out**(np.prod(dim_in))):
        PC = inf.detChannel( dim_out, dim_in, l)
        for k in range(0, np.prod(dim_in)):
            coefftpl = inf.coeffOfNo(k, dim_in)
            print(PC[:, coefftpl[0], coefftpl[1]])

# Check the reduced intrinsic information upper bound
if False:
    P = pr.marginal(P,(3))
    PC_UXYZ = inf.randChannelMP( (np.prod(P.shape),), P.shape)
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
        PC_UZ = inf.randChannelMP( (P.shape[2:]), (P.shape[2:]))
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
        PC_UZ = inf.randChannelMP( (np.prod(P.shape[2:]),), (P.shape[2:]))
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

# IntrInf ThreePDstrb from FourPDstrb
if False:
    P = bv.FourPDstrb()
    P = pr.marginal(P, 3)
    print( "Test MCupperBoundIntrinInfMP with Marginal over U of FourPDstrb()")
    for dimBZU in range(2,5):
        print( dimBZU, inf.MCupperBoundIntrinInfMP( P, dimBZU, 20))
 
# IntrInf FourPDstrb
if False:
    P = bv.FourPDstrb()
    print( "Test MCupperBoundIntrinInfMP with FourPDstrb()")
    for dimBZU in range(2,5):
        print( dimBZU, inf.MCupperBoundIntrinInfMP( P, dimBZU, 2000))
 
# IntrInfDet ThreePDstrb from FourPDstrb
if False:
    P = bv.FourPDstrb()
    P = pr.marginal(P, 3)
    print( "Test MCupperBoundIntrinInfMPDet with Marginal over U of FourPDstrb()")
    for dimBZU in range(2,5):
        print( dimBZU, inf.MCupperBoundIntrinInfMPDet( P, dimBZU))
 
# IntrInfDet FourPDstrb
if False:
    P = bv.FourPDstrb()
    print( "Test MCupperBoundIntrinInfMPDet with FourPDstrb()")
    for dimBZU in range(2,5):
        print( dimBZU, inf.MCupperBoundIntrinInfMPDet( P, dimBZU))

# RedIntrInf
if False:
    P = bv.FourPDstrb()
    P = pr.marginal(P, 3)
    print( "Test MCupperBoundRedIntrinInfX(Y) with FourPDstrb()")
    for dimU in range(2,5):
        for dimBZU in range(2,5):
            print( "dimBZU = ", dimBZU, ", dimU = ", dimU)
            print( inf.MCupperBoundRedIntrinInfXY( P, dimU, dimBZU, 200, 200))
            print( inf.MCupperBoundRedIntrinInfX ( P, dimU, dimBZU, 200, 200))

# RedIntrInfDet
if False:
    P.bv.FourPDstrb()
    P = pr.marginal(P, 3)
    print( "Test MCupperBoundRedIntrinInfX(Y)Det with FourPDstrb()")
    for dimU in range(2,5):
        for dimBZU in range(2,5):
            print( "dimBZU = ", dimBZU, ", dimU = ", dimU)
            print( inf.MCupperBoundRedIntrinInfXYDet( P, dimU, dimBZU, 200))
            print( inf.MCupperBoundRedIntrinInfXDet ( P, dimU, dimBZU, 200))
        
# RedIntrInfDD
if False:
    P = bv.FourPDstrb()
    P = pr.marginal(P, 3)
    print( "Test MCupperBoundRedIntrinInfX(Y)DD with FourPDstrb()")
    for dimU in range(2,5):
        for dimBZU in range(2,5):
            print( "dimBZU = ", dimBZU, ", dimU = ", dimU)
            print( inf.MCupperBoundRedIntrinInfXYDD( P, dimU, dimBZU))
            print( inf.MCupperBoundRedIntrinInfXDD ( P, dimU, dimBZU))
        
# Loop over different random channels
if True:
    P = bv.FourPDstrb()
    for k in range(0, 10):
        PC = inf.randChannel(2,2)
        print(PC)
        # Print P_Z after channel.
        # NB: last parties are swapped after applying the channel
        print( pr.marginal( inf.applyChannel( P, PC, 3), (0,1,2)))
        print( inf.mutInf( pr.marginal( inf.applyChannel(P, PC, 3), (2,3))))
        print( inf.MCupperBoundIntrinInf( pr.marginal(P, 3), 100))
        print( inf.MCupperBoundRedIntrinInfXY(pr.marginal(P,3), 2, 2, 10, 10))
        # Test the new RedIntrinInfo function
        print( inf.MCupperBoundRedIntrinInf_( pr.marginal( P, 3), 10, 10))
        pass
    print("*** END LOOPS ***")

# Test random bipartite channel
if False:
    CMulti = inf.randChannelMP( (4,2), (2,2))
    print( CMulti.shape )
    print( CMulti.min()) 
    print( np.sum( CMulti , axis=(0,1)))
    print("---")

    # Test deterministic and general uniform behaviors and then the respective entropy
    print( bv.determBhv( (2,2), 3 ) )
    print("---")
    print( bv.determBhv( (2,2), 2 ) )
    print("---")
    print( bv.determBhv( (4,), 2 ) )
    print("---")
    print( inf.entropy(bv.determBhv( (4,), 2 ) ))
    print("---")

    print( bv.unifBhv( (2,2) ))
    print("---")
    print( bv.unifBhv( (4,2) ))
    print("---")
    print( bv.unifBhv( (2,) ))
    print("---")
    print( inf.entropy(bv.unifBhv( (2,4)) ))
    print( inf.entropy(bv.unifBhv( (2,2)) ))
    print( inf.entropy(bv.unifBhv( (2,)) ))

# Test the entropy
if False:
    values = []
    for p in np.linspace(0,1,num=100):
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
if False:
    dimsChn = (4,5)
    bhv = bv.randBhv( (2,2,2,2) )
    rChn = inf.randChannelMP( dimsChn, (2,2))
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
    # Test on binarization channel
    rChnB = inf.randChannelMP((2,),(2,2))
    bhvAfterChn1 = np.zeros( (2,2,2))
    for x in range(0,2):
        for z in range(0,2):
            for u in range(0,2):
                for xp in range(0,2):
                    for yp in range(0,2):
                        bhvAfterChn1[z,u,x] += bhv[xp,yp,z,u]*rChnB[x,xp,yp] 
    bhvAfterChn = inf.applyChannel( bhv, rChnB, (0,1))
    print( np.amax(np.absolute(bhvAfterChn-bhvAfterChn1)))
    # Test as in MCupperBoundIntrInfMP
    bhvFoo = bv.randBhv( (32,4,4,2) )
    rChnFoo = inf.randChannelMP((2,),(32,2))
    bhvAfterChn1 = np.zeros( (4,4,2))
    for x in range(0,2):
        for y in range(0,4):
            for z in range(0,4):
                for xp in range(0,32):
                    for up in range(0,2):
                        bhvAfterChn1[ y,z,x ] += bhvFoo[xp,y,z,up]*rChnFoo[x, xp, up] 
    bhvAfterChn = inf.applyChannel( bhvFoo, rChnFoo, (0,3))
    print( np.amax(np.absolute(bhvAfterChn-bhvAfterChn1)))