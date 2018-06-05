import bhvs as bv
import info as inf
import numpy as np
import prob as pr
import matplotlib.pyplot as plt

PC = inf.randChannel(3,2)
print(PC.shape)
print(PC)

P = bv.FourPDstrb()
print(P.shape)
print(inf.applyChannel(P, PC, 2).shape)
print(inf.applyChannel(P, PC, 3).shape)
print(inf.applyChannel(P, PC, 2))
print(inf.applyChannel(P, PC, 3))
print(inf.mutInf(pr.marginal(P, (2,3))))

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
cCMulti = True
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
cEntr = True
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
cApplChannel = True
if cApplChannel:
    bhv = bv.randBhv( (2,2,2,2) )
    rChn = inf.randChannelMultipart( (4,4), (2,2))
    # Apply the channel to the first two parties
    bhvAfterChn = inf.applyChannel( bhv, PC, (0,1))
    bhvAfterChn1 = np.zeros( (2,2,4,4))
    for x in range(0,2):
        for y in range(0,4):
            for z in range(0,4):
                for u in range(0,2):
                    for xp in range(0,2):
                        for yp in range(0,2):
                            bhvAfterChn1[ z,u,x,y ] += bhv[xp,yp,z,u]*rChn[x,y,xp, yp] 
    print( np.amax(np.absolute(bhvAfterChn-bhvAfterChn1)))
