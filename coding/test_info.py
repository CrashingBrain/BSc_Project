import bhvs as bv
import info as inf
import numpy as np
import prob as pr

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
#for k in range(0, 10):
#    PC = inf.randChannel(2,2)
#    print(PC)
#    # Print P_Z after channel.
#    # NB: last parties are swapped after applying the channel
#    print( pr.marginal( inf.applyChannel( P, PC, 3), (0,1,2)))
#    print( inf.mutInf( pr.marginal( inf.applyChannel(P, PC, 3), (2,3))))
#    print( inf.MCupperBoundIntrinInf( pr.marginal(P, 3), 100))
#    print( inf.MCupperBoundRedIntrinInf( pr.marginal( P, 3), 10, 10))
#    pass

# Test random bipartite channel
CMulti = inf.randChannelMultipart( (4,2), (2,2))
print( CMulti.shape )
print( CMulti.min()) 
print( np.sum( CMulti , axis=(0,1)))

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
