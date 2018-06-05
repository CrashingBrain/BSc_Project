import bhvs as bv
import info as inf
import numpy as np
import prob as pr
import time
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

# Loop over different random channels
for k in range(0, 2):
   PC = inf.randChannel(2,2)
   print(PC)
   # Print P_Z after channel.
   # NB: last parties are swapped after applying the channel
   print( pr.marginal( inf.applyChannel( P, PC, 3), (0,1,2)))
   print( inf.mutInf( pr.marginal( inf.applyChannel(P, PC, 3), (2,3))))
   print( inf.MCupperBoundIntrinInf( pr.marginal(P, 3), 100))
   print( inf.MCupperBoundRedIntrinInf( pr.marginal( P, 3), 10, 10))
   # Test the new RedIntrinInfo function
   print( inf.MCupperBoundRedIntrinInf_( pr.marginal( P, 3), 10, 10))
   pass

# Test random bipartite channel
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

values = []
for p in np.arange(0, 1, 0.01):
    values.append( inf.entropy( bv.coin( p )))

plt.plot(values)
plt.show()
