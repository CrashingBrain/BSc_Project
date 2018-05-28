import bhvs as bv
import info as inf
import numpy as np
import prob as pr
import time

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
for k in range(0, 10):
    PC = inf.randChannel(2,2)
    print(PC)
    # Print P_Z after channel.
    # NB: last parties are swapped after applying the channel
    print( pr.marginal( inf.applyChannel( P, PC, 3), (0,1,2)))
    print( inf.mutInf( pr.marginal( inf.applyChannel(P, PC, 3), (2,3))))
    print( inf.MCupperBoundIntrinInf( pr.marginal(P, 3), 100))
    print( inf.MCupperBoundRedIntrinInf( pr.marginal( P, 3), 10, 10))
    # test New MCupperBoundRedIntrinInf
    print( inf.MCupperBoundRedIntrinInf_( pr.marginal( P, 3), 10, 10))
    pass
