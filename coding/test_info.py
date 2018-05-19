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
for k in range(10):
    PC = inf.randChannel(2,2)
    print( inf.mutInf( pr.marginal( inf.applyChannel(P, PC, 3), (2,3))))
    pass
