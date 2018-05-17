import prob as pr
import info as inf
import numpy as np

PC = inf.randChannel(3,2)
print(PC)
P = pr.FourPDstrb()
print(PC.shape)
print(P.shape)
print(inf.applyChannel(P, PC, 2).shape)
print(inf.applyChannel(P, PC, 3).shape)
print(inf.applyChannel(P, PC, 2))
print(inf.applyChannel(P, PC, 3))

print(inf.mutInf(pr.marginal(P, (2,3))))
