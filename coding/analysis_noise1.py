import numpy as np
import prob as pr
import bhvs as bv
import analysis as alys

# P = bv.ThreePDstrb()
P = bv.FourPDstrb()
P = pr.marginal(P, 3)

print('Initial distr:')
pr.PrintThreePDstrb(P)
print() # new line

# compute `iter` steps towards the uniform distribution
iter = 100
uniform = bv.ThreePNoise1_()
alys.testInfoAlongPath(P, uniform, iter=iter)