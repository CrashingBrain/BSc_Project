import numpy as np
import prob as pr
import bhvs as bv
import analysis as alys

P = bv.FourPDstrb()
Pm = pr.marginal(P,3)

print('Initial distr:')
pr.PrintFourPDstrb(P)
print() # new line

# compute `iter` steps towards the uniform distribution
iter = 10
uniform = bv.ThreePUniformNoise2()
print('applying uniform noise...')
alys.testInfoAlongPath(Pm, uniform, iter=iter)