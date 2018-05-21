import numpy as np
import prob as pr
import bhvs as bv
import analysis as alys

P = bv.ThreePDstrb()

print('Initial distr:')
pr.PrintThreePDstrb(P)
print() # new line

# compute `iter` steps towards the uniform distribution
iter = 10
uniform = bv.ThreePUniformNoise()
print('applying uniform noise...')
alys.testAlongPath(P, uniform, iter=iter)