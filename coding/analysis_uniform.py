import numpy as np
import prob as pr
import bhvs as bv
import analysis as alys

P = bv.ThreePDstrb()

print('Initial distr:')
pr.PrintThreePDstrb(P)
print() # new line

# compute `iter` steps towards the uniform distribution
iter = 25
uniform = bv.ThreePUniformNoise()
print('applying uniform noise...')
alys.testInfoAlongPath(P, uniform, iter=iter)
# 1'15" for testInfo 1 iter
# 5'59" for testAll 1 iter