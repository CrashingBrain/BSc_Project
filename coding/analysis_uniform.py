import numpy as np
import prob as pr
import bhvs as bv
import analysis as alys

P = bv.ThreePDstrb()

print('Initial distr:')
pr.PrintThreePDstrb(P)
print() # new line

# compute `iter` steps towards the uniform distribution
iter = 100
uniform = bv.ThreePUniformNoise()
print('applying uniform noise...')
alys.testAlongPath(P, uniform, iter=iter)

# print('mutual information of marginals:')
# print(mutIs); print() # new line
# print(m); print()
# print('estimated upper bound to intrinsic info:')
# print(intrIs); print()
# print('estimated upper bound to reduced intr. info:')
# print(redIntrIs); print()
# print('estiamted gap between the two:')
# diff = list(map(operator.sub, intrIs, redIntrIs))
# print(diff); print()