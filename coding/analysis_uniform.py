import prob as pr
import bhvs as bv
import info as inf
import numpy as np
import operator

P = bv.ThreePDstrb()

print('Initial distr:')
pr.PrintThreePDstrb(P)
print() # new line

# compute `iter` steps towards the uniform distribution
iter = 100
alpha = np.linspace(0,1,num=iter)
Ps = []
uniform = bv.ThreePUniformNoise()
print('applying uniform noise...')
for a in alpha:
    Ps.append(pr.mixBhvs(P, uniform, a) )

mutIs = []
inIter = 10
outIter = 10
intrIs = []
redIntrIs = []
print('getting values...')
for i, p in enumerate(Ps):
    m = pr.marginal(p, (2,))
    # calculate mutual information of the marginal X,Y
    mutIs.append(inf.mutInf(m))
    # use MC to get upper bound to intrinsic info
    intrIs.append(inf.MCupperBoundIntrinInf(p, inIter))
    # use MC to get upper bound on reduced intrinsic info
    # redIntrIs.append(inf.MCupperBoundRedIntrinInf(p, inIter, outIter))
    print("done %d/%d" % (i, len(Ps)))
# print results

print('mutual information of marginals:')
print(mutIs); print() # new line
print(m); print()
print('estimated upper bound to intrinsic info:')
print(intrIs); print()
print('estimated upper bound to reduced intr. info:')
print(redIntrIs); print()
print('estiamted gap between the two:')
diff = list(map(operator.sub, intrIs, redIntrIs))
print(diff); print()