import numpy as np
import prob as pr
import bhvs as bv
import analysis as alys

# compute `iter` steps towards the uniform distribution
# test for different range until endDim
iter = 10
endDim = 10

for dim in range(4,endDim):
    P = bv.FourPDistribN(dim)
    P = pr.marginal(P,3)
    uniform = np.ones_like(P)
    uniform = pr.normalize(uniform)
    print("# X,Y range: {0}\t\tBhv shape: {1}".format(dim, P.shape))

    alys.testInfoAlongPath(P, uniform, iter=iter)
    print() # enpty line