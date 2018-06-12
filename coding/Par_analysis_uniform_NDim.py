import numpy as np
import prob as pr
import bhvs as bv
import analysis as alys

from joblib import Parallel, delayed
import multiprocessing


# compute `iter` steps towards the uniform distribution
# test for different range until endDim
iter = 5
endDim = 7

inputs = range(4,endDim) 

def processInput(dim):
    P = bv.FourPDistribN(dim)
    P = pr.marginal(P,3)
    uniform = np.ones_like(P)
    uniform = pr.normalize(uniform)
    # print("# X,Y range: {0}\t\tBhv shape: {1}".format(dim, P.shape))
    return alys.PtestInfoAlongPath(P, uniform, iter=iter)

num_cores = multiprocessing.cpu_count()
    
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

for res in results: 
    print(res)