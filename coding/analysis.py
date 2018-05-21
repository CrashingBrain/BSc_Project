import prob as pr
import bhvs as bv
import info as inf
import numpy as np
import quantum as qm


def allTests(P):
    """
        performs all tests on a tripartite behaviour
    """
    # number of iterations for MC channels
    inIter = 10
    outIter = 10

    # get marginal for X,Y
    l = len(P.shape)
    t = tuple(np.arange(2,l))
    m = pr.marginal(P, t)
    # calculate mutual information of the marginal X,Y
    mutInf = inf.mutInf(m)
    intrInf = inf.MCupperBoundIntrinInf(P, inIter)
    # redIntrInf = inf.MCupperBoundRedIntrinInf(P, inIter, outIter)
    redIntrInf = 0.0

    # quantum part. very slow.
    rho = qm.PrToRho(m)
    dims = tuple(t//2 for t in np.shape(rho))
    ppt = qm.ppt(rho, dims) == 1
    # ppt = False
    return (mutInf, intrInf, redIntrInf, ppt)

def testAlongPath(P1, P2, iter=100):
    """
        test a Pdistr P1 along a path given by behaviour P2
        for iter steps
    """

    outStr = str()
    outStr += str('#Testing for %d iters\n' % iter)
    outStr += str('#alpha\t\tI(X,Y)\t\tintrInf\t\tredIntr\t\tseparable\n')
    print(outStr)
    for alpha in np.linspace(0,1,num=iter):
        newP = pr.mixBhvs(P1, P2, alpha=alpha)

        results = allTests(newP)
        results = (alpha,)+results
        outStr = str()
        outStr += str('%.3f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%s' % results)
        print(outStr)
    print('#-----')
    pass