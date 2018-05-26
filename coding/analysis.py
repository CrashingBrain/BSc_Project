import prob as pr
import bhvs as bv
import info as inf
import numpy as np
import quantum as qm


def allTests(P, inIter=10, outIter=10):
    """
        performs all tests on a tripartite behaviour
    """

    results = []
    # get marginal for X,Y
    l = len(P.shape)
    t = tuple(np.arange(2,l))
    m = pr.marginal(P, t)
    # calculate mutual information of the marginal X,Y
    mutInf = inf.mutInf(m)
    results.append(mutInf)
    # print('#Done mutInf')
    intrInf = inf.MCupperBoundIntrinInf(P, inIter)
    results.append(intrInf)
    # print('#Done intrInf')
    redIntrInf = inf.MCupperBoundRedIntrinInf(P, inIter, outIter)
    results.append(redIntrInf)
    # print('#Done redIntrInf')
    # redIntrInf = 0.0

    # quantum part. very slow.
    rho = qm.PrToRho(m)
    dims = tuple(int(np.sqrt(t)) for t in np.shape(rho))
    ppt = qm.ppt(rho, dims) == 1
    # ppt = False
    results.append(ppt)
    # trace with witness
    # trace = 0.0
    trace = np.trace( np.matmul( rho, qm.wtns44()))
    results.append(trace)
    return results

def allInfoTests(P, inIter=10, outIter=10):
    """
        performs information theoretic tests on a tripartite behaviour
    """

    results = []
    # get marginal for X,Y
    l = len(P.shape)
    t = tuple(np.arange(2,l))
    m = pr.marginal(P, t)
    # calculate mutual information of the marginal X,Y
    mutInf = inf.mutInf(m)
    results.append(mutInf)
    # print('#Done mutInf')
    intrInf = inf.MCupperBoundIntrinInf(P, inIter)
    results.append(intrInf)
    # print('#Done intrInf')
    redIntrInf = inf.MCupperBoundRedIntrinInf_(P, inIter, outIter)
    results.append(redIntrInf)

    return results

def testAllAlongPath(P1, P2, iter=100):
    """
        test a Pdistr P1 along a path given by behaviour P2
        for iter steps
    """

    outStr = str()
    outStr += str('#Testing for %d iters\n' % iter)
    outStr += str('#alpha\t\tI(X,Y)\t\tintrInf\t\tredIntr\t\tseparable\ttrace\n')
    print(outStr)
    for alpha in np.linspace(0,1,num=iter):
        newP = pr.mixBhvs(P1, P2, alpha=alpha)

        results = tuple(allTests(newP, 5, 1))
        results = (alpha,)+results
        outStr = str()
        outStr += str('%.3f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%s\t\t%.4f' % results)
        print(outStr)
    print('#-----')
    pass

def testInfoAlongPath(P1,P2,iter=100):
    """
        test a Pdistr P1 along a path given by behaviour P2
        for iter steps
        only tests from info.py are applied
    """

    outStr = str()
    outStr += str('#Testing for %d iters\n' % iter)
    outStr += str('#alpha\t\tI(X,Y)\t\tintrInf\t\tredIntr\n')
    print(outStr)
    for alpha in np.linspace(0,1,num=iter):
        newP = pr.mixBhvs(P1, P2, alpha=alpha)

        results = tuple(allInfoTests(newP, 2, 2))
        results = (alpha,)+results
        outStr = str()
        outStr += str('%.3f\t\t%.4f\t\t%.4f\t\t%.4f' % results)
        print(outStr)
    print('#-----')
    pass