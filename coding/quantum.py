import numpy as np

def pt(rho, dims):
    # flatten rho to array
    rho.shape = np.power(np.prod( dims),2)
    # build matrix for PT
    PT = np.zeros( (np.power(np.prod(dims),2), np.power(np.prod(dims),2)))
    for m in range(0, dims[0]):
        for n in range(0, dims[0]):
            for k in range(0,dims[1]):
                for l in range(0,dims[1]):
                    PT[ m + dims[0] * (k + dims[1] * ( n + dims[0] * l)), m + dims[0] * (l + dims[1] * ( n + dims[0] * k))] = 1
    rho_pt = np.matmul( PT, rho)
    rho_pt.shape = ( np.prod(dims), np.prod(dims))
    return rho_pt

# expect rho to be 2 partite density matrix
# return 1 if positive partial transpose
def ppt(rho, dims):
    min_val = np.linalg.eig( pt(rho, dims))[0].min()
    print('qm.ppt.minval ' + str(min_val))
    if min_val > -1e-8:
        res = 1
    else:
        res = 0
    return res

# define bipartite rho from bipartite probability distribution
def PrToRho(P):
    rho = np.zeros( (np.prod( P.shape), np.prod( P.shape)))
    for i in range(0, P.shape[0]):
        for j in range(0, P.shape[0]):
            for k in range(0, P.shape[1]):
                for l in range(0, P.shape[1]):
                    rho[ i + P.shape[0]*k, j + P.shape[0]*l] = np.sqrt( P[i,k]*P[j,l])
    return rho

def proj(v):
    v.shape = (len(v), 1)
    return np.matmul( v, np.transpose(v))

# Witness from quant-ph/0308032  (VII, B)
# For all separable states: tr(W*rho) >=0
# NB: something wrong.... doesn't yield the expected violation
def wtns44():
    v = np.zeros( (8, 16))
    v[0,  2 + 4 * 2] = 1
    v[0,  0 + 4 * 0] = -1
    v[1,  2 + 4 * 2] = 1
    v[1,  1 + 4 * 1] = -1
    v[2,  3 + 4 * 3] = 1
    v[2,  0 + 4 * 1] = -1
    v[3,  3 + 4 * 3] = 1
    v[3,  1 + 4 * 0] = -1
    v[4,  2 + 4 * 3] = 1
    v[5,  3 + 4 * 2] = 1
    v[6,  2 + 4 * 2] = 1
    v[7,  3 + 4 * 3] = 1
    w = np.zeros((16,16))
    for k in range(0,4):
        w += proj(v[k, :])
    w += proj( v[4, :]) + proj( v[5, :])
    w += -proj( v[6, :]) - proj( v[7, :])
    return w

def rho_a(alpha=0.):
    psi = np.zeros( (2, 16))
    psi[ 0, 0 + 4 * 0] = 1./2.
    psi[ 0, 1 + 4 * 1] = 1./2.
    psi[ 0, 2 + 4 * 2] = np.sqrt(2.)/2.
    psi[ 1, 0 + 4 * 1] = 1./2.
    psi[ 1, 1 + 4 * 0] = 1./2.
    psi[ 1, 3 + 4 * 3] = np.sqrt(2.)/2.
    sigma = np.zeros( (8, 16))
    sigma[ 0, 0 + 4 * 2] = 1.
    sigma[ 1, 0 + 4 * 3] = 1.
    sigma[ 2, 1 + 4 * 2] = 1.
    sigma[ 3, 1 + 4 * 3] = 1.
    sigma[ 4, 2 + 4 * 0] = 1.
    sigma[ 5, 2 + 4 * 1] = 1.
    sigma[ 6, 3 + 4 * 0] = 1.
    sigma[ 7, 3 + 4 * 1] = 1.
    rho = proj( psi[0,:]) + proj(psi[1,:])
    rho_s = np.zeros((16,16))
    for k in range(0,8):
        rho_s += 1./8.*proj( sigma[k, :])
    return ( 1./(2+alpha)*(rho+alpha*rho_s))
