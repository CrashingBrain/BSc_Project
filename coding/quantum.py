import numpy as np

def pt(rho, dims):
    # flatten rho to array
    rho.shape = np.power(np.prod( dims),2)
    # build matrix for PT
    PT = np.zeros( (np.power(np.prod(dims),2), np.power(np.prod(dims),2)))
    x = dims[0] # performance
    y = dims[1] # performance
    for m in range(0, x):
        for n in range(0, x):
            for k in range(0,y):
                for l in range(0,y):
                    PT[ m + x * (k + y * ( n + x * l)), m + x * (l + y * ( n + x * k))] = 1
    rho_pt = np.matmul( PT, rho)
    rho_pt.shape = ( np.prod(dims), np.prod(dims))
    return rho_pt

# expect rho to be 2 partite density matrix
# return 1 if positive partial transpose
def ppt(rho, dims):
    min_val = np.linalg.eig( pt(rho, dims))[0].min()
    # print('qm.ppt.minval ' + str(min_val))
    if min_val > -1e-16:
        res = 1
    else:
        res = 0
    return res

# define bipartite rho from bipartite probability distribution
def PrToRho(P):
    rho = np.zeros( (np.prod( P.shape, dtype=int), np.prod( P.shape, dtype=int)))
    x = P.shape[0] # performance
    y = P.shape[1] # performance
    for i in range(0, x):
        for j in range(0, x):
            for k in range(0, y):
                for l in range(0, y):
                    rho[ i + x*k, j + x*l] = np.sqrt( P[i,k]*P[j,l])
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

def bell_phi_minus():
    return proj( np.array([1./np.sqrt(2.), 0,0, -1./np.sqrt(2.) ]))

def bell_phi_plus():
    return proj( np.array([1./np.sqrt(2.), 0,0, 1./np.sqrt(2.) ]))

def bell_psi_minus():
    return proj( np.array([0, 1./np.sqrt(2.), -1./np.sqrt(2.), 0]))

def bell_phi_plus():
    return proj( np.array([0, 1./np.sqrt(2.), 1./np.sqrt(2.), 0]))
