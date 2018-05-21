import numpy as np

def pt(rho, dims):
    # flatten rho to array
    # rho.shape = np.power(np.prod( dims),2)
    # not sure this is what you meant...
    rho = rho.flatten()
    rho = np.resize(rho, np.power(np.prod( dims),2))
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
    # print('qm.ppt.minval ' + str(min_val))
    if min_val > -1e-8: #why the minus? also why only 1e-8? python uses 64bit by default
        res = 1
    else:
        res = 0
    return res

# define bipartite rho from bipartite probability distribution
def PrToRho(P):
    rho = np.zeros( (np.prod(P.shape, dtype=int), np.prod(P.shape, dtype=int)))
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
def wtns44():
    v = np.zeros( (8, 16, 1))
    v[0,  2 + 4*2, 0] = 1
    v[0,  0 + 4*0, 0] = -1
    v[1,  2 + 4*2, 0] = 1
    v[1,  1 + 4*1, 0] = -1
    v[2,  3 + 4*3, 0] = 1
    v[2,  0 + 4*1, 0] = -1
    v[3,  3 + 4*3, 0] = 1
    v[3,  1 + 4*0, 0] = -1
    v[4,  2 + 4*3, 0] = 1
    v[5,  3 + 4*2, 0] = 1
    v[6,  2 + 4*2, 0] = -1
    v[7,  3 + 4*3, 0] = -1
    w = np.zeros((16,16))
    for k in range(0,8):
        w += proj(v[k, :])
    return w

# WANTED: SDP from quant-ph/0308032 eqn (23) resp (A1)
# But careful: might not be so easy (needs inclusion of an SDP solver (see e.g. https://peterwittek.com/sdp-in-python.html) 
    
