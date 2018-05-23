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
    
