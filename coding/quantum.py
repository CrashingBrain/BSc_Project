import numpy as np

# expect rho to be 2 partite density matrix
# return 1 if positive partial transpose
def ppt(rho, dims):
    # flatten rho to array
    rho.shape = np.power(np.prod( dims),2)
    T = np.zeros( np.power(dims[1],2), np.power(dims[1],2))
    for k in range(0,dims[1]):
        for l in range(0,dims[1]):
            T[ k + dims[1]*l, l + dims[1]*k] = 1
    PT_op = np.tensordot( np.identity( dims[0]), T)
    rho_pt = np.matmul( PT_op, rho)
    rho_pt.shape = ( np.prod(dims), np.prod(dims))
    min_val = np.min( np.eig(rho_pt)).w )
    if min_val > -1e-8:
        res = 1
    else:
        res 0
    return res

# define bipartite rho from bipartite probability distribution
def PrToRho(P):
    rho = np.zeros( np.prod( P.shape), np.prod( P.shape))
    for i in P.shape[0]:
        for j in P.shape[0]:
            for k in P.shape[1]:
                for l in P.shape[1]:
                    rho[ i + P.shape[0]*k, j + P.shape[0]*l] = np.sqrt( P[i,k]*P[j,l])
    pass

def proj(v):
    if v.shape[1] > 1:
        prj = np.matmul( v, np.transpose(v))
    else:
        prj = np.matmul( np.transpose(v), v)
    return prj

# Witness from quant-ph/0308032  (VII, B)
# For all separable states: tr(W*rho) >=0
def wtns44():
    v = np.zeros( 8, 16, 1)
    v[0][ 2 + 4*2, 1] = 1
    v[0][ 0 + 4*0, 1] = -1
    v[1][ 2 + 4*2, 1] = 1
    v[1][ 1 + 4*1, 1] = -1
    v[2][ 3 + 4*3, 1] = 1
    v[2][ 0 + 4*1, 1] = -1
    v[3][ 3 + 4*3, 1] = 1
    v[3][ 1 + 4*0, 1] = -1
    v[4][ 2 + 4*3, 1] = 1
    v[5][ 3 + 4*2, 1] = 1
    v[6][ 2 + 4*2, 1] = -1
    v[7][ 3 + 4*3, 1] = -1
    w = np.zeros(16,16)
    for k in range(0,8):
        w += proj(v[i])
    return w

# WANTED: SDP from quant-ph/0308032 eqn (23) resp (A1)
# But careful: might not be so easy (needs inclusion of an SDP solver (see e.g. https://peterwittek.com/sdp-in-python.html) 
    
