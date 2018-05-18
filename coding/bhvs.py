import numpy as np
import prob as pr

# Four partite distrib
def FourPDstrb():
    Ptable = np.zeros((4,4,2,2))
    Ptable[0,0,0,0] = Ptable[1,1,0,0] = Ptable[0,1,1,0] = Ptable[1,0,1,0] = 1.0/8.0
    Ptable[2,2,0,1] = Ptable[3,3,1,1] = 1.0/4.0
    return Ptable

# Alternative implementation of the behavior above
def FourPDstrb2():
    P = np.zeros((4,4,2,2))
    for x in range(0,2):
        for y in range(0,2):
            P[x,y, (x+y)%2, x//2] = 1./8.
    for x in range(2,4):
        P[x, x, x%2, x//2] = 1./4.
    return P

def FourPDstrb3():
    P = np.zeros((4,4,16,2))
    for x in range(0,2):
        for y in range(0,2):
            P[x,y, (x+y)%2, x//2] = 1./8.
    for x in range(2,4):
        P[x, x, x%2, x//2] = 1./4.
    return P

def ThreePDstrb():
    P = np.zeros((4,4,16))
    for x in range(0,2):
        for y in range(0,2):
            P[x,y, (x+y)%2] = 1./8.
    for x in range(2,4):
        P[x, x, x%2] = 1./4.
    return P

# Four partite noise -> to be mixed with FourPDistrib
# Is to yield as a marginal the candidate from 2003 paper
def ThreePNoise1():
    P = np.zeros((4,4,16))
    # Fill upper triangle
    for idx in [[0,2],[0,3],[1,2],[1,3]]:
        P[ idx[0], idx[1], idx[0]*4 + idx[1]] = 1
        P[ idx[1], idx[0], idx[1]*4 + idx[0]] = 1
    return pr.normalize(P)        

def ThreePNoise2():
    P = np.zeros((4,4,16))
    # Fill upper triangle
    for idx in [[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3]]:
        P[ idx[0], idx[1], idx[0]*4 + idx[1]] = 1
        P[ idx[1], idx[0], idx[1]*4 + idx[0]] = 1
    return pr.normalize(P)        

def ThreePNoise3():
    P = np.zeros((4,4,16))
    for x in range(0, P.shape[0]):
        for y in range(0, P.shape[1]):
            P[ x, y, x*4 + y] = 1
    return pr.normalize(P)        

def ThreePUniformNoise():
    P = np.zeros((4,4,16))
    for x in range(0, P.shape[0]):
        for y in range(0, P.shape[1]):
            for z in range(0, P.shape[2]):
                P[ x,y,z] = 1
    return pr.normalize(P)        
