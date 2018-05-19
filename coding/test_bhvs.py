import prob as pr
import bhvs as bv
import info as inf
import numpy as np

P = bv.FourPDstrb()
P1 = bv.FourPDstrb2()
print('# Compare different methods')
print(np.subtract(P,P1).max())
print() # new line
P_XY = pr.marginal(P,(2,3))
P_XYU = pr.marginal(P,2)
print('# P_XY_ZU:') 
pr.PrintFourPDstrb(P)
# pr.PrintFourPDstrb(P1)
print('# P_(X,Y):')
print(np.sum(bv.ThreePDstrb(), axis=2))
print() # new line
pr.PrintThreePDstrb(bv.ThreePDstrb())
# print('# Noises functions:')
# pr.PrintThreePDstrb( bv.ThreePNoise1())
# pr.PrintThreePDstrb( bv.ThreePNoise2())
# pr.PrintThreePDstrb( bv.ThreePNoise3())
# pr.PrintThreePDstrb( bv.ThreePUniformNoise())

pr.PrintThreePDstrb( pr.mixBhvs( bv.ThreePDstrb(), bv.ThreePNoise2(), .1))
