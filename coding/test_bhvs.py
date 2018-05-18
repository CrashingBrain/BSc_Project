import prob as pr
import bhvs as bv
import info as inf
import numpy as np

P = bv.FourPDstrb()
P1 = bv.FourPDstrb2()
# Compare different methods
print(np.subtract(P,P1).max())
P_XY = pr.marginal(P,(2,3))
P_XYU = pr.marginal(P,2)
pr.PrintFourPDstrb(P)
pr.PrintFourPDstrb(P1)
pr.PrintThreePDstrb( bv.ThreePNoise1())
