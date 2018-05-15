import prob as pr
import infro as inf

P = pr.FourPDstrb()
P_XY = pr.marginal(P,(2,3))
P_XYU = pr.marginal(P,2)
P1 = pr.FourDimToTwo(P)
P2 = pr.FourDimToTwo(P).T
print(P1)
print(P2)
print(P_XY)
print(P_XYU)
P3 = pr.marginal(P1,1)
P4 = pr.marginal(P1,0)
print(P3)
print(P4)
print(pr.OneDimToTwo( P3))

