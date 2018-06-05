import entmnt_sdp as es
import quantum as qm
import bhvs as bv
import prob as pr

# Do some consistency checks here
rho1 = qm.bell_psi_minus()
dim = (2,2)
c = es.c( dim)
G = es.assembleGx( dim)
print(c.size)
print(c)
print(G.size)


# Transfer prob to state and check witness
#rho1 = qm.PrToRho( pr.marginal( bv.ThreePDstrb(), 2))
#print( rho1.shape)
#print( es.PPTsymmExt(rho1, (4,4)))
