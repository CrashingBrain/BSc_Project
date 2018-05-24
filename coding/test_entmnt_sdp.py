import entmnt_sdp as es
import quantum as qm
import bhvs as bv
import prob as pr


# Transfer prob to state and check witness
rho1 = qm.PrToRho( pr.marginal( bv.ThreePDstrb(), 2))
print( rho1.shape)
print( es.PPTsymmExt(rho1, (4,4)))
