import bhvs as bv
import prob as pr
import info as inf
import numpy as np
import quantum as qm

# Get some entangled vector
psi = np.multiply( 1./np.sqrt(2), np.identity(4)[:, 0]+np.identity(4)[:,3])
rho = qm.proj(psi)
print( rho)
print( qm.pt(rho, (2,2)))
print( qm.ppt(rho, (2,2)))

# Transfer prob to state and check witness
rho1 = qm.PrToRho( pr.marginal( bv.ThreePDstrb(), 2))
print(rho1)
print( np.trace( np.matmul( rho1, qm.wtns44())))
print( qm.pt(rho1, (4,4)))
# TO BE DONE
