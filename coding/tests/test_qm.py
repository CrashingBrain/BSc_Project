import numpy as np
import prob as pr
import quantum as qm
import bhvs as bv

# Get some entangled vector
psi = np.multiply( 1./np.sqrt(2), np.identity(4)[:, 0]+np.identity(4)[:,3])
rho = qm.proj(psi)
print( rho)
print( qm.pt(rho, (2,2)))
print( qm.ppt(rho, (2,2)))

# Transfer prob to state and check witness
rho1 = qm.PrToRho( pr.marginal( bv.ThreePDstrb(), 2))
print( "Witness from paper")
print( qm.wtns44())
print( "QmState from paper")
print( qm.rho_a())

print(rho1)
alpha = .5
threshold = -2.*(np.sqrt(2.)-1.)/(2.+alpha)
print( np.trace( np.matmul( qm.rho_a(alpha), qm.wtns44())))
print( "Val from paper: " + str(threshold))
alpha = .05
threshold = -2.*(np.sqrt(2.)-1.)/(2.+alpha)
print( np.trace( np.matmul( qm.rho_a(alpha), qm.wtns44())))
print( "Val from paper: " + str(threshold))
print( np.trace( np.matmul( rho1, qm.wtns44())))

# TO BE TESTED: that rho_a coincides with the transfered state...
