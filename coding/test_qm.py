import bhvs as bv
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
# TO BE DONE
