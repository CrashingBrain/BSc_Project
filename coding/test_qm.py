import bhvs as bv
import info as inf
import numpy as np
import quantum as qm

# Get some entangled vector
psi = np.multiply( 1./np.sqrt(2), np.identity(4)[0, :]+np.identity(4)[3,:])
rho = qm.proj(psi)
print( qm.ppt(rho))

