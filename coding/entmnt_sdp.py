import cvxopt as co

# Goal is the alternative formulation of the SDP -> appendix A1

# Objective function: t
c = co.matrix(0., (1,8))
c[0] = 1.
print(c)

# Compute the function F (G in cvxopt manual)
A = [ co.matrix( identity ....)]
