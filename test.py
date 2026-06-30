import numpy as np
x = 1
p = 1
eps = 1
k =1
lam = 1
print(np.linalg.solve(p_star -(p - 0.5*eps*(k*x + lam*x**3 + 0.5*p_star**2 ))), p_star)
