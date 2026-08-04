import numpy as np
k = -1
lam = 1
d = 0.01
L = 100
eps = 0.01
# Define the metric tensor (second derivative of the potential term)
def G(x,k,lam):
    return k + 3*lam*x**2    
# Define M (including delta).... to be used to avoid division by 0 errors
def M(x,k, lam, d):
    return np.sqrt(abs(G(x,k,lam)**2+d**2))

def dHdx(x, p):
    return k*x + lam*x**3\
                         + 0.5*p**2*(6*lam*x)\
                                             + 0.5*abs(-6*lam*x)/M(x,k,lam, d)
# Initialise 
x = [1]
p = 0
p_star = 4
p_star_vals = []
for l in range(L):
    print("Iteration", l)
    count = 0
    max_iters = 100
    while count < max_iters:
        count = count +1
        print("count", count)
        f = p_star - p + 0.5*eps*dHdx(x[0],p_star)
        f_prime = 1 + 0.5*eps*p_star*(6*lam*x[0])

        p_star = p_star - f/f_prime
        print("p_star is", p_star)
        print("f is", f)
        if f < 1e-6:
            p_star_vals.append(p_star)
            break
# print(p_star_vals)