import numpy as np

# Define the metric tensor (second derivative of the potential term)
def G(x,k,lam):
    return k + 3*lam*x**2
        
# Define M (including delta).... to be used to avoid division by 0 errors
def M(x,k, lam, d):
    return np.sqrt(abs(G(x,k,lam)**2+d**2))

M_vals = []
for i in range(100):
    x = np.random.normal(0, 1)
    print("x is", x)
    M_vals.append(M(x, -1, 1, 0.01))
print(M_vals)

p_star = p_current - eps\
                        *(k*x_star + lam*x_star**3\
                                             + 0.5*p_guess**2*(-6*lam*x_star)\
                                             + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam, d))