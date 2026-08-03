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