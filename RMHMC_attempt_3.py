'''
Upon failure of the Secant method for fixed point iteration, we revert to the basic Piccard method
(apparently the method that was originally used by Girolami and Calderhead). I attempted this before and had 
momentum values that got very large but I will try again in the hope that this is an appropriate method and
try and solve problems from there.
I will also not use a class because that was unnecessary and made the code so complicated. 
'''
import numpy as np

# Define the anharmonic potential term
def an_V(x,k,lam):
    return 0.5*k*x**2 + 0.25*lam*x**4

# Define the metric tensor (second derivative of the potential term)
def G(x,k,lam):
    '''
    We define the metric tensor and include a condition that prevents it from being 0
    '''
    G = k + 3*lam*x**2
    if G < 1e-14:
        G = 1e-5
    return G

# Define the kinetic energy term (include correction term)
def K(p, x):
    return 0.5*p*G(x)*p + 0.5*np.log(2*np.pi*np.abs(G(x)))

# Define the Hamiltonian
def H(x, p,k,lam):
    return an_V(x,k,lam) + K(p, x) 

def RMHMC_alg(n,eps,L,k,lam):
    '''
    Carry out the RMHMC algorithm to generate x values.
    We use the Piccard method for the fixed point iteration
    '''
    # Initialise the x list
    x = [0]
    # Start the loop for filling in the x values
    for t in range(n+1):
        # Draw mom from a normal distrbn
        p = np.random.normal(0, np.sqrt(G(x[t],k,lam)))
        # Begin the leapfrog method
        # Initialise p_star and x_star lists
        p_stars = [p]
        x_stars = [0]
        x_star = x_stars[-1]
        # Define variables that will be used for simplicty 
        inner_val = 1-4*eps*(G(x_stars[-1],k,lam))\
                                        *(p_stars[-1] + 0.5*eps*k*x_stars[-1] \
                                          +0.5*eps*lam*x_stars[-1]**3 \
                                            + 0.25*eps*abs(6*lam*x_stars[-1])/(G(x_stars[-1],k,lam)))
        root = np.sqrt(max(0.0, inner_val)) # Ensure no negative sqrt
        # First step 
        p_star = p_stars[-1] - 0.5*eps*(1+root) # Note that we took the + soln
        p_stars.append(p_star)
        x_star = x_star + eps*G(x_star,k,lam)*p_star
        x_stars.append(x_star)
        # Loop over leapfrog steps
        for i in range(1,L+1):
            inner_val = 1-2*eps*(G(x_stars[-1],k,lam))\
                                        *(p_stars[-1] + eps*k*x_stars[-1] \
                                          +eps*lam*x_stars[-1]**3 \
                                            + 0.5*eps*abs(6*lam*x_stars[-1])/(G(x_stars[-1],k,lam)))
            root = np.sqrt(max(0.0, inner_val))
            p_star = p_stars[-1] - eps*(1+root)
            p_stars.append(p_star)
            x_star = x_star +eps*G(x_star,k,lam)*p_star
            x_stars.append(x_star)
        # Final leapfrog step
        inner_val = 1-4*eps*(G(x_stars[-1],k,lam))\
                                        *(p_stars[-1] + 0.5*eps*k*x_stars[-1] \
                                          +0.5*eps*lam*x_stars[-1]**3 \
                                            + 0.25*eps*abs(6*lam*x_stars[-1])/(G(x_stars[-1],k,lam)))
        root = np.sqrt(max(0.0,inner_val))
        p_star = p_stars[-1] - 0.5*eps*(1+root)
        p_stars.append(p_star)
        # Carry out the Metropolis test
        # Compute the acceptance ratio
        r = np.exp(-H(x_stars[-1], p_stars[-1]) + H(x[t], p))
        # Draw W from a Uniform distribution
        W = np.random.uniform(0, 1)            
        # Carry out the Metropolis test
        if W <= min(1, r):
            x.append(x_stars[-1])
        else:
            x.append(x[t])
        return x
    
print(RMHMC_alg(10,0.1,10,1,1))
