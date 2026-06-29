import numpy as np
import matplotlib.pyplot as plt

class RMHMC:
    '''
    Rewriting the anharmonic HMC class but for RMHMC in one dimension. 
    Mass is non-constant and is instead represented by the metric tensor which I 
    shall denote G. In the 1D case, this is just a scalar. 
    '''
    # Define the variables to be used
    def __init__(self, L=None, eps=None, k=None, lam=None):
        self.L = L
        self.eps = eps
        self.k = k
        self.lam = lam

    # Define the anharmonic potential term
    def an_V(self, x):
        return 0.5*self.k*x**2 + 0.25*self.lam*x**4

    # Define the anharmonic Hamiltonian
    def an_H(self, x, p):
        return self.an_V(x) + 0.5*p**2

    # Run the HMC algorithm
    def an_HMC_alg(self, n):
        # Initialise the x values
        x = [0]
        # Start the loop to generate x values
        for t in range(n+1):
            # Draw the momentum from a Normal distribution
            p = np.random.normal(0, self.m)
            # Compute the first leapfrog step
            p_star = p - 0.5*self.eps*(self.k*x[t] + self.lam*x[t]**3)
            x_star = x[t] + self.eps*p_star/self.m
            # Compute (x*, - p*) using L leapfrog steps of size eps
            for l in range(1, self.L):
                p_star = p_star - self.eps*(self.k*x_star + self.lam*x_star**3)
                x_star = x_star + self.eps*p_star/self.m
            # Compute the final step of the leapfrog method
            p_star = p_star - 0.5*self.eps*(self.k*x_star + self.lam*x_star**3)
            # Compute the acceptance ratio
            r = np.exp(-self.an_H(x_star, p_star) + self.an_H(x[t], p))
            # Draw W from a Uniform distribution
            W = np.random.uniform(0, 1)
            # Carry out the Metropolis test
            if W <= min(1, r):
                x.append(x_star)
            else:
                x.append(x[t])
        return x

# Find the expected value of x
def exp_val(x):
    '''
    Given a list of x values, compute the expected value
    '''
    return np.mean(x)