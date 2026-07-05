import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.optimize import root

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

    # Define the metric tensor (negative second derivative of the potential term)
    def G(self, x):
        return -np.abs(self.k) - 3*self.lam*x**2
        
    # Define the kinetic energy term (include correction term)
    def K(self, p, x):
        return 0.5*p*self.G(x)*p + 0.5*np.log(2*np.pi*np.abs(self.G(x)))

    # Define the Hamiltonian
    def H(self, x, p):
        return self.an_V(x) + self.K(p, x) 

    # Run the HMC algorithm
    def RMHMC_alg(self, n):
        # Initialise the x values
        x = [0]
        # Start the loop to generate x values
        for t in range(n+1):
            # Compute the first leapfrog step
            p_star = p - 0.5*self.eps*(self.k*x[t] + self.lam*x[t]**3 + 0.5*p_star**2(-6*self.lam*x[t]/self.G(x[t])**2)+ 0.5/2*np.pi(-6*lam*x[t]))
            x_star = [x[t] + self.eps*self.G(x[t])*p_star]
            # Compute (x*, - p*) using L leapfrog steps of size eps 
            # Use convergence method to solve implicit equations for p_star
            for l in range(1, self.L):
                for i in range(1, 100):
                    p_star_guess  = p_star 
                    p_star_new = p_star - 0.5*self.eps*(self.k*x_star + self.lam*x_star**3 + 0.5*p_star_guess**2(-6*self.lam*x_star/self.G(x_star)**2)+ 0.5/2*np.pi(-6*lam*x_star))
                    if p_star_new - p_star_guess < 1e-6:
                        p_star_new = p_star_guess
            x_star = x_star + self.eps*self.G(x_star)*p_star_new
            # Compute the final step of the leapfrog method and append to the list
            p_star.append(1/(-3*(self.eps/self.G(x_star[self.L-1])**2)*self.lam*x_star[self.L-1])*(-1 - (1 + 6*self.lam*x_star[self.L-1]*self.eps/self.G(x_star[self.L-1])**2*(abs(-p_star[self.L-1] +0.5*self.eps*self.k*x_star[self.L-1] + 0.5*self.eps*self.lam*x_star[self.L-1]**3 + 0.25*self.eps*(1/self.G(x_star[self.L-1]))*(-6)*self.lam*x_star[self.L-1]))**0.5)))
            # Compute the acceptance ratio 
            r = np.exp(-self.H(x_star[self.L-1], p_star[self.L]) + self.H(x_star[0], p_star[0]))
            # Draw W from a Uniform distribution
            W = np.random.uniform(0, 1)
            # Carry out the Metropolis test
            if W <= min(1, r):
                x.append(x_star[self.L-1])
            else:
                x.append(x[t])
        return x

# Find the expected value of x
def exp_val(x):
    '''
    Given a list of x values, compute the expected value
    '''
    return np.mean(x)

# Testing the code
RMHMC_test = RMHMC(L=10, eps=0.01, k=1, lam=1)
print(RMHMC_test.RMHMC_alg(10))