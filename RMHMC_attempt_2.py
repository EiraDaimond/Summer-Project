import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

class RMHMC:
    '''
    Rewriting the anharmonic HMC class but for RMHMC in one dimension. 
    Mass is non-constant and is instead represented by the metric tensor which I 
    shall denote G. In the 1D case, this is just a scalar. 
    '''
    # Define the variables to be used
    def __init__(self, L=None, eps=None, k=None, lam=None, tol=1e-6):
        self.L = L
        self.eps = eps
        self.k = k
        self.lam = lam
        self.tol = tol

    # Define the anharmonic potential term
    def an_V(self, x):
        return 0.5*self.k*x**2 + 0.25*self.lam*x**4

    # Define the metric tensor (negative second derivative of the potential term)
    def G(self, x):
        return self.k + 3*self.lam*x**2

    # Define the kinetic energy term (include correction term)
    def K(self, p, x):
        return 0.5*p*self.G(x)*p + 0.5*np.log(2*np.pi*np.abs(self.G(x)))

    # Define the Hamiltonian
    def H(self, x, p):
        return self.an_V(x) + self.K(p, x) 

    # Run the RMHMC algorithm
    def RMHMC_alg(self, n, tol):
        '''
        Carry out the RMHMC algorithm to generate x values. 
        We will use the Generalised Leapfrog Method with the fixed point iteration.
        '''
        # Initialise the x values
        x = [0]
        # Start the loop to generate x values
        for t in range(n+1):
            # Initialise the x_star and p_star lists
            x_stars = []
            p_stars = []
            # Draw the momentum from a Normal distribution
            p = np.random.normal(0, (np.abs(self.G(x[t])))**0.5)
            # Provide an initial guess value for p, initialise p_star
            p_guess = p 
            p_star = 0
            # Start the fixed point iteration for the first leapfrog step
            while True:
                p_star = p_guess - 0.5*self.eps*\
                    (self.k*x[t] + self.lam*x[t]**3 \
                     + 0.5*p_guess**2*(-6*self.lam*x[t]) \
                     + 0.5*abs(-6*self.lam*x[t])/abs(-self.k-3*self.lam*x[t]**2))
                if abs(p_star - p_guess) < tol: 
                    break
                p_guess = p_star  
            print(p_star)  
            p_stars.append(p_star)
            x_star = x[t] + self.eps*self.G(x[t])*p_star
            x_stars.append(x_star)
            # Compute (x*, - p*) using L leapfrog steps of size eps
            for l in range(1, self.L):
                p_current = p_star
                p_guess = p_star
                while True:
                    p_star = p_current - self.eps\
                                            *(self.k*x_star + self.lam*x_star**3\
                                                + 0.5*p_guess**2*(-6*self.lam*x_star)\
                                                + 0.5*abs(-6*self.lam*x_star)/abs(-self.k-3*self.lam*x_star**2))
                    print("In while loop, p_star tracking:",p_star, p_star - p_guess)
                    if abs(p_star - p_guess) < tol:
                        break
                    p_guess = p_star
                p_stars.append(p_star)
                x_star = x_star + self.eps*self.G(x_star)*p_star
                x_stars.append(x_star)
            # Compute the final step of the leapfrog method
            p_current = p_star
            p_guess = p_star
            while True:
                p_star = p_guess - 0.5*self.eps\
                                        *(self.k*x[t] + self.lam*x[t]**3 + 0.5*p_guess**2*(-6*self.lam*x[t])\
                                           + 0.5*abs(-6*self.lam*x[t])/abs(-self.k-3*self.lam*x[t]**2))
                if abs(p_star - p_guess) < tol:
                    break
                p_guess = p_star
            # Compute the acceptance ratio
            r = np.exp(-self.H(x_star, p_star) + self.H(x[t], p))
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

# Testing the code
RMHMC_test = RMHMC(L=10, eps=0.005, k=1, lam=1, tol = 1e-6)
print(RMHMC_test.RMHMC_alg(10,tol=1e-6))