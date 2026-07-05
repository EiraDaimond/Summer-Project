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

    # Define the metric tensor (negative second derivative of the potential term)
    def G(self, x):
        return -self.k - 3*self.lam*x**2

    # Define the kinetic energy term (include correction term)
    def K(self, p, x):
        return 0.5*p*self.G(x)*p + 0.5*np.log(2*np.pi*np.abs(self.G(x)))

    # Define the Hamiltonian
    def H(self, x, p):
        return self.an_V(x) + self.K(p, x) 

    # Run the HMC algorithm
    def RMHMC_alg(self, n):
        # Initialise the x values
        x = [1e-6]
        # Start the loop to generate x values
        for t in range(n+1):
            # Initialise the x_star and p_star lists
            x_star = [0]*(self.L+1)
            p_star = [0]*(self.L+1)
            # Draw the momentum from a Normal distribution
            p = np.random.normal(0, np.abs(self.G(x[t])))
            print("Random p:",p)
            # Compute the first leapfrog step
            p_star[0] = (2/self.eps*self.G(x[t]))*(-1 + (1- self.eps*self.G(x[t])*(p + 0.5*self.eps*self.k*x[t] + 0.5*self.eps*self.lam*(x[t])**3 + 0.25*self.eps*abs(6*self.lam*x[t])/(self.G(x[t]))**2)**0.5)) # Note: Chose + term
            print("p_star[0]:", p_star[0])
            x_star[0] = x[t] + self.eps*self.G(x[t])*p_star[0]
            print(x_star[0])
            # Compute (x*, - p*) using L leapfrog steps of size eps
            for l in range(1, self.L):
                print("p_star[l-1]:", p_star[l-1])
                p_star[l] = (1/self.eps*self.G(x_star[l-1]))*(-1 + (1- 2*self.eps*self.G(x_star[l-1])*(p + self.eps*self.k*x_star[l-1] + self.eps*self.lam*(x_star[l-1])**3 + 0.5*self.eps*abs(6*self.lam*x_star[l-1])/(self.G(x_star[l-1]))**2)**0.5))
                print("p_star[0]:", p_star[0])
                x_star[l] = x_star[l-1] + self.eps*self.G(x_star[l-1])*p_star[l]
            # Compute the final step of the leapfrog method
            p_star[self.L] = (2/self.eps*self.G(x_star[self.L-1]))*(-1 + (1- self.eps*self.G(x_star[self.L-1])*(p + 0.5*self.eps*self.k*x_star[self.L-1] + 0.5*self.eps*self.lam*(x_star[self.L-1])**3 + 0.25*self.eps*abs(*6*self.lam*x_star[self.L-1])/(self.G(x_star[self.L-1]))**2)**0.5))
            print("p_star[0]:", p_star[0])
            # Compute the acceptance ratio
            r = np.exp(-self.H(x_star[self.L-1], p_star[self.L]) + self.H(x[t], p))
            # Draw W from a Uniform distribution
            W = np.random.uniform(0, 1)            
            # Carry out the Metropolis test
            if W <= min(1, r):
                x.append(x_star[self.L])
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
RMHMC_test = RMHMC(L=10, eps=0.1, k=1, lam=1)
print(RMHMC_test.RMHMC_alg(10))