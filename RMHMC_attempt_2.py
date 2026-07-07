import numpy as np
import matplotlib.pyplot as plt
import math

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
        return self.k + 3*self.lam*x**2

    # Define the kinetic energy term (include correction term)
    def K(self, p, x):
        return 0.5*p*self.G(x)*p + 0.5*np.log(2*np.pi*np.abs(self.G(x)))

    # Define the Hamiltonian
    def H(self, x, p):
        return self.an_V(x) + self.K(p, x) 

    # Run the RMHMC algorithm
    def RMHMC_alg(self, n):
        '''
        Carry out the RMHMC algorithm to generate x values. 
        We will use the Secant method for the fixed iteration in the Generalise Leapfrog Method. 
        '''
        # Initialise the x values
        x = [0]
        # Start the loop to generate x values
        for t in range(n+1):
            # Draw the momentum from a Normal distribution
            p = np.random.normal(0, (np.abs(self.G(x[t])))**0.5)
            # Initialise the x_star and p_star lists
            x_stars = [0]*(self.L+1)
            p_stars = [p,p+1]*math.ceil(0.5*(self.L+1))
            # Use Secant method for fixed point iteration
            # Leapfrog first step
            p_stars[2] = p_stars[1] - 2/(self.eps*(self.G(x_stars[1])))\
                                *(-1 + (1-4*self.eps*(self.G(x_stars[1]))\
                                        *(p_stars[1] + 0.5*self.eps*self.k*x_stars[1] \
                                          +0.5*self.eps*self.lam*x_stars[1]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[1])/(self.G(x_stars[1]))))**0.5)\
                                            *(p_stars[1] - p_stars[0])/(2/(self.eps*(self.G(x_stars[1])))\
                                *(-1 + (1-4*self.eps*(self.G(x_stars[1]))\
                                        *(p_stars[1] + 0.5*self.eps*self.k*x_stars[1] \
                                          +0.5*self.eps*self.lam*x_stars[1]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[1])/(self.G(x_stars[1]))))**0.5)\
                                            -(2/(self.eps*(self.G(x_stars[0])))\
                                *(-1 + (1-4*self.eps*(self.G(x_stars[0]))\
                                        *(p_stars[0] + 0.5*self.eps*self.k*x_stars[0] \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[0])/(self.G(x_stars[0]))))**0.5)))  
            print("p_stars[2] = ", p_stars[2])
            x_stars[2] = x_stars[1] + self.eps*self.G(x_stars[1])*p_stars[2]
            print("x_stars[2]= ", x_stars[2])
            # Start leapfrog loop
            for i in range(3, self.L):
                p_stars[i] = p_stars[i-1] - 1/(self.eps*(self.G(x_stars[i-1])))\
                                *(-1 + (1-2*self.eps*(self.G(x_stars[i-1]))\
                                        *(p_stars[i-1] + self.eps*self.k*x_stars[i-1] \
                                          +0.5*self.eps*self.lam*x_stars[i-1]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[i-1])/(self.G(x_stars[i-1]))))**0.5)\
                                            *(p_stars[i-1] - p_stars[i-2])/(2/(self.eps*(self.G(x_stars[i-1])))\
                                *(-1 + (1-2*self.eps*(self.G(x_stars[i-1]))\
                                        *(p_stars[i-1] + 0.5*self.eps*self.k*x_stars[i-1] \
                                          +0.5*self.eps*self.lam*x_stars[i-1]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[i-1])/(self.G(x_stars[i-1]))))**0.5)\
                                            -(2/(self.eps*(self.G(x_stars[i-2])))\
                                *(-1 + (1-2*self.eps*(self.G(x_stars[i-2]))\
                                        *(p_stars[i-2] + 0.5*self.eps*self.k*x_stars[i-2] \
                                          +0.5*self.eps*self.lam*x_stars[i-2]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[i-2])/(self.G(x_stars[i-2]))))**0.5)))    
                print("p_stars[",i,"] = ", p_stars[i])                            
                x_stars[i] = x_stars[i-1] + self.eps*self.G(x_stars[i-1])*p_stars[i]
                print("x_stars[",i,"] =", x_stars[i])
            # Leapfrog final step
            p_stars[self.L] = p_stars[self.L-1] - 2/(self.eps*(self.G(x_stars[self.L-1])))\
                                *(-1 + (1-2*self.eps*(self.G(x_stars[self.L-1]))\
                                        *(p_stars[self.L-1] + 0.5*self.eps*self.k*x_stars[self.L-1] \
                                          +0.5*self.eps*self.lam*x_stars[self.L-1]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[self.L-1])/(self.G(x_stars[self.L-1]))))**0.5)\
                                            *(p_stars[self.L-1] - p_stars[self.L-2])/(2/(self.eps*(self.G(x_stars[self.L-1])))\
                                *(-1 + (1-2*self.eps*(self.G(x_stars[self.L-1]))\
                                        *(p_stars[self.L-1] + 0.5*self.eps*self.k*x_stars[self.L-1] \
                                          +0.5*self.eps*self.lam*x_stars[self.L-1]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[self.L-1])/(self.G(x_stars[self.L-1]))))**0.5)\
                                            -(2/(self.eps*(self.G(x_stars[self.L-2])))\
                                *(-1 + (1-2*self.eps*(self.G(x_stars[self.L-2]))\
                                        *(p_stars[self.L-2] + 0.5*self.eps*self.k*x_stars[self.L-2] \
                                          +0.5*self.eps*self.lam*x_stars[self.L-2]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[self.L-2])/(self.G(x_stars[self.L-2]))))**0.5)))
            # Compute the acceptance ratio
            r = np.exp(-self.H(x_stars[self.L], p_stars[self.L]) + self.H(x[t], p))
            # Draw W from a Uniform distribution
            W = np.random.uniform(0, 1)            
            # Carry out the Metropolis test
            if W <= min(1, r):
                x.append(x_stars[self.L])
            else:
                x.append(x[t])
        return x

# Find the expected value of x
#def exp_val(x):
    '''
    Given a list of x values, compute the expected value
    '''
    #return np.mean(x)

# Testing the code
RMHMC_test = RMHMC(L=10, eps=0.1, k=1, lam=1)
print(RMHMC_test.RMHMC_alg(10))