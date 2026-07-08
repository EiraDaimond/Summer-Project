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
    def __init__(self, L=None, eps=None, k=None, lam=None, tol=None):
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
        '''
        We define the metric tensor and include a condition that prevents it from being 0
        '''
        G = self.k + 3*self.lam*x**2
        if G < 1e-14:
          G = 1e-5
        return G

    # Define the kinetic energy term (include correction term)
    def K(self, p, x):
        return 0.5*p*self.G(x)*p + 0.5*np.log(2*np.pi*np.abs(self.G(x)))

    # Define the Hamiltonian
    def H(self, x, p):
        return self.an_V(x) + self.K(p, x) 

    # Run the RMHMC algorithm
    def RMHMC_alg(self, n, tol=None):
        '''
        Carry out the RMHMC algorithm to generate x values. 
        We will use the Secant method for the fixed iteration in the Generalised Leapfrog Method. 
        '''
        # Initialise the x and KE values
        x = [0]
        KE_vals =[]
        # Start the loop to generate x values
        for t in range(10):
            print("ON MAIN ITERATION",t)
            # Draw the momentum from a Normal distribution
            p = np.random.normal(0, (np.abs(self.G(x[t])))**0.5)
            # Initialise the x_star and p_star lists
            x_stars = [x[t]]*(self.L+1)
            p_stars = [p]*(self.L+1)
            # Initialise the p_conv lists
            p_convs = [p,p+1]
            print(p_convs)
            # Use Secant method for fixed point iteration
            # Leapfrog first step
            for i in range(2, 10):
              print("ON FIRST STEP ITERATION", i)
              # We first force the square root to be real
              inner_val_1 = 1-4*self.eps*(self.G(x_stars[1]))\
                                        *(p_convs[i-1] + 0.5*self.eps*self.k*x_stars[1] \
                                          +0.5*self.eps*self.lam*x_stars[1]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[1])/(self.G(x_stars[1])))
              p_conv_root_1 = np.sqrt(max(0.0, inner_val_1))
              inner_val_2 = 1-4*self.eps*(self.G(x_stars[2]))\
                                        *(p_convs[i-2] + 0.5*self.eps*self.k*x_stars[2] \
                                          +0.5*self.eps*self.lam*x_stars[2]**3 \
                                            + 0.25*self.eps*abs(6*self.lam*x_stars[2])/(self.G(x_stars[2])))
              p_conv_root_2 = np.sqrt(max(0.0, inner_val_2))
              # Ensure denominator is non-zero
              denominator = (2/(self.eps*(self.G(x_stars[1])))\
                                *(-1 + p_conv_root_1))\
                                            -(2/(self.eps*(self.G(x_stars[0])))\
                                *(-1 + p_conv_root_2)) 
              if abs(denominator) < 1e-12:
                print("WARNING: Denominator 0")
                break
              p_conv = p_convs[i-1] - 2/(self.eps*(self.G(x_stars[1])))\
                                *(-1+p_conv_root_1)\
                                            *(p_convs[i-1] - p_convs[i-2])/denominator
              print("Added term=", 2/(self.eps*(self.G(x_stars[1])))\
                                *(-1+p_conv_root_1)\
                                            *(p_convs[i-1] - p_convs[i-2])/denominator)
              p_convs.append(p_conv)
              if abs(p_convs[-1] - p_convs[-2]) < self.tol:
                  break 
              print("p_convs[",i," is now=", p_convs[i])
            p_stars[2] = p_convs[len(p_convs)-1] 
            x_stars[2] = x_stars[1] + self.eps*self.G(x_stars[1])*p_stars[2]
            # Start leapfrog loop
            for j in range(3, self.L+1):
            # Initialise new p_convs list
                p_convs_new = [p_stars[j-1], p_stars[j]]
                for i in range(2,10):
                    print("ON MIDDLE STEPS ITERATION", j,i)
                # Force the square root to be real 
                    inner_val_1 = 1-2*self.eps*(self.G(x_stars[j-1]))\
                                        *(p_convs_new[i-1] + self.eps*self.k*x_stars[j-1] \
                                           +self.eps*self.lam*x_stars[j-1]**3 \
                                             + 0.5*self.eps*abs(6*self.lam*x_stars[j-1])/(self.G(x_stars[j-1])))
                    p_conv_root_1 = np.sqrt(max(0.0,inner_val_1))
                    inner_val_2 = 1-2*self.eps*(self.G(x_stars[j-2]))\
                                         *(p_convs_new[i-2] + self.eps*self.k*x_stars[j-2] \
                                           +self.eps*self.lam*x_stars[j-2]**3 \
                                            + 0.5*self.eps*abs(6*self.lam*x_stars[j-2])/(self.G(x_stars[j-2])))
                    p_conv_root_2 = np.sqrt(max(0.0, inner_val_2))
                    print("x_stars[",j-1,"]=", x_stars[j-1])
                    print("G here =",self.G(x_stars[j-1]))
                    print("Denominator=", self.eps*(self.G(x_stars[j-1])))
                    # Ensure denominator is non-zero
                    denominator = (2/(self.eps*(self.G(x_stars[1])))\
                                *(-1 + p_conv_root_1))\
                                            -(2/(self.eps*(self.G(x_stars[0])))\
                                *(-1 + p_conv_root_2)) 
                    if abs(denominator) < 1e-12:
                        print("WARNING: Denominator 0")
                        break
                    p_conv_new = p_convs_new[i-1] - 1/(self.eps*(self.G(x_stars[j-1])))\
                                 *(-1 + p_conv_root_1)\
                                            *(p_convs_new[i-1] - p_convs_new[i-2])/denominator  
                    p_convs_new.append(p_conv_new) 
                    if abs(p_convs_new[-1] - p_convs_new[-2]) < self.tol:
                        break                             
            p_stars[j] = p_convs_new[len(p_convs_new)-1]
            x_stars[j] = x_stars[j-1] + self.eps*self.G(x_stars[j-1])*p_stars[j]
            print("p_stars =", p_stars, "x_stars=", x_stars)
            # Leapfrog final step
            # Initialise new p_convs list
            p_convs_fin = [p_stars[self.L-2], p_stars[self.L-1]]
            for i in range(1, 2):
                print("ON END STEP ITERATION", i)
                # Force the square root to be real
                inner_val_1 = 1-2*self.eps*(self.G(x_stars[self.L-1]))\
                                         *(p_convs_fin[i-1] + 0.5*self.eps*self.k*x_stars[self.L-1] \
                                           +0.5*self.eps*self.lam*x_stars[self.L-1]**3 \
                                             + 0.25*self.eps*abs(6*self.lam*x_stars[self.L-1])/(self.G(x_stars[self.L-1])))
                p_conv_root_1 = np.sqrt(max(0.0,inner_val_1))
                inner_val_2 = 1-2*self.eps*(self.G(x_stars[self.L-2]))\
                                         *(p_convs_fin[i-2] + 0.5*self.eps*self.k*x_stars[self.L-2] \
                                           +0.5*self.eps*self.lam*x_stars[self.L-2]**3 \
                                             + 0.25*self.eps*abs(6*self.lam*x_stars[self.L-2])/(self.G(x_stars[self.L-2])))
                p_conv_root_2 = np.sqrt(max(0.0,inner_val_2))
                p_conv_fin = p_convs_fin[i-1] - 2/(self.eps*(self.G(x_stars[self.L-1])))\
                                 *(-1 + p_conv_root_1)\
                                             *(p_convs_fin[i-1] - p_convs_fin[i-2])/(2/(self.eps*(self.G(x_stars[self.L-1])))\
                                 *(-1 + p_conv_root_1)\
                                             -(2/(self.eps*(self.G(x_stars[self.L-2])))\
                                 *(-1 + p_conv_root_2)))
                p_convs_fin.append(p_conv_fin)
                if abs(p_convs_fin[-1] - p_convs_fin[-2]) < self.tol:
                    break
            p_stars[self.L] = p_convs_fin[len(p_convs_fin)-1]
            print("p_stars =", p_stars, "x_stars=", x_stars)
            # Compute the acceptance ratio
            r = np.exp(-self.H(x_stars[self.L-1], p_stars[self.L]) + self.H(x[t], p))
            # Draw W from a Uniform distribution
            W = np.random.uniform(0, 1)            
            # Carry out the Metropolis test
            if W <= min(1, r):
                x.append(x_stars[self.L])
            else:
                x.append(x[t])
            # Compute KE vals
            print("KE here", self.K(p_stars[self.L], x[t+1]))
            KE_vals.append(self.K(p_stars[self.L], x[t+1]))
        return x, KE_vals

# Find the expected value of x
def exp_val(x):
    '''
    Given a list of x values, compute the expected value
    '''
    return np.mean(x)

# Testing the codes
RMHMC_test = RMHMC(L=10, eps=0.000001, k=1, lam=1, tol = 1e-6)
print(RMHMC_test.RMHMC_alg(10))

'''FAILURE: I think this fixed point iteration method cannot be 
applied here because the added term varies too darmatically; need another fixed point iteration method

- Maybe works with very small epsilon.'''