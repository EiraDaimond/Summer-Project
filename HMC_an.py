import numpy as np
import matplotlib.pyplot as plt
import math

# Define the variables to be used
m = 1
L = 10
eps = 0.1
k = 1
lam = 1

# Define the anharmonic potential
def an_V(x):
    return 0.5*k*x**2 + 0.25*lam*x**4

# Define the anharmonic Hamiltonian
def an_H(x, p,m):
    return an_V(x) + 0.5*p**2/m
  
# def test_normal_p(n,m):
#     '''
#     Before running the HMC algorithm, it is sensible to check that genrating p values from a normal distribution gives a correct kinetic energy distribution.
#     We generate n p samples taken from a normal distribution, compute corresponding kinetic energies, and plot them to find the distribution.  
#     '''
#     # Initialise the p_normals and KE lists
#     p_normals = []
#     KE_p = []
#     # Loop over n
#     for t in range(n+1):
#         # Generate the p values from the normal distribution
#         p = np.random.normal(0, m**0.5)
#         # Calculate the corresponding KE
#         KE = 0.5*p**2/m
#         # Append to the lists
#         p_normals.append(p)
#         KE_p.append(KE)
#     # Plot
#     plt.figure()
#     plt.hist(KE_p, bins = 20, edgecolor = 'black')
#     plt.show()
#     return p_normals, KE_p
# print(test_normal_p(10,1))
    
# Run the HMC algorithm
def an_HMC_alg(n, L, eps):
    '''
    -Carry out the HMC algorithm using the leafrog method to generate x values. 
    -Simultaenously compute and store KE, PE, exp(-delH).
    -Calculate acceptance ratio. 
    -Another way to check that the algorithm is working correctly is to check 
    reversibility with each trajectory, so we also include this test.
    '''
    # Initialise the x values, KE values, errors, and the accepted values lists
    x = [0]
    KE_vals = []
    errors = []
    exps_delH = []
    accepted = []
    # Start the loop to generate x values
    for t in range(n+1):
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0, m**0.5)
        # Compute the first leapfrog step
        p_star = p - 0.5*eps*(k*x[t] + lam*x[t]**3)
        x_star = x[t] + eps*p_star/m
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, L):
            p_star = p_star - eps*(k*x_star + lam*x_star**3)
            x_star = x_star + eps*p_star/m
        # Compute the final step of the leapfrog method
        p_star = p_star - 0.5*eps*(k*x_star + lam*x_star**3)
        # Compute the acceptance ratio
        r = np.exp(-an_H(x_star, p_star,m ) + an_H(x[t], p,m))
        # Draw W from a Uniform distribution
        W = np.random.uniform(0, 1)
        # Carry out the Metropolis test
        if W <= min(1, r):
            x.append(x_star)
            accepted.append(x_star)
        else:
            x.append(x[t])
        # Compute the KE terms for this trajectory and append to list
        KE = 0.5*p_star**2/m
        KE_vals.append(KE)
        # Calculate exp(-delH) terms
        exp_minus_del_H_ = np.exp(an_H(x_star,p_star,m) - an_H(x[t], p,m))
        exps_delH.append(exp_minus_del_H_)
        # Check reversibility
        p_star = p_star + 0.5*eps*(k*x[t] + lam*x[t]**3)
        x_star = x_star - eps*p_star/m
        for l in range(1, L):
            p_star = p_star + eps*(k*x_star + lam*x_star**3)
            x_star = x_star - eps*p_star/m
        p_backwards = p_star + 0.5*eps*(k*x_star + lam*x_star**3)
        error = (p_backwards - p)
        errors.append(error)
    # Compute acceptance ratio
    acc_rat = (len(accepted)/len(x))*100    
    return x, KE_vals, exps_delH, errors, acc_rat

# Find the expected value and standard deviation of x
def mean_and_sd(x,m ,n):
    '''
    Given a list of x values, compute the expected value
      and standardised standard deviation (rejecting burn-in).
    '''
    length = len(x)
    values_to_use = x[math.ceil(length/10):]
    stand_sd = m**0.5/(n-1)**0.5
    return np.mean(values_to_use), stand_sd

print("Expected x =", mean_and_sd(an_HMC_alg(100000,L,eps)[0],1,100000)[0],\
      "Standardised standard deviation of x=", mean_and_sd(an_HMC_alg(100000,L,eps)[0],1,100000)[1],\
       "Expected KE = ",mean_and_sd(an_HMC_alg(100000, L, eps)[1],1,100000)[0], \
       "Standardised standard deviation of KE = ", mean_and_sd(an_HMC_alg(100000, L, eps)[1],1,100000)[1],\
        "Expected PE =", mean_and_sd(an_HMC_alg(100000,L,eps)[2],1,100000)[0],\
        "Standardised standard deviation of PE = ", mean_and_sd(an_HMC_alg(100000, L, eps)[2],1,100000)[1],\
        "Expected exp(-delH)= " ,mean_and_sd(an_HMC_alg(100000,L,eps)[3],1,100000)[0],\
        "Standardised standard deviation of exp(-delH) = ", mean_and_sd(an_HMC_alg(100000,L,eps)[3],1,100000)[1],\
        "Expected error =", mean_and_sd(an_HMC_alg(100000, L, eps)[4],1,100000)[0],\
        "Standardised standard deviation of error=", mean_and_sd(an_HMC_alg(100000,L,eps)[4],1,100000)[1],\
        "Acceptance ratio =" ,an_HMC_alg(100000, L, eps)[5])


