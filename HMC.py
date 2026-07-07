import numpy as np
import math
import matplotlib.pyplot as plt

# Define the variables to be used
m = 1 
L = 100
eps = 0.1
k = 1

# Define the potental
def V(x,k):
    '''
    Given x, compute the potential
    '''
    return 0.5*k*x**2 

# Define the Hamiltonian function
def H(x,p):
    '''
    Given x,p, and V(x),compute the Hamiltonian
    '''
    return V(x,k) + 0.5*p**2/m
 
def test_normal_p(n,m):
    '''
    Before running the HMC algorithm, it is sensible to check that genrating p values from a normal distribution gives a correct kinetic energy distribution.
    We generate n p samples taken from a normal distribution, compute corresponding kinetic energies, and plot them to find the distribution.  
    '''
    # Initialise the p_normals and KE lists
    p_normals = []
    KE_p = []
    # Loop over n
    for t in range(n+1):
        # Generate the p values from the normal distribution
        p = np.random.normal(0, m**0.5)
        # Calculate the corresponding KE
        KE = 0.5*p**2/m
        # Append to the lists
        p_normals.append(p)
        KE_p.append(KE)
    # Plot
    plt.figure()
    plt.hist(KE_p, bins = 20, edgecolor = 'black')
    plt.show()
    return p_normals, KE_p
print(test_normal_p(10,m))

def HMC(n,L,eps):
    '''
    -Carry out the HMC algorithm using the leafrog method to generate x values. 
    -Simultaenously compute and store KE, PE, exp(-delH).
    -Calculate acceptance ratio. 
    -Another way to check that the algorithm is working correctly is to check 
    reversibility with each trajectory, so we also include this test.
    '''
    # Initialise the x values, KE values, PE values, errors, and the accepted values lists
    x = [0]
    KE_vals = []
    PE_vals =[]
    errors = []
    exps_delH = []
    accepted = []
    # Start the loop to generate the x values
    for t in range(n+1):
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0,m**0.5)
        # Carry out step 1 of the leapfrog method
        p_star = p - 0.5*eps*k*x[t]
        x_star = x[t] + eps*p_star/m
        # Compute (x*,p*) using L leapfrog steps of size eps
        for l in range(1, L):
            p_star = p_star - eps*k*x_star
            x_star = x_star + eps*p_star/m
        # Carry out the final step of the leapfrog method
        p_star = p_star - 0.5*eps*k*x_star
        # Compute the acceptance ratio
        r = np.exp(-H(x_star, p_star) + H(x[t],p))
        # Draw W from a Uniform distribution
        W = np.random.uniform(0,1)
        # Carry out the Metropolis test
        if W <= min(1,r):
            x.append(x_star)
            accepted.append(x_star)
        else:
            x.append(x[t])
        # Compute the KE and PE terms for this trajectory and append to list
        KE = 0.5*p_star**2/m
        PE = 0.5*k*x_star**2 
        KE_vals.append(KE)
        PE_vals.append(PE)
        # Calculate exp(-delH) terms
        exp_minus_del_H_ = np.exp(H(x_star,p_star) - H(x[t], p))
        exps_delH.append(exp_minus_del_H_)
        # Check reversibility
        p_star = p_star + 0.5*eps*k*x[t]
        x_star = x_star - eps*p_star/m
        for l in range(1, L):
            p_star = p_star + eps*k*x_star
            x_star = x_star - eps*p_star/m
        p_backwards = p_star + 0.5*eps*k*x_star
        error = (p_backwards - p)
        errors.append(error)
    # Compute acceptance ratio
    acc_rat = (len(accepted)/len(x))*100
    return x, KE_vals, PE_vals, exps_delH, errors, acc_rat

# Find the expected value of x
def exp_val(x):
    '''
    Given a list of values, compute the expected value (with burn-in removed)
    '''
    values_to_use = x[math.ceil(len(x)/10):]
    return np.mean(values_to_use)   

print("Expected x =", exp_val(HMC(100000,L,eps)[0]),\
       "Expected KE = ",exp_val(HMC(100000, L, eps)[1]), \
        "Expected PE =", exp_val(HMC(100000,L,eps)[2]),\
        "Expected exp(-delH)= " ,exp_val(HMC(100000,L,eps)[3]),\
        "Expected error =", exp_val(HMC(100000, L, eps)[4]),\
        "Acceptance ratio =" ,HMC(100000, L, eps)[5])
