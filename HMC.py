import numpy as np

# Define the variables to be used
m = 1 
L = 100
eps = 0.1

# Define the Hamiltonian function
def H(x,p):
    '''
    Given x and p, compute the Hamiltonian
    '''
    return 0.5*x**2 + 0.5*x**2

def HMC(n,L,eps):
    '''
    Carry out the HMC algorithm using the leafrog method to generate x values
    '''
    # Initialise the x values
    x = [0]
    # Start the loop to generate the x values
    for t in range(n+1):
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0,m)
        # Carry out step 1 of the leapfrog method
        p_star = p - 0.5*eps*x[t]
        x_star = x[t] + eps*p_star/m
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, L):
            p_star = p_star - eps*x_star
            x_star = x_star + eps*p_star/m
        # Carry out the final step of the leapfrog method
        p_star = p_star - 0.5*eps*x_star
        # Compute the acceptance ratio
        r = np.exp(-H(x_star, p_star) + H(x[t],p))
        # Draw W from a Uniform distribution
        W = np.random.uniform(0,1)
        # Carry out the Metropolis test
        if W <= min(1,r):
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

print(exp_val(HMC(1000,L,eps)))