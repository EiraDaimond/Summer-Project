import numpy as np
import matplotlib.pyplot as plt

class RMHMC:
    # Define the variables to be used
    def __init__(self, L=None, eps=None, k=None, lam=None):
        self.L = L
        self.eps = eps
        self.k = k
        self.lam = lam
        
    # Define the metric tensor to be used
    def G(self, x):
        return np.array(0.5*self.k + 0.5*self.lam*x**2)

    # Define the kinetic energy function (with correction)
    def K_eff(self, p, G):
        return 0.5*p@np.linalg.inv(G)@p + 0.5*np.log(2*np.pi*np.linalg.det(G))

    # Define the anharmonic potential
    def an_V(self, x):
        return 0.5*self.k*x**2 + 0.25*self.lam*x**4

    # Define the Riemmanian Hamiltonian
    def R_H(self, x, p, G):
        return self.an_V(x) + self.K_eff(p, G)

    # Run the RMHMC algorithm
    def RMHMC_alg(self, n):
        # Initialise the x values
        x = [0]
        # Start the loop to generate x values
        for t in range(n+1):
            # Draw the momentum from a Normal distribution
            p = np.random.normal(0, self.G(x[t]))
            # Compute the first leapfrog step
            p_star = p - 0.5*self.eps*(self.k*x[t] + self.lam*x[t]**3 + self.lam )
            x_star = x[t] + self.eps*p_star/self.G
            # Compute (x*, - p*) using L leapfrog steps of size eps
            for l in range(1, self.L):
                p_star = p_star - self.eps*x_star
                x_star = x_star + self.eps*p_star/self.m
            # Compute the final step of the leapfrog method
            p_star = p_star - 0.5*self.eps*x_star
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

# Investigating effect of changing k on the expected value of x
# Initialise the lists
k = []
exp_x = []
# Start the loop over different k values
for i in range(1, 101):   
    # Create an instance of the an_HMC class
    an_HMC_instance = an_HMC(m=1, L=1000, eps=0.1, k=i, lam=1)
    # Run the an_HMC algorithm
    an_HMC_result = an_HMC_instance.an_HMC_alg(1000)
    # Append the lists appropriately
    k.append(i)
    exp_x.append(exp_val(an_HMC_result))
# Plot the results
plt.figure()
plt.plot(k, exp_x)
plt.xlabel('k')
plt.ylabel('<x>')
plt.title('Expected value of x vs k')
plt.grid(True)
plt.savefig('exp_x_vs_k.png')

# Investigating effect of changing lam on the expected value of x
# Initialise the lists
lam = []
exp_x = []
# Start the loop over different lam values
for i in range(1, 101):   
    # Create an instance of the an_HMC class
    an_HMC_instance = an_HMC(m=1, L=1000, eps=0.1, k=1, lam=i)
    # Run the an_HMC algorithm
    an_HMC_result = an_HMC_instance.an_HMC_alg(1000)
    # Append the lists appropriately
    lam.append(i)
    exp_x.append(exp_val(an_HMC_result))
# Plot the results
plt.figure()
plt.plot(lam, exp_x)
plt.xlabel('lam')
plt.ylabel('<x>')
plt.title('Expected value of x vs lam')
plt.grid(True)
plt.savefig('exp_x_vs_lam.png')