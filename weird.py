import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def RMHMC(L=None,eps=None,k=None,lam=None,tol=None,n=None, d=None):
    '''
    Rewriting the anharmonic HMC class but for RMHMC in one dimension. 
    Mass is non-constant and is instead represented by the metric tensor which I 
    shall denote G. In the 1D case, this is just a scalar. 
    ''' 

    # Define the anharmonic potential term
    def an_V(x):
        return 0.5*k*x**2 + 0.25*lam*x**4

    # Define the metric tensor (second derivative of the potential term)
    def G(x):
        return k + 3*lam*x**2
    
    # Define M (including delta).... to be used to avoid division by 0 errors
    def M(x, d):
        return np.sqrt(abs(G(x)**2+d**2))

    # Define the kinetic energy term (include correction term)
    def K(p, x,d):
        return 0.5*p*M(x,d)*p + 0.5*np.log(np.abs(M(x,d)))

    # Define the Hamiltonian
    def H(x, p):
        return an_V(x) + K(p, x,d) 

    # Run the RMHMC algorithm
    '''
    Carry out the RMHMC algorithm to generate x values. 
    We will use the Generalised Leapfrog Method with the fixed point iteration.
    '''
    # Initialise the x values
    x = [0.1] # COMMENT: Wanted to use 0 but that doesn't work for the fixed point iteration
    # Start the loop to generate x values
    for t in range(n+1):
        # Initialise the x_star and p_star lists
        x_stars = []
        p_stars = []
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0, (np.abs(G(x[t])))**0.5)
        # Provide an initial guess value for p, initialise p_star
        p_guess = p 
        p_star = 0
        # Start the fixed point iteration for the first leapfrog step
        # p convergence
        count = 1
        while True:
            print("Count =",count)
            count = count +1 
            p_star = p - 0.5*eps*\
                 (k*x[t] + lam*x[t]**3 \
                     + 0.5*p_guess**2*(-6*lam*x[t]) \
                + 0.5*abs(-6*lam*x[t])\
                     /M(x[t],d))
            if p_star > 1e14:
                print("BROKE p_star too big")
                break
            else:
                if p_star < -1e14:
                    print("BROKE p_star too big -ve")
                    break
                else:
                    print("1st step p_star is:",p_star)
                    print("1st step p_guess is:",p_guess)
                    print("Difference in ps:", abs(p_guess - p_star))
                    if abs(p_star - p_guess) < tol: 
                        print("STOPPING loop for p")
                        break
                    else:
                        p_guess = p_star  
            print()
        print("Moving on from 1st step with p_star", p_star)  
        p_stars.append(p_star)
        # x convergence
        x_guess = x[t]
        x_star = 0
        count = 1
        while True:
            print("Count =",count)
            count = count + 1
            print("1st step x_star is :", x_star)
            x_star = x[t] + 0.5*eps\
                            *(p_star*M(x[t],d)+p_star*M(x_guess,d))
            if x_star > 1e14:
                print("BROKE x_star too big")
                break
            else:
                if x_star < -1e14:
                    print("BROKE x_star too big -ve")
                    break
                else:
                    if abs(x_star - x_guess) < tol:
                        print("STOPPING while loop for x_star")
                        break
                    else:
                        x_guess = x_star
            print()
        print("Moving on from 1st step with x_star", x_star)
        x_stars.append(x_star)
        print("CODE WORKS UP TO HERE")
        print()
        print("STARTING MIDDLE STEPS")
        print()
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, n+1):
            p_current = p_star
            p_guess = p_star
            p_star = 0
            count = 1
            print("On iter", l, "with p_star =", p_star, "p_guess =", p_guess)
            # PROBLEM IS HERE WHERE P VALUES AREN'T CONVERGING
            while True:
                print("Count=",count)
                count = count +1
                print("Middle step iter[",l,"] p_star is :", p_star)
                print("Using x_star:", x_star)
                p_star = p_current - eps\
                                        *(k*x_star + lam*x_star**3\
                                             + 0.5*p_guess**2*(-6*lam*x_star)\
                                             + 0.5*abs(-6*lam*x_star)/M(x_star,d))
                print("Calculated p_star =", p_star)
                print("p_guess is", p_guess)
                print("Difference in ps", abs(p_star - p_guess))
                if p_star > 1e14:
                    print("BROKE p_star too big")
                    break
                else:
                    if p_star < -1e14:
                        print("BROKE p_star too big -ve")
                        break
                    else:
                        if abs(p_star - p_guess) < tol:
                            print("STOPPING WHILE LOOP for p")
                            break 
                        else:
                            p_guess = p_star
                print()
            print("Moving on from middle step iter [",l,"] with p_star", p_star)
            p_stars.append(p_star)
            print()
            print("STARTING x convergence")
            print()
            # x convergence
            x_current = x_star
            x_guess = x_star
            x_star = 0
            count = 1
            while True:
                print("Count=",count)
                count = count+1
                print("Middle step iter[",l,"] x_star is :", x_star)
                print("Using p_star", p_star)
                x_star = x_current + 0.5*eps\
                            *(p_star*M(x_current,d)+p_star*M(x_guess,d))
                if x_star > 1e14:
                    print("BROKE x_star too big")
                    break
                else:
                    if x_star < -1e14:
                        print("BROKE x_star too big -ve")
                        break
                    else:
                        if abs(x_star - x_guess) < tol:
                            print("STOPPING while loop for x_star")
                            break
                        else:
                            x_guess = x_star
                print()
            print("Moving on from middle step iter[",l,"] with x_star", x_star)
            x_stars.append(x_star)
        print()
        print("STARTING FINAL STEPS")
        print()
        # Compute the final step of the leapfrog method
        p_current = p_star
        p_guess = p_star
        count = 1
        while True:
            print("Count=",count)
            count = count+1
            p_star = p_guess - 0.5*eps\
                                    *(k*x[t] + lam*x[t]**3 + 0.5*p_guess**2*(-6*lam*x[t])\
                                        + 0.5*abs(-6*lam*x[t])/M(x[t],d))
            print("p_star is :", p_star)
            print("p_guess is", p_guess)
            print("Difference in ps:", abs(p_star - p_guess))
            if p_star > 1e14:
                print("BROKE p_star too big")
                break
            else:
                if p_star < -1e14:
                    print("BROKE p_star too big -ve")
                    break
                else:
                    if abs(p_star - p_guess) < tol:
                        break
                    else:
                        p_guess = p_star
            print()
        # Compute the acceptance ratio
        r = np.exp(-H(x_star, p_star) + H(x[t], p))
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

print(RMHMC(1e8,1e-8,1,1,1e-6,100000,1e-6))
