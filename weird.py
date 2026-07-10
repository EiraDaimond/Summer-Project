import numpy as np
import matplotlib.pyplot as plt
import math

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
    # Initialise the x, KE, PE, exps_delH, errors, accepted values list
    x = [0.1] # COMMENT: Wanted to use 0 but that doesn't work for the fixed point iteration
    KE_vals =[]
    PE_vals= []
    exps_delH = []
    errors = []
    accepted = []
    # Start the loop to generate x values
    for t in range(n+1):
        print("On iteration:", t)
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
            #("Count =",count)
            count = count +1 
            p_star = p - 0.5*eps*\
                 (k*x[t] + lam*x[t]**3 \
                     + 0.5*p_guess**2*(-6*lam*x[t]) \
                + 0.5*abs(-6*lam*x[t])\
                     /M(x[t],d))
            if p_star > 1e14:
                #("BROKE p_star too big")
                break
            else:
                if p_star < -1e14:
                    #("BROKE p_star too big -ve")
                    break
                else:
                    #("1st step p_star is:",p_star)
                    #("1st step p_guess is:",p_guess)
                    #("Difference in ps:", abs(p_guess - p_star))
                    if abs(p_star - p_guess) < tol: 
                        #("STOPPING loop for p")
                        break
                    else:
                        p_guess = p_star  
            #()
        #("Moving on from 1st step with p_star", p_star)  
        p_stars.append(p_star)
        # x convergence
        x_guess = x[t]
        x_star = 0
        count = 1
        while True:
            #("Count =",count)
            count = count + 1
            #("1st step x_star is :", x_star)
            x_star = x[t] + 0.5*eps\
                            *(p_star*M(x[t],d)+p_star*M(x_guess,d))
            if x_star > 1e14:
                #("BROKE x_star too big")
                break
            else:
                if x_star < -1e14:
                    #("BROKE x_star too big -ve")
                    break
                else:
                    if abs(x_star - x_guess) < tol:
                        #("STOPPING while loop for x_star")
                        break
                    else:
                        x_guess = x_star
            #()
        #("Moving on from 1st step with x_star", x_star)
        x_stars.append(x_star)
        #("CODE WORKS UP TO HERE")
        #()
        #("STARTING MIDDLE STEPS")
        #()
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, n+1):
            p_current = p_star
            p_guess = p_star
            p_star = 0
            count = 1
            #("On iter", l, "with p_star =", p_star, "p_guess =", p_guess)
            # PROBLEM IS HERE WHERE P VALUES AREN'T CONVERGING
            while True:
                #("Count=",count)
                count = count +1
                #("Middle step iter[",l,"] p_star is :", p_star)
                #("Using x_star:", x_star)
                p_star = p_current - eps\
                                        *(k*x_star + lam*x_star**3\
                                             + 0.5*p_guess**2*(-6*lam*x_star)\
                                             + 0.5*abs(-6*lam*x_star)/M(x_star,d))
                #("Calculated p_star =", p_star)
                #("p_guess is", p_guess)
                #("Difference in ps", abs(p_star - p_guess))
                if p_star > 1e14:
                    #("BROKE p_star too big")
                    break
                else:
                    if p_star < -1e14:
                        #("BROKE p_star too big -ve")
                        break
                    else:
                        if abs(p_star - p_guess) < tol:
                            #("STOPPING WHILE LOOP for p")
                            break 
                        else:
                            p_guess = p_star
                #()
            #("Moving on from middle step iter [",l,"] with p_star", p_star)
            p_stars.append(p_star)
            #()
            #("STARTING x convergence")
            #()
            # x convergence
            x_current = x_star
            x_guess = x_star
            x_star = 0
            count = 1
            while True:
                #("Count=",count)
                count = count+1
                #("Middle step iter[",l,"] x_star is :", x_star)
                #("Using p_star", p_star)
                x_star = x_current + 0.5*eps\
                            *(p_star*M(x_current,d)+p_star*M(x_guess,d))
                if x_star > 1e14:
                    #("BROKE x_star too big")
                    break
                else:
                    if x_star < -1e14:
                        #("BROKE x_star too big -ve")
                        break
                    else:
                        if abs(x_star - x_guess) < tol:
                            #("STOPPING while loop for x_star")
                            break
                        else:
                            x_guess = x_star
                #()
            #("Moving on from middle step iter[",l,"] with x_star", x_star)
            x_stars.append(x_star)
        #()
        #("STARTING FINAL STEPS")
        #()
        # Compute the final step of the leapfrog method
        p_current = p_star
        p_guess = p_star
        count = 1
        while True:
            #("Count=",count)
            count = count+1
            p_star = p_guess - 0.5*eps\
                                    *(k*x_star + lam*x_star**3 + 0.5*p_guess**2*(-6*lam*x_star)\
                                        + 0.5*abs(-6*lam*x_star)/M(x_star,d))
            #("p_star is :", p_star)
            #("p_guess is", p_guess)
            #("Difference in ps:", abs(p_star - p_guess))
            if p_star > 1e14:
                #("BROKE p_star too big")
                break
            else:
                if p_star < -1e14:
                    #("BROKE p_star too big -ve")
                    break
                else:
                    if abs(p_star - p_guess) < tol:
                        break
                    else:
                        p_guess = p_star
            #()
        # Compute the acceptance ratio
        r = np.exp(-H(x_star, p_star) + H(x[t], p))
        # Draw W from a Uniform distribution
        W = np.random.uniform(0, 1)            
        # Carry out the Metropolis test
        if W <= min(1, r):
            x.append(x_star)
            accepted.append(x_star)
        else:
            x.append(x[t])
        # Compute the KE and append to list
        KE = K(x[-1],p_star,d)
        KE_vals.append(KE)
        # Compute the PE and append to list
        PE = an_V(x[-1])
        PE_vals.append(PE)
        # Compute the exp(-delH)
        exp_minus_del_H_ = np.exp(H(x[-1],p_star) - H(x[t], p))
        exps_delH.append(exp_minus_del_H_)
        # Check reversibility
        p_star = p_star + 0.5*eps\
                            *(k*x_star + lam*x_star**3 + 0.5*p_guess**2*(-6*lam*x_star)\
                                + 0.5*abs(-6*lam*x_star)/M(x_star,d))
        x_star = x_star - 0.5*eps\
                            *(p_star*M(x_current,d)+p_star*M(x_guess,d))
        for l in range(1, L):
            p_star = p_star + eps\
                                *(k*x_star + lam*x_star**3\
                                    + 0.5*p_guess**2*(-6*lam*x_star)\
                                    + 0.5*abs(-6*lam*x_star)/M(x_star,d))
            x_star = x_star - 0.5*eps\
                            *(p_star*M(x[t],d)+p_star*M(x_guess,d))
        p_backwards = p_star + 0.5*eps\
                            *(k*x_star + lam*x_star**3 + 0.5*p_guess**2*(-6*lam*x_star)\
                                + 0.5*abs(-6*lam*x_star)/M(x_star,d))
        error = (p_backwards - p)
        errors.append(error)
    # Compute acceptance ratio
    acc_rat = (len(accepted)/len(x))*100
    return x, KE_vals, PE_vals, exps_delH, errors, acc_rat
    
# Find the expected value of x and corresponding standardised standard deviation
def mean_and_sd(x, L, eps, k, lam, tol, n, d):
    '''
    Given a list of values, compute the expected value (with burn-in removed), 
    and corresponding standardised standar deviation.
    '''
    values_to_use = x[math.ceil(len(x)/10):]
    # Initialise the sd_list
    sd_list = [0]*(len(values_to_use))
    for i in range(len(values_to_use)+1):
        sd_list[i] = RMHMC(L,eps,k,lam,tol,n,d).M(values_to_use[i], d)
    stand_sd = np.sqrt((np.mean(sd_list))/(n-1))
    return np.mean(values_to_use), stand_sd      

print(RMHMC(1e8,1e-8,1,1,1e-6,10000,1e-6))

print("Expected x =", mean_and_sd(RMHMC(1e8,1e-8,1,1,1e-6,10000,1e-6)[6][0] ,\
      "Standardised standard deviation of x=", RMHMC(1e8,1e-8,1,1,1e-6,10000,1e-6)[6][1],\
       "Expected KE = ",RMHMC(1e8,1e-8,1,1,1e-6,10000,1e-6)[], \
       "Standardised standard deviation of KE = ", mean_and_sd(HMC(100000, L, eps)[1])[1],\
        "Expected PE =", mean_and_sd(HMC(100000,L,eps)[2])[0],\
        "Standardised standard deviation of PE = ", mean_and_sd(HMC(100000, L, eps)[2][1]),\
        "Expected exp(-delH)= " ,mean_and_sd(HMC(100000,L,eps)[3])[0],\
        "Standardised standard deviation of exp(-delH) = ", mean_and_sd(HMC(100000,L,eps)[3])[1],\
        "Expected error =", mean_and_sd(HMC(100000, L, eps)[4])[0],\
        "Standardised standard deviation of error=", mean_and_sd(HMC(100000,L,eps)[4])[1],\
        "Acceptance ratio =" ,HMC(100000, L, eps)[5])
'''
COMMENTS:
- Want Leps = 1 (will later change), at mo this is resulting in many L iterations, code takes a long time to run.
- Tolerance I have chosen as 1e-6... I feel like this is still pretty high but maybe I can adapt it later. 
'''
