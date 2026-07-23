import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Define the variables to be used
L = 10000
eps = 1e-8
k = -1
lam = 1
n = 10
tol = 1e-12
d = 1e-6

# Define the anharmonic potential term
def an_V(x,k,lam):
    if k >0:
        an_V = 0.25*lam*x**4 + 0.5*lam*k*x**2
    else:
        an_V = 0.25*lam*x**4 - 0.5*lam*k*x**2
    return an_V

# Define the metric tensor (second derivative of the potential term)
def G(x,k,lam):
    return k + 3*lam*x**2
    
# Define M (including delta).... to be used to avoid division by 0 errors
def M(x, d):
    return np.sqrt(abs(G(x,k,lam)**2+d**2))

# Define the kinetic energy term (include correction term)
def K(p, x,d):
    return 0.5*p*M(x,d)*p + 0.5*np.log(np.abs(M(x,d)))

# Define the Hamiltonian
def H(x, p,d):
    return an_V(x,k,lam) + K(p, x, d) 

def RMHMC(L=None,eps=None,k=None,lam=None,tol=None,n=None, d=None):
    '''
    Rewriting the anharmonic HMC class but for RMHMC in one dimension. 
    Mass is non-constant and is instead represented by the metric tensor which I 
    shall denote G. In the 1D case, this is just a scalar. 
    ''' 
 
    # Run the RMHMC algorithm
    '''
    Carry out the RMHMC algorithm to generate x values. 
    We will use the Generalised Leapfrog Method with the fixed point iteration.
    '''
    # Initialise the x, KE, PE, exps_delH, errors, accepted values list
    x = [0] 
    KE_vals =[]
    PE_vals= [0]
    exps_delH = []
    errors = []
    accepted = []
    for_animation_x = np.zeros((L+1,2),dtype=float)
    for_animation_p = np.zeros((L+1,2), dtype = float)
    # Start the loop to generate x values
    for t in range(n+1):
        print("On iteration:", t)
        # Initialise the x_star and p_star lists
        x_stars = []
        p_stars = []
        # Initialise the V(x) lists
        V_x = []
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0, M(x[t], d))
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
        for_animation_p[0] = [p_star, 1]
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
        #("x_star is now", x_star)
        x_stars.append(x_star)
        V_x.append(an_V(x_star,k,lam))
        for_animation_x[0] = [x_star,1]
        #("CODE WORKS UP TO HERE")
        #()
        print("STARTING MIDDLE STEPS")
        #()
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, L+1):
            p_current = p_star
            p_guess = p_star
            p_star = 0
            count = 1
            #("On iter", l, "with p_star =", p_star, "p_guess =", p_guess)
            # PROBLEM IS HERE WHERE P VALUES AREN'T CONVERGING
            while True:
                print("Count=",count)
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
            #("Moving on from middle step iter [",l,"] with p_star", p_star)
            p_stars.append(p_star)
            for_animation_p[l] = [p_star, l]
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
                    print("BROKE x_star too big")
                    break
                else:
                    if x_star < -1e14:
                        print("BROKE x_star too big -ve")
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
            V_x.append(an_V(x_star,k,lam))  
            for_animation_x[l] = [x_stars[-1],l]
        #()
        print("STARTING FINAL STEPS")
        #()
        # Compute the final step of the leapfrog method
        p_current = p_star
        p_guess = p_star
        count = 1
        max_iter = 100
        while count < max_iter:
            #("Count=",count)
            count = count+1
            p_star = p_current - 0.5*eps\
                                    *(k*x_star + lam*x_star**3 + 0.5*p_guess**2*(-6*lam*x_star)\
                                        + 0.5*abs(-6*lam*x_star)/M(x_star,d))
            #("p_star is :", p_star)
            #("p_guess is", p_guess)
            #("Difference in ps:", abs(p_star - p_guess))
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
            #()
            for_animation_x[L] = [x_stars[-1],L]
            for_animation_p[L] = [p_stars[-1],L]
        # Compute the acceptance ratio
        r = np.exp(-H(x_star, p_star,d) + H(x[t], p,d))
        # Draw W from a Uniform distribution
        W = np.random.uniform(0, 1)            
        # Carry out the Metropolis test
        if W <= min(1, r):
            x.append(x_star)
            accepted.append(x_star)
        else:
            x.append(x[t])
        #print("x looks like:", x)
        # Compute the KE and append to list
        KE = K(x[-1],p_star,d)
        KE_vals.append(KE)
        #print("KE_vals looks like:", KE_vals)
        # Compute the PE and append to list
        PE = an_V(x[-1],k,lam)
        PE_vals.append(PE)
        #print("PE_vals looks like:", PE_vals)
        # Compute the exp(-delH)
        exp_minus_del_H_ = np.exp(H(x[-1],p_star,d) - H(x[t], p,d))
        exps_delH.append(exp_minus_del_H_)
        #print("exps_minus_delH looks like:", exps_delH)
        # Check reversibility
        p_star = p_star + 0.5*eps\
                            *(k*x_star + lam*x_star**3 + 0.5*p_guess**2*(-6*lam*x_star)\
                                + 0.5*abs(-6*lam*x_star)/M(x_star,d))
        x_star = x_star - 0.5*eps\
                            *(p_star*M(x_current,d)+p_star*M(x_guess,d))
        for l in range(1, L):
            #print("On reversibility check, iter", l)
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
    return x, KE_vals, PE_vals, exps_delH, errors, acc_rat, for_animation_x, for_animation_p
    
# # Find the expected value of x and corresponding standardised standard deviation
# def mean_and_sd(list, n, d):
#     '''
#     Given a list of values, compute the expected value (with burn-in removed), 
#     and corresponding standardised standar deviation.
#     '''
#     length = len(list)
#     values_to_use = list[math.ceil(length/10):]
#     # Initialise the sd_list
#     sd_list = [0]*(len(values_to_use))
#     for i in range(len(values_to_use)):
#         sd_list[i] = M(values_to_use[i], d)
#     return np.mean(values_to_use), np.sqrt((np.mean(sd_list))/(n-1))  

#print(RMHMC(L,eps, k,lam,tol,n,d)[0])
# print("Expected x =", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[0]),n, 1e-6)[0],\
#       "Standardised standard deviation of x=",mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[0]),n, 1e-6)[1] ,\
#        "Expected KE = ",mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[1]),n, 1e-6)[0], \
#        "Standardised standard deviation of KE = ",mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[1]),n, 1e-6)[1],\
#        "Expected PE =", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[2]),n, 1e-6)[0],\
#        "Standardised standard deviation of PE = ", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[2]),n, 1e-6)[1],\
#        "Expected exp(-delH)= " ,mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[3]),n, 1e-6)[0],\
#        "Standardised standard deviation of exp(-delH) = ", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[3]),n, 1e-6)[1],\
#         "Expected error =", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[4]),n, 1e-6)[0],\
#         "Standardised standard deviation of error=", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[4]),n, 1e-6)[1],\
#         "Acceptance ratio =" ,RMHMC(L,eps,1,1,1e-6,n,1e-6)[5])

# Store the results from running the RMHMC alg
results = RMHMC(L,eps,k,lam,tol,n,d)
x_anim = np.array(results[6])[:,1]
y_anim = np.array(results[6])[:,0]
x_anim_p = np.array(results[7])[:,1]
y_anim_p = np.array(results[7])[:,0]
stride = 20
x_anim = x_anim[::stride]
y_anim = y_anim[::stride]
x_anim_p = x_anim_p[::stride]
y_anim_p = y_anim_p[::stride]

# Setting up the plot for the dynamics
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(min(x_anim)-1,max(x_anim)+1)
fig.supxlabel("Leapfrog step")
ax.set_ylim(min(y_anim)-0.0000001,max(y_anim)+0.0000001)
fig.supylabel(")Value")
ax.set_title("x dynamics")
trace, = ax.plot([],[])
current_plot, = ax.plot([],[]) 

# Setting up the plot for the dynamics
fig_p, ax_p = plt.subplots(figsize=(10,10))
ax_p.set_xlim(min(x_anim_p)-1,max(x_anim_p)+1)
fig_p.supxlabel("Leapfrog step")
ax_p.set_ylim(min(y_anim_p)-0.0000000001,max(y_anim_p)+0.000000001)
fig_p.supylabel("Value")
ax_p.set_title("p dynamics")
trace_p, = ax.plot([],[])
current_plot_p, = ax.plot([],[]) 

# Functions for the dynamics
def init():
    trace.set_data([],[])
    current_plot.set_data([],[])
    trace.set_color('blue')
    current_plot.set_color('green')
    trace_p.set_data([],[])
    current_plot_p.set_data([],[])
    trace_p.set_color('red')
    current_plot_p.set_color('green')
    return trace, current_plot, trace_p, current_plot_p
def update(frame):
    trace_x = x_anim[:frame+1]
    trace_y = y_anim[:frame+1]
    trace.set_data(trace_x, trace_y)
    current_x = [x_anim[frame]]
    current_y = [y_anim[frame]]
    current_plot.set_data(current_x, current_y)
    trace_x_p = x_anim_p[:frame+1]
    trace_y_p = y_anim_p[:frame+1]
    trace_p.set_data(trace_x_p, trace_y_p)
    current_x_p = [x_anim_p[frame]]
    current_y_p = [y_anim_p[frame]]
    current_plot_p.set_data(current_x_p, current_y_p)
    return trace, current_plot, trace_p, current_plot_p
# def init_p():
#     trace_p.set_data([],[])
#     current_plot_p.set_data([],[])
#     trace_p.set_color('red')
#     current_plot_p.set_color('green')
#     return trace_p, current_plot_p
# def update_p(frame):
#     trace_x_p = x_anim_p[:frame+1]
#     trace_y_p = y_anim_p[:frame+1]
#     trace_p.set_data(trace_x_p, trace_y_p)
#     current_x_p = [x_anim_p[frame]]
#     current_y_p = [y_anim_p[frame]]
#     current_plot_p.set_data(current_x_p, current_y_p)
#     return trace_p, current_plot_p

animate_x = ani.FuncAnimation(fig, update, frames=len(x_anim), init_func=init, blit=False, interval=20, repeat=False)
fig.canvas.manager.window.attributes('-topmost', 1)
animate_x.save("animate_x.gif", writer = 'pillow')
# animate_p = ani.FuncAnimation(fig, update_p, frames=(len(x_anim_p)-1), init_func=init_p, blit=False, interval=50, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_p.save("animate_p.gif", writer = 'pillow')

'''
COMMENTS:
RMHMC ALG
- Want Leps = 1 (will later change), at mo this is resulting in many L iterations (1e8,1e-8), code takes a long time to run.
- Tolerance I have chosen as 1e-6... I feel like this is still pretty high but maybe I can adapt it later.
- For some reason a very high acceptance ratio (+90%) 
- PROBLEM: Algorithm is bad; if given initial x as positive, then all xs positive, if started negative then all xs negative; 
only 0 allows x to take both positive and negative values.... this seems bad. 
ANIMATION
- Won't appear 
'''
