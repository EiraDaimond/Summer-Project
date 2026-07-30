import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D

# Define the anharmonic potential term
def an_V(x,k,lam):
    an_V = 0.25*lam*x**4 + 0.5*k*x*2
    return an_V

# Define the metric tensor (second derivative of the potential term)
def G(x,k,lam):
    return k + 3*lam*x**2
        
# Define M (including delta).... to be used to avoid division by 0 errors
def M(x,k, lam, d):
    return np.sqrt(abs(G(x,k,lam)**2+d**2))

# Define the kinetic energy term (include correction term)
def K(x,p, k, lam, d):
    return 0.5*p*M(x,k, lam,d)*p + 0.5*np.log(np.abs(M(x,k,lam, d)))

# Define the Hamiltonian
def H(x, p, k,lam,d):
    return an_V(x,k,lam) + K(p, x,k, lam, d) 

# Find the expected value of x and corresponding standardised standard deviation
def mean_and_sd(list, n,k, lam, d):
    '''
    Given a list of values, compute the expected value (with burn-in removed), 
    and corresponding standardised standar deviation.
    '''
    length = len(list)
    values_to_use = list[math.ceil(length/10):]
    # Initialise the sd_list
    sd_list = [0]*(len(values_to_use))
    for i in range(len(values_to_use)):
        sd_list[i] = M(values_to_use[i],k, lam, d)
    return np.mean(values_to_use), np.sqrt((np.mean(sd_list))/(n-1))  

def RMHMC(L = 10000,
            eps = 1e-8,
            k = -1,
            lam = 1,
            n = 1000,
            tol = 1e-12,
            d = 0.1):
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
    x = [2] 
    KE_vals =[]
    PE_vals= [0]
    exps_delH = []
    errors = []
    accepted = []
    for_animation_x = np.zeros((L+1,2),dtype=float)
    for_animation_p = np.zeros((L+1,2), dtype = float)
    # Start the loop to generate x values
    for t in range(n+1):
        # Initialise the x_star and p_star lists
        x_stars = []
        p_stars = []
        # Initialise the V(x) lists
        V_x = []
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0, M(x[t],k, lam, d))
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
                     /M(x[t],k, lam, d))
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
                            *(p_star*M(x[t],k,lam, d)+p_star*M(x_guess,k, lam,d))
            #("x_star=", x_star)
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
        #("STARTING MIDDLE STEPS")
        #()
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, L+1):
            # print("L=", l)
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
                                             + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam, d))
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
                            #("STOPPING WHILE LOOP for p")
                            break 
                        else:
                            p_guess = p_star
                #()
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
                            *(p_star*M(x_current,k,lam,d)+p_star*M(x_guess,k,lam,d))
                #("x_star=", x_star)
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
        #("STARTING FINAL STEPS")
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
                                        + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam,d))
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
        r = np.exp(-H(x_star, p_star,k, lam, d) + H(x[t], p,k,lam, d))
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
        KE = K(x[-1],p_star,k,lam,d)
        KE_vals.append(KE)
        #print("KE_vals looks like:", KE_vals)
        # Compute the PE and append to list
        PE = an_V(x[-1],k,lam)
        PE_vals.append(PE)
        #print("PE_vals looks like:", PE_vals)
        # Compute the exp(-delH)
        exp_minus_del_H_ = np.exp(H(x[-1],p_star,k,lam,d) - H(x[t], p,k,lam,d))
        exps_delH.append(exp_minus_del_H_)
        #print("exps_minus_delH looks like:", exps_delH)
        # Check reversibility
        # p_star = p_star + 0.5*eps\
        #                     *(k*x_star + lam*x_star**3 + 0.5*p_guess**2*(-6*lam*x_star)\
        #                         + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam,d))
        # x_star = x_star - 0.5*eps\
        #                     *(p_star*M(x_current,k,lam,d)+p_star*M(x_guess,k,lam,d))
        # for l in range(1, L):
        #     #print("On reversibility check, iter", l)
        #     p_star = p_star + eps\
        #                         *(k*x_star + lam*x_star**3\
        #                             + 0.5*p_guess**2*(-6*lam*x_star)\
        #                             + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam,d))
        #     x_star = x_star - 0.5*eps\
        #                     *(p_star*M(x[t],k,lam,d)+p_star*M(x_guess,k,lam,d))
        #     p_backwards = p_star + 0.5*eps\
        #                     *(k*x_star + lam*x_star**3 + 0.5*p_guess**2*(-6*lam*x_star)\
        #                         + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam,d))
        #     error = (p_backwards - p)
        #     errors.append(error)
    # Compute acceptance ratio
    acc_rat = (len(accepted)/len(x))*100

    # # Plot the potential
    # fig, ax = plt.subplots(figsize=(10,10))
    # x_wbi = x[math.ceil(len(x)*0.1):]
    # # print("x_wbi=", x_wbi)
    # V_vals_raw = []
    # for i in range(len(x)):
    #     V_vals_raw.append(an_V(x[i],k,lam))
    # V_wbi = V_vals_raw[math.ceil(len(V_vals_raw)*0.1):]
    # # print("V_wbi=", V_wbi)
    # ax.set_xlim(0,3)
    # fig.supxlabel("x")
    # ax.set_ylim(0,5)
    # fig.supylabel("V(x)")
    # ax.set_title("Anharmonic potential from Metropolis")
    # ax.scatter(x_wbi, V_wbi )
    # fig.savefig("x_RMHMC.png")
    # print()
    return x, KE_vals, PE_vals, exps_delH, errors, acc_rat, p_star , for_animation_x, for_animation_p

# print("x with k = 1", RMHMC(L = 10000,
#             eps = 1e-8,
#             k = 1,
#             lam = 1,
#             n = 100,
#             tol = 1e-12,
#             d = 1e-6)[0])
# print("x with k=-1",RMHMC(L = 10000,
#             eps = 1e-8,
#             k = -1,
#             lam = 1,
#             n = 1000,
#             tol = 1e-12,
#             d = 1e-6)[0])

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

# # Store the results from running the RMHMC alg
# results_1 = RMHMC(L=10000,
#                   eps = 1e-8, 
#                   k = 1,
#                   lam = 1,
#                   n=1,
#                   tol = 1e-12,
#                   d = 1e-6)
# x_anim_1 = np.array(results_1[7])[:,1]
# y_anim_1 = results_1[7][:,0]
# x_anim_p_1 = np.array(results_1[8])[:,1]
# y_anim_p_1 = np.array(results_1[8])[:,1]
results_2 = RMHMC(L=10000,
                  eps = 1e-8, 
                  k = -1,
                  lam = 1,
                  n=1,
                  tol = 1e-12,
                  d = 0.1)
x_anim_2 = np.array(results_2[7])[:,1]
# print("x_anim", x_anim)
y_anim_2 = np.array(results_2[7])[:,0]
# x_anim_p_2 = np.array(results_2[8])[:,1]
# y_anim_p_2 = np.array(results_2[8])[:,0]

# print(x_anim_1)
# print()
# print(y_anim_1)
# print()
# print(x_anim_2)
# print()
# print(y_anim_2)
# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,10000)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-0.001,0.001)
# fig.supylabel("x")
# ax.set_title("x dynamics for k =1")
# ax.scatter(x_anim_1, y_anim_1)
# fig.savefig("anHMC_ani_set_1.png")
# trace_1, = ax.plot([],[])

# # Functions for the dynamics
# def init():
#     trace_1.set_data([],[])
#     trace_1.set_color('blue')
#     return trace_1
# def update(frame):
#     trace_x = x_anim_1[:frame+1]
#     trace_y = y_anim_1[:frame+1]
#     trace_1.set_data(trace_x, trace_y)
#     return trace_1

# animate_x_1 = ani.FuncAnimation(fig, update, frames=len(x_anim_1), init_func=init, blit=False, interval=50, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_x_1.save("RMHMC_animate_x_1.gif", writer = 'pillow')

# Setting up the plot for the dynamics
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(0,10000)
fig.supxlabel("Leapfrog step")
ax.set_ylim(0,4)
fig.supylabel("x")
ax.set_title("x dynamics for k = -1")
ax.scatter(x_anim_2, y_anim_2)
fig.savefig("RMHMC_ani_set_2.png")

# trace_2, = ax.plot([],[]) 

# # Functions for the dynamics
# def init_2():
#     trace_2.set_data([],[])
#     trace_2.set_color('blue')
#     return trace_2
# def update_2(frame):
#     trace_x = x_anim_2[:frame+1]
#     trace_y = y_anim_2[:frame+1]
#     trace_2.set_data(trace_x, trace_y)
#     return trace_2

# animate_x_2 = ani.FuncAnimation(fig, update_2, frames=len(x_anim_2), init_func=init, blit=False, interval=50, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_x_2.save("RMHMC_animate_x_2.gif", writer = 'pillow')

# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,10000)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-0.001,0.001)
# fig.supylabel("p")
# ax.set_title("p dynamics for k =1 ")
# # ax.scatter(x_anim_p_1, y_anim_p_1)
# # fig.savefig("RMHMC_ani_set_3.png")
# trace_3, = ax.plot([],[]) 

# # Functions for the dynamics
# def init_3():
#     trace_3.set_data([],[])
#     trace_3.set_color('blue')
#     return trace_3
# def update_3(frame):
#     trace_x = x_anim_1[:frame+1]
#     trace_y = y_anim_1[:frame+1]
#     trace_3.set_data(trace_x, trace_y)
#     return trace_3

# animate_p_1 = ani.FuncAnimation(fig, update_3, frames=len(x_anim_p_1), init_func=init, blit=False, interval=50, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_p_1.save("RMHMC_animate_p_1.gif", writer = 'pillow')

# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,10000)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-0.001,0.001)
# fig.supylabel("p")
# ax.set_title("p dynamics for k = -1")
# # ax.scatter(x_anim_p_2, y_anim_p_2)
# # fig.savefig("RMHMC_ani_set_4.png")

# trace_4, = ax.plot([],[])

# # Functions for the dynamics
# def init_4():
#     trace_4.set_data([],[])
#     trace_4.set_color('blue')
#     return trace_2
# def update_4(frame):
#     trace_x = x_anim_2[:frame+1]
#     trace_y = y_anim_2[:frame+1]
#     trace_4.set_data(trace_x, trace_y)
#     return trace_4

# animate_p_2 = ani.FuncAnimation(fig, update_4, frames=len(x_anim_p_2), init_func=init, blit=False, interval=20, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_p_2.save("RMHMC_animate_p_2.gif", writer = 'pillow')

# # Calculate gradient
# L = 10000
# x_grad_1 = (max(y_anim_1) - min(y_anim_1))/(L-1)
# print("Gradient of x ani for k =1 =", x_grad_1 )
# x_grad_2 = (max(y_anim_2) - min(y_anim_2))/(L-1)
# print("Gradient of x ani for k =-1 =", x_grad_2)
# p_grad_1 = (max(y_anim_p_1) - min(y_anim_p_1))/(L-1)
# print("Gradient of p ani for k =1 =", p_grad_1 )
# p_grad_2 = (max(y_anim_p_2) - min(y_anim_p_2))/(L-1)
# print("Gradient of p ani for k =-1 =", p_grad_2)

# # Big test for algorithm
# eps_vals = [1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1]
# L_vals = [100000, 50000, 10000,5000, 1000,500, 100,50, 10, 5, 1]
# eps_for_plotting = []
# L_for_plotting = []
# xs = []
# exp_xs = []
# exp_xs_KE =[]
# std_xs = []
# s = []
# for i in range(len(eps_vals)-1):
#     print("Running eps_vals[",i,"]")
#     for j in range(i+1, len(L_vals)):
#         print("Running L_vals[",j,"]")
#         eps_for_plotting.append(eps_vals[i])
#         L_for_plotting.append(L_vals[j])
#         updated_results = RMHMC(L = L_vals[j],
#             eps = eps_vals[i],
#             k = -1,
#             lam = 1,
#             n = 1,
#             tol = 1e-12,
#             d = 1e-6)
#         new_xs = updated_results[0]
#         new_p = updated_results[6]
#         xs.append(new_xs)
#         s.append(eps_vals[i]*L_vals[j])
# for k in range(len(xs)):
#     new_exp_x_val = mean_and_sd(xs[k], n=len(xs), k = -1, lam = 1, d=1e-6)[0]
#     exp_xs.append(new_exp_x_val)
#     exp_xs_KE.append(K(new_exp_x_val, new_p,k=-1, lam=1, d=1e-6))
#     std_xs.append(mean_and_sd(xs[k], n=len(xs), k = -1, lam = 1, d=1e-6)[1])
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(eps_for_plotting, L_for_plotting, exp_xs)
# plt.title("Epsilon, L, and Expected Value of x")
# plt.savefig("s_and_exp_xs.png")
# fig_2 = plt.figure()
# ax_2 = plt.axes(projection='3d')
# ax_2.scatter(eps_for_plotting, L_for_plotting, std_xs)
# plt.title("Epsilon, L, and Standard Deviation of x")
# plt.savefig("s_and_std.png")
# fig, ax = plt.subplots()  
# ax.scatter(s, exp_xs)
# ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
# ax.set_xlabel("s")
# ax.set_ylabel("Expected x")
# ax.set_title("s against expected value of x")
# fig.savefig("s_exp_x.png")
# plt.close(fig)

'''COMMENTS:
- If epsilon too big then overflow error... x_star values get too large and don't converge.
- If epsilon too small, change is very very small.
'''