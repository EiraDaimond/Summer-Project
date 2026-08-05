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
    return np.sqrt(G(x,k,lam)**2+d**2)

# Define the kinetic energy term (include correction term)
def K(x,p, k, lam, d):
    return 0.5*p**2/M(x,k, lam,d) + 0.5*np.log(M(x,k,lam, d))

# Define the Hamiltonian
def H(x, p, k,lam,d):
    return an_V(x,k,lam) + K(p, x,k, lam, d) 

# Find the expected value of x and corresponding standardised standard deviation
def mean_and_sd(list, n,k, lam, d):
    '''
    Given a list of values, compute the expected value (with burn-in removed), 
    and corresponding standardised standard deviation.
    '''
    length = len(list)
    values_to_use = list[math.ceil(length/10):]

    M_vals = [0]*(len(values_to_use))
    for i in range(len(values_to_use)):
        M_vals[i] = M(values_to_use[i],k, lam, d)
    return np.mean(values_to_use), np.sqrt((np.mean(M_vals))/(n-1))*np.std(values_to_use)  

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
    # Initialise the x, KE, PE, exps_delH, errors, accepted values, animation lists
    x = [np.random.normal(0, np.sqrt(M(0,k, lam, d)))] 
    p_vals = []
    KE_vals =[]
    PE_vals= [0]
    exps_delH = []
    errors_p = []
    errors_x = []
    accepted = []
    for_animation_x = np.zeros((L+1,2), dtype = float)
    for_animation_p = np.zeros((L+1,2), dtype = float)
    # Start the loop to generate x values
    for t in (range(n)):
        # print("x is=", x)
        # print("x length=", len(x))
        # print("x[",t,"] =", x[t])
        # Initialise the x_star and p_star lists
        x_stars = []
        p_stars = []
        # Initialise the V(x) lists
        V_x = []
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0,np.sqrt(M(x[t],k, lam, d)))
        print("On iter", [t], "with x[t]", x[t], "p", p)
        # Provide an initial guess value for p, initialise p_star
        p_guess = p 
        p_star = 0
        # Start the fixed point iteration for the first leapfrog step
        # p convergence
        count = 0
        max_iter = 100
        while count < max_iter:
            #("Count =",count)
            count = count +1 
            # if count == max_iter:
                # print("Hit max iterations")
            p_star = p - 0.5*eps*\
                 ((k*x[t] + lam*x[t]**3 \
                     + 0.5*p_guess**2*(G(x[t],k,lam))*6*lam*x[t])/(M(x[t], k, lam, d))**2 \
                + 0.5*(6*lam*x[t])\
                      /M(x[t],k, lam, d))
            if p_star > 1e14:
                # print("BROKE p_star too big")
                break
            else:
                if p_star < -1e14:
                    # print("BROKE p_star too big -ve")
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
        # print("Moving on from 1st step with p_star", p_star)  
        # print("Broke p for first step on count=", count)
        p_stars.append(p_star)
        for_animation_p[0] = [p_star, 1]
        # x convergence
        x_guess = x[t]
        x_star = 0
        count = 0
        while count < max_iter:
            #("Count =",count)
            count = count + 1
            # if count == max_iter:
                # print("Hit max iterations")
            #("1st step x_star is :", x_star)
            x_star = x[t] + 0.5*eps\
                             *(p_star/M(x[t],k,lam, d)+p_star/M(x_guess,k, lam,d))
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
        # print("Moving on from 1st step with x_star", x_star)
        # print("Broke x for first step on count=", count)
        #("x_star is now", x_star)
        x_stars.append(x_star)
        V_x.append(an_V(x_star,k,lam))
        for_animation_x[0] = [x_star,1]
        # print(for_animation_x)
        #("STARTING MIDDLE STEPS")
    #     #()
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, L+1):
            if p_star > 1e14:
                print("BROKE p_star too big on iter", l)
                break
            else:
                if p_star < -1e14:
                    print("BROKE p_star too big -ve", l)
                    break
                else: 
                    pass
            # print("L=", l)
            p_current = p_star
            p_guess = p_star
            p_star = 0
            count = 0
            #("On iter", l, "with p_star =", p_star, "p_guess =", p_guess)
            while count < max_iter:
                #("Count=",count)
                count = count +1
                # if count == max_iter:
                    # print("Hit max iterations")
                #("Middle step iter[",l,"] p_star is :", p_star)
                #("Using x_star:", x_star)
                p_star = p_current - eps*\
                                 ((k*x_star + lam*x_star**3 \
                                     + 0.5*p_guess**2*(G(x_star,k,lam))*6*lam*x_star)/(M(x_star, k, lam, d))**2 \
                                + 0.5*(6*lam*x_star)\
                                      /M(x_star,k, lam, d))
                print("p_star", p_star)
                # print("Change",eps\
                #                         *(k*x_star + lam*x_star**3\
                #                              + 0.5*p_guess**2*(6*lam*x_star)\
                #                              + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam, d)) )
                #("Calculated p_star =", p_star)
                #("p_guess is", p_guess)
                #("Difference in ps", abs(p_star - p_guess))
                if p_star > 1e14:
                    # print("BROKE p_star too big")
                    break
                else:
                    if p_star < -1e14:
                        # print("BROKE p_star too big -ve")
                        break
                    else:
                        if abs(p_star - p_guess) < tol:
                            #("STOPPING WHILE LOOP for p")
                            break 
                        else:
                            p_guess = p_star
            # print("Moving on from middle step iter [",l,"] with p_star", p_star)
            # print("Change",eps\
            #                                         *(k*x_star + lam*x_star**3\
            #                                              + 0.5*p_guess**2*(6*lam*x_star)\
            #                                              + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam, d)) )
            # print("Broke for p iteration", l," count=", count)
            p_stars.append(p_star)
            for_animation_p[l] = [p_star, l]
            ()
            # x convergence
            x_current = x_star
            x_guess = x_star
            x_star = 0
            count = 1
            while count < max_iter:
                #("Count=",count)
                count = count+1
                # if count == max_iter:
                    # print("Hit max iterations")
                #     break
                #("Middle step iter[",l,"] x_star is :", x_star)
                #("Using p_star", p_star)
                x_star = x_current + 0.5*eps\
                                *(p_star/M(x_current,k,lam,d)+p_star/M(x_guess,k,lam,d))
                # print("Change", 0.5*eps\
                #             *(p_star*M(x_current,k,lam,d)+p_star*M(x_guess,k,lam,d)) )
                #("x_star=", x_star)
                if x_star > 1e14:
                    # print("BROKE x_star too big")
                    break
                elif x_star < -1e14:
                    # print("BROKE x_star too big -ve")
                    break
                elif abs(x_star - x_guess) < tol:
                    #("STOPPING while loop for x_star")
                    break
                else:
                    x_guess = x_star
                #()
            # ("Moving on from middle step iter[",l,"] with x_star", x_star)
            #("Broke for x iteration", l," count=", count)
            x_stars.append(x_star)
            V_x.append(an_V(x_star,k,lam))  
            for_animation_x[l] = [x_stars[-1],l]
            # print(for_animation_x)
        #()
        # Compute the final step of the leapfrog method
        p_current = p_star
        p_guess = p_star
        count = 0
        while count < max_iter:
            #("Count=",count)
            count = count+1
            if count == max_iter:
                            # print("Hit max iterations")
                            p_star = p_current - 0.5*eps*\
                                                    ((k*x_star + lam*x_star**3 \
                                                                 + 0.5*p_guess**2*(G(x_star,k,lam))*6*lam*x_star)/(M(x_star, k, lam, d))**2 \
                                                            + 0.5*(6*lam*x_star)\
                                                                  /M(x_star,k, lam, d))
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
            x.append(x_star)
        print("x_star is", x_star)
        # Compute the acceptance ratio
        r = np.exp(-H(x_star, p_star,k, lam, d) + H(x[t], p,k,lam, d))
        exps_delH.append(r)
        # print("r",r)
        # Draw W from a Uniform distribution
        W = np.random.uniform(0, 1)  
        # print("W", W)          
        # Carry out the Metropolis test
        if W <= min(1, r):
            x.append(x_star)
            accepted.append(x_star)
            p_vals.append(p_star)
            print("I appended")
        else:
            x.append(x[t])
            p_vals.append(p)
        #print("x looks like:", x)
        # # Compute the KE and append to list
        # KE = K(x[-1],p_star,k,lam,d)
        # KE_vals.append(KE)
        # #print("KE_vals looks like:", KE_vals)
        # # Compute the PE and append to list
        # PE = an_V(x[-1],k,lam)
        # PE_vals.append(PE)
        #print("PE_vals looks like:", PE_vals)
        #print("exps_minus_delH looks like:", exps_delH)
        # # Check reversibility
        # p_star = p_current - 0.5*eps*\
        #                             ((k*x_star + lam*x_star**3 \
        #                                                          + 0.5*p_guess**2*(G(x_star,k,lam))*6*lam*x_star)/(M(x_star, k, lam, d))**2 \
        #                                                             + 0.5*(6*lam*x_star)\
        #                                                                   /M(x_star,k, lam, d))
        # x_star = x_star - 0.5*eps\
        #                     *(p_star/M(x_star,k,lam,d)+p_star/M(x_guess,k,lam,d))
        # for l in range(1, L):
        #     #print("On reversibility check, iter", l)
        #     p_star = p_current - eps*\
        #                         ((k*x_star + lam*x_star**3 \
        #                                                 + 0.5*p_guess**2*(G(x_star,k,lam))*6*lam*x_star)/(M(x_star, k, lam, d))**2 \
        #                                                                 + 0.5*(6*lam*x_star)\
        #                                                                       /M(x_star,k, lam, d))
        #     x_star = x_star - 0.5*eps\
        #                     *(p_star/M(x_star,k,lam,d)+p_star/M(x_guess,k,lam,d))
        #     p_backwards = p_star - 0.5*eps*\
        #                                     ((k*x_star + lam*x_star**3 \
        #                                                     + 0.5*p_star**2*(G(x_star,k,lam))*6*lam*x_star)/(M(x_star, k, lam, d))**2 \
        #                                                             + 0.5*(6*lam*x_star)\
        #                                                                     /M(x_star,k, lam, d))
        #     error_p = (p_backwards - p)
        #     errors_p.append(error_p)
        #     error_x = x_star - x[-2]
        #     errors_x.append(error_x)
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
    return x, p_vals, KE_vals, PE_vals, exps_delH, errors_p, errors_x, acc_rat, for_animation_x, for_animation_p

# print("x with k = 1", RMHMC(L = 10000,
#             eps = 1e-8,
#             k = 1,
#             lam = 1,
#             n = 1,
#             tol = 1e-12,
#             d = 1e-6)[0])
# print("x with k=-1",RMHMC(L = 10000,
#             eps = 1e-8,
#             k = -1,
#             lam = 1,
#             n = 1000,
#             tol = 1e-12,
#             d = 1e-6)[0])

results_pos = RMHMC(L=10000,
                  eps = 0.001, 
                  k = 1,
                  lam = 1,
                  n=100,
                  tol = 1e-12,
                  d = 0.1)

# results_neg = RMHMC(L = 10000,
#                 eps = 1e-8,
#                 k = -0.1,
#                 lam = 1,
#                 n = 100,
#                 tol = 1e-12,
#                 d = 0.1)
# k = 1
# lam = 1
# n = 100
# d = 0.1 
# print("Expected x =", mean_and_sd((results_pos[0]),n, k, lam, d)[0],\
#       "Standardised standard deviation of x=",mean_and_sd(((results_pos)[0]),n, k, lam, d)[1]) 
#     #    "Expected KE = ",mean_and_sd(results[1]),n, 1e-6)[0], \
#     #    "Standardised standard deviation of KE = ",mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[1]),n, 1e-6)[1],\
#     #    "Expected PE =", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[2]),n, 1e-6)[0],\
#     #    "Standardised standard deviation of PE = ", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[2]),n, 1e-6)[1],\
# print(      "Expected exp(-delH)= " ,mean_and_sd((results_pos[3]),n, k,lam, d)[0],\
#       "Standardised standard deviation of exp(-delH) = ", mean_and_sd((results_pos[3]),n, k, lam, d)[1],\
#        "Expected error =", mean_and_sd((results_pos[4]),n, k, lam, d)[0],\
#         "Standardised standard deviation of error=", mean_and_sd((results_pos[4]),n, k, lam,d)[1],\
# print("Acceptance ratio =" ,results_pos[5]) 
# print()
# print("Now for negative k")
# print()
# k = -0.1
# print(results_neg[4])
# print("Expected x =", mean_and_sd((results_pos[0]),n, k, lam, d)[0],\
#       "Standardised standard deviation of x=",mean_and_sd(((results_pos)[0]),n,k, lam,d)[1]) ,\
#     #    "Expected KE = ",mean_and_sd(results[1]),n, 1e-6)[0], \
#     #    "Standardised standard deviation of KE = ",mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[1]),n, 1e-6)[1],\
#     #    "Expected PE =", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[2]),n, 1e-6)[0],\
#     #    "Standardised standard deviation of PE = ", mean_and_sd((RMHMC(L,eps,1,1,1e-6,n,1e-6)[2]),n, 1e-6)[1],\
# print(     "Expected exp(-delH)= " ,mean_and_sd((results_pos[3]),n,k, lam,d)[0],\
#        "Standardised standard deviation of exp(-delH) = ", mean_and_sd((results_pos[3]),n, k, lam, d)[1])
# print("Expected error in p =", mean_and_sd((results_pos[4]),n, k, lam, d)[0],\
#         "Standardised standard deviation of error in p=", mean_and_sd((results_pos[4]),n, k, lam, d)[1],\
#         "Expected error in x=", mean_and_sd((results_pos[5]),n, k, lam, d)[0],\
#         "Standardised standard deviation of error in x=", mean_and_sd((results_pos[5]),n, k, lam, d)[1],\
#          "Acceptance ratio =" ,results_pos[6]) 
# Store the results from running the RMHMC alg
# results_1 = RMHMC(L=10000,
#                   eps = 0.001, 
#                   k = 1,
#                   lam = 1,
#                   n=1,
#                   tol = 1e-12,
#                   d = 0.1)
# x_anim_1 = np.array(results_1[8])[:,1]
# y_anim_1 = results_1[8][:,0]
# # print(x_anim_1, y_anim_1)
# # print(results_1[8])
# x_anim_p_1 = np.array(results_1[9])[:,1]
# y_anim_p_1 = np.array(results_1[9])[:,0]
# print(x_anim_p_1, y_anim_p_1)
# results_2 = RMHMC(L=10000,
#                   eps = 0.001, 
#                   k = -1,
#                   lam = 1,
#                   n=1,
#                   tol = 1e-12,
#                   d = 0.1)
# x_anim_2 = np.array(results_2[8])[:,1]
# y_anim_2 = np.array(results_2[8])[:,0]
# # print(y_anim_2)
# x_anim_p_2 = np.array(results_2[9])[:,1]
# y_anim_p_2 = np.array(results_2[9])[:,0]
# # print(y_anim_p_2)

# # print(x_anim_1)
# # print()
# # print(y_anim_1)
# # print()
# # print(x_anim_2)
# # print()
# # print(y_anim_2)
# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,10000)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-2,2)
# fig.supylabel("x")
# ax.set_title("x dynamics for k =1")
# ax.scatter(x_anim_1, y_anim_1)
# fig.savefig("RMHMC_ani_set_1.png")
# trace_1, = ax.plot([],[])

# # # Functions for the dynamics
# # def init():
# #     trace_1.set_data([],[])
# #     trace_1.set_color('blue')
# #     return trace_1
# # def update(frame):
# #     trace_x = x_anim_1[:frame+1]
# #     trace_y = y_anim_1[:frame+1]
# #     trace_1.set_data(trace_x, trace_y)
# #     return trace_1

# # animate_x_1 = ani.FuncAnimation(fig, update, frames=len(x_anim_1), init_func=init, blit=False, interval=100, repeat=False)
# # fig.canvas.manager.window.attributes('-topmost', 1)
# # animate_x_1.save("RMHMC_animate_x_1.gif", writer = 'pillow')

# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,10000)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-2,2)
# fig.supylabel("x")
# ax.set_title("x dynamics for k = -1")
# ax.scatter(x_anim_2, y_anim_2)
# fig.savefig("RMHMC_ani_set_2.png")

# # trace_2, = ax.plot([],[]) 

# # # Functions for the dynamics
# # def init_2():
# #     trace_2.set_data([],[])
# #     trace_2.set_color('blue')
# #     return trace_2
# # def update_2(frame):
# #     trace_x = x_anim_2[:frame+1]
# #     trace_y = y_anim_2[:frame+1]
# #     trace_2.set_data(trace_x, trace_y)
# #     return trace_2

# # animate_x_2 = ani.FuncAnimation(fig, update_2, frames=len(x_anim_2), init_func=init, blit=False, interval=100, repeat=False)
# # fig.canvas.manager.window.attributes('-topmost', 1)
# # animate_x_2.save("RMHMC_animate_x_2.gif", writer = 'pillow')

# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,10000)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-4,4)
# fig.supylabel("p")
# ax.set_title("p dynamics for k =1 ")
# ax.scatter(x_anim_p_1, y_anim_p_1, c='#D32F2F')
# fig.savefig("RMHMC_ani_set_3.png")
# # trace_3, = ax.plot([],[]) 

# # # Functions for the dynamics
# # def init_3():
# #     trace_3.set_data([],[])
# #     trace_3.set_color('blue')
# #     return trace_3
# # def update_3(frame):
# #     trace_x = x_anim_1[:frame+1]
# #     trace_y = y_anim_1[:frame+1]
# #     trace_3.set_data(trace_x, trace_y)
# #     return trace_3

# # animate_p_1 = ani.FuncAnimation(fig, update_3, frames=len(x_anim_p_1), init_func=init, blit=False, interval=50, repeat=False)
# # fig.canvas.manager.window.attributes('-topmost', 1)
# # animate_p_1.save("RMHMC_animate_p_1.gif", writer = 'pillow')

# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,10000)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-4,4)
# fig.supylabel("p")
# ax.set_title("p dynamics for k = -1")
# ax.scatter(x_anim_p_2, y_anim_p_2, c='#D32F2F')
# fig.savefig("RMHMC_ani_set_4.png")

# # trace_4, = ax.plot([],[])

# # # Functions for the dynamics
# # def init_4():
# #     trace_4.set_data([],[])
# #     trace_4.set_color('blue')
# #     return trace_2
# # def update_4(frame):
# #     trace_x = x_anim_2[:frame+1]
# #     trace_y = y_anim_2[:frame+1]
# #     trace_4.set_data(trace_x, trace_y)
# #     return trace_4

# # animate_p_2 = ani.FuncAnimation(fig, update_4, frames=len(x_anim_p_2), init_func=init, blit=False, interval=20, repeat=False)
# # fig.canvas.manager.window.attributes('-topmost', 1)
# # animate_p_2.save("RMHMC_animate_p_2.gif", writer = 'pillow')

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

# # Parameter space testing

# k_lam_vals = [-10, -5, -1, -0.1]
# d_vals = [0.001, 0.01, 0.1, 1]
# grads = []

# for i in range(len(k_lam_vals)):
#         for j in range(len(k_lam_vals)):
#             for k in range(len(d_vals)):
#                 results = RMHMC(L=100,
#                     eps = 1e-3, 
#                     k = k_lam_vals[i],
#                     lam = k_lam_vals[j],
#                     n=1,
#                     tol = 1e-6,
#                     d = d_vals[k])
#             x_vals = np.array(results[7])[:,0]
#             grad = (max(x_vals) - min(x_vals))/100
#             grads.append([grad, (i, j , k)])
#             print("Grad for k =", k_lam_vals[i], "lam =", k_lam_vals[j], "d = ", k_lam_vals[k], "=", grad)
# print(grads)

# # Epsilon and L testing

# epsilon = [1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1]
# L_vals = [100000, 50000, 10000,5000, 1000,500, 100,50, 10, 5, 1]
# grads = []

# for i in range(len(epsilon)):
#         for j in range(len(L_vals)):
#                 results = RMHMC(L=j,
#                     eps = i, 
#                     k = -1,
#                     lam = -1,
#                     n=1,
#                     tol = 1e-6,
#                     d = 0.01)
#                 x_vals = np.array(results[7])[:,0]
#                 grad = (max(x_vals) - min(x_vals))/100
#                 grads.append([grad, (i, j , k)])
#                 print("Grad for eps =", epsilon[i], "L =", L_vals[j], "is", grad)
# print(grads)

# results = RMHMC(L = 100,
#             eps = 0.01,
#             k = -1,
#             lam = 1,
#             n = 10,
#             tol = 1e-6,
#             d = 0.01)
# print("x=",results[0])
# print("p starting=", results[8])
# # print("x in leapfrog=",results[6])
# # print("p in leapfrog=",results[7])

