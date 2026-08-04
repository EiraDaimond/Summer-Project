import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math

# Define the variables to be used
m = 1.0
L = 5000
eps = 0.001
n = 10000
k = 1
lam = 1

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
def an_HMC_alg(k, lam, n, L, eps):
    '''
    -Carry out the HMC algorithm using the leafrog method to generate x values. 
    -Simultaenously compute and store KE, PE, exp(-delH).
    -Calculate acceptance ratio. 
    -Another way to check that the algorithm is working correctly is to check 
    reversibility with each trajectory, so we also include this test.
    '''
        
    # Define the anharmonic potential term
    def an_V(x,k,lam):
        an_V = 0.5*k*x**2 + 0.25**lam*x**4
        return an_V

    # Define the anharmonic Hamiltonian
    def an_H(x, p,m):
        return an_V(x,k,lam) + 0.5*p**2/m
    
    # Initialise the x values, KE values, errors, and the accepted values lists
    x = [1]
    p_vals = []
    KE_vals = []
    errors_p = []
    errors_x = []
    exps_delH = []
    accepted = []
    for_animation_x = np.zeros((L,2),dtype=float)
    for_animation_p = np.zeros((L,2),dtype=float)
    # Start the loop to generate x values
    for t in range(n+1):
        print("On iteration", t)
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0, m**0.5)
        # print("p=", p)
        # Compute the first leapfrog step
        p_star = p - 0.5*eps*(k*x[t] + lam*x[t]**3)
        # print("p_star before leapfrog=",p_star)
        x_star = x[t] + eps*p_star/m
        for_animation_x[0] = x_star, 0
        for_animation_p[0] = p_star, 0
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, L):
            p_star = p_star - eps*(k*x_star + lam*x_star**3)
            #print("p_star in step", l, p_star)
            x_star = x_star + eps*p_star/m
            # print("x_star", x_star)
            # print("change for x", eps*p_star/m)
            for_animation_x[l] = [x_star, l]
            for_animation_p[l] = [p_star, l]
        # Compute the final step of the leapfrog method
        p_star = p_star - 0.5*eps*(k*x_star + lam*x_star**3)
        # Compute the acceptance ratio
        r = np.exp(-an_H(x_star, p_star,m ) + an_H(x[t], p,m))
        exps_delH.append(r)
        # Draw W from a Uniform distribution
        W = np.random.uniform(0, 1)
        # Carry out the Metropolis test
        if W <= min(1, r):
            x.append(x_star)
            accepted.append(x_star)
            p_vals.append(p_star)
        else:
            x.append(x[t])
            p_vals.append(p)
        # Compute the KE terms for this trajectory and append to list
        KE = 0.5*p_vals[-1]**2/m
        KE_vals.append(KE)
        # Check reversibility
        p_star = p_star + 0.5*eps*(k*x_star + lam*x_star**3)
        x_star = x_star - eps*p_star/m
        for l in range(1, L):
            p_star = p_star + eps*(k*x_star + lam*x_star**3)
            x_star = x_star - eps*p_star/m
        p_backwards = p_star + 0.5*eps*(k*x_star + lam*x_star**3)
        error_p = (p_backwards - p)
        errors_p.append(error_p)
        error_x = x_star - x[-2]
        errors_x.append(error_x)
    # Compute acceptance ratio
    acc_rat = (len(accepted)/len(x))*100    

    # # Plot the potential
    # fig, ax = plt.subplots(figsize=(10,10))
    # x_wbi = x[math.ceil(len(x)*0.1):]
    # V_vals_raw = []
    # for i in range(len(x)):
    #     V_vals_raw.append(an_V(x[i],k,lam))
    # V_wbi = V_vals_raw[math.ceil(len(V_vals_raw)*0.1):]
    # ax.set_xlim(-2,2)
    # fig.supxlabel("x")
    # ax.set_ylim(-2,2)
    # fig.supylabel("V(x)")
    # ax.set_title("Anharmonic potential from Metropolis")
    # ax.scatter(x_wbi, V_wbi )
    # fig.savefig("x_anHMC.png")
    return x, KE_vals, exps_delH, errors_p, errors_x, acc_rat, for_animation_x, for_animation_p

# print(an_HMC_alg(1,1,1000,L, eps))
# print(an_HMC_alg(-1, 1, 1000, L, eps))
# Find the expected value and standard deviation of x
# def mean_and_sd(x,m ,n):
#     '''
#     Given a list of x values, compute the expected value
#       and standardised standard deviation (rejecting burn-in).
#     '''
#     length = len(x)
#     values_to_use = x[math.ceil(length/10):]
#     stand_sd = m**0.5/(n-1)**0.5
#     return np.mean(values_to_use), stand_sd*np.std(values_to_use)

# print("Expected x =", mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[0],1,100000)[0],\
#       "Standardised standard deviation of x=", mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[0],1,100000)[1],\
#        "Expected KE = ",mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[1],1,100000)[0], \
#        "Standardised standard deviation of KE = ", mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[1],1,100000)[1],\
#         "Expected exp(-delH)= " ,mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[2],1,100000)[0],\
#         "Standardised standard deviation of exp(-delH) = ", mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[2],1,100000)[1],\
#         "Expected error =", mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[3],1,100000)[0],\
#         "Standardised standard deviation of error=", mean_and_sd(an_HMC_alg(k,lam,n,L,eps)[3],1,100000)[1],\
#         "Acceptance ratio =" ,an_HMC_alg(k,lam,n,L,eps)[4])

# # Store the results from running the RMHMC alg
results_1 = an_HMC_alg(1, 1, n=1, L= L, eps = eps)
x_anim_1 = np.array(results_1[5])[:,1]
# print("x_anim", x_anim)
y_anim_1 = np.array(results_1[5])[:,0]
x_anim_1_p = np.array(results_1[6])[:,1]
# print("x_anim", x_anim)
y_anim_1_p = np.array(results_1[6])[:,0]
results_2 = an_HMC_alg(-1, 1, n=1, L= L, eps = eps)
x_anim_2 = np.array(results_2[5])[:,1]
# print("x_anim", x_anim)
y_anim_2 = np.array(results_2[5])[:,0]
x_anim_2_p = np.array(results_2[6])[:,1]
# print("x_anim", x_anim)
y_anim_2_p = np.array(results_2[6])[:,0]
# # print("y_anim", y_anim)
# stride = 20
# x_anim = x_anim[::stride]
# y_anim = y_anim[::stride]

# Setting up the plot for the dynamics
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(0,L)
fig.supxlabel("Leapfrog step")
ax.set_ylim(-2,2)
fig.supylabel("x")
ax.set_title("x dynamics with k > 0")
ax.scatter(x_anim_1, y_anim_1)
fig.savefig("anHMC_ani_set_1.png")
# trace_1, = ax.plot([],[])
# current_plot_1, = ax.plot([],[]) 

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

# animate_x = ani.FuncAnimation(fig, update, frames=len(x_anim_1), init_func=init, blit=False, interval=100, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_x.save("HMC_animate_x_1.gif", writer = 'pillow')

# Setting up the plot for the dynamics
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(0,L)
fig.supxlabel("Leapfrog step")
ax.set_ylim(-3,3)
fig.supylabel("Value")
ax.set_title("x dynamics with k < 0")
ax.scatter(x_anim_2, y_anim_2)
fig.savefig("anHMC_ani_set_2.png")

fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(0,L)
fig.supxlabel("Leapfrog step")
ax.set_ylim(-3,3)
fig.supylabel("Value")
ax.set_title("p dynamics with k > 0")
ax.scatter(x_anim_1_p, y_anim_1_p, c='#D32F2F')
fig.savefig("anHMC_ani_set_1_p.png")

fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(0,L)
fig.supxlabel("Leapfrog step")
ax.set_ylim(-3,3)
fig.supylabel("Value")
ax.set_title("p dynamics with k < 0")
ax.scatter(x_anim_2_p, y_anim_2_p,c='#D32F2F')
fig.savefig("anHMC_ani_set_2_p.png")

# trace_2, = ax.plot([],[])
# current_plot_2, = ax.plot([],[]) 

# # Functions for the dynamics
# def init():
#     trace_2.set_data([],[])
#     trace_2.set_color('blue')
#     return trace_2
# def update(frame):
#     trace_x = x_anim_2[:frame+1]
#     trace_y = y_anim_2[:frame+1]
#     trace_2.set_data(trace_x, trace_y)
#     return trace_2

# animate_x = ani.FuncAnimation(fig, update, frames=len(x_anim_2), init_func=init, blit=False, interval=100, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_x.save("HMC_animate_x_2.gif", writer = 'pillow')



