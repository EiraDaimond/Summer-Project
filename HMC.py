import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Define the variables to be used
m = 1 
L = 100
eps = 0.01
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
#     plt.xlabel("KE_p values")
#     plt.ylabel("Frequency")
#     plt.title("KE distribution")
#     plt.show()
#     return p_normals, KE_p
# print(test_normal_p(100000,1))

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
    for_animation_x = np.zeros((L,2),dtype=float)
    # Start the loop to generate the x values
    for t in range(n+1):
        print("On iteration", t)
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0,m**0.5)
        print("p=",p)
        # Carry out step 1 of the leapfrog method
        p_star = p - 0.5*eps*k*x[t]
        # print("p_star is=", p_star)
        x_star = x[t] + eps*p_star/m
        for_animation_x[0] = x_star, 0
        # print("Change is", eps*p_star/m)
        # print("x_star is=", x_star)
        # Compute (x*,p*) using L leapfrog steps of size eps
        for l in range(1, L):
            #print("On leapfrog step", l)
            p_star = p_star - eps*k*x_star
            x_star = x_star + eps*p_star/m
            for_animation_x[l] = x_star, l
        #     print("Change is", eps*p_star/m)
        # print("x_star=", x_star)
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
        PE = 0.5*k*x[t]**2 
        KE_vals.append(KE)
        PE_vals.append(PE)
        # Calculate exp(-delH) terms
        exp_minus_del_H_ = np.exp(H(x[-1],p_star) - H(x[t], p))
        exps_delH.append(exp_minus_del_H_)
        # # Check reversibility
        # p_star = p_star + 0.5*eps*k*x[t]
        # x_star = x_star - eps*p_star/m
        # for l in range(1, L):
        #     p_star = p_star + eps*k*x_star
        #     x_star = x_star - eps*p_star/m
        # p_backwards = p_star + 0.5*eps*k*x_star
        # error = (p_backwards - p)
        # errors.append(error)
        # print(x)
    # Compute acceptance ratio
    acc_rat = (len(accepted)/len(x))*100
    return x, KE_vals, PE_vals, exps_delH, errors, acc_rat, for_animation_x

# # Find the expected value of x and corresponding standardised standard deviation
# def mean_and_sd(x,m,n):
#     '''
#     Given a list of values, compute the expected value (with burn-in removed), 
#     and corresponding standardised standard deviation.
#     '''
#     values_to_use = x[math.ceil(len(x)/10):]
#     stand_sd = m**0.5/(n-1)**0.5
#     return np.mean(values_to_use), stand_sd*np.std(values_to_use)   

# print("Expected x =", mean_and_sd(HMC(100000,L,eps)[0],1,100000)[0],\
#       "Standardised standard deviation of x=", mean_and_sd(HMC(100000,L,eps)[0],1,1000000)[1],\
#        "Expected KE = ",mean_and_sd(HMC(100000, L, eps)[1],1,1000000)[0], \
#        "Standardised standard deviation of KE = ", mean_and_sd(HMC(100000, L, eps)[1],1,100000)[1],\
#         "Expected PE =", mean_and_sd(HMC(100000,L,eps)[2],1,100000)[0],\
#         "Standardised standard deviation of PE = ", mean_and_sd(HMC(100000, L, eps)[2],1,100000)[1],\
#         "Expected exp(-delH)= " ,mean_and_sd(HMC(100000,L,eps)[3],1,1000000)[0],\
#         "Standardised standard deviation of exp(-delH) = ", mean_and_sd(HMC(100000,L,eps)[3],1,100000)[1],\
#         "Expected error =", mean_and_sd(HMC(100000, L, eps)[4],1,100000)[0],\
#         "Standardised standard deviation of error=", mean_and_sd(HMC(100000,L,eps)[4],1,100000)[1],\
#         "Acceptance ratio =" ,HMC(100000, L, eps)[5])

# # Store the results from running the RMHMC alg
results = HMC(n=10000, L= L, eps = eps)
x_anim = np.array(results[6][:,1])
y_anim = np.array(results[6][:,0])
# print("x_anim=", x_anim)
# print("y_anim=", y_anim)
# stride = 20
# x_anim = x_anim[::stride]
# y_anim = y_anim[::stride]

# Setting up the plot for the dynamics
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(0,L)
fig.supxlabel("Leapfrog step")
ax.set_ylim(-2,2)
fig.supylabel("Value")
ax.set_title("x dynamics")
# ax.scatter(x_anim, y_anim)
# fig.savefig("HMC_ani_set.png")
trace, = ax.plot([],[])
current_plot, = ax.plot([],[]) 

# Functions for the dynamics
def init():
    trace.set_data([],[])
    current_plot.set_data([],[])
    trace.set_color('blue')
    current_plot.set_color('green')
    return trace, current_plot
def update(frame):
    trace_x = x_anim[:frame+1]
    trace_y = y_anim[:frame+1]
    trace.set_data(trace_x, trace_y)
    current_x = [x_anim[frame]]
    current_y = [y_anim[frame]]
    current_plot.set_data(current_x, current_y)
    return trace, current_plot

animate_x = ani.FuncAnimation(fig, update, frames=len(x_anim), init_func=init, blit=False, interval=20, repeat=False)
fig.canvas.manager.window.attributes('-topmost', 1)
animate_x.save("HMC_ani.gif", writer = 'pillow')

# # Plot the potential
# fig, ax = plt.subplots(figsize=(10,10))
# x_vals_raw = results[0]
# x_vals = x_vals_raw[math.ceil(len(x_vals_raw)*0.1):]
# V_vals_raw = []
# for i in range(len(x_vals_raw)):
#     V_vals_raw.append(V(x_vals_raw[i],k))
# V_vals = V_vals_raw[math.ceil(len(V_vals_raw)*0.1):]
# ax.set_xlim(-2,2)
# fig.supxlabel("x")
# ax.set_ylim(-2,2)
# fig.supylabel("V(x)")
# ax.set_title("Harmonic potential from Metropolis")
# ax.scatter(x_vals, V_vals)
# fig.savefig("x_HMC.png")

