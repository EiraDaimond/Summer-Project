import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math

# Define the variables to be used
m = 1.0
L = 1000
eps = 0.001
k = 1
lam = 1

# Define the anharmonic potential term
def an_V(x,k,lam):
    if k >0:
        an_V = 0.25*lam**2*x**4 + 0.5*lam*k*x**2
    else:
        an_V = 0.25*lam**2*x**4 - 0.5*lam*k*x**2
    return an_V

# Define the anharmonic Hamiltonian
def an_H(x, p,m):
    return an_V(x,k,lam) + 0.5*p**2/m
  
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
def an_HMC_alg(n, L, eps):
    '''
    -Carry out the HMC algorithm using the leafrog method to generate x values. 
    -Simultaenously compute and store KE, PE, exp(-delH).
    -Calculate acceptance ratio. 
    -Another way to check that the algorithm is working correctly is to check 
    reversibility with each trajectory, so we also include this test.
    '''
    # Initialise the x values, KE values, errors, and the accepted values lists
    x = [1]
    KE_vals = []
    errors = []
    exps_delH = []
    accepted = []
    for_animation_x = np.zeros((L,2),dtype=float)
    # Start the loop to generate x values
    for t in range(n+1):
        # Draw the momentum from a Normal distribution
        p = np.random.normal(0, m**0.5)
        print("p=", p)
        # Compute the first leapfrog step
        p_star = p - 0.5*eps*(k*x[t] + lam*x[t]**3)
        print("p_star before leapfrog=",p_star)
        x_star = x[t] + eps*p_star/m
        # Compute (x*, - p*) using L leapfrog steps of size eps
        for l in range(1, L):
            p_star = p_star - eps*(k*x_star + lam*x_star**3)
            #print("p_star in step", l, p_star)
            x_star = x_star + eps*p_star/m
            print("x_star", x_star)
            print("change for x", eps*p_star/m)
            for_animation_x[l] = [x_star, l]
        # Compute the final step of the leapfrog method
        p_star = p_star - 0.5*eps*(k*x_star + lam*x_star**3)
        # Compute the acceptance ratio
        r = np.exp(-an_H(x_star, p_star,m ) + an_H(x[t], p,m))
        # Draw W from a Uniform distribution
        W = np.random.uniform(0, 1)
        # Carry out the Metropolis test
        if W <= min(1, r):
            x.append(x_star)
            accepted.append(x_star)
        else:
            x.append(x[t])
        # Compute the KE terms for this trajectory and append to list
        KE = 0.5*p_star**2/m
        KE_vals.append(KE)
        # Calculate exp(-delH) terms
        exp_minus_del_H_ = np.exp(an_H(x_star,p_star,m) - an_H(x[t], p,m))
        exps_delH.append(exp_minus_del_H_)
        # Check reversibility
        p_star = p_star + 0.5*eps*(k*x[t] + lam*x[t]**3)
        x_star = x_star - eps*p_star/m
        for l in range(1, L):
            p_star = p_star + eps*(k*x_star + lam*x_star**3)
            x_star = x_star - eps*p_star/m
        p_backwards = p_star + 0.5*eps*(k*x_star + lam*x_star**3)
        error = (p_backwards - p)
        errors.append(error)
    # Compute acceptance ratio
    acc_rat = (len(accepted)/len(x))*100    
    return x, KE_vals, exps_delH, errors, acc_rat, for_animation_x

# Find the expected value and standard deviation of x
def mean_and_sd(x,m ,n):
    '''
    Given a list of x values, compute the expected value
      and standardised standard deviation (rejecting burn-in).
    '''
    length = len(x)
    values_to_use = x[math.ceil(length/10):]
    stand_sd = m**0.5/(n-1)**0.5
    return np.mean(values_to_use), stand_sd*np.std(values_to_use)

# print("Expected x =", mean_and_sd(an_HMC_alg(100000,L,eps)[0],1,100000)[0],\
#       "Standardised standard deviation of x=", mean_and_sd(an_HMC_alg(100000,L,eps)[0],1,100000)[1],\
#        "Expected KE = ",mean_and_sd(an_HMC_alg(100000, L, eps)[1],1,100000)[0], \
#        "Standardised standard deviation of KE = ", mean_and_sd(an_HMC_alg(100000, L, eps)[1],1,100000)[1],\
#         "Expected exp(-delH)= " ,mean_and_sd(an_HMC_alg(100000,L,eps)[2],1,100000)[0],\
#         "Standardised standard deviation of exp(-delH) = ", mean_and_sd(an_HMC_alg(100000,L,eps)[2],1,100000)[1],\
#         "Expected error =", mean_and_sd(an_HMC_alg(100000, L, eps)[3],1,100000)[0],\
#         "Standardised standard deviation of error=", mean_and_sd(an_HMC_alg(100000,L,eps)[3],1,100000)[1],\
#         "Acceptance ratio =" ,an_HMC_alg(100000, L, eps)[4])

# # Store the results from running the RMHMC alg
results = an_HMC_alg(n=1, L= L, eps = eps)
print("x_list out of alg", results[0][len(L)*0.1:])
print("for_animation list", results[5][len(L)*0.1:])
x_anim = np.array(results[5])[:,1]
print("x_anim", x_anim)
y_anim = np.array(results[5])[:,0]
print("y_anim", y_anim)
# stride = 20
# x_anim = x_anim[::stride]
# y_anim = y_anim[::stride]

# # Setting up the plot for the dynamics
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(0,9)
# fig.supxlabel("Leapfrog step")
# ax.set_ylim(-2,2)
# fig.supylabel("Value")
# ax.set_title("x dynamics")
# ax.plot(x_anim, y_anim)
# fig.savefig("anHMC_ani_set.png")
# trace, = ax.plot([],[])
# current_plot, = ax.plot([],[]) 

# # Functions for the dynamics
# def init():
#     trace.set_data([],[])
#     current_plot.set_data([],[])
#     trace.set_color('blue')
#     current_plot.set_color('green')
#     return trace, current_plot
# def update(frame):
#     trace_x = x_anim[:frame+1]
#     trace_y = y_anim[:frame+1]
#     trace.set_data(trace_x, trace_y)
#     current_x = [x_anim[frame]]
#     current_y = [y_anim[frame]]
#     current_plot.set_data(current_x, current_y)
#     return trace, current_plot

# animate_x = ani.FuncAnimation(fig, update, frames=len(x_anim), init_func=init, blit=False, interval=20, repeat=False)
# fig.canvas.manager.window.attributes('-topmost', 1)
# animate_x.save("HMC_animate_x.gif", writer = 'pillow')

# Plot the potential
fig, ax = plt.subplots(figsize=(10,10))
x_vals = results[0][len(x_vals)*0.1:]
V_vals = []
for i in range(len(x_vals)):
    V_vals.append(an_V(x_vals[i],k,lam))
ax.set_xlim(-2,2)
fig.supxlabel("x")
ax.set_ylim(-2,2)
fig.supylabel("V(x)")
ax.set_title("Anharmonic potential from Metropolis")
ax.plot(x_vals, V_vals )
fig.savefig("x_anHMC.png")


