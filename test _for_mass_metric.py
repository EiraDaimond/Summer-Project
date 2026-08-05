import numpy as np

x = 1.808012119454852
k = 1
lam = 1
d = 1
p = -3.822302833566044
# Define the metric tensor (second derivative of the potential term)
def G(x,k,lam):
    return k + 3*lam*x**2
        
# Defin M (including delta).... to be used to avoid division by 0 errors
def M(x,k, lam, d):
    return np.sqrt(abs(G(x,k,lam)**2+d**2))

# M_vals = []
# for i in range(100):
#     x = np.random.normal(0, 1)
#     print("x is", x)
#     M_vals.append(M(x, -1, 1, 0.01))
# print(M_vals)

# p_star = p_current - eps\
#                         *(k*x_star + lam*x_star**3\
#                                              + 0.5*p_guess**2*(-6*lam*x_star)\
#                                              + 0.5*abs(-6*lam*x_star)/M(x_star,k,lam, d))
# p_vals = []
# for i in range(100):
#     p = np.random.normal(0,np.sqrt(M(x,k, lam, d)))
#     p_vals.append(p)
# print(p_vals)

# p_star = 6 - 0.001*\
#                 ((k*x+ lam*x**3 \
#                         + 0.5*p**2*(G(x,k,lam))*6*lam*x)/(M(x, k, lam, d))**2 \
#                             + 0.5*(6*lam*x)\
#                                     /M(x,k, lam, d))
# print(p_star)

p_current = 6
p_guess = 6
p_star = 0
count = 0
while count < 100:
                #("Count=",count)
                count = count +1
                # if count == max_iter:
                    # print("Hit max iterations")
                #("Middle step iter[",l,"] p_star is :", p_star)
                #("Using x_star:", x_star)
                p_star = p_current - 0.001*\
                                 ((k*x + lam*x**3 \
                                     + 0.5*p_guess**2*(G(x,k,lam))*6*lam*x)/(M(x, k, lam, d))**2 \
                                + 0.5*(6*lam*x)\
                                      /M(x,k, lam, d))
                if abs(p_star - p_guess) < 1e-12:
                                            #("STOPPING WHILE LOOP for p")
                    break 
                else:
                                            p_guess = p_star
print(p_star)