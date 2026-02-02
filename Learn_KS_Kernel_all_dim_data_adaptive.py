#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:05:06 2024

@author: chenqian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:39:45 2024

@author: chenqian
"""

import numpy as np 
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist 
import patsy # for comparison
import KS_data_distribution as KS


number_of_initials = 500 # M = number_of_initials
epsilon = 0

d = 4
 
if d == 1:
    Chi = 0.55
    r_c = 0.01
    
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    with open(path +'KS_1D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
    # with open(path +'KS_Particle_cut_off_50_t_0.05_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_X = np.load(f) 
        h = np.load(f)
        tau = np.load(f)  
        observed_time_step = np.load(f)
        BIG_Derivative = np.load(f)

    N = np.size(BIG_X[0][0]) # number of particles in each data 
    L = np.shape(BIG_X)[1]   # number of time steps in the data
    BIG_Data = BIG_X

    # To get the interval of domain [a,b] from the pairwise difference of data 
    a_min = []; b_max = []

    for s in range(np.shape(BIG_X)[0]):
        W = np.reshape(BIG_X[s], [np.shape(BIG_X)[1], np.shape(BIG_X)[2]])
        abs_pairwise_diff = np.abs(np.diff(W))        
        a_min.append(np.min(abs_pairwise_diff)); b_max.append(np.max(abs_pairwise_diff))
        
    # Use the data-defined interval to do function approximation 
    a_min = np.min(a_min); b_max = np.max(b_max)
       
if d == 2: 
    r_c = 0.01 
    omega = 2.0
    
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    with open(path + 'KS_2D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        BIG_X = np.load(f) 
        BIG_Y = np.load(f)
        h = np.load(f)
        tau = np.load(f)  
        observed_time_step = np.load(f)
        BIG_Derivative = np.load(f) 
        
    N = np.shape(BIG_X)[2]   # number of particles 
    L = np.shape(BIG_X)[1]   # How many time steps of the particle trajectories are recorded
    
    # To get the interval of domain [a, b] from the norm of the pairwise difference of data 
    a_min = []; b_max = []
    for s in range(np.shape(BIG_X)[0]): # 2-norm 
        W_x = np.reshape(BIG_X[s], [np.shape(BIG_X)[1], np.shape(BIG_X)[2]])
        W_y = np.reshape(BIG_Y[s], [np.shape(BIG_Y)[1], np.shape(BIG_Y)[2]])
        square_pairwise_diff_x = np.power(np.diff(W_x), 2)
        square_pairwise_diff_y = np.power(np.diff(W_y), 2)
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y)
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
    # Use the data-defined interval to do function approximation 
    a_min = np.min(a_min); b_max = np.max(b_max)

if d == 3: 
    r_c = 0.01
    omega = 2.0 
    
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    
    with open(path + 'KS_3D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        h = np.load(f) 
        tau = np.load(f) 
        observed_time_step = np.load(f) 
        BIG_Derivative = np.load(f) 
    
    BIG_X = BIG_Data[:,:,:, 0, None]
    BIG_Y = BIG_Data[:,:,:, 1, None]
    BIG_Z = BIG_Data[:,:,:, 2, None]
        
    N = np.shape(BIG_X)[2]   # number of particles 
    L = np.shape(BIG_X)[1]   # How many time steps of the particle trajectories are recorded
    
    # To get the interval of domain [a, b] from the norm of the pairwise difference of data 
    a_min = []; b_max = []
      
    for s in range(np.shape(BIG_X)[0]): # 2-norm 
        W_x = np.reshape(BIG_X[s], [np.shape(BIG_X)[1], np.shape(BIG_X)[2]])
        W_y = np.reshape(BIG_Y[s], [np.shape(BIG_Y)[1], np.shape(BIG_Y)[2]])
        W_z = np.reshape(BIG_Z[s], [np.shape(BIG_Z)[1], np.shape(BIG_Z)[2]])
        square_pairwise_diff_x = np.power(np.diff(W_x), 2)
        square_pairwise_diff_y = np.power(np.diff(W_y), 2)
        square_pairwise_diff_z = np.power(np.diff(W_z), 2)
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y + square_pairwise_diff_z)
        
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
    # Use the data-defined interval to do function approximation 
    a_min = np.min(a_min); b_max = np.max(b_max)

if d == 4: 
    r_c = 0.05
    omega = 1.0 
    
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    
    with open(path + 'KS_4D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        h = np.load(f) 
        tau = np.load(f) 
        omega = np.load(f) 
        observed_time_step = np.load(f) 
        BIG_Derivative = np.load(f) 
    
    BIG_X = BIG_Data[:,:,:, 0, None]
    BIG_Y = BIG_Data[:,:,:, 1, None]
    BIG_Z = BIG_Data[:,:,:, 2, None]
    BIG_W = BIG_Data[:,:,:, 3, None]
    
    N = np.shape(BIG_X)[2]   # number of particles 
    L = np.shape(BIG_X)[1]   # How many time steps of the particle trajectories are recorded
    
    # To get the interval of domain [a, b] from the norm of the pairwise difference of data 
    a_min = []; b_max = []
      

    for s in range(np.shape(BIG_X)[0]): # 2-norm 
        W_x = np.reshape(BIG_X[s], [np.shape(BIG_X)[1], np.shape(BIG_X)[2]])
        W_y = np.reshape(BIG_Y[s], [np.shape(BIG_Y)[1], np.shape(BIG_Y)[2]])
        W_z = np.reshape(BIG_Z[s], [np.shape(BIG_Z)[1], np.shape(BIG_Z)[2]])
        W_w = np.reshape(BIG_W[s], [np.shape(BIG_W)[1], np.shape(BIG_W)[2]])
        square_pairwise_diff_x = np.power(np.diff(W_x), 2)
        square_pairwise_diff_y = np.power(np.diff(W_y), 2)
        square_pairwise_diff_z = np.power(np.diff(W_z), 2)
        square_pairwise_diff_w = np.power(np.diff(W_w), 2)
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y + square_pairwise_diff_z + square_pairwise_diff_w)     
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
    # Use the data-defined interval to do function approximation 
    a_min = np.min(a_min); b_max = np.max(b_max)

# For computing the linear system: Ax = b at each iteration, we need to preprocess the data first
# to store the pairwise difference and pairwise distance of each point from each initial data distribution   
BIG_R = np.zeros([number_of_initials, L, N, N]) # M * L * N * N numpy array 
BIG_R_hat = np.zeros([number_of_initials, L, N * d, N]) # M * L * (Nd) * N numpy array 
BIG_Delta_X = np.zeros([number_of_initials, L, N *d, N]) # M * L * (Nd) * N numpy array 
BIG_LHS = np.zeros([number_of_initials, L, N * d, 1]) # M * L * (Nd) * 1 numpy array 
# To store all the data we need in advance to the adpative algorithm for constructing the learned kernel 
for m in range(number_of_initials): 
    for l in range(L-1): 
        # To compute the A matrix for Ax = b  
        X = np.reshape(BIG_Data[m][l+1], [N * d, 1]) # X = [x1; y1; z1; x2; y2; z2, ... ] (Nd by 1) vector 
        X_old = np.reshape(BIG_Data[m][l],[N * d, 1]) # Nd by 1 vector 
        X_tild = BIG_Data[m][l+1] # check !!!
        
        # Create the pairwise distance matrix at each time step for each initial data 
        BIG_R[m,l][np.triu_indices(N,1)] = pdist(X_tild)
        BIG_R[m,l] += BIG_R[m,l].T
        BIG_R_hat[m,l] = np.kron(BIG_R[m,l], np.ones([d,1]))
        
        XX =  np.tile(X, [1,N]) # To repeat the column of X to create an Nd by N matrix 
        X_tild_tild = np.tile(X_tild.T, [N,1]) # To repeat the row of X_tild to create an Nd by N matrix 

        BIG_Delta_X[m,l] = X_tild_tild - XX # pairwise diff for each component of X 
        # Delta_X = X_tild_tild - XX # pairwise diff for each component of X  
        
        
        # To construct b vector 
        K_h_0 = np.exp(-np.power(BIG_R[m,l] / h, 2)) / np.power(h * np.sqrt(np.pi), d) # N by N Gaussian kernel matrix
        K_h = np.exp(- np.power(BIG_R_hat[m,l] / h, 2)) / np.power((h * np.sqrt(np.pi)), d) # Nd by N Gaussian kernel matrix 
        Sum_K_h_0 = np.sum(K_h_0, axis = 0)   # N by 1 vector; summation along axis 0 
        Sum_K_h_1 = np.sum(K_h[:,:,None], axis = 1) # Nd by 1 vector; summation along axis 1 
        
        first_moment = BIG_Delta_X[m,l] * K_h # Nd by N matrix  
                
        dist_term = np.reshape(BIG_Derivative[m][l], [N * d, 1], order = 'C') # make it x1_dot, y1_dot ...
        nonlocal_grad1 = (2 / np.power(h,2)) * np.sum((first_moment / Sum_K_h_0)[:,:,None], axis = 1)  # Nd by 1 vector 
        
        nonlocal_grad2 = (2 / np.power(h,2)) * np.sum(first_moment[:,:,None], axis = 1) / Sum_K_h_1 # Nd by 1 vector 
        BIG_LHS[m,l] = dist_term + nonlocal_grad1 + nonlocal_grad2 


nodes = np.linspace(a_min, b_max, 200)
number_of_nodes = np.size(nodes)

if d == 1: 
    # Cut-off Kernel 
    functions = [0, lambda x:  2 * Chi / (r_c ** 2)  ,lambda x: 2 * Chi / np.power(x, 2) ] 
    W = np.piecewise(nodes, [nodes == 0, (nodes <= r_c) * (nodes > 0 ), nodes > r_c], functions) 

elif d == 2:
    # Cut-off Kernel 
    functions = [0, omega / (r_c ** 2) / (2 * np.pi), lambda x: omega / np.power(x,2) / (2 * np.pi)]
    W = np.piecewise(nodes, [nodes == 0, (nodes > 0) * (nodes <= r_c), nodes > r_c], functions)

elif d == 3: 
    # Cut-off Kernel 
    functions = [0, omega / (r_c ** 3) / (4 * np.pi), lambda x: omega / np.power(x,3) / (4 * np.pi)]
    W = np.piecewise(nodes, [nodes == 0, (nodes > 0) * (nodes <= r_c), nodes > r_c], functions)

elif d == 4: 
    # Cut-off Kernel 
    functions = [0, omega / (r_c ** 4) / (2 * np.pi ** 2), lambda x: omega / np.power(x , 4) / (2 * np.pi ** 2)]
    W = np.piecewise(nodes, [nodes == 0, (nodes > 0) * (nodes <= r_c), nodes > r_c], functions)

# Using adaptive algorithm to generate the knots 
maxIter = 6
tol = 1e-2
P_init = np.array([a_min, b_max])  # initial partition of the interval [a_min, b_max]
I_init = np.ones([1])          # initial associated set of indicators 
total = number_of_initials * L * N 

# Data distribution 
interp_density, bins, discrete_density = KS.data_distribution(a_min, b_max, BIG_Data)

basis_old = patsy.bs(nodes, knots = P_init[:-1], degree = 3, lower_bound = P_init[0], upper_bound = P_init[-1])
num_interp_old = np.shape(basis_old)[1]

Interp_func_old = []
for m in range(num_interp_old):
    interp_func = interp1d(nodes, basis_old[:,m], fill_value = 'extrapolate')
    Interp_func_old.append(interp_func)

PSI_old = np.zeros([N * d, num_interp_old]) # Nd * n_old
AA_old  = np.zeros([num_interp_old, num_interp_old]) # n_old * n_old 
Id_old = np.eye(num_interp_old) # n_old * n_old 
b_old = np.zeros([num_interp_old, 1]) # n_old * 1

for m in range(number_of_initials): 
    for l in range(L-1):
        for eta in range(num_interp_old):
            Psi_old = Interp_func_old[eta](BIG_R_hat[m,l]) / N # Nd * N matrix; M = num_of_initials   
            Psi_vec_old = Psi_old * BIG_Delta_X[m,l] 
            PSI_old[:,eta] = np.sum(Psi_vec_old, axis = 1) # check carefully with array (high dimension): Nd * 1 vector 
                   
        AA_old += np.matmul(PSI_old.T, PSI_old) # n by n matrix 
        b_old += np.matmul(PSI_old.T, BIG_LHS[m,l])  
b_old /= total 
AA_old /= total

alpha_old = np.linalg.solve(AA_old + epsilon * Id_old, b_old) 
        
def adaptive_basis_funcs(P_init, I_init, BIG_R, BIG_R_hat, BIG_Delta_X, BIG_LHS, alpha, Interp_func, tol = 1e-2, maxIter = 6, epsilon = 0):
    # return the basis functions and the coefficients; 
    alpha_old = np.copy(alpha) 
    Interp_func_old = np.copy(Interp_func)
    num_interp_old = np.size(Interp_func_old)
    P_old = np.copy(P_init) 
    I_old = np.copy(I_init)
    for j in range(maxIter):
        print(j) 
        # Refined knot points        
        P = np.append(P_old, (P_old[0:-1] + P_old[1:]) / 2)
        P = np.sort(P)
        
        basis_new = patsy.bs(nodes, knots = P[:-1], degree = 3, lower_bound = P[0], upper_bound = P[-1])
        num_interp_new = np.shape(basis_new)[1]
        
        Interp_func_new = [] 
        for m in range(num_interp_new): 
            interp_func = interp1d(nodes, basis_new[:,m], fill_value = 'extrapolate')
            Interp_func_new.append(interp_func) 
        
        PSI_new = np.zeros([N * d, num_interp_new]) # Nd * n_new 
        AA_new = np.zeros([num_interp_new, num_interp_new]) # n_new * n_new 
        Id_new = np.eye(num_interp_new) # n_new * n_new 
        b_new = np.zeros([num_interp_new, 1]) # n_new * 1 
            
        # Part III: Solve the linear system Ax = b at each iteration 
        for m in range(number_of_initials): 
            for l in range(L-1):
                for eta in range(num_interp_new):
                    Psi_new = Interp_func_new[eta](BIG_R_hat[m,l]) / N # Nd * N matrix 
                    Psi_vec_new = Psi_new * BIG_Delta_X[m,l]
                    PSI_new[:,eta] = np.sum(Psi_vec_new, axis = 1) # Nd * 1 vector 
                        
                    
                AA_new += np.matmul(PSI_new.T, PSI_new) # n by n matrix 
                    
                b_new += np.matmul(PSI_new.T, BIG_LHS[m,l])
        
        AA_new /= total 
        b_new /= total 
        alpha_new = np.linalg.solve(AA_new + epsilon * Id_new, b_new)
            
        Phi_new = np.zeros([num_interp_new, np.size(P)]) # num_interp_new * np.size(P) matrix 
        Phi_old = np.zeros([num_interp_old, np.size(P)]) # num_interp_old * np.size(P) matrix 
        for eta in range(num_interp_new):
            Phi_new[eta, :] = Interp_func_new[eta](P) # old index is the new point 
            
        for eta in range(num_interp_old):
            Phi_old[eta, :] = Interp_func_old[eta](P) # 
            
        Phi_old = np.matmul(alpha_old.T, Phi_old)
        Phi_new = np.matmul(alpha_new.T, Phi_new) 
            
        # Apply midpoint rule to compute the L2 error of Phi_{j-1} - Phi_{j}
        for k in range(np.size(I_old)):  # associated set of indicators 
                
            if np.abs((Phi_new[0,2 * k + 1] - Phi_old[0,2 * k + 1]) / Phi_old[0, 2 * k + 1]) < tol: 
                    
                I_old[k] = 0 
            else: 
                I_old[k] = 1 
            
        if np.sum(I_old[I_old == 0]) == np.size(I_old):
                
            return P # , Interp_func_new  
                    
        else: 
            index = np.where(I_old == 0)[0]
            P_old = np.delete(P, 2 * index + 1)
                
            I_old = np.ones([np.size(P_old) - 1])
                
            alpha_old = np.copy(alpha_new) 
            Interp_func_old = np.copy(Interp_func_new) 
            num_interp_old = num_interp_new 
    return P_old # adaptive knot positions 

P =  adaptive_basis_funcs(P_init, I_init, BIG_R, BIG_R_hat, BIG_Delta_X, BIG_LHS, alpha_old, Interp_func_old)

basis_funcs = patsy.bs(nodes, knots = P[0:-1], degree = 3, lower_bound = P[0], upper_bound = P[-1])

num_of_interp = np.shape(basis_funcs)[1]

Interp_func = []
for m in range(num_of_interp):
    interp_func = interp1d(nodes, basis_funcs[:,m], fill_value = 'extrapolate')
    Interp_func.append(interp_func)

PSI = np.zeros([N * d, num_of_interp]) # Nd * n_old
AA  = np.zeros([num_of_interp, num_of_interp]) # n_old * n_old 
Id = np.eye(num_of_interp) # n_old * n_old 
b = np.zeros([num_of_interp, 1]) # n_old * 1

for m in range(number_of_initials): 
    for l in range(L-1):
        for eta in range(num_of_interp):
            Psi = Interp_func[eta](BIG_R_hat[m,l]) / N # Nd * N matrix; M = num_of_initials   
            Psi_vec = Psi * BIG_Delta_X[m,l] 
            PSI[:,eta] = np.sum(Psi_vec, axis = 1) # check carefully with array (high dimension): Nd * 1 vector 
            
        
        AA += np.matmul(PSI.T, PSI) # n by n matrix 
        
        
        b += np.matmul(PSI.T, BIG_LHS[m,l])  
b /= total 
AA /= total

alpha = np.linalg.solve(AA + epsilon * Id, b) 


Learned_Kernel = np.zeros([number_of_nodes, 1])
Learned_Kernel[:,0] = np.sum(alpha.T * basis_funcs, axis = 1)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()


ax1.plot(nodes, W, label = 'Cut off kernel', markersize = 7, color = 'black', linestyle = 'solid')
ax1.plot(nodes, Learned_Kernel, label = 'Learned kernel', markersize = 5, color = 'b', linestyle = '--', dashes = (5,5) )
ax2.plot(nodes, interp_density(nodes), color = [0.85, 0.325, 0.098], alpha = 0.2)
ax2.fill_between(nodes, interp_density(nodes), color = [0.85, 0.325, 0.098], alpha = 0.2, zorder = 1)
ax1.set_xlabel('nodes')
ax1.set_ylabel('Kernel values')
ax2.set_ylabel('data density', color = 'g')
if d == 1: 
    ax1.set_title(('Kernel Comparison with \u03C7 =' + str(Chi)))
else: 
    ax1.set_title(('Kernel Comparison with \u03C9 =' + str(omega)))

ax1.legend()
plt.show()


r_c = 0.01 
if d == 1:
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    with open(path +'KS_1D_Particle_50_Learned_adaptive_cut_off_' + str(r_c) + '_t_0.2_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open(path +'KS_Particle_cut_off_50_t_0.05_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, P)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, Chi)
        np.save(f, observed_time_step) 
        np.save(f, a_min)
        np.save(f, b_max) 
        
elif d == 2:
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    with open(path +'KS_2D_Particle_50_Learned_adaptive_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, P)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min)
        np.save(f, b_max) 
        
elif d == 3 :
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    with open(path +'KS_3D_Particle_50_Learned_adaptive_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, P)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min) 
        np.save(f, b_max) 

elif d == 4 :
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    with open(path +'KS_4D_Particle_50_Learned_adaptive_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, P)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min) 
        np.save(f, b_max) 

        
        
        
        
        
        
        
        