#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:16:39 2025

@author: chenqian
"""

import numpy as np 
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt 
import matplotlib 
from matplotlib.collections import LineCollection # For multicolored lines plots 
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist 
import patsy # for comparison
import KS_data_distribution as KS

number_of_initials = 500
observed_time_step = 1e-3 
dt = 1e-4 
record_time_step = int(observed_time_step / dt) 
N_par = 50 

d = 4
        
if d == 2: 
    omega = 2.0
    epsilon = 1e-2 
    T = 0.2
    N_par = 50
    eta = 0.01
    with open('KS_2D_SDE_Par_' + str(N_par) + '_eta_' + str(eta) + '_reg_' + str(epsilon) + '_t_' + str(T) + '_Chi_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        sigma = np.load(f) # load the noise level for the stochastic particle trajectories 
        epsilon = np.load(f) # load the regularization parameter    
        observed_time_step = np.load(f) # load the time step for the trajectory data 
        BIG_noise = np.load(f) # load the noise for the random processes 
        
    BIG_X = BIG_Data[:,:,:,0,None]; BIG_Y = BIG_Data[:,:,:,1,None] 
    
    N = np.shape(BIG_Data)[2]   # number of particles 
    L = np.shape(BIG_Data)[1]   # How many time steps of the particle trajectories are recorded including the initial position 
    M = np.shape(BIG_Data)[0]   # Number of initial data  
    # To get the interval of domain [a, b] from the norm of the pairwise difference of data 
    a_min = []; b_max = []
      

    for s in range(np.shape(BIG_Data)[0]): # 2-norm 
        W_x = np.reshape(BIG_X[s], [np.shape(BIG_X)[1], np.shape(BIG_X)[2]])
        W_y = np.reshape(BIG_Y[s], [np.shape(BIG_Y)[1], np.shape(BIG_Y)[2]])
        square_pairwise_diff_x = np.power(np.diff(W_x), 2)
        square_pairwise_diff_y = np.power(np.diff(W_y), 2)
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y)
        
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
    # Use the data-defined interval to do function approximation 
    a_min = np.min(a_min); b_max = np.max(b_max)

if d == 4: 
    number_of_initials = 500
    omega = 2.0 
    epsilon = 1e-2
    # N_par = 50 
    eta = 0.01 
    with open('KS_4D_SDE_Par_' + str(N_par) + '_eta_' + str(eta) + '_reg_' + str(epsilon) + '_observed_time_step_' + str(observed_time_step) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        sigma = np.load(f) # load the noise level for the stochastic particle trajectories 
        epsilon = np.load(f) # load the regularization parameter    
        observed_time_step = np.load(f) # load the time step for the trajectory data 
        BIG_Noise = np.load(f) # load the noise of the random processes 
    
    BIG_X = BIG_Data[:,:,:, 0, None]; BIG_Y = BIG_Data[:,:,:, 1, None] 
    BIG_Z = BIG_Data[:,:,:, 2, None]; BIG_W = BIG_Data[:,:,:, 3, None] 
    
    N = np.shape(BIG_Data)[2]   # number of particles 
    L = np.shape(BIG_Data)[1]   # How many time steps of the particle trajectories are recorded
    M = np.shape(BIG_Data)[0]   # number of initial data
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
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y \
                                    + square_pairwise_diff_z + square_pairwise_diff_w)
    
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
    # Use the data-defined interval to do function approximation 
    a_min = np.min(a_min); b_max = np.max(b_max)
    
# TO create B-spline basis functions 
these_knots = np.linspace(a_min, b_max, 15) # fix knots number for 10, 20, 40 ... 

# If we want to make nonuniform grid points 
  
# these_knots_1 = np.linspace(a_min, 0.2, 6) 
# these_knots_2 = np.linspace(0.2, b_max, 10)
# these_knots = np.concatenate([these_knots_1[:-1], these_knots_2], axis = 0) 
    

# nodes = np.linspace(0., 1., 200) # nodes for generating BSpline basis functions 
nodes = np.linspace(a_min, b_max, 200)
number_of_nodes = np.size(nodes)
Truncated_point = int(number_of_nodes / 100)
nodes_1 = np.copy(nodes) 
basis_funcs = patsy.bs(nodes, knots = these_knots[:-1], degree = 3, lower_bound = these_knots[0], upper_bound = these_knots[-1])
num_of_interp_funcs = np.shape(basis_funcs)[1]


Interp_func = []
for m in range(num_of_interp_funcs):
    interp_func = interp1d(nodes, basis_funcs[:,m], fill_value = 'extrapolate')
    Interp_func.append(interp_func)

    
if d == 2:
    W = omega / (np.power(nodes,2) + epsilon ** 2) / (2 * np.pi)

elif d == 4: 
    W = omega / (np.power(nodes, 4) + epsilon ** 2) / (2 * np.pi ** 2)     
    
    
# Part III: Solve the linear system Ax = b 
# To generate a pairwise distance matrix, we create a zero matrix first 
R = np.zeros([N,N])
PSI = np.zeros([N*d, num_of_interp_funcs])
AA = np.zeros([num_of_interp_funcs, num_of_interp_funcs])
I = np.eye(num_of_interp_funcs)
b = np.zeros([num_of_interp_funcs, 1])

for m in range(number_of_initials): 
    print(m) 
    for l in range(L-1): 
        # To compute the A matrix for Ax = b  
        X = np.reshape(BIG_Data[m][l+1], [N * d, 1]) # X = [x1; y1; z1; x2; y2; z2, ... ] (Nd by 1) vector 
        X_old = np.reshape(BIG_Data[m][l],[N * d, 1]) # Nd by 1 vector 
        X_tild = np.reshape(BIG_Data[m][l+1], [N,d]) # (N by d) matrix with each column representing each axis for each particle  
        # Create the pairwise distance matrix at each time step for each initial data 
        R[:,:] = 0.0; 
        R[np.triu_indices(N,1)] = pdist(X_tild)
        R += R.T     # N by N distance matrix 
        R_hat = np.kron(R, np.ones([d,1])) # Nd by N distance matrix 
        XX =  np.tile(X, [1,N]) # To repeat the column of X to create an Nd by N matrix 
        
        X_tild_tild = np.tile(X_tild.T, [N,1]) # To repeat the row of X_tild to create an Nd by N matrix 
        
        Delta_X = X_tild_tild - XX # pairwise diff for each component of X  
                                   # [[x1 - x1; x1 - x2; ... ; x1 - xN],[x2 - x1; x2 - x2; ...  ]]    
        for eta in range(num_of_interp_funcs):
            Psi = Interp_func[eta](R_hat) / N # Nd * N matrix   
            Psi_vec = Psi * Delta_X 
            PSI[:,eta] = np.sum(Psi_vec, axis = 1)
        AA += np.matmul(PSI.T, PSI) # n by n matrix 
        
        
        dist_term = X - X_old # Nd by 1 vector: X(t_{l+1}) - X(t_{l}) 
        b += np.matmul(PSI.T, dist_term) 
        

b /= (number_of_initials * N)        
AA /= (number_of_initials * N); AA *= observed_time_step 
# alpha = np.linalg.solve(AA + epsilon * I, b)
alpha = np.linalg.solve(AA, b)

 
Learned_Kernel = np.zeros([number_of_nodes,1])
Learned_Kernel[:,0] = np.sum(alpha.T * basis_funcs, axis = 1)


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

density, bins, discrete_density = KS.data_distribution(a_min, b_max, BIG_Data)
interp_density = density(nodes)
ax1.plot(nodes, W, label = 'Cut off kernel', markersize = 7, color = 'black', linestyle = 'solid')
ax1.plot(nodes, Learned_Kernel, label = 'Learned kernel', markersize = 5, color = 'b', linestyle = '--', dashes = (5,5) )
ax2.plot(nodes, interp_density, color = [0.85, 0.325, 0.098], alpha = 0.2)
ax2.fill_between(nodes, interp_density, color = [0.85, 0.325, 0.098], alpha = 0.2, zorder = 1)
ax1.set_xlabel('nodes', fontsize = 16)
ax1.set_ylabel('Kernel values', fontsize = 16)
ax2.set_ylabel('data density', color = 'g', fontsize = 16)
ax1.set_title(('Kernel Comparison with \u03C9 =' + str(omega)), fontsize = 16)
ax1.legend()

""" 
# Create inset
axins = inset_axes(ax1, width="40%", height="30%", loc="upper right")
axins.plot(nodes, W, label = 'Cut off kernel', markersize = 7, color = 'black', linestyle = 'solid')
axins.plot(nodes, Learned_Kernel, label = 'Learned kernel', markersize = 5, color = 'b', linestyle = '--', dashes = (5,5) )


# Set zoomed region
axins.set_xlim(0, 0.2)
axins.set_ylim(50000, 70000)
axins.set_title("Zoom In", fontsize=10)
""" 
plt.show()


# save data for reconstruction of particle trajectories 
knot_num = np.size(these_knots)
"""         
if d == 2:
    # path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    with open('KS_SDE_2D_Par_reg_' + str(epsilon) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, BIG_Noise) # save the noise for the particle trajectories 
        np.save(f, alpha)
        np.save(f, these_knots)         
        # np.save(f, h) 
        # np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min) 
        np.save(f, b_max)

elif d == 4 :
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    # with open(path +'KS_4D_Particle_50_Learned_reg_' + str(epsilon) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
    # with open(path +'KS_4D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
     
    with open('KS_SDE_4D_Par_reg_' + str(epsilon) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, BIG_Noise) # save the noise for the particle trajectories 
        np.save(f, alpha)
        np.save(f, these_knots)         
        # np.save(f, h) 
        # np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min) 
        np.save(f, b_max) 
""" 
     
   
    
    
