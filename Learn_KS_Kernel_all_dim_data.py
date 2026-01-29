#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:39:45 2024

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

d = 1
if d == 1:
    Chi = 0.55
    path = 'data/'
    r_c = 0.01
    with open(path +'KS_1D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_X = np.load(f) 
        # x = np.load(f)
        h = np.load(f)
        tau = np.load(f)  
        # Chi = np.load(f)
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
    omega = 2.0
    path = 'data/'
    r_c = 0.01
    with open(path + 'KS_2D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        BIG_X = np.load(f) 
        BIG_Y = np.load(f)
        # x = np.load(f)
        h = np.load(f)
        tau = np.load(f)  
        # Chi = np.load(f)
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
    omega = 2.0 
    r_c = 0.01
    path = 'data/'
    with open(path + 'KS_3D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        # BIG_X = np.load(f)
        # BIG_Y = np.load(f) 
        # BIG_Z = np.load(f) 
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
    number_of_initials = 500
    omega = 1.0 
    epsilon = 1e-4
    r_c = 0.05
    path = '/Users/chenqian/Desktop/python/Particle_method_2024/KS_particle_trajectory/'
    # with open(path + 'KS_4D_Particle_50_reg_' + str(epsilon) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
    with open(path + 'KS_4D_Particle_50_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        # BIG_X = np.load(f)
        # BIG_Y = np.load(f) 
        # BIG_Z = np.load(f) 
        h = np.load(f) 
        tau = np.load(f) 
        omega = np.load(f)
        observed_time_step = np.load(f)
        BIG_Derivative = np.load(f)
    BIG_X = BIG_Data[:,:,:, 0, None]; BIG_Y = BIG_Data[:,:,:, 1, None] 
    BIG_Z = BIG_Data[:,:,:, 2, None]; BIG_W = BIG_Data[:,:,:, 3, None] 
    
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
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y \
                                    + square_pairwise_diff_z + square_pairwise_diff_w)
    
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
    # Use the data-defined interval to do function approximation 
    a_min = np.min(a_min); b_max = np.max(b_max)
    
# TO create B-spline basis functions 
these_knots = np.linspace(a_min, b_max, 30) # fix knots number for 10, 20, 40 ... 

# If we want to make nonuniform grid points 
  
# these_knots_1 = np.linspace(a_min, 0.2, 6) 
# these_knots_2 = np.linspace(0.2, b_max, 10)
# these_knots = np.concatenate([these_knots_1[:-1], these_knots_2], axis = 0) 
    

# nodes = np.linspace(0., 1., 200) # nodes for generating BSpline basis functions 
nodes = np.linspace(a_min, b_max, 200)
number_of_nodes = np.size(nodes)
Truncated_point = int(number_of_nodes / 100)
nodes_1 = np.copy(nodes) 
basis_funcs = patsy.bs(nodes, knots=these_knots[:-1], degree = 3, lower_bound = these_knots[0], upper_bound = these_knots[-1])
num_of_interp_funcs = np.shape(basis_funcs)[1]


Interp_func = []
for m in range(num_of_interp_funcs):
    interp_func = interp1d(nodes, basis_funcs[:,m], fill_value = 'extrapolate')
    Interp_func.append(interp_func)

if d == 1: 
    # Cut-off Kernel 
    functions = [0, lambda x: 2 * Chi / (r_c ** 2)  ,lambda x: 2 * Chi / np.power(x, 2) ] 
    W = np.piecewise(nodes, [nodes == 0, (nodes <= r_c) * (nodes > 0 ), nodes > r_c], functions) 

elif d == 2:
    
    # functions = [0, 10000 * omega / (2 * np.pi), lambda x: omega / np.power(x,2) / (2 * np.pi)]
    
    # W = np.piecewise(nodes, [nodes == 0, (nodes > 0) * (nodes <= 0.01), nodes > 0.01], functions)
    
    functions = [0, omega / (r_c ** 2) / (2 * np.pi), lambda x: omega / np.power(x,2) / (2 * np.pi)]

    
    W = np.piecewise(nodes, [nodes == 0, (nodes > 0) * (nodes <= r_c), nodes > r_c], functions)

elif d == 3: 
    
    functions = [0, omega / (r_c ** 3) / (4 * np.pi), lambda x: omega / np.power(x,3) / (4 * np.pi)]

    W = np.piecewise(nodes, [nodes == 0, (nodes > 0) * (nodes <= r_c), nodes > r_c], functions)

elif d == 4: 
    
    functions = [0, omega / (r_c ** 4) / (2 * np.pi ** 2), lambda x: omega / np.power(x,4) / (2 * np.pi ** 2)]
    
    W = np.piecewise(nodes, [nodes == 0, (nodes > 0) * (nodes <= r_c), nodes > r_c], functions)
    # W = 1 / (np.power(nodes, 4) + epsilon) 

# Part III: Solve the linear system Ax = b 
# To generate a pairwise distance matrix, we create a zero matrix first 
R = np.zeros([N,N])
PSI = np.zeros([N*d, num_of_interp_funcs])
AA = np.zeros([num_of_interp_funcs, num_of_interp_funcs])
I = np.eye(num_of_interp_funcs)
b = np.zeros([num_of_interp_funcs, 1])

for m in range(number_of_initials): 
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
        
        # To construct b vector 
        K_h_0 = np.exp(-np.power(R / h, 2)) / np.power(h * np.sqrt(np.pi), d) # N by N Gaussian kernel matrix
        K_h = np.exp(- np.power(R_hat / h, 2)) / np.power((h * np.sqrt(np.pi)), d) # Nd by N Gaussian kernel matrix 
        Sum_K_h_0 = np.sum(K_h_0, axis = 0)   # N by 1 vector; summation along axis 0 
        Sum_K_h_1 = np.sum(K_h[:,:,None], axis = 1) # Nd by 1 vector; summation along axis 1 
        
        first_moment = Delta_X * K_h # Nd by N matrix  
        
        # Approx_derivative = 
        # dist_term = (X - X_old) / observed_time_step  # Nd by 1 vector 
        
        dist_term = np.reshape(BIG_Derivative[m][l], [N * d, 1], order = 'C') # make it x1_dot, y1_dot ...
        nonlocal_grad1 = (2 / np.power(h,2)) * np.sum((first_moment / Sum_K_h_0)[:,:,None], axis = 1)  # Nd by 1 vector 
        
        nonlocal_grad2 = (2 / np.power(h,2)) * np.sum(first_moment[:,:,None], axis = 1) / Sum_K_h_1 # Nd by 1 vector 
        b += np.matmul(PSI.T, dist_term + nonlocal_grad1 + nonlocal_grad2) 
        

b /= (number_of_initials * L * N)        
AA /= (number_of_initials * L * N)
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
#ax1.set_xlabel('nodes')
ax1.set_ylabel('Profile values', fontsize = 16)
ax2.set_ylabel('data density', color = 'g', fontsize = 16)
if d == 1: 
    ax1.set_title(('\u03c7 =' + str(Chi)), fontsize = 16)
else:     
    ax1.set_title(('\u03c7 =' + str(omega)), fontsize = 16)

# plt.xlabel('x axis')
# plt.ylabel('y axis')
ax1.legend()


# Create inset
axins = inset_axes(ax1, width="40%", height="30%", loc="upper right")
axins.plot(nodes, W, label = 'Cut off kernel', markersize = 7, color = 'black', linestyle = 'solid')
axins.plot(nodes, Learned_Kernel, label = 'Learned kernel', markersize = 5, color = 'b', linestyle = '--', dashes = (5,5) )


# Set zoomed region
if d == 1: 
    axins.set_xlim(0, 0.02)
    axins.set_ylim(8000, 12000)
    # axins.set_title("Zoom In", fontsize=10)
elif d == 2: 
    axins.set_xlim(0, 0.02) 
    axins.set_ylim(2000, 3500)
elif d == 3: 
    axins.set_xlim(0, 0.02) 
    axins.set_ylim(20000, 30000) 
elif d == 4: 
    axins.set_xlim(0, 0.02) 
    axins.set_ylim(7000, 8500)
plt.show()



# save data for reconstruction of particle trajectories 
knot_num = np.size(these_knots)
""" 
if d == 1:
    with open(path +'KS_1D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Chi_' + str(Chi) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
    # with open(path +'KS_Particle_cut_off_50_t_0.05_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, these_knots)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, Chi)
        np.save(f, observed_time_step)
        np.save(f, a_min)
        np.save(f, b_max) 
        
elif d == 2:
    with open(path +'KS_2D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, these_knots)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min) 
        np.save(f, b_max)

elif d == 3 :
    with open(path +'KS_3D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, these_knots)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min) 
        np.save(f, b_max) 

elif d == 4 :
    with open(path +'KS_4D_Particle_50_Learned_reg_' + str(epsilon) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
    # with open(path +'KS_4D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)  # save the initial data for the particle trajectories 
        np.save(f, alpha)
        np.save(f, these_knots)         
        np.save(f, h) 
        np.save(f, tau)   
        np.save(f, omega)
        np.save(f, observed_time_step) 
        np.save(f, a_min) 
        np.save(f, b_max) 
   
"""          


        
    
    
    
    



        
        
        
        
        
        
        
        
