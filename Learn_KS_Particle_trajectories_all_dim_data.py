#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:11:11 2025

@author: chenqian
"""
import numpy as np 
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt
import matplotlib  
from matplotlib.collections import LineCollection # For multicolored lines plots 
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import warnings 
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist 
import patsy # for comparison
import KS_data_distribution as KS
import contextlib

number_of_initials = 500
epsilon = 0 
outer_iter = 2001 # number of iterations for the nonlinear optimization problem  

d = 2
# read data 
if d == 1:
    Chi = 0.55
    r_c = 0.01
    knot = 30 
    with open('KS_1D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Chi_' + str(Chi) + '_knot_' + str(knot)+ '_' + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f) 
        alpha = np.load(f) 
        these_knots = np.load(f)
        h = np.load(f)
        tau = np.load(f)  
        Chi = np.load(f)
        observed_time_step = np.load(f)
        a_min = np.load(f) 
        b_max = np.load(f)
 
    # adaptive data 
    """     
    with open('KS_1D_Particle_50_Learned_adaptive_cut_off_0.01_t_0.2_Chi_0.55_500.npy', 'rb') as f: 
        BIG_Data = np.load(f)  # save the initial data for the particle trajectories 
        alpha = np.load(f)
        P = np.load(f)         
        h = np.load(f) 
        tau = np.load(f)   
        Chi = np.load(f)
        observed_time_step = np.load(f) 
        a_min = np.load(f)
        b_max = np.load(f)
    """     
    
if d == 2: 
    omega = 2.0
    r_c = 0.05
    knot = 20 
    with open('KS_2D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_knot_' + str(knot) + '_' + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        alpha = np.load(f) 
        these_knots = np.load(f) 
        h = np.load(f)
        tau = np.load(f)  
        omega = np.load(f) 
        observed_time_step = np.load(f)
        a_min = np.load(f) 
        b_max = np.load(f)
    
    # adaptive data 
    """
    with open('KS_2D_Particle_50_Learned_adaptive_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)  # save the initial data for the particle trajectories 
        alpha = np.load(f)
        P = np.load(f)         
        h = np.load(f) 
        tau = np.load(f)   
        omega = np.load(f)
        observed_time_step = np.load(f) 
        a_min = np.load(f)
        b_max = np.load(f) 
    """     
if d == 3: 
    omega = 2.0
    r_c = 0.05
    knot = 25 
    with open('KS_3D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_knot_' + str(knot) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)
        alpha = np.load(f) 
        these_knots = np.load(f)
        h = np.load(f) 
        tau = np.load(f) 
        omega = np.load(f) 
        observed_time_step = np.load(f) 
        a_min = np.load(f) 
        b_max = np.load(f)
    
    # adaptive data 
    """ 
    with open('KS_3D_Particle_50_Learned_adaptive_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'rb') as f:
       BIG_Data = np.load(f)  # save the initial data for the particle trajectories 
       alpha = np.load(f)
       P = np.load(f)         
       h = np.load(f) 
       tau = np.load(f)   
       omega = np.load(f)
       observed_time_step = np.load(f) 
       a_min = np.load(f)
       b_max = np.load(f)
    """ 


if d == 4: 
    omega = 1.0 
    r_c = 0.05    
    knot = 30 
    with open('KS_4D_Particle_50_Learned_cut_off_' + str(r_c) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot) + '_' + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)  # save the initial data for the particle trajectories 
        alpha = np.load(f)
        these_knots = np.load(f)         
        h = np.load(f) 
        tau = np.load(f)   
        omega = np.load(f)
        observed_time_step = np.load(f) 
        a_min = np.load(f) 
        b_max = np.load(f)
        
    BIG_X = BIG_Data[:,:,:, 0, None]; BIG_Y = BIG_Data[:,:,:, 1, None] 
    BIG_Z = BIG_Data[:,:,:, 2, None]; BIG_W = BIG_Data[:,:,:, 3, None] 

M = np.shape(BIG_Data)[0]   # number of initial data    
N = np.shape(BIG_Data)[2] # number of particles in each data 
L = np.shape(BIG_Data)[1]   # number of time steps in the data

Final_time_step = int(observed_time_step / tau)
Stopping_time = np.around(tau * outer_iter, decimals = 2) # Final time 

""" 
nodes = np.linspace(a_min, b_max, 200)
number_of_nodes = np.size(nodes)
nodes_1 = np.copy(nodes) 
basis_funcs = patsy.bs(nodes, knots=these_knots[:-1], degree = 3, lower_bound = these_knots[0], upper_bound = these_knots[-1])
num_of_interp = np.shape(basis_funcs)[1]
""" 

def Interp_func(a, b, coeff, node_num = 200):
    nodes = np.linspace(a, b, node_num)
    basis_funcs = basis_funcs = patsy.bs(nodes, knots=these_knots[:-1], degree = 3, lower_bound = these_knots[0], upper_bound = these_knots[-1]) # 
    coeff1 = coeff.reshape(np.size(coeff))
    interp = np.dot(coeff1, basis_funcs.T)
    interpolant = interp1d(nodes, interp, fill_value = 'extrapolate')
    
    return interpolant
# Interpolation function 
Interpolant = Interp_func(a_min, b_max, alpha)

# save the data for each time 
observed_time = np.arange(0, Stopping_time + observed_time_step , observed_time_step)

if d == 1: 
    M = 1 
    BIG_X =[]
    x_final = []
    BIG_X_dot = [] 
    for m in range(M):
        print(f"iteration: {m}")
        x_init = np.copy(BIG_Data[m][0])
        x = np.copy(x_init)
        
        # V = 0; W(x) = 2 * Chi * log(|x|); H(x) = xlog(x) 
        grad_old = 0
        x_old = 0
        X_old = []; X_old.append(x) 
        X_dot = []; 

        for i in range(outer_iter):            
                
            for j in range(100): # maximum number of iterations for BB method (or other optimization method)
                    
                # Applying BB method for optimization 
                    
                # To compute the gradient of the target function including computing the Gaussian Kernel 
                diff = x - np.transpose(x) # A N by N matrix consisting of x_{i} - x_{j}  
                np.fill_diagonal(diff, 1) # Warning: change the diagonal terms to be zero 
                    
                diff_reciprocal = 1 / diff  # It will be legal to take reciprocal of diff; N by N by 1 array  
                np.fill_diagonal(diff_reciprocal, 0)
                np.fill_diagonal(diff, 0)
                    
                dist_term = (x - x_init) / tau # First term of the optimization problem: ||y - x^n||^2 / 2(tau)
                                                   # N by 1 vector 
                    
                # Gaussian Kernel terms without constant factor 
                K_h = np.exp(- np.power(diff / h, 2)) / (h * np.sqrt(np.pi)) # An N by N matrix
                    
                sum_K_h = np.sum(K_h, axis = 1)
                first_moment = diff * K_h                    
            
                # Gradient of the learned Keller_Segel kernel
                Learned_Keller_grad = np.diag(np.matmul(Interpolant(np.abs(diff)), diff))[:,np.newaxis]                    
                    
                # First term of the gradient of nonlocal Keller approximation of rho * log(rho)  
                nonlocal_grad1 = -2 / (h ** 2) \
                * np.sum((first_moment / sum_K_h)[:,:,None], axis = 1)     # N by 1 vector
                    
                # Second term if the gradient of nonlocal Keller approximation of rho * log(rho)
                nonlocal_grad2 = -2 / (h ** 2) \
                * np.sum(first_moment[:,:,None], axis = 1) / sum_K_h[:,None] # N by 1 vector 
                    
                # Total gradient                                                                         # An N by 1 vctor 
                grad = dist_term + nonlocal_grad1 + nonlocal_grad2 - Learned_Keller_grad / N 
                    
                if np.sqrt(np.tensordot(grad, grad)) < 1e-6:   # Stopping criterion of the inner loop
                    print(j)
                    break                 
                    
                step_length = 1e-6
                    
                # Compute the BB step length 
                if j > 0: 
                    y = grad - grad_old 
                    s = x - x_old 
                    step_length = np.tensordot(s, s) / np.tensordot(s,y)
                    
                grad_old = grad 
                x_old = x 
                x = x - step_length * grad 
                
            if (i > 0) and  (i % Final_time_step)  == 0: 
                X_old.append(x_init)
                X_dot.append((x_old - x_init) / tau) # (x^{n+1} - x^{n}) / tau 
                              
            x_init = x
                
        BIG_X.append(X_old)
        BIG_X_dot.append(X_dot)

    BIG_X = np.array(BIG_X)
    X_dot = np.array(X_dot)
    """ 
    with open('KS_1D_Particle_learned_trajectory_unif_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_1D_Particle_learned_trajectory_adap_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Chi_' + str(Chi) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_X)
        np.save(f, h)
        np.save(f, tau)
        np.save(f, observed_time_step)
        np.save(f, BIG_X_dot)     
    """  
    fig, ax = plt.subplots(1,2, figsize=(12, 4))    
    time = np.linspace(0, 0.2, 21); time = np.reshape(time, [np.size(time), 1])
    
    for k in range(50): 
        ax[0].plot(BIG_X[0,:,k], time, 'b', markersize = 1, linestyle = 'solid')
        ax[1].plot(BIG_Data[0,:,k], time, 'r', markersize = 1, linestyle = 'solid')
        
    ax[0].set_xlabel('x position')
    ax[0].set_ylabel('time') # , fontsize = 16)
    ax[0].set_title('Learned Particle trajectories') # , fontsize = 16)
    ax[1].set_xlabel('x position') #, fontsize = 16)
    ax[1].set_ylabel('time') # , fontsize = 16)
    ax[1].set_title('True Particle trajectories') # , fontsize = 16)
    plt.show()



if d == 2:  
    BIG_X =[]; BIG_Y = []   # Now we have two components 
    BIG_X_dot = []; BIG_Y_dot = [] 
    M = 1
    
    for m in range(M):
        x_init = BIG_Data[m][0,:, 0, None]
        y_init = BIG_Data[m][0,:, 1, None]
        x = np.copy(x_init)
        y = np.copy(y_init)
        
        
        
        # V = 0; W(x) = 2 * Chi * log(|x|); H(x) = xlog(x) 
        grad_old_x =  np.zeros(np.shape(x_init));    grad_old_y = np.zeros(np.shape(y_init))
        x_old = np.zeros(np.shape(x_init));     y_old = np.zeros(np.shape(y_init))
        X_old = []; Y_old = []
        X_old.append(x); Y_old.append(y) 
        X_dot = []; Y_dot = []
        
        
        for i in range(outer_iter):
            
        
            for j in range(100): # maximum number of iterations for BB method (or other optimization method)
                
                # Applying BB method for optimization 
                
                # For the dissipation term: 
                dist_term_x = (x - x_init) / tau 
                dist_term_y = (y - y_init) / tau 
                
                # To compute the gradient of the Keller-Segel kernel and the Gaussian Kernel 
                diff_x = x - np.transpose(x) # An N by N matrix consisting of x_{i} - x_{j}  
                diff_y = y - np.transpose(y) # An N by N matrix consisting of y_{i} - y_{j} 
            
                diff_sum_square = np.power(diff_x, 2) + np.power(diff_y, 2)  
                diff = np.sqrt(diff_sum_square)
                
                # Gradient of the learned Keller_Segel kernel
                interp = Interpolant(np.abs(diff))
                Learned_Keller_grad_x = np.diag(np.matmul(interp, diff_x))[:,np.newaxis]                
                Learned_Keller_grad_y = np.diag(np.matmul(interp, diff_y))[:,np.newaxis] 
                
                np.fill_diagonal(diff_sum_square, 0)
                
                # Gaussian Kernel terms without constant factor 
                K_h = np.exp(- diff_sum_square / (h ** 2))
                
                sum_K_h = np.sum(K_h, axis = 1) # An N by 1 column vector 
                
                first_moment_x = diff_x * K_h
                first_moment_y = diff_y * K_h
                
                # Now we have all the components of the gradient terms for our optimization problem J 
                # We can compute the gradient of J now 
                
                # First term of the gradient of nonlocal Keller approximation of rho * log(rho)  
                nonlocal_grad1_x = -2 / (h ** 2) \
                * np.sum((first_moment_x / sum_K_h)[:,:,None], axis = 1)     # N by 1 vector for x-axis 
                
                nonlocal_grad1_y = -2 / (h ** 2) \
                    * np.sum((first_moment_y / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for y-axis 
                
                # Second term of the gradient of nonlocal Keller approximation of rho * log(rho)
                nonlocal_grad2_x = -2 / (h ** 2) \
                * np.sum(first_moment_x[:,:,None], axis = 1) / sum_K_h[:, None] # N by 1 vector 
                                
                nonlocal_grad2_y = -2 / (h ** 2) \
                * np.sum(first_moment_y[:,:,None], axis = 1) / sum_K_h[:, None]
                
                # Total gradient 
                grad_x = dist_term_x + nonlocal_grad1_x + nonlocal_grad2_x - Learned_Keller_grad_x / N # total gradient for the optimization problem 
                
                grad_y = dist_term_y + nonlocal_grad1_y + nonlocal_grad2_y - Learned_Keller_grad_y / N  
                
                if np.sqrt(np.tensordot(grad_x, grad_x) + np.tensordot(grad_y, grad_y)) < 1e-6:   # Stopping criterion of the inner loop
                    print(j)
                    break                 
                
                step_length_x = 1e-6
                step_length_y = 1e-6 
                # Compute the BB step length 
                if j > 0: 
                    z_x = grad_x - grad_old_x
                    z_y = grad_y - grad_old_y
                    s_x = x - x_old 
                    s_y = y - y_old
                    step_length_x = np.tensordot(s_x, s_x) / np.tensordot(s_x, z_x)
                    step_length_y = np.tensordot(s_y, s_y) / np.tensordot(s_y, z_y)
                
                grad_old_x = grad_x 
                grad_old_y = grad_y 
                x_old = x; y_old = y  
                x = x - step_length_x * grad_x 
                y = y - step_length_y * grad_y 
            
            if (i > 0) and (i % Final_time_step) == 0: 
                X_old.append(x_old)
                Y_old.append(y_old) 
                X_dot.append((x - x_init) / tau); Y_dot.append((y - y_init) / tau)
            x_init = x; y_init = y 
            
            
        BIG_X.append(X_old); BIG_Y.append(Y_old)
        BIG_X_dot.append(X_dot); BIG_Y_dot.append(Y_dot)
    
    BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y)
    BIG_Data_learned = np.concatenate([BIG_X, BIG_Y], axis = 3)
    BIG_X_dot = np.array(BIG_X_dot); BIG_Y_dot = np.array(BIG_Y_dot)
    BIG_Derivative_learned = np.concatenate([BIG_X_dot, BIG_Y_dot], axis = 3)
    """ 
    with open('KS_2D_Particle_learned_trajectory_unif_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_2D_Particle_learned_trajectory_adap_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data_learned)
        np.save(f, h)
        np.save(f, tau)
        np.save(f, observed_time_step)
        np.save(f, BIG_Derivative_learned) 
    """ 
    fig, ax = plt.subplots(1,2, figsize = (12,4)) 
    norm = plt.Normalize(observed_time[0], observed_time[-1]) 

    colormap = matplotlib.colormaps["plasma"]
    colormap = colormap.reversed() 

    for k in range(N): 
         
        xy = np.array([BIG_X[0][:,k,0], BIG_Y[0][:,k,0]]).T 
        segments = np.stack([xy[:-1, :], xy[1:, :]], axis = 1)
        
        xy_true = np.array([BIG_Data[0][:,k,0], BIG_Data[0][:,k,1]]).T
        segments_true = np.stack([xy_true[:-1,:], xy_true[1:,:]], axis = 1)
        
        lc = LineCollection(segments, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)
        lc_true = LineCollection(segments_true, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)
        ax[0].add_collection(lc)
        ax[1].add_collection(lc_true)        
        if k == 0:
            lc.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
            lc_true.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
            fig.colorbar(lc, ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
            fig.colorbar(lc_true, ticks = [0.0, 0.05, 0.1, 0.15, 0.2])
            ax[0].grid()
            ax[1].grid() 
            
    ax[0].set_xlabel("x_axis", fontsize = 16)
    ax[0].set_ylabel("y_axis", fontsize = 16)
    ax[0].set_title("Learned Particle trajectory", fontsize = 16)
    ax[1].set_xlabel("x_axis", fontsize = 16)
    ax[1].set_ylabel("y_axis", fontsize = 16)
    ax[1].set_title("True Particle trajectory", fontsize = 16)
    plt.show()
    fig.savefig('KS_2D_trajectory_comparison_chi_4.png', dpi = 600, bbox_inches = 'tight')

if d == 3:  
    BIG_X =[]; BIG_Y = []; BIG_Z = []   # Now we have two components 
    BIG_X_dot = []; BIG_Y_dot = []; BIG_Z_dot = []
    M = 1
    
    for m in range(M):
        x_init = BIG_Data[m][0, :, 0, None]
        y_init = BIG_Data[m][0, :, 1, None]
        z_init = BIG_Data[m][0, :, 2, None]
        x = np.copy(x_init)
        y = np.copy(y_init)
        z = np.copy(z_init)
        
        # V = 0; W(x) = -1 / (4 * pi * |x|); H(x) = xlog(x) 
        grad_old_x =  np.zeros(np.shape(x_init)); grad_old_y = np.zeros(np.shape(y_init)); grad_old_z = np.zeros(np.shape(z_init))
        x_old = np.zeros(np.shape(x_init)); y_old = np.zeros(np.shape(y_init)); z_old = np.zeros(np.shape(z_init))
        X_old = []; Y_old = []; Z_old = []
        X_old.append(x); Y_old.append(y); Z_old.append(z) 
        X_dot = []; Y_dot = []; Z_dot = [] 
        
        for i in range(outer_iter):
            for j in range(100): # maximum number of iterations for BB method (or other optimization method)
                
                # Applying BB method for optimization 
                
                # For the dissipation term: 
                dist_term_x = (x - x_init) / tau 
                dist_term_y = (y - y_init) / tau 
                dist_term_z = (z - z_init) / tau 
                
                # To compute the gradient of the Keller-Segel kernel and the Gaussian Kernel 
                diff_x = x - np.transpose(x) # An N by N matrix consisting of x_{i} - x_{j}  
                diff_y = y - np.transpose(y) # An N by N matrix consisting of y_{i} - y_{j} 
                diff_z = z - np.transpose(z) # An N by N matrix consisting of z_{i} - z_{j} 
                
                diff_sum_square = np.power(diff_x, 2) + np.power(diff_y, 2) + np.power(diff_z, 2)  
                
                diff = np.sqrt(diff_sum_square)
                
                # Gradient of the learned Keller_Segel kernel
                interp = Interpolant(np.abs(diff))
                Learned_Keller_grad_x = np.diag(np.matmul(interp, diff_x))[:,np.newaxis]
                
                Learned_Keller_grad_y = np.diag(np.matmul(interp, diff_y))[:,np.newaxis]
                
                Learned_Keller_grad_z = np.diag(np.matmul(interp, diff_z))[:, np.newaxis]
                
                np.fill_diagonal(diff_sum_square, 0)
                
                # Gaussian Kernel terms without constant factor 
                K_h = np.exp(- diff_sum_square / (h ** 2))
                
                sum_K_h = np.sum(K_h, axis = 1) # An N by 1 column vector 
                
                first_moment_x = diff_x * K_h
                first_moment_y = diff_y * K_h
                first_moment_z = diff_z * K_h
                
                # Now we have all the components of the gradient terms for our optimization problem J 
                # We can compute the gradient of J now 
                
                # First term of the gradient of nonlocal Keller approximation of rho * log(rho)  
                nonlocal_grad1_x = -2 / (h ** 2) \
                    * np.sum((first_moment_x / sum_K_h)[:,:,None], axis = 1)     # N by 1 vector for x-axis 
                
                nonlocal_grad1_y = -2 / (h ** 2) \
                    * np.sum((first_moment_y / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for y-axis
                
                nonlocal_grad1_z = -2 / (h ** 2) \
                    * np.sum((first_moment_z / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for z-axis  
                
                # Second term of the gradient of nonlocal Keller approximation of rho * log(rho)
                nonlocal_grad2_x = -2 / (h ** 2) \
                    * np.sum(first_moment_x[:,:,None], axis = 1) / sum_K_h[:, None] # N by 1 vector 
                
                
                nonlocal_grad2_y = -2 / (h ** 2) \
                    * np.sum(first_moment_y[:,:,None], axis = 1) / sum_K_h[:, None]
                
                nonlocal_grad2_z = -2 / (h ** 2) \
                    * np.sum(first_moment_z[:,:,None], axis = 1) / sum_K_h[:, None]    
                
                
                # Total gradient 
                grad_x = dist_term_x + nonlocal_grad1_x + nonlocal_grad2_x - Learned_Keller_grad_x / N # total gradient for the optimization problem 
                
                grad_y = dist_term_y + nonlocal_grad1_y + nonlocal_grad2_y - Learned_Keller_grad_y / N  
                
                grad_z = dist_term_z + nonlocal_grad1_z + nonlocal_grad2_z - Learned_Keller_grad_z / N 
                
                if np.sqrt(np.tensordot(grad_x, grad_x) + np.tensordot(grad_y, grad_y) + np.tensordot(grad_z, grad_z)) < 1e-6:   # Stopping criterion of the inner loop
                    print(j)
                    break                 
                
                step_length_x = 1e-6
                step_length_y = 1e-6 
                step_length_z = 1e-6 
                # Compute the BB step length 
                if j > 0: 
                    w_x = grad_x - grad_old_x
                    w_y = grad_y - grad_old_y
                    w_z = grad_z - grad_old_z 
                    s_x = x - x_old 
                    s_y = y - y_old
                    s_z = z - z_old 
                    step_length_x = np.tensordot(s_x, s_x) / np.tensordot(w_x, s_x)
                    step_length_y = np.tensordot(s_y, s_y) / np.tensordot(w_y, s_y)
                    step_length_z = np.tensordot(s_z, s_z) / np.tensordot(w_z, s_z) 
                
                grad_old_x = grad_x 
                grad_old_y = grad_y 
                grad_old_z = grad_z 
                x_old = x; y_old = y; z_old = z   
                x = x - step_length_x * grad_x 
                y = y - step_length_y * grad_y 
                z = z - step_length_z * grad_z 
                
            if (i > 0) and (i % Final_time_step) == 0: 
                X_old.append(x_old)
                Y_old.append(y_old) 
                Z_old.append(z_old)
                X_dot.append((x - x_init) / tau) 
                Y_dot.append((y - y_init) / tau)
                Z_dot.append((z - z_init) / tau)
                
            x_init = x; y_init = y; z_init = z  
            
            
        BIG_X.append(X_old); BIG_Y.append(Y_old); BIG_Z.append(Z_old)
        BIG_X_dot.append(X_dot); BIG_Y_dot.append(Y_dot); BIG_Z_dot.append(Z_dot)
    
    BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y); BIG_Z = np.array(BIG_Z)
    BIG_Data_learned = np.concatenate([BIG_X, BIG_Y, BIG_Z], axis = 3)
    BIG_X_dot = np.array(BIG_X_dot); BIG_Y_dot = np.array(BIG_Y_dot); BIG_Z_dot = np.array(BIG_Z_dot)
    BIG_Derivative_learned = np.concatenate([BIG_X_dot, BIG_Y_dot, BIG_Z_dot], axis = 3)
    
    """ 
    with open('KS_3D_Particle_learned_trajectory_unif_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_3D_Particle_learned_trajectory_adap_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data_learned)
        np.save(f, h)
        np.save(f, tau)
        np.save(f, observed_time_step)
        np.save(f, BIG_Derivative_learned) 
    """ 
    colormap = matplotlib.colormaps["plasma"]
    colormap = colormap.reversed()

    norm = plt.Normalize(observed_time[0], observed_time[-1])
    
    fig, ax = plt.subplots(1, 2, figsize=(12,4),subplot_kw= dict(projection='3d'))
    
    for k in range(N): 
         
        xyz = np.array([BIG_X[0][:,k,0], BIG_Y[0][:,k,0], BIG_Z[0][:,k,0]]).T 
        segments = np.stack([xyz[:-1, :], xyz[1:, :]], axis = 1)
        lc = Line3DCollection(segments, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)

        xyz_true = np.array([BIG_Data[0][:,k,0], BIG_Data[0][:,k,1], BIG_Data[0][:,k,2]]).T 
        segments_true = np.stack([xyz_true[:-1,:], xyz_true[1:,:]], axis = 1)
        lc_true = Line3DCollection(segments_true, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)

        ax[0].add_collection3d(lc)
        ax[1].add_collection3d(lc_true)

        if k == 0:
            lc.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
            lc_true.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
            fig.colorbar(lc, location = 'left', ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
            fig.colorbar(lc_true, location = 'left', ticks = [0.0, 0.05, 0.1, 0.15, 0.2])
        
        
    ax[0].set_xlabel("x_axis")
    ax[0].set_ylabel("y_axis")
    ax[0].set_zlabel("z_axis")     
    ax[0].set_title("Learned Particle trajectory", fontsize = 16) 
    
    ax[1].set_xlabel("x_axis")
    ax[1].set_ylabel("y_axis")
    ax[1].set_zlabel("z_axis")     
    ax[1].set_title("True Particle trajectory", fontsize = 16)
    plt.show()
    
    fig.savefig('KS_3D_trajectory_comparison_chi_4.png', dpi = 600, bbox_inches = 'tight')


if d == 4:   
    M = 1
    BIG_X =[]; BIG_Y = []; BIG_Z = []; BIG_W = []   # Now we have two components 
    BIG_X_dot = []; BIG_Y_dot = []; BIG_Z_dot = []; BIG_W_dot = []
    
    for m in range(M):
        x_init = BIG_Data[m][0, :, 0, None]
        y_init = BIG_Data[m][0, :, 1, None]
        z_init = BIG_Data[m][0, :, 2, None]
        w_init = BIG_Data[m][0, :, 3, None]
        x = np.copy(x_init)
        y = np.copy(y_init)
        z = np.copy(z_init)
        w = np.copy(w_init)
        
        # V = 0; W(x) = -1 / (4 * pi * |x|); H(x) = xlog(x) 
        grad_old_x = np.zeros(np.shape(x_init)); grad_old_y = np.zeros(np.shape(y_init)) 
        grad_old_z = np.zeros(np.shape(z_init)); grad_old_w = np.zeros(np.shape(w_init))
        x_old = np.zeros(np.shape(x_init)); y_old = np.zeros(np.shape(y_init))
        z_old = np.zeros(np.shape(z_init)); w_old = np.zeros(np.shape(w_init))
        X_old = []; Y_old = []; Z_old = []; W_old = []
        X_old.append(x); Y_old.append(y); Z_old.append(z); W_old.append(w) 
        X_dot = []; Y_dot = []; Z_dot = []; W_dot = [] 
        
        for i in range(outer_iter):
            
        
            for j in range(100): # maximum number of iterations for BB method (or other optimization method)
                
                # Applying BB method for optimization 
                
                # For the dissipation term: 
                dist_term_x = (x - x_init) / tau 
                dist_term_y = (y - y_init) / tau 
                dist_term_z = (z - z_init) / tau 
                dist_term_w = (w - w_init) / tau 
                
                # To compute the gradient of the Keller-Segel kernel and the Gaussian Kernel 
                diff_x = x - np.transpose(x) # An N by N matrix consisting of x_{i} - x_{j}  
                diff_y = y - np.transpose(y) # An N by N matrix consisting of y_{i} - y_{j} 
                diff_z = z - np.transpose(z) # An N by N matrix consisting of z_{i} - z_{j} 
                diff_w = w - np.transpose(w) # An N by N matrix consisting of w_{i} - w_{j} 
                
                diff_sum_square = np.power(diff_x, 2) + np.power(diff_y, 2) + np.power(diff_z, 2) + np.power(diff_w, 2)
                
                diff = np.sqrt(diff_sum_square)
                
                # Gradient of the learned Keller_Segel kernel
                interp = Interpolant(np.abs(diff))
                Learned_Keller_grad_x = np.diag(np.matmul(interp, diff_x))[:,np.newaxis]
                
                Learned_Keller_grad_y = np.diag(np.matmul(interp, diff_y))[:,np.newaxis]
                
                Learned_Keller_grad_z = np.diag(np.matmul(interp, diff_z))[:, np.newaxis]
                
                Learned_Keller_grad_w = np.diag(np.matmul(interp, diff_w))[:, np.newaxis]
                
                np.fill_diagonal(diff_sum_square, 0)
                
                # Gaussian Kernel terms without constant factor 
                K_h = np.exp(- diff_sum_square / (h ** 2))
                
                sum_K_h = np.sum(K_h, axis = 1) # An N by 1 column vector 
                
                first_moment_x = diff_x * K_h
                first_moment_y = diff_y * K_h
                first_moment_z = diff_z * K_h
                first_moment_w = diff_w * K_h 

                # Now we have all the components of the gradient terms for our optimization problem J 
                # We can compute the gradient of J now 
                
                # First term of the gradient of nonlocal Keller approximation of rho * log(rho)  
                nonlocal_grad1_x = -2 / (h ** 2) \
                    * np.sum((first_moment_x / sum_K_h)[:,:,None], axis = 1)     # N by 1 vector for x-axis 
                
                nonlocal_grad1_y = -2 / (h ** 2) \
                    * np.sum((first_moment_y / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for y-axis
                
                nonlocal_grad1_z = -2 / (h ** 2) \
                    * np.sum((first_moment_z / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for z-axis  
                
                nonlocal_grad1_w = -2 / (h ** 2) \
                    * np.sum((first_moment_w / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for w-axis  
                
                # Second term of the gradient of nonlocal Keller approximation of rho * log(rho)
                nonlocal_grad2_x = -2 / (h ** 2) \
                    * np.sum(first_moment_x[:,:,None], axis = 1) / sum_K_h[:, None] # N by 1 vector 
                
                nonlocal_grad2_y = -2 / (h ** 2) \
                    * np.sum(first_moment_y[:,:,None], axis = 1) / sum_K_h[:, None]
                
                nonlocal_grad2_z = -2 / (h ** 2) \
                    * np.sum(first_moment_z[:,:,None], axis = 1) / sum_K_h[:, None]    
                    
                nonlocal_grad2_w = -2 / (h ** 2) \
                    * np.sum(first_moment_w[:,:,None], axis = 1) / sum_K_h[:, None]    
                
                
                # Total gradient 
                grad_x = dist_term_x + nonlocal_grad1_x + nonlocal_grad2_x - Learned_Keller_grad_x / N # total gradient for the optimization problem 
                
                grad_y = dist_term_y + nonlocal_grad1_y + nonlocal_grad2_y - Learned_Keller_grad_y / N  
                
                grad_z = dist_term_z + nonlocal_grad1_z + nonlocal_grad2_z - Learned_Keller_grad_z / N 
                
                grad_w = dist_term_w + nonlocal_grad1_w + nonlocal_grad2_w - Learned_Keller_grad_w / N 
                
                if np.sqrt(np.tensordot(grad_x, grad_x) + np.tensordot(grad_y, grad_y) + np.tensordot(grad_z, grad_z) + np.tensordot(grad_w, grad_w))< 1e-6:   # Stopping criterion of the inner loop
                    print(j)
                    break                 
                
                step_length_x = 1e-6
                step_length_y = 1e-6 
                step_length_z = 1e-6
                step_length_w = 1e-6
                # Compute the BB step length 
                if j > 0: 
                    w_x = grad_x - grad_old_x
                    w_y = grad_y - grad_old_y
                    w_z = grad_z - grad_old_z 
                    w_w = grad_w - grad_old_w 
                    s_x = x - x_old 
                    s_y = y - y_old
                    s_z = z - z_old 
                    s_w = w - w_old
                    step_length_x = np.tensordot(s_x, s_x) / np.tensordot(w_x, s_x)
                    step_length_y = np.tensordot(s_y, s_y) / np.tensordot(w_y, s_y)
                    step_length_z = np.tensordot(s_z, s_z) / np.tensordot(w_z, s_z) 
                    step_length_w = np.tensordot(s_w, s_w) / np.tensordot(w_w, s_w) 
                grad_old_x = grad_x 
                grad_old_y = grad_y 
                grad_old_z = grad_z 
                grad_old_w = grad_w 
                x_old = x; y_old = y; z_old = z; w_old= w   
                x = x - step_length_x * grad_x 
                y = y - step_length_y * grad_y 
                z = z - step_length_z * grad_z 
                w = w - step_length_w * grad_w 
            if (i > 0) and (i % Final_time_step) == 0: 
                X_old.append(x_old)
                Y_old.append(y_old) 
                Z_old.append(z_old)
                W_old.append(w_old)
                X_dot.append((x - x_init) / tau) 
                Y_dot.append((y - y_init) / tau)
                Z_dot.append((z - z_init) / tau)
                W_dot.append((w - w_init) / tau)
           
            x_init = x; y_init = y; z_init = z ; w_init = w 
            
            
        BIG_X.append(X_old); BIG_Y.append(Y_old); BIG_Z.append(Z_old); BIG_W.append(W_old)
        BIG_X_dot.append(X_dot); BIG_Y_dot.append(Y_dot)
        BIG_Z_dot.append(Z_dot); BIG_W_dot.append(W_dot)
    
    BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y)
    BIG_Z = np.array(BIG_Z); BIG_W = np.array(BIG_W)
    BIG_Data_learned = np.concatenate([BIG_X, BIG_Y, BIG_Z, BIG_W], axis = 3)
    BIG_X_dot = np.array(BIG_X_dot); BIG_Y_dot = np.array(BIG_Y_dot)
    BIG_Z_dot = np.array(BIG_Z_dot); BIG_W_dot = np.array(BIG_W_dot)
    BIG_Derivative_learned = np.concatenate([BIG_X_dot, BIG_Y_dot, BIG_Z_dot, BIG_W_dot], axis = 3)
    
    """ 
    with open('KS_4D_Particle_learned_trajectory_unif_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_3D_Particle_learned_trajectory_adap_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data_learned)
        np.save(f, h)
        np.save(f, tau)
        np.save(f, observed_time_step)
        np.save(f, BIG_Derivative_learned) 
    """ 

    
        
        
        
        
        
        
        