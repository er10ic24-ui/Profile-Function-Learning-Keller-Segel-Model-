#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 14:51:19 2025

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
# epsilon = 0 
outer_iter = 2001 # number of iterations for the nonlinear optimization problem  

d = 2
# read data 

    
if d == 2: 
    omega = 2.0
    epsilon = 0.01
    eta = 0.01 
    knot_num = 30
    T = 0.2 
    N_par = 50 
   
    with open('KS_2D_SDE_Particle_50_Learned_reg_' + str(epsilon) + '_t_0.2_Chi_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f)  # save the initial data for the particle trajectories 
        alpha = np.load(f)
        these_knots = np.load(f)         
        tau = np.load(f)   
        omega = np.load(f)
        observed_time_step = np.load(f) 
        a_min = np.load(f) 
        b_max = np.load(f)
        BIG_Noise = np.load(f) 
    
    N_time = int(T / tau)
    dt = tau 
    sigma = np.sqrt(2) * eta 
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


if d == 4: 
    omega = 2.0
    epsilon = 0.01
    eta = 0.01 
    knot_num = 15
    T = 0.2 
    tau = 1e-4 
    N_par = 50 
    with open('KS_SDE_4D_Par_reg_' + str(epsilon) + '_t_0.2_Omega_' + str(omega) + '_knot_'  + str(knot_num) + '_' + str(number_of_initials) + '.npy', 'rb') as f:
        BIG_Data = np.load(f) # load the particle trajectories 
        BIG_Noise = np.load(f) # load the noise for the particle trajectories 
        alpha = np.load(f)    # laod the coefficients of the basis functions 
        these_knots = np.load(f)  # load the positions of the knot points         
        omega = np.load(f)
        observed_time_step = np.load(f) 
        a_min = np.load(f) 
        b_max = np.load(f) 
        
    N_time = int(T / tau)
    dt = tau 
    sigma = np.sqrt(2) * eta 
    # adaptive data 
   
      

M = np.shape(BIG_Data)[0]   # number of initial data    
N = np.shape(BIG_Data)[2] # number of particles in each data 
L = np.shape(BIG_Data)[1]   # number of time steps in the data
    
Final_time_step = int(observed_time_step / tau)
Stopping_time = np.around(tau * outer_iter, decimals = 2) # Final time 
    
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
record_time_step = int(observed_time_step / dt) 

if d == 2: 
    BIG_X =[]; BIG_Y = []   # Now we have two components 
    x_final = []
    # For each inital distribution
    
    # for m in range(2):
    for m in range(number_of_initials): 
        print(m) 
        X_old = []; Y_old = [] 
        X_noise = []; Y_noise = []
        X_init = BIG_Data[m][0]
        X_0 = X_init[:,0,None]; Y_0 = X_init[:,1,None]
        X_old.append(X_0); Y_old.append(Y_0)
        # X = np.zeros([N_par, N_time + 1]); Y = np.zeros([N_par, N_time + 1]) 
        # X[:,0,None] = X_0; Y[:,0,None] = Y_0 
        
        # Diff_X = np.zeros([N_time + 1, N_par, N_par]); Diff_Y = np.zeros([N_time + 1, N_par, N_par])
        # Diff_X[0] = X_0 - X_0.T; Diff_Y[0] = Y_0 - Y_0.T  
        
        for i in range(N_time): 
            
            Diff_X = X_0 - X_0.T; Diff_Y = Y_0 - Y_0.T
            dist_diff = np.power(Diff_X, 2) + np.power(Diff_Y, 2) 
    
            x_noise = BIG_Noise[m,i,:,0, np.newaxis]; y_noise = BIG_Noise[m,i,:,1, np.newaxis]
            
            diff = np.sqrt(dist_diff)
            
            # cut_off = np.piecewise(diff_sum_square, [diff_sum_square <= r_c_square , diff_sum_square > r_c_square], [inverse_square_r_c, lambda y: 1 / y])
            
            # Gradient of the learned Keller_Segel kernel
            # Learned = np.stack([Interp_func[m](np.abs(diff)) for m in range(num_of_interp)]) # save all the data 
            # Interpolants_x = np.stack([np.diag(np.matmul(Learned[m], diff_x))[:, np.newaxis] for m in range(num_of_interp)]) # num_interp
            interp = Interpolant(np.abs(diff))
            Learned_Keller_grad_x = np.diag(np.matmul(interp, Diff_X))[:,np.newaxis]
            
            # Interpolants_y = np.stack([np.diag(np.matmul(Learned[m], diff_y))[:, np.newaxis] for m in range(num_of_interp)]) # num_interp]) 
            Learned_Keller_grad_y = np.diag(np.matmul(interp, Diff_Y))[:,np.newaxis] 
            
            # X[:,i+1] += sigma * np.random.normal(0, np.sqrt(dt), N_par) + chi / N_par * (- np.sum(Diff_X[i] / (2 * np.pi) / (dist_diff + epsilon_sqr), axis = 1) ) * dt      
            # Y[:,i+1] += sigma * np.random.normal(0, np.sqrt(dt), N_par) + chi / N_par * (- np.sum(Diff_Y[i] / (2 * np.pi) / (dist_diff + epsilon_sqr), axis = 1) ) * dt      
            X = X_0 + sigma * np.sqrt(dt) * x_noise +  ( Learned_Keller_grad_x) * dt  / N_par
            Y = Y_0 + sigma * np.sqrt(dt) * y_noise +  ( Learned_Keller_grad_y) * dt  / N_par  
            X_0 = np.copy(X); Y_0 = np.copy(Y) 
    
            
            if ((i + 1) % record_time_step) == 0: 
                X_old.append(X_0); Y_old.append(Y_0) 
    
        BIG_X.append(X_old); BIG_Y.append(Y_old)
                
      
        
    
    BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y)
    BIG_Data_learned = np.concatenate([BIG_X, BIG_Y], axis = 3)
    
    """ 
    with open('KS_SDE_2D_Particle_learned_trajectory_unif_50_reg_' + str(epsilon) + '_t_' + str(Stopping_time) + '_Chi_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data_learned)
        np.save(f, tau)
        np.save(f, observed_time_step)
    """ 
    
    fig, ax = plt.subplots(1,2, figsize = (12,4)) 
    # ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    # cbaxes = fig.add_axies([0.15, 0.03, 0.7])
    # cbaxes = fig.add_axies([0.15, 0.1, 0.03, 0.7])
    norm = plt.Normalize(observed_time[0], observed_time[-1]) 

    colormap = matplotlib.colormaps["plasma"]
    colormap = colormap.reversed() 

    for k in range(N): 
         
        xy = np.array([BIG_X[1][:,k,0], BIG_Y[1][:,k,0]]).T 
        segments = np.stack([xy[:-1, :], xy[1:, :]], axis = 1)
        
        xy_true = np.array([BIG_Data[1][:,k,0], BIG_Data[1][:,k,1]]).T
        segments_true = np.stack([xy_true[:-1,:], xy_true[1:,:]], axis = 1)
        
        lc = LineCollection(segments, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)
        lc_true = LineCollection(segments_true, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)
        ax[0].add_collection(lc)
        ax[1].add_collection(lc_true)
        # ax.add_patch(lc)
        
        if k == 0:
            # sm = fig.cm.ScalarMappable(cmap = colormap)
            # sm.set_clim(vmin = 0.0, vmax = 0.2)
            lc.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
            lc_true.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
            # fig.colorbar(lc, cax = cbaxes, ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
            fig.colorbar(lc, ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
            fig.colorbar(lc_true, ticks = [0.0, 0.05, 0.1, 0.15, 0.2])
            ax[0].grid()
            ax[1].grid() 
            
    ax[0].set_xlabel("x_axis", fontsize = 16)
    ax[0].set_ylabel("y_axis", fontsize = 16)
    # ax[0].set_title("Learned Particle trajectory 2d with \u03c7 =" + str(omega), fontsize = 16)
    ax[0].set_title("Learned Particle trajectory", fontsize = 16)
    ax[1].set_xlabel("x_axis", fontsize = 16)
    ax[1].set_ylabel("y_axis", fontsize = 16)
    ax[1].set_title("True Particle trajectory", fontsize = 16)
    plt.show()
    
    # fig.savefig('KS_SDE_2D_trajectory_comparison_chi_4.png', dpi = 600, bbox_inches = 'tight')

    
    




      





if d == 4: 
    BIG_X =[]; BIG_Y = []; BIG_Z = []; BIG_W = []    # Now we have two components 
    x_final = []
    # For each inital distribution
    
    # for m in range(2):
    for m in range(number_of_initials): 
        print(m) 
        X_old = []; Y_old = []; Z_old = []; W_old = []  
        X_noise = []; Y_noise = []; Z_noise = []; W_noise = []
        X_init = BIG_Data[m][0]
        X_0 = X_init[:,0,None]; Y_0 = X_init[:,1,None]; Z_0 = X_init[:,2,None]; W_0 =X_init[:,3,None]
        X_old.append(X_0); Y_old.append(Y_0); Z_old.append(Z_0); W_old.append(W_0)
        # X = np.zeros([N_par, N_time + 1]); Y = np.zeros([N_par, N_time + 1]) 
        # X[:,0,None] = X_0; Y[:,0,None] = Y_0 
        
        # Diff_X = np.zeros([N_time + 1, N_par, N_par]); Diff_Y = np.zeros([N_time + 1, N_par, N_par])
        # Diff_X[0] = X_0 - X_0.T; Diff_Y[0] = Y_0 - Y_0.T  
        
        for i in range(N_time): 
            
            Diff_X = X_0 - X_0.T; Diff_Y = Y_0 - Y_0.T; Diff_Z = Z_0 - Z_0.T; Diff_W = W_0 - W_0.T
            dist_diff = np.power(Diff_X, 2) + np.power(Diff_Y, 2) + np.power(Diff_Z, 2) + np.power(Diff_W, 2)
    
            x_noise = BIG_Noise[m,i,:,0, np.newaxis]; y_noise = BIG_Noise[m,i,:,1, np.newaxis]
            z_noise = BIG_Noise[m,i,:,2, np.newaxis]; w_noise = BIG_Noise[m,i,:,3, np.newaxis]
            
            diff = np.sqrt(dist_diff)
            
            # cut_off = np.piecewise(diff_sum_square, [diff_sum_square <= r_c_square , diff_sum_square > r_c_square], [inverse_square_r_c, lambda y: 1 / y])
            
            # Gradient of the learned Keller_Segel kernel
            # Learned = np.stack([Interp_func[m](np.abs(diff)) for m in range(num_of_interp)]) # save all the data 
            # Interpolants_x = np.stack([np.diag(np.matmul(Learned[m], diff_x))[:, np.newaxis] for m in range(num_of_interp)]) # num_interp
            interp = Interpolant(np.abs(diff))
            Learned_Keller_grad_x = np.diag(np.matmul(interp, Diff_X))[:,np.newaxis]
            
            # Interpolants_y = np.stack([np.diag(np.matmul(Learned[m], diff_y))[:, np.newaxis] for m in range(num_of_interp)]) # num_interp]) 
            Learned_Keller_grad_y = np.diag(np.matmul(interp, Diff_Y))[:,np.newaxis] 
            Learned_Keller_grad_z = np.diag(np.matmul(interp, Diff_Z))[:,np.newaxis]
            Learned_Keller_grad_w = np.diag(np.matmul(interp, Diff_W))[:,np.newaxis]


            
            # X[:,i+1] += sigma * np.random.normal(0, np.sqrt(dt), N_par) + chi / N_par * (- np.sum(Diff_X[i] / (2 * np.pi) / (dist_diff + epsilon_sqr), axis = 1) ) * dt      
            # Y[:,i+1] += sigma * np.random.normal(0, np.sqrt(dt), N_par) + chi / N_par * (- np.sum(Diff_Y[i] / (2 * np.pi) / (dist_diff + epsilon_sqr), axis = 1) ) * dt      
            X = X_0 + sigma * np.sqrt(dt) * x_noise +  ( Learned_Keller_grad_x) * dt  / N_par
            Y = Y_0 + sigma * np.sqrt(dt) * y_noise +  ( Learned_Keller_grad_y) * dt  / N_par  
            Z = Z_0 + sigma * np.sqrt(dt) * z_noise +  ( Learned_Keller_grad_z) * dt  / N_par
            W = W_0 + sigma * np.sqrt(dt) * w_noise +  ( Learned_Keller_grad_w) * dt  / N_par

            X_0 = np.copy(X); Y_0 = np.copy(Y); Z_0 = np.copy(Z); W_0 = np.copy(W_0) 
    
            
            if ((i + 1) % record_time_step) == 0: 
                X_old.append(X_0); Y_old.append(Y_0); Z_old.append(Z_0); W_old.append(W_0) 
    
        BIG_X.append(X_old); BIG_Y.append(Y_old); BIG_Z.append(Z_old); BIG_W.append(W_old)
                
      
        
    
    BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y); BIG_Z = np.array(BIG_Z); BIG_W = np.array(BIG_W)
    BIG_Data_learned = np.concatenate([BIG_X, BIG_Y, BIG_Z, BIG_W], axis = 3)
    
    
    with open('KS_SDE_4D_Particle_learned_trajectory_unif_50_reg_' + str(epsilon) + '_t_' + str(Stopping_time) + '_Chi_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data_learned)
        np.save(f, tau)
        np.save(f, observed_time_step)
     
    


        
        
        
        
        
        
        
