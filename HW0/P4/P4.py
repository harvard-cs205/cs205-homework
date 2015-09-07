###############################

# CS 205 Fall 2015 Homework 0 Parts 4-6
# Submitted by Kendrick Lo (Harvard ID: 70984997)
# Github username: ppgmg

# Note to reader: 
# I am in the process of acquainting myself
# with Python; therefore, I may have included extraneous
# comments to document my process and observations for my
# own reference. These would be unnecessary if writing 
# for an audience that already understands the language.
# Commands used for debugging will be removed for 
# submission but can be found in earlier commits.

###############################

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    # Model parameters
    K = 121    # Number of times steps
    dt = 0.01  # Delta t
    c = 0.1    # Coefficient of drag
    g = -9.81  # Gravity
    # For constructing matrix B
    bx = 0
    by = 0
    bz = 0
    bvx = 0.25
    bvy = 0.25
    bvz = 0.1
    # For constructing matrix C
    rx = 1.0
    ry = 5.0
    rz = 5.0

    # Create 3D axes for plotting
    ax = Axes3D(plt.figure())

    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    #####################

    # We load the data from the file P4_trajectory.txt
    # which we assume is in the current working directory

    # We previewed the data and noted it appeared to consist
    # of 120 lines of six comma-separated values, with no
    # obvious anomalies or data entry errors/missing values

    filename_true = "P4_trajectory.txt"
    s_true = np.loadtxt(filename_true, delimiter=",")
    ## print s_true.shape, s_true.dtype 

    # extract first three columns to obtain series of positions
    x_coords = s_true[:, 0]
    y_coords = s_true[:, 1]
    z_coords = s_true[:, 2]
    ## print x_coords.shape, y_coords.shape, z_coords.shape

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # We load the data from the file P4_measurements.txt
    # which we assume is in the current working directory

    # We previewed the data and noted it appeared to consist
    # of 121 lines of three comma-separated values, with no
    # obvious anomalies or data entry errors/missing values

    filename_obs = "P4_measurements.txt"
    mT_obs = np.loadtxt(filename_obs, delimiter=",")
    ## print mT_obs.shape, mT_obs.dtype 

    # given the values of the scaling matrix, we could simply divide
    # all entries in the read array by the appropriate (rx, ry, rx)
    # and pull the coordinates for plotting;
    # while more complicated, we constructed the formula for x_est
    # to practice implementation of equations involving matrices
    m_obs = mT_obs.T  
    r_values = [rx, ry, rz]
    r_array = np.diag(np.array(r_values))

    # take inverse of diagonal matrix to get 1/rx, 1/ry, 1/rz
    x_est = np.dot(inv(r_array), m_obs)
    ## print m_obs.shape, r_array.shape, x_est.shape
    
    # extract first three ROWS to obtain series of positions
    x_coords = x_est[0]
    y_coords = x_est[1]
    z_coords = x_est[2]
    ## print x_coords.shape, y_coords.shape, z_coords.shape
 
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    ## debug check
    ## x_coords2 = mT_obs[:, 0]/float(rx)
    ## y_coords2 = mT_obs[:, 1]/float(ry)
    ## z_coords2 = mT_obs[:, 2]/float(rz)
    ## ax.plot(x_coords2, y_coords2, z_coords2,
    ##        '.r', label='test Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # initialize the array for s
    s_model = np.empty((6, K))
    # initialize s at t=0 with values from problem
    s_model[:,0] = np.array([0, 0, 2, 15, 3.5, 4.0]).T
    ## print s_model[:, 0]
    ## print s_model.shape

    # construct the array for A by filling in diagonals
    Adiag = np.array([1, 1, 1, 1-c*dt, 1-c*dt, 1-c*dt])
    Adiag3 = np.array([dt, dt, dt])
    A_prop = np.add(np.diag(Adiag), np.diag(Adiag3, k=3))
    ## print A_prop 
    
    # initialize column array a
    a = np.array([0, 0, 0, 0, 0, g*dt]).T
    ## print a, a.shape

    # compute the rest of s using Eq (1)
    # starting with column 1 (column 0 already initialized)
    # last column is (k-1)th
    for i in xrange(1, K, 1):
        s_model[:, i] = np.dot(A_prop, s_model[:, i-1])  # A*sk
        s_model[:, i] = np.add(s_model[:, i], a)  # +a
        ## print s_model[:, i]       

    # extract first three ROWS to obtain series of positions
    x_coords = s_model[0]
    y_coords = s_model[1]
    z_coords = s_model[2]    

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
