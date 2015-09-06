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

###############################

import numpy as np
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
    s_true = np.loadtxt(filename_true,delimiter=",")
    # print s_true.shape, s_true.dtype 

    # extract first three columns to obtain series of positions
    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]
    # print x_coords.shape, y_coords.shape, z_coords.shape

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # ax.plot(x_coords, y_coords, z_coords,
    #         '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-k', label='Blind trajectory')

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
