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

    s_true = np.loadtxt( "P4_trajectory.txt", delimiter = ',', usecols = ( 0, 1, 2 ) ).transpose()
    ax.plot( s_true[0], s_true[1], s_true[2], '--b', label = 'True trajectory' )

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    measurements_matrix = np.mat( np.loadtxt( "P4_measurements.txt", delimiter = ',' ).transpose() )
    dim_stretching_inverse_matrix = np.mat( [ [ 1/rx, 0, 0 ], [ 0, 1/ry, 0 ], [ 0, 0, 1/rz ] ] )
    position_estimate_matrix = dim_stretching_inverse_matrix * measurements_matrix
    
    position_estimate_array = np.array( position_estimate_matrix )
    
    ax.plot( position_estimate_array[0], position_estimate_array[1], position_estimate_array[2], 
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    s_initial = array( [ 0, 0, 2, 15, 3.5, 4.0 ] )
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
