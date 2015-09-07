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

    s_true = np.loadtxt('P4_trajectory.txt', delimiter = ",") 
        #Load the trajectory text file into an array

    x_coords = s_true[:, 0]  # extract x coordinates
    y_coords = s_true[:, 1]  # extract y coordinates
    z_coords = s_true[:, 2]  # extract z coordinates

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_meas = np.loadtxt('P4_measurements.txt', delimiter = ",")
        #Load the measurements text file into an array


    x_coords = s_meas[:, 0]  # extract x coordinates
    y_coords = s_meas[:, 1]  # extract y coordinates
    z_coords = s_meas[:, 2]  # extract z coordinates

    s_meas_matrix = np.matrix([x_coords, y_coords, z_coords])
        # Put them into a matrix to do matrix multiplication with

    stretch_matrix = np.matrix([[1/rx, 0, 0], [0, 1/ry, 0], [0, 0, 1/rz]])
        # Matrix with the stretching factors

    corrected = stretch_matrix * s_meas_matrix
        # Correct position by multiplying the stretch matrix by the measured position

    # Now you have to get the coordinates out a different way because it's in a matrix,
    #    not an array.  So I convert it to an array first.  
    x_coords = np.array(corrected)[0, :]  # extract x coordinates
    y_coords = np.array(corrected)[1, :]  # extract y coordinates
    z_coords = np.array(corrected)[2, :]  # extract z coordinates


    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Propagation Matrix
    A = np.matrix([[1, 0, 0,         dt,          0,          0],
                   [0, 1, 0,          0,         dt,          0],
                   [0, 0, 1,          0,          0,         dt],
                   [0, 0, 0, (1 - c*dt),          0,          0],
                   [0, 0, 0,          0, (1 - c*dt),          0],
                   [0, 0, 0,          0,          0, (1 - c*dt)]])
    
    # Matrix a is the component of the motion that is changed by gravity in 
    #    the time interval delta-t
    a = np.transpose(np.matrix([0, 0, 0, 0, 0, g*dt]))
    
    # Initial conditions for s0
    s = np.transpose(np.matrix([0, 0, 2, 15, 3.5, 4.0]))

    # Empty matrix to be filled with predicted values
    s_pred = np.matrix(np.empty(shape=(6, (K-1))))

    # Insert inistial conditions into the first row
    s_pred[:, 0] = s
    
    # Compute the rest of sk using Eq (1), one column at a time
    for i in range(1, K-1):
        s_pred[:, i] = A * s_pred[:, i-1] + a

    # Extract coordinate values from the top three rows
    x_coords = np.array(s_pred)[0, :]  # extract x coordinates
    y_coords = np.array(s_pred)[1, :]  # extract y coordinates
    z_coords = np.array(s_pred)[2, :]  # extract z coordinates

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
