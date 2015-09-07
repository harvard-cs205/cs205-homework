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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    # print(s_true)
    # print("2nd column")
    # print(s_true[:,1])
    
    x_coords_truth = s_true[:,0]
    y_coords_truth = s_true[:,1]
    z_coords_truth = s_true[:,2]

    # Plot True Trajectory Coordinates
    ax.plot(x_coords_truth, y_coords_truth, z_coords_truth,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_obs = np.loadtxt('P4_measurements.txt', delimiter=',')

    ax.plot(s_obs[:,0]/rx, s_obs[:,1]/ry, s_obs[:,2]/rz,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    left_top = np.zeros((3, 3), float)
    np.fill_diagonal(left_top, 1)
    right_top = np.zeros((3, 3), float)
    np.fill_diagonal(right_top, dt)

    left_bot = np.zeros((3, 3), float)
    right_bot = np.zeros((3, 3), float)
    np.fill_diagonal(right_bot, 1-c*dt)

    top = np.hstack([left_top, right_top])
    bottom = np.hstack([left_bot, right_bot])
    top
    bottom

    A = np.vstack([top, bottom])
    A

    # a
    a = np.matrix([0,0,0,0,0,g*dt])
    a = a.transpose()

    # s 6x121
    s = np.matrix(np.zeros((6, K-1), float))


    # Initial conditions for s0
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0])
    s0 = s0.transpose()
    s[:,0] = s0

    # Compute the rest of sk using Eq (1)
    for i in range(1, K-1):
        s[:,i] = A * s[:,i-1] + a

    s = np.asarray(s)
    ax.plot(s[0,:], s[1,:], s[2,:], '-k', label='Blind trajectory')

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
