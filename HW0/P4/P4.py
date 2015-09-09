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
    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_m = np.loadtxt('P4_measurements.txt', delimiter=',')
    x_coords = 1.0 / rx * s_m[:,0]
    y_coords = 1.0 / ry * s_m[:,1]
    z_coords = 1.0 / rz * s_m[:,2]
    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.identity(6)
    A[0:3,3:6] = np.identity(3) * dt
    A[3:6,3:6] = np.identity(3) - c * dt * np.identity(3)
    a = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, g * dt]]).transpose()

    # Initial conditions for s0
    s0 = np.matrix([[0, 0, 2, 15, 3.5, 4.0]]).transpose()


    # Compute the rest of sk using Eq (1)
    s = np.zeros((6, K))
    s[0:6,0] = s0.transpose()


    # compute 
    for i in range(1, K):
        s[0:6, i] = (A * np.matrix(s[0:6, (i-1)]).transpose() + a).reshape((6)) # somehow python needs this reshaping command...

    x_coords = s[0,:]
    y_coords = s[1,:]
    z_coords = s[2,:]


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
