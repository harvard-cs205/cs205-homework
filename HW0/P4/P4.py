import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from StringIO import StringIO

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

    s_true = np.loadtxt('P4_trajectory.txt', 
             delimiter = ',', unpack = True)
    ax.plot(s_true[0], s_true[1], s_true[2],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    #C = np.matrix([1.0/rx,0,0],[0,1.0/ry,0],[0,0,1.0/rz])

    s_measured = np.loadtxt('P4_measurements.txt', 
             delimiter = ',', unpack = True)

    ax.plot(s_measured[0] * 1.0/rx, s_measured[1] * 1.0/ry, s_measured[2] * 1.0/rz,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    drag_per_dt = 1 - c*dt
    A = np.matrix([
                    [1,0,0,dt,0,0],
                    [0,1,0,0,dt,0],
                    [0,0,1,0,0,dt],
                    [0,0,0,drag_per_dt,0,0],
                    [0,0,0,0,drag_per_dt,0],
                    [0,0,0,0,0,drag_per_dt]
                 ])
    #print A
    a = np.matrix([
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [g*dt]
                 ])
    #print a

    # Initial conditions for s0
    s_column = np.matrix([
                    [s_true[0][0]],
                    [s_true[1][0]], 
                    [s_true[2][0]], 
                    [s_true[3][0]],
                    [s_true[4][0]],
                    [s_true[5][0]]
                 ])
    #print s_column

    s = np.zeros([6,K])
    s = np.asmatrix(s)

    # construct s using Eq (1)
    s[:,0] = s_column
    for k in range(1,K):
        s[:,k] = A*s_column + a
        s_column = s[:,k]

    s = np.array(s)
    x_coords = s[0]
    y_coords = s[1]
    z_coords = s[2]

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
