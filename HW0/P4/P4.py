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
    # s_true = np.loadtxt("P4_trajectory.txt", delimiter=',')
    # x_coords = s_true[:,0]
    # y_coords = s_true[:,1]
    # z_coords = s_true[:,2]
    # ax.plot(x_coords, y_coords, z_coords, '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    # m_k = np.loadtxt("P4_measurements.txt", delimiter=',')
    # matrix = np.array([[1/rx, 0, 0], [0, 1/ry, 0], [0, 0, 1/rz]])
    # x_k = np.dot(m_k, matrix)
    # x_coords2 = x_k[:,0]
    # y_coords2 = x_k[:,1]
    # z_coords2 = x_k[:,2]
    # ax.plot(x_coords2, y_coords2, z_coords2, '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.array([[1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt], [0,0,0,1-c*dt,0,0], [0,0,0,0,1-c*dt,0], [0,0,0,0,0,1-c*dt]])
    a = np.transpose(np.matrix([0,0,0,0,0,g*dt]))
    s = np.asmatrix(np.zeros([6,K]))

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    s_0 = np.transpose(np.matrix([0,0,2,15,3.5,4.0]))
    s[:,0] = s_0
    for i in xrange(1,K):
        # Update the ith column in s according the (i-1)th column
        s[:,i] = A * s[:,i-1] + a

    # Convert s back to array; take first 3 rows as x,y,z coords respectively
    s = np.asarray(s)
    x_coords3 = s[0,:]
    y_coords3 = s[1,:]
    z_coords3 = s[2,:]

    ax.plot(x_coords3, y_coords3, z_coords3,
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
