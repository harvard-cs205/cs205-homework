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
    s_true = np.loadtxt('P4_trajectory.txt',delimiter = ',')
    x_coords = [x[0] for x in s_true]
    y_coords = [y[1] for y in s_true]
    z_coords = [z[2] for z in s_true]
    ax.plot(x_coords, y_coords, z_coords,'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_ob = np.loadtxt('P4_measurements.txt',delimiter = ',')
    x_coords_ob = [x[0] for x in s_ob]
    y_coords_ob = [x[1] for x in s_ob]
    z_coords_ob = [x[2] for x in s_ob]
    #print s_ob
    m_matrix = np.array([x_coords_ob,y_coords_ob,z_coords_ob])
    #print m_matrix.shape
    c_matrix = np.array([1/rx,1/ry,1/rz])
    c_matrix = np.diag(c_matrix)
    #print m_matrix.shape
    x_matrix = np.dot(c_matrix,m_matrix)
    #print x_matrix.shape
    ax.plot(x_matrix[0], x_matrix[1], x_matrix[2],'.g', label='Observed trajectory')
    #print z_coords
    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.zeros(36).reshape(6,6)
    np.fill_diagonal(A,1)
    np.fill_diagonal(A[:-3,3:],dt)
    np.fill_diagonal(A[3:6,3:6],1-c*dt)
    #print A
    a = np.zeros(6)
    #print a
    a = np.matrix(a)
    a.itemset(5,g*dt)
    a = a.T
    #print a
    s_0 = np.matrix([0.0,0.0,2.0,15.0,3.5,4.0])
    s_0 = s_0.T
    s_total = s_0
    s_k_1 = s_0
    for i in xrange(0,K):
        s_k_1 = A*s_k_1+a
        s_total = np.c_[s_total,s_k_1]
    dd = s_total[0]
    #print np.array(s_total[0]).flatten(),np.array(s_total[1]).flatten(),np.array(s_total[2]).flatten()
    x_coords_s = [x for x in np.array(s_total[0]).flatten()]
    y_coords_s = [x for x in np.array(s_total[1]).flatten()]
    z_coords_s = [y for y in np.array(s_total[2]).flatten()]
    #print x_coords_s[0]
    # A = ?
    # a = ?
    # s = ?
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    ax.plot(x_coords_s, y_coords_s, z_coords_s,'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    A_2 = np.zeros(36).reshape(6,6)
    A_2[3,3] = bvx
    A_2[4,4] = bvy
    A_2[5,5] = bvz
    print A_2
    # B = ?
    # C = ?

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    #ax.legend()
    #plt.show()