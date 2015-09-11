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
    x_true = s_true[:, 0]
    y_true = s_true[:, 1]
    z_true = s_true[:, 2]
    ax.plot(x_true, y_true, z_true, '--b', label = 'True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_measure = np.loadtxt('P4_measurements.txt', delimiter=',')
    correction = [[1.0/rx, 0, 0], [0, 1.0/ry, 0], [0, 0, 1.0/rz]]
    s_measure_trans = s_measure.transpose()
    s_measure_adj = np.dot(correction, s_measure_trans)
    x_measure = s_measure_adj[0]
    y_measure = s_measure_adj[1]
    z_measure = s_measure_adj[2]
    ax.plot(x_measure, y_measure, z_measure, '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?
    s_matrix = np.zeros([6,K], dtype=float)
    propagation = np.zeros([6,6],dtype=float)
    propagation_diag = np.array([1.0, 1.0, 1.0, 1.0-c*dt, 1.0-c*dt, 1.0-c*dt])
    propagation_diag3 = np.array([dt, dt, dt])
    propagation = propagation + np.diag(propagation_diag) + np.diag(propagation_diag3, k=3)
    acceleration = np.array([0.0, 0.0, 0.0, 0.0, 0.0, g*dt]).transpose()

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    s_matrix[:, 0] = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0]).transpose()

    for k in range(1, K):
        s_matrix[:, k] = np.add(np.dot(propagation, s_matrix[:, k-1]), acceleration)
    x_calc = s_matrix[0]
    y_calc = s_matrix[1]
    z_calc = s_matrix[2]

    ax.plot(x_calc, y_calc, z_calc, '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?
    A_matrix = propagation
    B_diag = np.array([bx, by, bz, bvx, bvy, bvz])
    B_matrix = np.diag(B_diag)
    C_matrix = np.hstack([np.diag(np.array([rx, ry, rz])), np.zeros([3,3])])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    s2_matrix = np.zeros([6, K])
    s2_matrix[:, 0] = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0]).transpose()
    sig_diag = np.array([1, 1, 1, 1, 1, 1]) * 0.01
    sig = np.diag(sig_diag)

    def predictS():
        s2_est = np.add(np.dot(propagation, s2_prev), acceleration)
        return s2_est

    def predictSig():
        sig_est = np.linalg.inv(((A_matrix * sig) * A_matrix.transpose()) + B_matrix * B_matrix.transpose())
        return sig_est

    def updateSig():
        sig = np.linalg.inv(sig_est + np.dot(C_matrix.transpose(), C_matrix))
        return sig

    def updateS():
        s2 = np.dot(sig, (np.dot(sig_est, s2_est) + np.dot(C_matrix.transpose(), s_measure_trans[:, k])))
        return s2

    for k in range(1, K):
        s2_prev = s2_matrix[:, k-1]
        s2_est = predictS()
        sig_est = predictSig()
        sig = updateSig()
        s2 = updateS()
        s2_matrix[:, k] = s2

    x_s2 = s2_matrix[0]
    y_s2 = s2_matrix[1]
    z_s2 = s2_matrix[2]


    ax.plot(x_s2, y_s2, z_s2, '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
