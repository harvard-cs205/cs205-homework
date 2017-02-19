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
    x_coords, y_coords, z_coords, x_velocity, y_velocity, z_velocity = s_true.T
    ax.plot(x_coords, y_coords, z_coords, '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_measured = np.loadtxt('P4_measurements.txt', delimiter=',')

    # store for part 4 before adjusting
    m = s_measured

    # adjust for dimensional stretching
    s_measured = np.multiply(s_measured, np.array([1. / rx, 1. / ry, 1. / rz]))

    x_coords, y_coords, z_coords = s_measured.T
    ax.plot(x_coords, y_coords, z_coords, '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1 - c * dt, 0, 0],
        [0, 0, 0, 0, 1 - c * dt, 0],
        [0, 0, 0, 0, 0, 1 - c * dt]
    ])

    a = np.array([0, 0, 0, 0, 0, g * dt])
    s_0 = np.array([0, 0, 2, 15, 3.5, 4.0])
    s = np.zeros([6, K])
    s[:, 0] = s_0
    s_k = s_0

    # n is k+1 in Eq. 1
    for n in xrange(1, K):
        s_n = (np.dot(A, s_k) + a)
        s[:, n] = s_n
        s_k = s_n

    x_coords, y_coords, z_coords = s[:3, :]

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.array([
        [bx, 0, 0, 0, 0, 0],
        [0, by, 0, 0, 0, 0],
        [0, 0, bz, 0, 0, 0],
        [0, 0, 0, bvx, 0, 0],
        [0, 0, 0, 0, bvy, 0],
        [0, 0, 0, 0, 0, bvz]
    ])

    C = np.array([
        [rx, 0, 0, 0, 0, 0],
        [0, ry, 0, 0, 0, 0],
        [0, 0, rz, 0, 0, 0],
    ])

    sigma = .01 * np.eye(6)
    s_matrix = np.zeros((6, K))
    s = np.array([0, 0, 2, 15, 3.5, 4.0])
    s_matrix[:, 0] = s.T

    def predictS(A, s_k, a):
        return np.dot(A, s_k) + a

    def predictSig(A, sigma_k, B):
        return np.linalg.inv(np.dot(np.dot(A, sigma_k), A.T) + np.dot(B, B.T))

    def updateSig(sigma_tilde, C):
        return np.linalg.inv(sigma_tilde + np.dot(C.T, C))

    def updateS(sigma_k1, sigma_tilde, s_tilde, C, m_k1):
        return np.dot(sigma_k1, np.dot(sigma_tilde, s_tilde) + np.dot(C.T, m_k1))

    for k in xrange(0, K - 1):
        s_tilde = predictS(A, s, a)
        sigma_tilde = predictSig(A, sigma, B)
        sigma = updateSig(sigma_tilde, C)
        s = updateS(sigma, sigma_tilde, s_tilde, C, m[k + 1])
        s_matrix[:, k + 1] = s

    x_coords, y_coords, z_coords = s_matrix[:3, :]

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
