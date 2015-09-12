import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

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
    x_coords = s_true[:, 0]
    y_coords = s_true[:, 1]
    z_coords = s_true[:, 2]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    meas = np.loadtxt('P4_measurements.txt', delimiter=',')
    x_coords = (1/rx) * meas[:, 0]
    y_coords = (1/ry) * meas[:, 1]
    z_coords = (1/rz) * meas[:, 2]
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='measerved trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    A = np.matrix([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1 - c*dt, 0, 0],
            [0, 0, 0, 0, 1 - c*dt, 0],
            [0, 0, 0, 0, 0, 1 - c*dt]
        ])
    a = np.matrix([0, 0, 0, 0, 0, g*dt]).transpose()
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()

    s_blind = np.asmatrix(np.zeros([6, K]))
    s_blind[:, 0] = s0

    for k in xrange(1, K):
        s_blind[:, k] = A * s_blind[:, k-1] + a

    s_blind = np.array(s_blind)
    x_coords = s_blind[0]
    y_coords = s_blind[1]
    z_coords = s_blind[2]

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

    B = np.matrix([
            [bx, 0, 0, 0, 0, 0],
            [0, by, 0, 0, 0, 0],
            [0, 0, bz, 0, 0, 0],
            [0, 0, 0, bvx, 0, 0],
            [0, 0, 0, 0, bvy, 0],
            [0, 0, 0, 0, 0, bvz]
        ])

    C = np.matrix([
            [rx, 0, 0, 0, 0, 0],
            [0, ry, 0, 0, 0, 0],
            [0, 0, rz, 0, 0, 0]
        ])
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()
    sigma0 = np.asmatrix(0.1 * np.identity(6))

    sk_matrix = np.asmatrix(np.zeros([6, K]))
    sk_matrix[:, 0] = s0

    def predictS(sk):
        return A * sk + a

    def predictSig(sigma_k):
        return inv(A * sigma_k * A.transpose() + B * B.transpose())

    def updateSig(sigma_tilde):
        return inv(sigma_tilde + C.transpose() * C)

    def updateS(sigma, sigma_tilde, s_tilde, mk):
        return sigma * (sigma_tilde * s_tilde + C.transpose() * mk)

    curr_sigma = sigma0
    for k in xrange(K-1):
        s_tilde = predictS(sk_matrix[:, k])
        sigma_tilde = predictSig(curr_sigma)
        curr_sigma = updateSig(sigma_tilde)
        mk = np.asmatrix(meas[k+1]).transpose()
        curr_sk = updateS(curr_sigma, sigma_tilde, s_tilde, mk)
        sk_matrix[:, k+1] = curr_sk

    sk_matrix = np.array(sk_matrix)
    x_coords = sk_matrix[0]
    y_coords = sk_matrix[1]
    z_coords = sk_matrix[2]

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
