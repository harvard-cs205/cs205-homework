import numpy as np
import pdb
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

    s_true = np.transpose(np.loadtxt('P4_trajectory.txt', delimiter=','))
    ax.plot(s_true[0,:], s_true[1,:], s_true[2,:],
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    m = np.transpose(np.loadtxt('P4_measurements.txt', delimiter=','))
    ax.plot(m[0,:] / rx, m[1,:] / ry, m[2,:] / rz,
            '--g', label='True trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1 - c * dt, 0, 0], [0, 0, 0, 0, 1 - c * dt, 0], [0, 0, 0, 0, 0, 1 - c * dt]])
    a = np.matrix(np.zeros((6, 1)))
    a[5] = g * dt
    s = np.matrix(np.zeros((6, K)))

    s[:, 0] = np.transpose([[0, 0, 2, 15, 3.5, 4.0]])
    for i in xrange(1, K, 1):
        s[:, i] = A * s[:, i - 1] + a

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    s = np.array(s)
    ax.plot(s[0,:], s[1,:], s[2,:],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([[bx, 0, 0, 0, 0, 0], [0, by, 0, 0, 0, 0], [0, 0, bz, 0, 0, 0], [0, 0, 0, bvx, 0, 0], [0, 0, 0, 0, bvy, 0], [0, 0, 0, 0, 0, bvz]])
    C = np.matrix([[rx, 0, 0, 0, 0, 0], [0, ry, 0, 0, 0, 0], [0, 0, rz, 0, 0, 0]])

    # Initial conditions for s0 and Sigma0
    s = np.matrix(np.zeros((6, K)))
    s[:, 0] = np.transpose([[0, 0, 2, 15, 3.5, 4.0]])
    sigma = 0.01 * np.matrix(np.identity(6))
    m = np.matrix(m)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    def predictS(sk):
        return A * sk + a

    def predictSig(sigk):
        return np.linalg.pinv(A * sigk * np.transpose(A) + B * np.transpose(B))

    def updateSig(sigtil):
        return np.linalg.pinv(sigtil + np.transpose(C) * C)

    def updateS(sigk, stil, sigtil, m, k):
        return updateSig(predictSig(sigk)) * (sigtil * stil + np.transpose(C) * m[:, k + 1])

    # pdb.set_trace()
    for k in xrange(0, K - 1, 1):
        stil = predictS(s[:, k])
        s[:, k + 1] = updateS(sigma, stil, predictSig(sigma), m, k)
        sigma = updateSig(stil)

    s = np.array(s)
    ax.plot(s[0,:], s[1,:], s[2,:],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
