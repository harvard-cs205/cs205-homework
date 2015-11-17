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
    strue = np.loadtxt('P4_trajectory.txt', delimiter=',')
    ax.plot(strue[:, 0], strue[:, 1], strue[:, 2],
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measured = np.loadtxt('P4_measurements.txt', delimiter=',')[:120, :3]  # My file contains only 120 entries!

    position = strue[:, :3]

    r_array = np.divide(position, measured)

    r = [np.mean(r_array[:, 0]), np.mean(r_array[:, 1]), np.mean(r_array[:, 2])]

    C = np.matrix([[1/r[0], 0, 0], [0, 1/r[1], 0], [0, 0, 1/r[2]]])

    def mult(matrix_row):
        return C * matrix_row

    x = measured.copy()

    i = 0
    for row in measured:
        x[:][i] = np.dot(C, row.T)
        i += 1

    ax.plot(x[:, 0], x[:, 1], x[:, 2], '.g', label='Observed trajectory')


    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    A = np.matrix([[1, 0, 0, dt, 0, 0],
                   [0, 1, 0, 0, dt, 0],
                   [0, 0, 1, 0, 0, dt],
                   [0, 0, 0, 1 - c*dt, 0, 0],
                   [0, 0, 0, 0, 1 - c*dt, 0],
                   [0, 0, 0, 0, 0, 1 - c*dt]])

    a = np.matrix([[0, 0, 0, 0, 0, g*dt]]).T

    s0 = np.matrix('0 0 2 15 3.5 4.0').T

    result = np.zeros((6, K))

    i = 0
    s_current = s0
    for k in range(K):
        s_new = (A * s_current) + a
        result[:, i] = s_new[0]
        s_current = s_new
        i += 1

    ax.plot(result[0], result[1], result[2], '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    B = np.matrix([[bx, 0, 0, 0, 0, 0],
                   [0, by, 0, 0, 0, 0],
                   [0, 0, bz, 0, 0, 0],
                   [0, 0, 0, bvx, 0, 0],
                   [0, 0, 0, 0, bvy, 0],
                   [0, 0, 0, 0, 0, bvz]])



    def predictS(s):
        return (A * s) + a

    def predictSig(sigk):
        return np.linalg.inv(A * sigk * A.T + B * B.T)

    def updateSig(sig):
        return np.linalg.inv(predictSig(sig) + C.T * C)

    def updateS(s, sigk, m):
        return updateSig(sigk) * (predictSig(sigk) * s + C.T * m)

    sigma0 = 0.01 * np.identity(6)

    sigmak = sigma0
    sk = s0
    for mk in measured:
        predicted_s = predictS(s0)
        predicted_sigma = predictSig(sigmak)
        sigma = updateSig(predicted_sigma)
        sk = updateS(sk, sigmak, mk)
        print sk
        # store sk after


    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
