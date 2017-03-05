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
    ax.plot(strue[:, 0], strue[:, 1], strue[:, 2], '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measured = np.loadtxt('P4_measurements.txt', delimiter=',')[:120, :3]  # My file contains only 120 entries!

    position = strue[:, :3]

    r = [rx, ry, rz]

    r_matrix = np.matrix([[1/r[0], 0, 0], [0, 1/r[1], 0], [0, 0, 1/r[2]]])

    x = measured.copy()

    i = 0
    for row in measured:
        x[:][i] = np.dot(r_matrix, row.T)
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

    result = np.zeros((6, len(strue)))
    result = np.asmatrix(result)

    result[:, 0] = s0

    for i in range(len(strue) - 1):
        result[:, i+1] = A * result[:, i] + a

    result = np.asarray(result)

    ax.plot(result[0, :], result[1, :], result[2, :], '-k', label='Blind trajectory')

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

    C = np.matrix([[rx, 0, 0, 0, 0, 0],
                   [0, ry, 0, 0, 0, 0],
                   [0, 0, rz, 0, 0, 0]])

    measured_matrix = np.matrix(measured)

    def predictS(s):
        return np.dot(A, s) + a

    def predictSig(sigk):
        return np.linalg.inv(np.dot(np.dot(A, sigk), A.T) + (B * B.T))

    def updateSig(predicted_sig):
        return np.linalg.inv(predicted_sig + (C.T * C))

    def updateS(updated_sig, predicted_sig, predicted_S, m_next):
        return updated_sig * ((predicted_sig * predicted_S) + (C.T * m_next))

    sigma0 = 0.01 * np.identity(6)

    s_array = np.empty((6, measured_matrix.shape[0] - 1))  # K-1 since there are only 120 entries
    s_matrix = np.asmatrix(s_array)

    sigmak = sigma0
    sk = s0

    for n in range(0, measured_matrix.shape[0] - 1):

        # Predict
        predicted_s = predictS(sk)
        predicted_sigma = predictSig(sigmak)

        # Update
        updated_sigma = updateSig(predicted_sigma)
        s_next = updateS(updated_sigma, predicted_sigma, predicted_s, measured_matrix[n + 1].T)
        s_matrix[:, n] = s_next[:, 0]

        # Next steps
        sk = s_next
        sigmak = updated_sigma

    s_matrix = np.asarray(s_matrix)  # For matplotlib
    ax.plot(s_matrix[0, :], s_matrix[1, :], s_matrix[2, :], '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
