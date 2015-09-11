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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter = ',', unpack = True)
    #s_true = np.loadtxt('P4_trajectory.txt', delimiter = ',', usecols=[0,1,2])
    # Normally, this data wouldn't be available in the real world
    #####################

    ax.plot(s_true[0], s_true[1], s_true[2], '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measure = np.loadtxt('P4_measurements.txt', delimiter = ',', unpack = True)

    ax.plot((1 / rx) * measure[0], (1 / ry) * measure[1], (1 / rz) * measure[2], 
             '.g', label = 'Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################


    # Initial conditions for s0
    # s = ?
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()
    # Initial conditions for a
    # a = ?
    a = np.matrix([0, 0, 0, 0, 0, g * dt]).transpose()

    # Construct a large 6K matrix A
    # A = ?
    A = np.asmatrix(np.zeros([6, 6]))
    A[0, 0], A[1, 1], A[2, 2] = 1, 1, 1
    A[0, 3], A[1, 4], A[2, 5] = dt, dt, dt
    A[3, 3], A[4, 4], A[5, 5] = 1 - c * dt, 1 - c * dt, 1 - c * dt
    
    # Fill it in one column at a time
    prePosMat = np.asmatrix(np.zeros([6, K]))
    prePosMat[:, 0] = s0

    # Compute the rest of sk using Eq (1)
    for i in range(0, K - 1):
        prePosMat[:, (i + 1)] = ( A * prePosMat[:, i] ) + a
        i += 1

    predPos = np.asarray(prePosMat)
    # Plot
    ax.plot(predPos[0], predPos[1], predPos[2],
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    B = np.asmatrix(np.zeros([6, 6]))
    B[0, 0] = bx
    B[1, 1] = by
    B[2, 2] = bz
    B[3, 3] = bvx
    B[4, 4] = bvy
    B[5, 5] = bvz
    # C = ?
    C = np.asmatrix(np.zeros([3, 6]))
    C[0, 0] = rx
    C[1, 1] = ry
    C[2, 2] = rz

    # Define Four Python Functions
    def predictS(A, sk, a):
        return (A * sk + a)

    def predictSig(A, sigmak, B):
        return (A * sigmak * A.T + B * B.T).I

    def updateSig(sigma_pred, C):
        return (sigma_pred + C.T * C).I

    def updateS(sigmak1, sigma_pred, s_pred, C, mk1):
        return (sigmak1 * (sigma_pred * s_pred + C.T * mk1))

    # Initial conditions for s0 and Sigma0
    Sigma0 = 0.01 * np.identity(6)
    Sigmak = Sigma0
    sk = s0
    m_f = np.asmatrix(measure)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    kalmanMat = np.asmatrix(np.zeros([6, K]))
    kalmanMat[:, 0] = sk


    for i in range(0, K - 1):
        s_pred = predictS(A, kalmanMat[:, i], a)
        sigma_pred = predictSig(A, Sigmak, B)
        sigmak = updateSig(sigma_pred, C)
        kalmanMat[:, (i + 1)] = updateS(sigmak, sigma_pred, s_pred, C, m_f[:, (i + 1)])
        i += 1

    kalman = np.asarray(kalmanMat)
    
    ax.plot(kalman[0, :], kalman[1, :], kalman[2, :],
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
