import numpy as np
from numpy.linalg import inv
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
    x_coords = [row[0] for row in s_true]
    y_coords = [row[1] for row in s_true]
    z_coords = [row[2] for row in s_true]

    ax.plot(x_coords, y_coords, z_coords,'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_obs = np.loadtxt('P4_measurements.txt', delimiter=',')
    x_coords = [row[0]/rx for row in s_obs]
    y_coords = [row[1]/ry for row in s_obs]
    z_coords = [row[2]/rz for row in s_obs]
    ax.plot(x_coords, y_coords, z_coords,'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1, 0, 0, dt, 0, 0],
                   [0, 1, 0, 0, dt, 0],
                   [0, 0, 1, 0, 0, dt],
                   [0, 0, 0, 1-c*dt, 0, 0],
                   [0, 0, 0, 0, 1-c*dt, 0],
                   [0, 0, 0, 0, 0, 1-c*dt]])
    a = np.matrix([0, 0, 0, 0, 0, g * dt])
    s = np.matrix([0, 0, 2, 15, 3.5, 4.0])

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    z = np.zeros([K, 6])
    model = np.asmatrix(z)
    model[0] = s
    for i in range(1, K):
        sk = A * s.transpose() + a.transpose()
        s = sk.transpose()
        model[i] = s

    x_coords = [model.item(i, 0) for i in xrange(K)]
    y_coords = [model.item(i, 1) for i in xrange(K)]
    z_coords = [model.item(i, 2) for i in xrange(K)]
    ax.plot(x_coords, y_coords, z_coords, '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([[bx, 0, 0, 0, 0, 0],
                   [0, by, 0, 0, 0, 0],
                   [0, 0, bz, 0, 0, 0],
                   [0, 0, 0, bvx, 0, 0],
                   [0, 0, 0, 0, bvy, 0],
                   [0, 0, 0, 0, 0, bvz]])
    C = np.matrix([[rx, 0, 0, 0, 0, 0],
                   [0, ry, 0, 0, 0, 0],
                   [0, 0, rz, 0, 0, 0]])

    def predictS(state):
        return A * state.transpose() + a.transpose()

    def predictSig(sig):
        return inv(A * sig * A.transpose() + B * B.transpose())

    def updateSig(pSig):
        return inv(pSig + C.transpose() * C)

    def updateS(uSig, pSig, pState, meas):
        return uSig * (pSig * pState + C.transpose() * meas)

    # Initial conditions for s0 and Sigma0

    s = np.matrix([0, 0, 2, 15, 3.5, 4.0])
    Sigma = np.multiply(0.01, np.identity(6))

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    s_meas = np.asmatrix(s_obs)
    z = np.zeros([K, 6])
    kalman = np.asmatrix(z)
    kalman[0, :] = s
    for i in range(1, K):
        pState = predictS(s)
        pSigma = predictSig(Sigma)
        Sigma = updateSig(pSigma)
        sk = updateS(Sigma, pSigma, pState, s_meas[i].transpose())
        s = sk.transpose()
        kalman[i, :] = s

    x_coords = [kalman.item(i, 0) for i in xrange(K)]
    y_coords = [kalman.item(i, 1) for i in xrange(K)]
    z_coords = [kalman.item(i, 2) for i in xrange(K)]

    ax.plot(x_coords, y_coords, z_coords, '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
