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

    # Load true trajectory data
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')

    # Plot
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # Load observed trajectory data
    m_measured = np.loadtxt('P4_measurements.txt', delimiter=',')

    # Scale values
    scale = np.array([1/rx, 1/ry, 1/rz])

    # Compute adjusted matrix
    x_adjusted = m_measured * scale

    # Plot
    ax.plot(x_adjusted[:,0], x_adjusted[:,1], x_adjusted[:,2],
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    # Set propogation matrix globally
    A = np.matrix([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1-c*dt, 0, 0],
        [0, 0, 0, 0, 1-c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]
    ])

    # Set gravity matrix globally
    a = np.matrix([0, 0, 0, 0, 0, g*dt]).transpose()

    # Init matrix to hold s values
    s_blind = np.zeros([6, K])
    s_blind = np.asmatrix(s_blind)

    # Populate values for s0
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()
    s_blind[:,0] = s0

    # Populate rest of the values
    for k in xrange(1, K):
        s_blind[:,k] = A * s_blind[:,k-1] + a

    # Set as array to display coordinates
    s_blind = np.array(s_blind)

    # Plot
    ax.plot(s_blind[0], s_blind[1], s_blind[2],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # Set scale of errors matrix globally
    B = np.matrix([
        [bx, 0, 0, 0, 0, 0],
        [0, by, 0, 0, 0, 0],
        [0, 0, bz, 0, 0, 0],
        [0, 0, 0, bvx, 0, 0],
        [0, 0, 0, 0, bvy, 0],
        [0, 0, 0, 0, 0, bvz]
    ])

    # Set capacity matrix globally
    C = np.matrix([
        [rx, 0, 0, 0, 0, 0],
        [0, ry, 0, 0, 0, 0],
        [0, 0, rz, 0, 0, 0]
    ])

    ###
    # Predict intermediary coordinates values
    #
    # @param sk  coordinates and velocities at kth time interval
    #
    # @return s matrix
    ###
    def predictS(sk):
        return A * sk + a

    ###
    # Predict intermediary Sigma value
    #
    # @param SigmaK  covariance matrix at kth interval
    #
    # @return Sigma~ matrix
    ###
    def predictSig(SigmaK):
        return inv(A * SigmaK * A.transpose() + B * B.transpose())

    ###
    # Return Sigma covariance matrix at k+1th time interval
    #
    # @param SigmaTilde  predicted intermediary covariance matrix
    #
    # @return SigmaKPlus1 matrix
    ###
    def updateSig(SigmaTilde):
        return inv(SigmaTilde + C.transpose() * C)

    ###
    # Predict s~ value
    #
    # @param SigmaKPlus1  covariance matrix at k+1th time interval
    # @param SigmaTilde   predicted intermediary covariance matrix
    # @param sTilde       intermediary coordinates matrix
    # @param MkPlus1      measurements at k+1th time interval 
    #
    # @return s~ matrix
    ###
    def updateS(SigmaKPlus1, SigmaTilde, sTilde, MkPlus1):
        return SigmaKPlus1 * (SigmaTilde * sTilde + C.transpose() * MkPlus1)

    # Set initial SigmaK
    SigmaK = 0.01*np.identity(6)

    # Init matrix to hold s values
    s_kalman = np.zeros([6, K])
    s_kalman = np.asmatrix(s_kalman)

    # Populate values for s0 (using s0 assigned before)
    s_kalman[:,0] = s0

    # Set measured data points as matrix
    m_matrix = np.matrix(m_measured)

    # Populate rest of the values
    for k in xrange(1, K):
        # Get previous values
        sk = s_kalman[:,k-1]

        # Get kth measurement data
        MkPlus1 = m_matrix[k,:].transpose()

        # Get next coordinate set
        sTilde = predictS(sk)
        SigmaTilde = predictSig(SigmaK)
        SigmaKPlus1 = updateSig(SigmaTilde)
        skPlus1 = updateS(SigmaKPlus1, SigmaTilde, sTilde, MkPlus1)
        s_kalman[:,k] = skPlus1

        # Set SigmaK for next iteration
        SigmaK = SigmaKPlus1
        
    # Set kalman matrix as array
    s_kalman = np.array(s_kalman)

    # Plot
    ax.plot(s_kalman[0], s_kalman[1], s_kalman[2],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
