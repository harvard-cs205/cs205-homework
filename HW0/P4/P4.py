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

    # Generate propogation matrix
    A = np.matrix([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1-c*dt, 0, 0],
        [0, 0, 0, 0, 1-c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]
    ])

    # Account for gravity
    a = np.matrix([0, 0, 0, 0, 0, g*dt]).transpose()

    # Init matrix to hold s values
    s_blind = np.zeros([6, K])
    s_blind = np.asmatrix(s_blind)

    # Populate values for s0
    s_blind[0,0] = 0
    s_blind[1,0] = 0
    s_blind[2,0] = 2
    s_blind[3,0] = 15
    s_blind[4,0] = 3.5
    s_blind[5,0] = 4.0

    # Populate rest of the values
    for k in xrange(1, K):
        sk = np.matrix([s_blind[0,k-1], s_blind[1,k-1], s_blind[2,k-1], s_blind[3,k-1], s_blind[4,k-1], s_blind[5,k-1]]).transpose()
        skplus1 = A*sk + a
        s_blind[0,k] = skplus1[0]
        s_blind[1,k] = skplus1[1]
        s_blind[2,k] = skplus1[2]
        s_blind[3,k] = skplus1[3]
        s_blind[4,k] = skplus1[4]
        s_blind[5,k] = skplus1[5]

    s_blind = np.array(s_blind)
    # Plot
    ax.plot(s_blind[0], s_blind[1], s_blind[2],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    def predictS(A, sk, a):
        return A*sk + a

    def predictSig(A, B, Ek):
        return inv(A*Ek*A.transpose() + B*B.transpose())

    def updateSig(E, C):
        return inv(E + C.transpose()*C)

    def updateS(Ekplus1, E, s, C, mkplus1):
        return Ekplus1*(E*s+C.transpose()*mkplus1)

    Ek = 0.01*np.identity(6)

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

    # Init matrix to hold s values
    s_kalman = np.zeros([6, K])
    s_kalman = np.asmatrix(s_kalman)

    # Populate values for s0
    s_kalman[0,0] = 0
    s_kalman[1,0] = 0
    s_kalman[2,0] = 2
    s_kalman[3,0] = 15
    s_kalman[4,0] = 3.5
    s_kalman[5,0] = 4.0

    m_matrix = np.matrix(m_measured)

    # Populate rest of the values
    for k in xrange(K-1):
        sk = np.matrix([s_kalman[0,k], s_kalman[1,k], s_kalman[2,k], s_kalman[3,k], s_kalman[4,k], s_kalman[5,k]]).transpose()

        # A and a were assigned before
        s = predictS(A, sk, a)

        E = predictSig(A, B, Ek)

        Ekplus1 = updateSig(E, C)

        mkplus1 = m_matrix[k+1,:].transpose()
        print(mkplus1)

        skplus1 = updateS(Ekplus1, E, s, C, mkplus1)
        Ek = Ekplus1
        s_kalman[0,k+1] = skplus1[0]
        s_kalman[1,k+1] = skplus1[1]
        s_kalman[2,k+1] = skplus1[2]
        s_kalman[3,k+1] = skplus1[3]
        s_kalman[4,k+1] = skplus1[4]
        s_kalman[5,k+1] = skplus1[5]    

    s_kalman = np.array(s_kalman)

    ax.plot(s_kalman[0], s_kalman[1], s_kalman[2],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
