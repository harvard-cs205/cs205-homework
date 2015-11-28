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
    x_coords = [row[0] for row in s_true]
    y_coords = [row[1] for row in s_true]
    z_coords = [row[2] for row in s_true]
    ax.plot(x_coords, y_coords, z_coords, '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    measured = np.loadtxt('P4_measurements.txt', delimiter=',')
    x_coords = [(1/rx)*row[0] for row in measured]
    y_coords = [(1/ry)*row[1] for row in measured]
    z_coords = [(1/rz)*row[2] for row in measured]
    ax.plot(x_coords, y_coords, z_coords, '.g', label='Observed trajectory')

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
    a = np.matrix([0, 0, 0, 0, 0, g*dt]).transpose()

    # Initial conditions for s0
    s = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()

    # Compute the rest of sk using Eq (1)
    for i in range(1, K):
        sk = A*(s[:, i-1])+a
        s = np.append(s, sk, axis=1)

    s = np.array(s)
    x_coords = s[0]
    y_coords = s[1]
    z_coords = s[2]
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

    # Initial conditions for s0 and Sigma0
    s = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()
    Sigma0 = 0.01*np.identity(6)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    # return ~s when k
    def predictS(k):
        return A*(s[:, k])+a

    # return ~sigma when k
    def predictSig(k):
        return inv(A*updateSig(k-1)*A.transpose() + B*B.transpose())

    # return sigma_k+1 when k
    def updateSig(k):
        if k==-1:
            return Sigma0
        else:
            return inv(predictSig(k) + C.transpose()*C)

    # return s_k+1 when k
    def updateS(k):
        return updateSig(k)*(predictSig(k)*predictS(k) + C.transpose()*np.asmatrix(measured[k+1]).transpose())


    for k in range(K-1):
        s_kplus1 = updateS(k)
        s = np.append(s, s_kplus1, axis=1)

    s = np.array(s)
    x_coords = s[0]
    y_coords = s[1]
    z_coords = s[2]
    ax.plot(x_coords, y_coords, z_coords, '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
