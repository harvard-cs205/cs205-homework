import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(s):
    return np.dot(A, s) + a

def predictSig(Sigma):
    return np.linalg.inv(np.dot(np.dot(A, Sigma), AT) + BBT)

def updateSig(Sigma_tilde):
    return np.linalg.inv(Sigma_tilde + CTC)

def updateS(s, Sigma, m_next):
    s_tilde     = predictS(s)
    Sigma_tilde = predictSig(Sigma)
    Sigma_new   = updateSig(Sigma_tilde)
    return np.dot(Sigma_new,
        np.dot(Sigma_tilde, s_tilde) + np.dot(CT , m_next))

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
    [x_coords, y_coords, z_coords] = np.transpose(s_true[:, 0:3])

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    m = np.loadtxt('P4_measurements.txt', delimiter=',')
    [x_coords, y_coords, z_coords] = np.transpose(m)

    ax.plot(x_coords/rx, y_coords/ry, z_coords/rz,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.eye(6)
    A[3,3] = A[4,4] = A[5,5] = 1-c*dt
    A[0,3] = A[1,4] = A[2,5] = dt

    a = np.zeros(6)
    a[-1] = g*dt

    s0 = np.array([0, 0, 2, 15, 3.5, 4.0])
    S = np.zeros((6,K))
    S[:,0] = s0

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    for k in range(0, K-1):
        S[:,k+1] = np.dot(A, S[:,k]) + a

    [x_coords, y_coords, z_coords] = S[0:3]

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    AT = np.transpose(A)

    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    BT = np.transpose(B)
    BBT = np.dot(B, BT)

    C = np.zeros((3,6))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz
    CT = np.transpose(C)
    CTC = np.dot(CT, C)

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    S = np.zeros((6,K))
    S[:,0] = s0
    Sigma = 0.01 * np.eye(6)

    for k in range(0, K-1):
        S[:,k+1] = updateS(S[:,k], Sigma, m[k+1])

    [x_coords, y_coords, z_coords] = S[0:3]

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
