import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(A, sk, a):
    return np.dot(A, sk) + a

def predictSig(A, Ek, B):
    return np.linalg.inv(np.dot(np.dot(A, Ek), A.T) + np.dot(B, B.T))

def updateSig(Ep, C):
    return np.linalg.inv(Ep + np.dot(C.T, C))

def updateS(Ek, Ep, sp, C, mk):
    return np.dot(Ek, np.dot(Ep, sp) + np.dot(C.T, mk)) 

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
    x_coords, y_coords, z_coords = s_true[:, 0], s_true[:, 1], s_true[:, 2]
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    m = np.loadtxt('P4_measurements.txt', delimiter=',') 
    R = np.diag([1.0/rx, 1.0/ry, 1.0/rz])
    approx_pos = np.dot(R, m.T).T
    ax.plot(approx_pos[:, 0], approx_pos[:, 1], approx_pos[:, 2],
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array(
        [[1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1 - c * dt, 0, 0],
        [0, 0, 0, 0, 1 - c * dt, 0],
        [0, 0, 0, 0, 0, 1 - c * dt]])
    a = np.array([0, 0, 0, 0, 0, g * dt])
    s = np.eye(6, K)
    # Initial conditions for s0
    s[:, 0] = np.array([0, 0, 2, 15, 3.5, 4.0])
     
    # Compute the rest of sk using Eq (1)
    for i in range(K-1):
        s[:, i+1] = np.dot(A, s[:, i]) + a

    x_coords, y_coords, z_coords  = s[0, :], s[1, :], s[2, :]
    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.array(
            [[bx, 0, 0, 0, 0, 0],
            [0, by, 0, 0, 0, 0],
            [0, 0, bz, 0, 0, 0],
            [0, 0, 0, bvx, 0, 0],
            [0, 0, 0, 0, bvy, 0],
            [0, 0, 0, 0, 0, bvz]])
    C = np.array(
            [[rx, 0, 0, 0, 0, 0],
             [0, ry, 0, 0, 0, 0],
             [0, 0, rz, 0, 0, 0]])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    s = np.eye(6, K)
    # Initial conditions for s0
    s[:, 0] = np.array([0, 0, 2, 15, 3.5, 4.0])
    last_E = 0.01 * np.eye(6, 6) 
    for i in range(K-1):
        cur_sp = predictS(A, s[:, i], a) 
        cur_Ep = predictSig(A, last_E, B)
        cur_Ek = updateSig(cur_Ep, C)
        s[:, i+1] = updateS(cur_Ek, cur_Ep, cur_sp, C, m[i+1, :])
        last_E = cur_Ek

    x_coords, y_coords, z_coords  = s[0, :], s[1, :], s[2, :]
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()


