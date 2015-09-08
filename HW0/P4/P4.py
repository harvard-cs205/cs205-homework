import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(s, A, a):
    return np.dot(A, s) + a

def predictSig(Sigma, A, B):
    return np.linalg.inv(np.dot(np.dot(A, Sigma), np.transpose(A)) + np.dot(B, np.transpose(B)))

def updateSig(Sigma_tilde, C):
    return np.linalg.inv(Sigma_tilde + np.dot(np.transpose(C), C))

def updateS(s_tilde, Sigma_tilde, Sigma, C, m):
    return np.dot(Sigma, np.dot(Sigma_tilde, s_tilde) + np.dot(np.transpose(C), m))


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

    s_true = np.loadtxt('P4_trajectory.txt',delimiter=',')
    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    m = np.loadtxt('P4_measurements.txt',delimiter=',')
    s_obs = np.dot(m, np.diag([1.0/rx, 1.0/ry, 1.0/rz]))
    x_coords = s_obs[:,0]
    y_coords = s_obs[:,1]
    z_coords = s_obs[:,2]

    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.diag(np.ones(6))
    for ii in range(3):
        A[ii, ii+3] = dt
        A[ii+3, ii+3] -= c * dt
    a = np.zeros(6)
    a[-1] = g * dt
    s = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0])

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    
    s_model = np.zeros((K, 6))
    s_model[0, :] = s

    for ii in range(1, K):
        s = np.dot(A, s) + a
        s_model[ii, :] = s
    
    x_coords = s_model[:,0]
    y_coords = s_model[:,1]
    z_coords = s_model[:,2]

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.zeros((3,6))
    C[0, 0] = rx
    C[1, 1] = ry
    C[2, 2] = rz

    # Initial conditions for s0 and Sigma0

    s = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0])
    Sigma = 0.01 * np.diag(np.ones(6))

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    
    s_filted = np.zeros((K, 6))
    s_filted[0, :] = s

    for ii in range(1, K):
        s_tilde = predictS(s, A, a)
        Sigma_tilde = predictSig(Sigma, A, B)
        Sigma = updateSig(Sigma_tilde, C)
        s = updateS(s_tilde, Sigma_tilde, Sigma, C, m[ii, :])
        s_filted[ii, :] = s
    
    x_coords = s_filted[:, 0]
    y_coords = s_filted[:, 1]
    z_coords = s_filted[:, 2]

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
