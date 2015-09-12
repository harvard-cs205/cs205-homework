import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

def predictS(sk, A, a):
    return np.dot(A, sk) + a

def predictSig(Sigma_k, A, B):
    ret = np.dot(np.dot(A, Sigma_k), A.T) + np.dot(B, B.T)
    ret = la.inv(ret)
    return ret

def updateSig(Sigma_bar, C):
    ret = Sigma_bar + np.dot(C.T, C)
    ret = la.inv(ret)
    return ret

def updateS(Sigma_k1, Sigma_bar, s_bar, C, m_k1):
    ret = np.dot(Sigma_bar, s_bar) + np.dot(C.T, m_k1)
    ret = np.dot(Sigma_k1, ret)
    return ret

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

    trajectory = np.loadtxt('P4_trajectory.txt', delimiter=',')
    x_coords, y_coords, z_coords = trajectory[:,0], trajectory[:,1], trajectory[:,2]
   
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    P4_measurements = np.loadtxt('P4_measurements.txt', delimiter=',')

    x_bar = np.dot(np.diag([1.0/rx, 1.0/ry, 1.0/rz]), P4_measurements.T)
    x_coords, y_coords, z_coords = x_bar[0,:], x_bar[1,:], x_bar[2,:]

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
    s = np.zeros((6, K))
    s[:,0] = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0])

    # Initial conditions for s0
    # Compute the rest of  using Eq (1)

    for k in xrange(1, K):
        s[:,k] = np.dot(A, s[:, k-1]) + a

    x_coords, y_coords, z_coords = s[0,:], s[1,:], s[2,:]
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
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    s = np.zeros((6, K))
    s[:,0] = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0])
    Sigma_k = 0.01 * np.eye(6)
    P4_measurements = P4_measurements.T
    for k in xrange(1, K):
        s_bar = predictS(s[:,k-1], A, a)
        Sigma_bar = predictSig(Sigma_k, A, B)
        Sigma_k = updateSig(Sigma_bar, C)
        s[:,k] = updateS(Sigma_k, Sigma_bar, s_bar, C, P4_measurements[:,k])
    
    x_coords = s[0,:]
    y_coords = s[1,:]
    z_coords = s[2,:]
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')


    ax.legend()
    plt.show()
