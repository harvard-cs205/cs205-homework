import numpy as np
import numpy.linalg as la
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
    fname = 'P4_trajectory.txt'
    s_true = np.loadtxt(fname, delimiter=',')
    x_coords, y_coords, z_coords = s_true[:,0], s_true[:,1], s_true[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    C_inv = np.eye(3)
    C_inv[0,0] = 1.0 / rx
    C_inv[1,1] = 1.0 / ry
    C_inv[2,2] = 1.0 / rz

    fname = 'P4_measurements.txt'
    s_measure = np.loadtxt(fname, delimiter=',')

    x_tilt = np.dot(C_inv, s_measure.T)
    x_coords, y_coords, z_coords = x_tilt[0,:], x_tilt[1,:], x_tilt[2,:]
    
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.eye(6)
    for i in xrange(3, 6):
        A[i,i] = 1 - c * dt
        A[i-3,i] = dt
    a = np.array([0.0] * 6)
    a[5] = g * dt
    s = np.zeros((6, K))

    # Initial conditions for s0
    # Compute the rest of  using Eq (1)
    s[:,0] = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0])
    for k in xrange(1, K):
        s[:,k] = np.dot(A, s[:, k-1]) + a

    x_coords, y_coords, z_coords = s[0,:], s[1,:], s[2,:]
    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    def predictS(sk, A, a):
        return np.dot(A, sk) + a

    def predictSig(Sigma_k, A, B):
        ret = np.dot(np.dot(A, Sigma_k), A.T) + np.dot(B, B.T)
        ret = la.inv(ret)
        return ret

    def updateSig(Sigma_tilt, C):
        ret = Sigma_tilt + np.dot(C.T, C)
        ret = la.inv(ret)
        return ret

    def updateS(Sigma_k1, Sigma_tilt, s_tilt, C, m_k1):
        ret = np.dot(Sigma_tilt, s_tilt) + np.dot(C.T, m_k1)
        ret = np.dot(Sigma_k1, ret)
        return ret
    
    B = np.zeros((6,6))
    B[0,0] = bx
    B[1,1] = by
    B[2,2] = bz
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz

    C = np.zeros((3, 6))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    s = np.zeros((6, K))
    s[:,0] = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0])
    Sigma_k = 0.01 * np.eye(6)
    s_measure = s_measure.T
    for k in xrange(1, K):
        s_tilt = predictS(s[:,k-1], A, a)
        Sigma_tilt = predictSig(Sigma_k, A, B)
        Sigma_k = updateSig(Sigma_tilt, C)
        s[:,k] = updateS(Sigma_k, Sigma_tilt, s_tilt, C, s_measure[:,k])
    
    x_coords, y_coords, z_coords = s[0,:], s[1,:], s[2,:]
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
