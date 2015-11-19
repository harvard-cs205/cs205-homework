import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(A, s_k, a):
    s_tild = np.dot(A,s_k) + a
    return s_tild


def predictSig(A, sig_k, B):
    sig_tild = np.linalg.inv( np.dot( np.dot(A,sig_k), np.transpose(A) ) +
                       np.dot(B, np.transpose(B)) )
    return sig_tild

def updateSig(sig_tild, C):
    sig_k1 = np.linalg.inv( sig_tild + np.dot( np.transpose(C), C ) )
    return sig_k1

def updateS(sig_k1, sig_tild, s_tild, C, m_k1):
    s_k1 = np.dot( sig_k1, np.dot( sig_tild, s_tild ) + np.dot( np.transpose(C), m_k1) )
    return s_k1

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

    # References: http://stackoverflow.com/questions/4151128/what-are-the-differences-between-numpy-arrays-and-matrices-which-one-should-i-u
    s_measured = np.loadtxt('P4_measurements.txt', delimiter=',')

    F = np.zeros((3,3))
    F[np.diag_indices(3)] = [1./rx, 1./ry, 1./rz]
    m = np.dot(F,np.transpose(s_measured))

    x_coords = m[0,:]
    y_coords = m[1,:]
    z_coords = m[2,:]

    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Propagation matrix
    A = np.zeros((6,6))
    A0, A1 = 1., 1 - c*dt
    A[np.diag_indices(6)] = [1., 1., 1., A1, A1, A1]
    A[0,3], A[1,4], A[2,5] = dt, dt, dt

    # 'a' vector
    a = np.zeros(6)
    a[-1] = g*dt
    a = np.transpose(a)

    # Initial conditions for s0
    s0 = np.transpose( np.array( ([0, 0, 2, 15, 3.5, 4.0]) ) )

    # 6xK 's' matrix
    s = np.zeros( (6,K) )
    s[:,0] = s0

    # Compute the rest of sk using Eq (1)
    for cur_k in xrange(1, K):
        s[:,cur_k] = np.dot(A,s[:,cur_k-1]) + a

    x_coords = s[0,:]
    y_coords = s[1,:]
    z_coords = s[2,:]

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Initial conditions for s0 and Sigma0
    sig_0 = 0.01 * np.eye(6)

    B = np.zeros((6,6))
    B[np.diag_indices(6)] = [bx, by, bz, bvx, bvy, bvz]

    C = np.zeros( (3,6) )
    C[0,0], C[1,1], C[2,2] = rx, ry, rz

    kal_s = np.zeros( (6,K) )
    kal_s[:,0] = s0

    sig_k = sig_0

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for cur_k in xrange(1, K):
        s_tild = predictS(A, kal_s[:, cur_k-1], a)
        sig_tild = predictSig(A, sig_k, B)
        sig_k = updateSig(sig_tild, C)
        kal_s[:, cur_k] = updateS(sig_k, sig_tild, s_tild, C, s_measured[cur_k,:])

    x_coords = kal_s[0,:]
    y_coords = kal_s[1,:]
    z_coords = kal_s[2,:]

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
