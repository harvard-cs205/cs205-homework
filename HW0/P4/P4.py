import numpy as np
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
    s_truth = np.loadtxt('P4_trajectory.txt', delimiter=',', dtype=float)
    x_coords = s_truth.T[0]
    y_coords = s_truth.T[1]
    z_coords = s_truth.T[2]

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')
    
    #plt.show()
    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measured_coord = np.loadtxt('P4_measurements.txt', delimiter=',', dtype=float)
    xm_coords = measured_coord.T[0]
    ym_coords = measured_coord.T[1]
    zm_coords = measured_coord.T[2]

    #ax.plot(xm_coords, ym_coords, zm_coords, '.g', label='Observed trajectory')
    
    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    A = np.zeros(shape=(6,6)) + np.eye(6,6)
    A[3][3]=1-c*dt; A[4][4]=1-c*dt; A[5][5]=1-c*dt
    A[0][3]=dt; A[1][4]=dt; A[2][5]=dt

    # a = ?
    a = np.zeros(6)
    a[-1] = g * dt
    
    # s = ?
    s0 = np.array([0, 0, 2, 15, 3.5, 4]).T

    # Initial conditions for s0
    s = np.zeros(shape=(6,K)) 
    s[:,0] = s0
    
    # Compute the rest of sk using Eq (1)
    for j in xrange(K-1):
        s[:,j+1] = np.dot(A, s[:,j]) + a

    xb_coords = s[:,0:-1][0]
    yb_coords = s[:,0:-1][1]
    zb_coords = s[:,0:-1][2]

    ax.plot(xb_coords, yb_coords, zb_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    ### Function Defintions ###
    # Implements equation (2) prediction step and returns s_tilde
    def predictS(sk):
        return np.dot(A, sk) + a

    # Implements equation (3) prediction step and returns cov_mat_tilde
    def predictSig(sig_k):
        return np.linalg.inv(np.dot(np.dot(A, sig_k), A.T) + np.dot(B, B.T))

    # Implements equation (4) and returns cov_mat_k+1
    def updateSig(sig_tilde):
        return np.linalg.inv(sig_tilde + np.dot(C.T, C))

    # Implments equation (5) and returns s_k+1
    def updateS(sig_k_1, sig_tilde, s_tilde, m_k_1):
        return np.dot(sig_k_1, (np.dot(sig_tilde, s_tilde) + np.dot(C.T, m_k_1)))

    # Initialize B scale of errors
    B = np.zeros(shape=(6,6))
    B[3,3] = bvx; B[4,4] = bvy; B[5,5] = bvz

    # Initialize C 
    C = np.zeros(shape=(3,6))
    C[0][0] = rx; C[1][1] = ry; C[2][2] = rz

    # Initial conditions for s0 and Sigma0
    Sig_0 = np.eye(6) * .01

    sk = np.zeros(shape=(6,K)) 
    sk[:,0] = s0

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for j in xrange(K-1):
        # Calculate (2)
        s_tilde = predictS(sk[:,j])
        
        # Calculate (3), then update Sigma 0 
        Sig_tilde = predictSig(Sig_0)
        Sig_0 = Sig_tilde
        
        # Calculate (4) 
        Sig_k_1 = updateSig(Sig_tilde)
        
        # Calculate s^k+1
        s_k_1 = updateS(Sig_k_1, Sig_tilde, s_tilde, measured_coord[j])
        
        # Save the answer
        sk[:,j+1] = s_k_1
        
    x_kf_coords = sk[:,0:-1][0]
    y_kf_coords = sk[:,0:-1][1]
    z_kf_coords = sk[:,0:-1][2]    

    ax.plot(x_kf_coords, y_kf_coords, z_kf_coords, '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
