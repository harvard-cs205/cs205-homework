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

    s_true = np.array(np.loadtxt("P4_trajectory.txt", delimiter=','))
    x_coords, y_coords, z_coords = s_true[:,0], s_true[:,1], s_true[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # TODO: this seems a bit silly
    C = np.zeros((3, 3))
    C[0,0] = 1./rx
    C[1,1] = 1./ry
    C[2,2] = 1./rz
    multByC = lambda x: np.dot(C, x)
    s_measured = np.array(np.loadtxt("P4_measurements.txt", delimiter=','))
    s_adjusted = np.apply_along_axis(multByC, axis=1, arr=s_measured)
    x_coords, y_coords, z_coords = s_adjusted[:,0], s_adjusted[:,1], s_adjusted[:,2]
    
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.zeros((6, 6))
    for i in range(3):
        A[i,i] = 1
        A[i, i+3] = dt
    for i in range(3, 6):
        A[i,i] = 1 - c*dt
    print A
    a = np.array([0, 0, 0, 0, 0, g*dt])
    s_blind = np.zeros((6, K))

    # Initial conditions for s0
    s0 = np.array([0, 0, 2, 15, 3.5, 4.0])

    # Compute the rest of sk using Eq (1)
    s_k = s0.copy()
    for i in range(K):
        s_blind[:,i] = s_k
        s_k = np.dot(A, s_k) + a
    
    s_blind = s_blind.transpose()
    x_coords, y_coords, z_coords = s_blind[:,0], s_blind[:,1], s_blind[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.zeros((3, 6))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz

    def predictS(s):
        return A.dot(s) + a

    def predictSig(sig):
        return np.linalg.inv(A.dot(sig).dot(A.T) + B.dot(B.T))

    def updateSig(sig_t):
        return np.linalg.inv(sig_t + C.T.dot(C))

    def updateS(s, sig, sig_new, m):
        return sig_new.dot(predictSig(sig).dot(predictS(s)) + C.T.dot(m))

    # Initial conditions for s0 and Sigma0
    s_filtered = np.zeros((6, K))
    s0 = np.array([0, 0, 2, 15, 3.5, 4.0])
    Sigma0 = 0.01*np.eye(6)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    s_k = s0.copy()
    Sigma = Sigma0.copy()
    for i in range(K):
        sig_new = updateSig(predictSig(Sigma))
        s_filtered[:,i] = s_k
        s_k = updateS(s_k, Sigma, sig_new, s_measured[i])
        Sigma = sig_new

    s_filtered = s_filtered.transpose() 
    x_coords, y_coords, z_coords = s_filtered[:,0], s_filtered[:,1], s_filtered[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.savefig('P4.png')
    plt.show()
