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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    #####################

    # ax.plot(x_coords, y_coords, z_coords,
    #         '--b', label='True trajectory')
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2], '--b', label='True trajectory')
    
    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    s_obs = np.loadtxt('P4_measurements.txt', delimiter=',')
    R = np.array([[1.0/rx, 0, 0], [0,1.0/ry,0], [0,0,1.0/5.0]])
    approx_T = np.dot(R, s_obs.transpose())
    approx = approx_T.transpose()
    #####################

    # ax.plot(x_coords, y_coords, z_coords,
    #         '.g', label='Observed trajectory')
    ax.plot(approx[:,0], approx[:,1], approx[:,2],
            '.g', label='Observed trajectory')
    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?
    
    A = np.array( [ [1,0,0, dt,  0,  0], 
                    [0,1,0,  0, dt,  0],
                    [0,0,1,  0,  0, dt],
                    [0,0,0, 1-c*dt, 0, 0],
                    [0,0,0, 0, 1-c*dt, 0],
                    [0,0,0, 0, 0, 1-c*dt] ])
    
    a = np.array( [0,0,0,0,0,g*dt] ).reshape([-1,1])
    
    s_prop = np.empty([6,K])
    
    # Initial conditions for s0
    s0 = np.array([0,0,2,15,3.5,4.0]).reshape([-1,1])
    # Compute the rest of sk using Eq (1)
    s_next = s0.copy()
    
    for i in xrange(K):
        s_prop[:,i] = s_next.reshape(-1)
        s_next = np.dot(A, s_next) + a
    s_prop = s_prop.transpose()

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-k', label='Blind trajectory')
    ax.plot(s_prop[:,0], s_prop[:,1], s_prop[:,2], '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?
    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.hstack( [np.diag([rx, ry, rz]), np.zeros([3,3])] )
    
    # Initial conditions for s0 and Sigma0
    s0 = np.array([0,0,2,15,3.5,4.0]).reshape([-1,1])
    sigma0 = 0.01 * np.identity(len(A))
    
    # (2)
    def predictS(sk):
        return np.dot(A,sk) + a
    # (3)
    def predictSig(sigmak):
        return np.linalg.inv( np.dot(np.dot(A, sigmak), A.T) + np.dot(B, B.T) )
    # (4)
    def updateSig(sigmatilt):
        return np.linalg.inv( sigmatilt + np.dot(C.T, C) )
    # (5)
    def updateS(sigmak1, sigmatilt, stilt, mk1):
        return np.dot( sigmak1, np.dot(sigmatilt, stilt) + np.dot(C.T, mk1) )

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)    
    s_kal = np.empty([6,K])
    
    s_nextkal = s0.copy()
    sigma_nextkal = sigma0.copy()
    for i in xrange(K):
        s_kal[:,i] = s_nextkal.reshape(-1)
        s_tilt = predictS(s_nextkal)
        sigma_tilt = predictSig(sigma_nextkal)
        sigma_nextkal = updateSig(sigma_tilt)
        mk1 = s_obs[i].reshape([-1,1])
        s_nextkal = updateS(sigma_nextkal, sigma_tilt, s_tilt, mk1)
    s_kal = s_kal.T

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')
    ax.plot(s_kal[:,0], s_kal[:,1], s_kal[:,2], '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
