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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    # print(s_true)
    # print("2nd column")
    # print(s_true[:,1])
    
    x_coords_truth = s_true[:,0]
    y_coords_truth = s_true[:,1]
    z_coords_truth = s_true[:,2]

    # Plot True Trajectory Coordinates
    ax.plot(x_coords_truth, y_coords_truth, z_coords_truth,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_obs = np.loadtxt('P4_measurements.txt', delimiter=',')

    ax.plot(s_obs[:,0]/rx, s_obs[:,1]/ry, s_obs[:,2]/rz,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    left_top = np.zeros((3, 3), float)
    np.fill_diagonal(left_top, 1)
    right_top = np.zeros((3, 3), float)
    np.fill_diagonal(right_top, dt)

    left_bot = np.zeros((3, 3), float)
    right_bot = np.zeros((3, 3), float)
    np.fill_diagonal(right_bot, 1-c*dt)

    top = np.hstack([left_top, right_top])
    bottom = np.hstack([left_bot, right_bot])
    top
    bottom

    A = np.vstack([top, bottom])
    A

    # a
    a = np.matrix([0,0,0,0,0,g*dt])
    a = a.transpose()

    # s 6x121
    s = np.matrix(np.zeros((6, K), float))


    # Initial conditions for s0
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0])
    s0 = s0.transpose()
    s[:,0] = s0

    # Compute the rest of sk using Eq (1)
    for i in range(1, K):
        s[:,i] = A * s[:,i-1] + a

    s = np.asarray(s)
    ax.plot(s[0,:], s[1,:], s[2,:], '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    B = np.zeros((6, 6), float)
    np.fill_diagonal(B, np.array([bx, by, bz, bvx, bvy, bvz]))
    B = np.matrix(B)

    # C = ?
    C = np.zeros((3, 3), float)
    np.fill_diagonal(C, np.array([rx, ry, rz]))
    C = np.matrix(np.hstack([C, np.zeros((3, 3), float)]))


    # Initial conditions for s0 and Sigma0
    Sigma0 = np.zeros((6, 6), float)
    np.fill_diagonal(Sigma0, 0.01)
    Sigma0 = np.matrix(Sigma0)


    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    s_k_ = np.matrix(np.zeros((6, K), float))
    s_k_[:,0] = s0
    sigma_k = Sigma0

    # Equ (2)
    def predictS(s_):
        return A * s_ + a

    # Equ (3)
    def predictSig(sig_):
        return np.linalg.inv(A*sig_*A.T + B*B.T)

    # Equ (4)
    def updateSig(sig_tilde_):
        return np.linalg.inv(sig_tilde_ + C.T*C)

    # Equ (5)
    def updateS(s_tilde_, sig_tilde_, sig_plus_1_, m_plus_1_):
        return sig_plus_1_ * (sig_tilde_*s_tilde_ + C.T*m_plus_1_)

    for i in range(1,K):
        s_k = s_k_[:,i-1]
        s_tilde = predictS(s_k)
        sigma_tilde = predictSig(sigma_k)
        sigma_plus_1 = updateSig(sigma_tilde)
        
        # measurement
        m_plus_1_ = np.matrix(s_obs[i,:]).T
        s_plus_1 = updateS(s_tilde, sigma_tilde, sigma_plus_1, m_plus_1_)
        
        # update
        s_k_[:, i] = s_plus_1
        sigma_k = sigma_plus_1
    
    # Convert into array
    s_k_ = np.asarray(s_k_)
    # Plot
    ax.plot(s_k_[0,:], s_k_[1,:], s_k_[2,:], '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
