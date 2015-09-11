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

    # Read in data from P4_trajectory.txt
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
        
    # Plot true trajectory
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # Read in observed data from P4_measurements.txt
    s_obs = np.loadtxt('P4_measurements.txt', delimiter=',')
    
    # Create matrix to fix dimensional stretching 
    stretch_fix = np.array([
        [(1/rx), 0, 0],
        [0, (1/ry), 0],
        [0, 0, (1/rz)]
        ])
    
    # Calculate approximate position
    s_appr = np.dot(stretch_fix, s_obs.transpose()).transpose()
    
    # Plot observed trajectory
    ax.plot(s_appr[:,0], s_appr[:,1], s_appr[:,2],
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initialize progation matrix
    A = np.mat([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1-c*dt, 0, 0],
        [0, 0, 0, 0, 1-c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]
        ])
    
    # Initialize a matrix (impact of gravity)
    a = np.mat([
        [0],
        [0],
        [0],
        [0],
        [0],
        [g*dt]
        ])
    
    # Construct 6xK matrix to fill in with predicted trajecory
    s = np.mat(np.zeros((6, K)))

    # Initial conditions for s0
    s[:,0] = np.mat(s_true[0,:]).transpose()
    
    # Compute the rest of sk using Eq (1)
    for k in range(1,K):
        s[:,k] = (A * s[:,k-1]) + a

    # Convert matrix to array for plotting purposes
    s = np.array(s)
        
    # Plot estimated trajectory (simple model)
    ax.plot(s[0,:], s[1,:], s[2,:],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    
    # Equation 2 from the instructions
    def predictS(s_kal):
        return ((A * s_kal) + a)
    
    # Equation 3 from the instructions
    def predictSig(sigma):
        return np.linalg.inv(A * sigma * A.transpose() + B * B.transpose())
    
    # Equation 4 from the instructions
    def updateSig(sigma_pred):
        return np.linalg.inv(sigma_pred + C.transpose() * C)
    
    # Equation 5 from instructions
    def updateS(s_kal, s_obs, sigma_pred, sigma):
        return (sigma * (sigma_pred * s_kal + C.transpose() * s_obs))
    
    # Initialize B matrix
    B = np.mat([
        [bx, 0, 0, 0, 0, 0],
        [0, by, 0, 0, 0, 0],
        [0, 0, bz, 0, 0, 0],
        [0, 0, 0, bvx, 0, 0],
        [0, 0, 0, 0, bvy, 0],
        [0, 0, 0, 0, 0, bvz]
        ])
    
    # Initialize C matrix
    C = np.mat([
        [rx, 0, 0, 0, 0, 0],
        [0, ry, 0, 0, 0, 0],
        [0, 0, rz, 0, 0, 0]
        ])

    # Construct 6xK matrix to fill in with predicted trajecory
    s_kal = np.mat(np.zeros((6, K)))

    # Initial conditions for s0
    s_kal[:,0] = np.mat(s_true[0,:]).transpose()
    
    # Initial conditions for sigma0
    sigma_init = 0.01
    sigma = np.mat(np.eye(6)) * sigma_init
    
    # Convert array to matrix to simplify calculations
    s_obs = np.mat(s_obs)
                                                  
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for k in range(1,K):
        s_kal[:,k] = predictS(s_kal[:,k-1])
        sigma_pred = predictSig(sigma)
        sigma = updateSig(sigma_pred)
        s_kal[:,k] = updateS(s_kal[:,k], s_obs[k,:].transpose(), sigma_pred, sigma)
    
    # Convert matrix to array for plotting purposes
    s_kal = np.array(s_kal)
    
    # Plot filtered trajectory
    ax.plot(s_kal[0,:], s_kal[1,:], s_kal[2,:],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()