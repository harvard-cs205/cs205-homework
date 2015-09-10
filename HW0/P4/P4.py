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

    inputs = np.loadtxt('P4_trajectory.txt', delimiter=',', usecols=(0,1,2))
    x_coords = inputs[:,0]
    y_coords = inputs[:,1]
    z_coords = inputs[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    measured_inputs = np.loadtxt('P4_measurements.txt', delimiter=',')
    stretch = np.array([[1.0/rx, 0.0, 0.0],[0.0, 1.0/ry, 0.0],[0.0, 0.0, 1.0/rz]])
    stretched_inputs = np.dot(stretch, measured_inputs.T)
    x_coords = stretched_inputs[0,:]
    y_coords = stretched_inputs[1,:]
    z_coords = stretched_inputs[2,:]
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initial conditions for s0
    A = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, dt, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                  [0.0, 0.0, 0.0, (1.0 - (c*dt)), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, (1.0 - (c*dt)), 0.0], 
                  [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - (c*dt))]])
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, g*dt]).T
    a.shape = (6,1)
    s_old = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0]).T
    s_old.shape = (6,1)

    # Compute the rest of sk using Eq (1)
    outputs = s_old
    for k in range(0, K):
        s_new = np.dot(A, s_old) + a
        s_new.shape = (6,1)
        outputs = np.append(outputs, s_new, axis=1)
        s_old = s_new
    
    x_coords = outputs[0,:]
    y_coords = outputs[1,:]
    z_coords = outputs[2,:]

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.array([[bx, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, by, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, bz, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, bvx, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, bvy, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, bvz]])
    C = np.array([[rx, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, ry, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, rz, 0.0, 0.0, 0.0]])

    # Initial conditions for s0 and Sigma0
    s_old = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0]).T
    s_old.shape = (6,1)
    
    sig_old = 0.01 * np.identity(6)
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def predictS (sk):
        guess = np.dot(A, sk) + a
        guess.shape = (6,1)
        return guess

    def predictSig (sigk):
        return np.linalg.inv(np.dot(np.dot(A, sigk) , A.T) + np.dot(B, B.T))
    
    def updateSig (sigGuess):
        return np.linalg.inv(sigGuess + np.dot(C.T, C))
        
    def updateS (sGuess, sig, sigGuess, measure):
        measure.shape = (3,1)
        return np.dot(sig, np.dot(sigGuess, sGuess) + np.dot(C.T, measure))
    
    outputs = s_old
    for k in range(0, K-1):
        sig_new = updateSig(predictSig(sig_old))
        #print sig_new
        s_new = updateS(predictS(s_old), sig_new, predictSig(sig_old), measured_inputs[k+1])
        #print s_new
        outputs = np.append(outputs, s_new, axis=1)
        s_old = s_new
        sig_old = sig_new
    
    x_coords = outputs[0,:]
    y_coords = outputs[1,:]
    z_coords = outputs[2,:]
        
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
