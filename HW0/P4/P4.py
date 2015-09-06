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

    # Read P4_trajectory.txt file and extract coordinates from the original arrays
    # for ploting ground truth

    trajectory_array = np.loadtxt('P4_trajectory.txt', delimiter = ',')
    x_coords = trajectory_array[:,0]
    y_coords = trajectory_array[:,1]
    z_coords = trajectory_array[:,2]

    ax.plot(x_coords, y_coords, z_coords,'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # Read P4_measurements.txt file and extract coordinates from the original arrays
    # for ploting measurements

    measurements_array = np.loadtxt('P4_measurements.txt', delimiter = ',')
    x_coords = measurements_array[:,0]
    y_coords = measurements_array[:,1]
    z_coords = measurements_array[:,2]
    ax.plot(x_coords, y_coords, z_coords,'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    #Initialize propagation matrix A
    A = np.zeros((6,6))
    (A[0,0],A[1,1],A[2,2]) = [1] * 3
    (A[0,3],A[1,4],A[2,5]) = [dt] * 3
    (A[3,3],A[4,4],A[5,5]) = [1-c * dt] * 3

    #Initialize vector a
    a = np.array([0,0,0,0,0,g*dt]).T

    #Initialize a 6 * K matrix s to store all states
    s = np.zeros((6, K))

    #Assign s0 as the initial state
    s[:,0] = np.array([0,0,2,15,3.5,4.0]).T

    #Compute all of the sk for k = 0, 1, . . . , K - 1 using Eq (1).
    for i in range(1, K):
        s[:, i] = np.dot(A, s[:, (i - 1)]) + a

    #Convert it to arrays before plotting the first three rows
    x_coords = s[0, :]
    y_coords = s[1, :]
    z_coords = s[2, :]

    # Plot the predicted positions, xk , using the commented ax.plot command.
    ax.plot(x_coords, y_coords, z_coords,'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    #Initialize matrix B and C
    B = np.diag((bx, by, bz, bvx, bvy, bvz))
    C = np.hstack([np.diag((rx,ry, rz)), np.zeros((3,3))])

    #Define four python functions for prediction steps and update steps

    #predictS: Implements equation (2) and returns predicted steps
    def predictS(sk):
        si = np.dot(A, sk) + a
        return(si)

    #predictSig: Implements equation (3) and returns predicted sigma.
    def predictSig(sigma):
        p_sig = np.dot(np.dot(A, sigma), A.T) + np.dot(B, B.T)
        p_sig = np.asmatrix(p_sig).I
        return(p_sig)

    #updateSig: Implements equation (3) and returns updated sigma.
    def updateSig(p_sig):
        sig = p_sig + np.dot(C.T, C)
        sig = np.asmatrix(sig).I
        return(sig)

    #updateSig: Implements equation (4) and returns updated step.
    def updateS(si, p_sig, sig, m):
        sk = np.dot(sig, (np.dot(p_sig, si) + np.dot(C.T, m)).T)
        return sk

    # Initial conditions for s0 and Sigma0
    p_sig = 0.01 * np.eye(6)

    kfS = np.zeros((6, K))
    kfS[:,0] = np.array([0,0,2,15,3.5,4.0]).T

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for i in range(1, K):
        si = predictS(kfS[:, i - 1])
        p_sig = predictSig(p_sig)
        sig = updateSig(p_sig)
        m = measurements_array[i, :].T
        kfS[:,i] = updateS(si, p_sig, sig, m).reshape(6,)

    #Convert it to arrays before plotting the first three rows
    x_coords = kfS[0, :]
    y_coords = kfS[1, :]
    z_coords = kfS[2, :]

    #plot the positions (the first three rows of kfS) to see the filtered predicted path.
    ax.plot(x_coords, y_coords, z_coords, '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
