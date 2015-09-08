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

    s_true = np.transpose(np.mat(np.loadtxt('P4_trajectory.txt', delimiter = ","))) 
        #Load the trajectory text file into a matrix

    x_coords = np.array(s_true)[0, :]  # extract x coordinates
    y_coords = np.array(s_true)[1, :]  # extract y coordinates
    z_coords = np.array(s_true)[2, :]  # extract z coordinates
        #Have to extract these as an array, otherwise each point gets its own label in the legend

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_meas = np.transpose(np.mat(np.loadtxt('P4_measurements.txt', delimiter = ",")))
        #Load the measurements text file into a matrix

    stretch_matrix = np.matrix([[1/rx, 0, 0], [0, 1/ry, 0], [0, 0, 1/rz]])
        # Matrix with the stretching factors

    corrected = stretch_matrix * s_meas
        # Correct position by multiplying the stretch matrix by the measured position
 
    x_coords = np.array(corrected)[0, :]  # extract x coordinates
    y_coords = np.array(corrected)[1, :]  # extract y coordinates
    z_coords = np.array(corrected)[2, :]  # extract z coordinates
    

    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Propagation Matrix
    A = np.matrix([[1, 0, 0,         dt,          0,          0],
                   [0, 1, 0,          0,         dt,          0],
                   [0, 0, 1,          0,          0,         dt],
                   [0, 0, 0, (1 - c*dt),          0,          0],
                   [0, 0, 0,          0, (1 - c*dt),          0],
                   [0, 0, 0,          0,          0, (1 - c*dt)]])
    
    # Matrix a is the component of the motion that is changed by gravity in 
    #    the time interval delta-t
    a = np.transpose(np.matrix([0, 0, 0, 0, 0, g*dt]))
    
    # Initial conditions for s0
    s0 = np.transpose(np.matrix([0, 0, 2, 15, 3.5, 4.0]))

    # Empty matrix to be filled with predicted values
    s_pred = np.matrix(np.empty(shape=(6, (K-1))))

    # Insert initial conditions into the first row
    s_pred[:, 0] = s0
    
    # Compute the rest of sk using Eq (1), one column at a time
    for i in range(1, K-1):
        s_pred[:, i] = A * s_pred[:, i-1] + a

    # Extract coordinate values from the top three rows
    x_coords = np.array(s_pred)[0, :]  # extract x coordinates
    y_coords = np.array(s_pred)[1, :]  # extract y coordinates
    z_coords = np.array(s_pred)[2, :]  # extract z coordinates

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Matrix B describes the scale of the errors
    B = np.matrix([[bx,  0,  0,  0,  0,  0],
                   [ 0, by,  0,  0,  0,  0],
                   [ 0,  0, bz,  0,  0,  0],
                   [ 0,  0,  0,bvx,  0,  0],
                   [ 0,  0,  0,  0,bvx,  0],
                   [ 0,  0,  0,  0,  0,bvz]])

    # Matrix C describes the dimensional stretching
    C = np.matrix([[rx,  0,  0, 0, 0, 0],
                   [ 0, ry,  0, 0, 0, 0],
                   [ 0,  0, rz, 0, 0, 0]])

    # Initial conditions for s0 and Sigma0
        # s0 was defined above in part 3
    Sigma0 = 0.01 * np.mat(np.identity(6))    # 6X6 identity matrix

    # Function predictS returns a predicted location as in Eq(2)
    def predictS(s):
        return (A * s + a)

    # Function predictSig returns a predicted Sigma value as in Eq(3)
    def predictSig(sig_k):
        return np.linalg.inv(A * sig_k * np.transpose(A) + B * np.transpose(B))

    # Function updateSig returns the next value of Sigma as in Eq(4)
    def updateSig(sig_hat):
        return np.linalg.inv(sig_hat + np.transpose(C) * C)

    # Function updateS returns the next position as in Eq(5)
    def updateS(m, sig_hat, sig_k, s):
        return (sig_k * (sig_hat * s + np.transpose(C) * m))

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # Empty matrix to be filled with Kalman filtering predicted values
    s_kalm = np.matrix(np.empty(shape=(6, (K-1))))

    # Insert initial conditions into the first column
    s_kalm[:, 0] = s0

    # Intermediate storage variables for Sigma and s that will be used in the for loop
    sig_k = Sigma0               # First value: Sigma0
    sig_hat = predictSig(Sigma0) # First value: based on Sigma0
    s1 = s0                      # Initial value is the starting point
                                 # After tweaking my code, this initial value is no longer used

    for i in range(1, K-1):
        m1 = s_meas[0:3, i]     # pick out the actual measurement to plug in to eq(5)
        s1 = predictS(s_kalm[:, i - 1])       # find the next intermediate guess for s
        sig_k = updateSig(sig_hat)  # find the next sig2 to plug into eq(5)
        s_kalm[:, i] = updateS(m1, sig_hat, sig_k, s1) # fill in next column of s_kalm
        sig_hat = predictSig(sig_k) # update sig1 for the next loop
        
        
    x_coords = np.array(s_kalm)[0, :]  # extract x coordinates
    y_coords = np.array(s_kalm)[1, :]  # extract y coordinates
    z_coords = np.array(s_kalm)[2, :]  # extract z coordinates



    ax.plot(x_coords, y_coords, z_coords,
           '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
