import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Given:
    A: a propogation matrix
    s^k: the state at position k
    a: 
Calculates an approximation for the state:
    s' = A * s^k + a
'''
def predictS(A, s_k, a):
    return A * s_k + a

'''
Given:
    A: a propogation matrix
    sigma_k: a covariance matrix representing confidence in state k
Calculates an approximate covariance function:
    sigma' = (A * sigma_k * A^t + B * B^t)^-1
        where B is a matrix defining the scale of errors
'''
def predictSig(A, sigma_k, B):       
    return np.linalg.inv(A * sigma_k * A.transpose() + B * B.transpose())

'''
Given: 
    sigma: an approximation of the covariance matrix representing confidence
    C: 
Calculates the updated value of sigma
'''
def updateSig(sigma, C):
    return np.linalg.inv((sigma + C.transpose() * C))

'''
Given: 
    sigma: the covariance matrix for position k+1
    sigma_predicted: the predicted covariance matrix
    s_predicted: the predicted position value
    C: 
    m: the next measurement
Calculates the updated position value of s
'''    
def updateS(sigma, sigma_predicted, s_predicted, C, m):
    return sigma * (sigma_predicted * s_predicted + C.transpose() * m)

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
    x_true, y_true, z_true, vx, vy, vz = np.loadtxt("P4_trajectory.txt", delimiter=",", unpack=True)

    ax.plot(x_true, y_true, z_true,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    x_obs, y_obs, z_obs = np.loadtxt("P4_measurements.txt", delimiter=",", unpack=True)    

    ax.plot(x_obs, y_obs, z_obs,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1-c*dt, 0, 0],
        [0, 0, 0, 0, 1-c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]
        ])
    
    a = np.matrix([0, 0, 0, 0, 0, g * dt]).transpose()

    s = np.asmatrix(np.zeros([6, K]))

    # Initial conditions for s0
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4]).transpose()  
    s[:,0] = s0

    # Compute the rest of sk using Eq (1)
    for i in range(1, K):
        s[:,i] = A * s[:,i-1] + a

    #unpacks the first three columns from the matrix
    x_adjusted = np.squeeze(np.asarray(s[0,:]))
    y_adjusted = np.squeeze(np.asarray(s[1,:]))
    z_adjusted = np.squeeze(np.asarray(s[2,:]))    

    ax.plot(x_adjusted, y_adjusted, z_adjusted, '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([
        [bx, 0, 0, 0, 0, 0],
        [0, by, 0, 0, 0, 0],
        [0, 0, bz, 0, 0, 0],
        [0, 0, 0, bvx, 0, 0],
        [0, 0, 0, 0, bvy, 0],
        [0, 0, 0, 0, 0, bvz],        
        ])

    C = np.matrix([
        [rx, 0, 0, 0, 0, 0],
        [0, ry, 0, 0, 0, 0],
        [0, 0, rz, 0, 0, 0],
        ])
    
    #Initial conditions for s0 and Sigma0    
    s_kalmann_filter = np.asmatrix(np.zeros([6, K]))
    s_kalmann_filter[:,0] = s[:,0]

    sigma = np.matrix([
        [0.01, 0, 0, 0, 0, 0],
        [0, 0.01, 0, 0, 0, 0],
        [0, 0, 0.01, 0, 0, 0],
        [0, 0, 0, 0.01, 0, 0],
        [0, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.01],
        ])

    #Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for i in range(1, K):
        s_predicted = predictS(A, s_kalmann_filter[:,i-1], a)
        sigma_predicted = predictSig(A, sigma, B)

        #predicts next sigma
        sigma = updateSig(sigma_predicted, C)

        #updates next state
        current_measurement = np.matrix([[x_obs[i], y_obs[i], z_obs[i]]]).transpose()
        s_kalmann_filter[:,i] = updateS(sigma, sigma_predicted, s_predicted, C, current_measurement)

    #unpacks the first three columns from the matrix
    x_adjusted = np.squeeze(np.asarray(s_kalmann_filter[0,:]))
    y_adjusted = np.squeeze(np.asarray(s_kalmann_filter[1,:]))
    z_adjusted = np.squeeze(np.asarray(s_kalmann_filter[2,:]))   

    ax.plot(x_adjusted, y_adjusted, z_adjusted, '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()