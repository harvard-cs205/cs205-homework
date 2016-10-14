###############################

# CS 205 Fall 2015 Homework 0 Part 4
# Submitted by Kendrick Lo (Harvard ID: 70984997)
# Github username: ppgmg

# Note to reader: 
# I am in the process of acquainting myself
# with Python; therefore, I may have included extraneous
# comments to document my process and observations for my
# own reference. These would be unnecessary if writing 
# for an audience that already understands the language.
# Commands used for debugging have been removed for 
# submission but can be found in earlier commits.

###############################

import numpy as np
from numpy.linalg import inv
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

    # We load the data from the file P4_trajectory.txt
    # which we assume is in the current working directory

    # We previewed the data and noted it appeared to consist
    # of 120 lines of six comma-separated values, with no
    # obvious anomalies or data entry errors/missing values

    filename_true = "P4_trajectory.txt"
    s_true = np.loadtxt(filename_true, delimiter=",")

    # Extract first three columns to obtain series of positions
    x_coords, y_coords, z_coords = (s_true[:, 0], s_true[:, 1], 
                                      s_true[:, 2]) 

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # We load the data from the file P4_measurements.txt
    # which we assume is in the current working directory

    # We previewed the data and noted it appeared to consist
    # of 121 lines of three comma-separated values, with no
    # obvious anomalies or data entry errors/missing values

    filename_obs = "P4_measurements.txt"
    mT_obs = np.loadtxt(filename_obs, delimiter=",") 

    # We used the fact that the inverse of a diagonal matrix
    # with non-zero entries a1,... on the diagonal is 1/a1,...
    m_obs = mT_obs.T  # matrix m 
    r_values = [rx, ry, rz]
    r_array = np.diag(np.array(r_values))
    x_est = np.dot(inv(r_array), m_obs)
    
    # Extract first three ROWS to obtain series of positions
    x_coords, y_coords, z_coords = x_est[0], x_est[1], x_est[2]
 
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initialize the array for s
    s_model = np.empty((6, K))
    # Initialize s at t=0 with values from problem
    s_model[:,0] = np.array([0, 0, 2, 15, 3.5, 4.0]).T

    # Construct the array for A
    Adiag = np.array([1, 1, 1, 1-c*dt, 1-c*dt, 1-c*dt])
    Adiag3 = np.array([dt, dt, dt])
    A_prop = np.add(np.diag(Adiag), np.diag(Adiag3, k=3))  # matrix A
    
    # Initialize column vector a
    a = np.array([0, 0, 0, 0, 0, g*dt]).T

    # Compute the rest of s using Eq (1)
    # starting with column 1 (column 0 already initialized);
    # last column is (K-1)th
    for i in xrange(1, K, 1):
        s_model[:, i] = np.add(np.dot(A_prop, s_model[:, i-1]), a)    

    # Extract first three ROWS to obtain series of positions
    x_coords, y_coords, z_coords = s_model[0], s_model[1], s_model[2]   

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Initialize the array for B using values from problem
    B_values = [bx, by, bz, bvx, bvy, bvz]
    B_array = np.diag(np.array(B_values))
 
    # Initialize the array for C using values from problem
    C_array = np.zeros((3, 6))
    C_array[0, 0], C_array[1, 1], C_array[2, 2] = rx, ry, rz

    # Initial conditions for s0 and Sigma0
    s0 = s_model[:,0]  # from previous part
    Sigma0 = 0.01 * np.identity(6) # A/B are 6x6 thus so is C from Eq (3)

    ####
    #
    # function definitions

    # In this implementation we cache values for one vector of Sigma
    # and corresponding time step, for use in calculating Sigma 
    # in the next time step
    current_sigma, current_sigma_k = Sigma0, 0 

    def set_sigma_k(column, k):
        '''Caches column vector and corresponding value for SigmaK'''
        global current_sigma, current_sigma_k
        current_sigma, current_sigma_k = column, k
        
    def predictS(sk):
        '''Calculates an intermediate guess for s at time t=k+1.

        Implements equation 2. Uses 'global' arrays A and a.
        
        Args:
            sk: A 6x1 array with values of s for time t=k.

        Returns: 
            A 6x1 array representing the guess for s at time t=k+1.
        '''
        return np.add(np.dot(A_prop, sk), a)

    def predictSig(sigmak):
        '''Calculates prediction for confidence matrix.

        Implements equation 3. Uses 'global' arrays A and B.

        Args:
            sigmak: A 6x6 array representing a covariance matrix.

        Returns:
            A 6x6 array representing the prediction at next time step.
        ''' 
        first_term = np.dot(np.dot(A_prop, sigmak), A_prop.T)
        second_term = np.dot(B_array, B_array.T)
        return inv(np.add(first_term, second_term))

    def updateSig(sigma_tilde):
        '''Calculates confidence/covariance matrix at next time step.
 
        Implements equation 4. Uses 'global' array C. 
        
        Args:
            sigma_tilde: A 6x6 array representing a prediction for
                         the covariance matrix.

        Returns:
            A 6x6 array representing sigma at time t=k+1.
        '''
        return inv(np.add(sigma_tilde, np.dot(C_array.T, C_array)))
    
    def updateS(old_sk):
        '''Calculates the next column of s, for time t=k+1.
      
        Implements equation 5. Uses 'global' arrays C and m.

        We also define this method to have only one input parameter,
        and call previously defined 'predict' and 'update' methods
        here. Since we are not passing vectors for Sigma to this 
        method, we cache one vector for Sigma at one time step required
        to call update Sig.

        We could have alternatively called the previously defined
        methods outside of updateS, and passed values for Sigma_tilde and
        and SigmaK+1 in by expanding the parameter list, but then
        updateS would simply be performing matrix multiplication and
        would have no direct connection to the other defined methods.

        Thus we chose to trade-off use of global variables in our
        caching function set_sigma_k for an (arguably) more intuitive
        form of an invocation of updateS.

        Args:
            sk: A 6x1 array with values of s for time t=k.

        Returns: 
            A 6x1 array with values of s at time t=k+1.  
        '''
        s_tilde = predictS(old_sk)  # from equation 2
        sigma_tilde = predictSig(current_sigma)

        bracket_term = np.add(np.dot(sigma_tilde, s_tilde), 
                              np.dot(C_array.T, 
                                     m_obs[:, current_sigma_k + 1]))

        # update sigma values and time step
        next_sigma = updateSig(sigma_tilde)  # sigma at t=k+1
        set_sigma_k(next_sigma, current_sigma_k + 1)  # cache it

        return np.dot(next_sigma, bracket_term)        

    ###########  
    #
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # Initialize new array for s
    s_kalman = np.empty((6, K))
    # Initialize s at t=0
    s_kalman[:, 0] = s_model[:, 0]

    # Compute the rest of s using Eq (1)
    # starting with column 1 (column 0 already initialized);
    # last column is (K-1)th
    for i in xrange(1, K, 1):
        s_kalman[:, i] = updateS(s_kalman[:, i-1])      

    # Extract first three ROWS to obtain series of positions
    x_coords, y_coords, z_coords = s_kalman[0], s_kalman[1], s_kalman[2]

    ax.plot(x_coords, y_coords, z_coords,
           '-r', label='Filtered trajectory')

    # Set up and show the plot with legend (see example plot: P4.png)
    plt.title("Plots of real and modeled trajectories",
               horizontalalignment='right')
    ax.legend(loc='best')
    plt.show()

# eof
