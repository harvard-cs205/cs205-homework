###############################

# CS 205 Fall 2015 Homework 0 Parts 4-6
# Submitted by Kendrick Lo (Harvard ID: 70984997)
# Github username: ppgmg

# Note to reader: 
# I am in the process of acquainting myself
# with Python; therefore, I may have included extraneous
# comments to document my process and observations for my
# own reference. These would be unnecessary if writing 
# for an audience that already understands the language.
# Commands used for debugging will be removed for 
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
    ## print s_true.shape, s_true.dtype 

    # Extract first three columns to obtain series of positions
    x_coords = s_true[:, 0]
    y_coords = s_true[:, 1]
    z_coords = s_true[:, 2]
    ## print x_coords.shape, y_coords.shape, z_coords.shape

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
    ## print mT_obs.shape, mT_obs.dtype 

    # Given the values of the scaling matrix, we could simply divide
    # all entries in the read array by the appropriate (rx, ry, rx)
    # and pull the coordinates for plotting;
    # while more complicated, we constructed the formula for x_est
    # to practice implementation of equations involving matrices
    m_obs = mT_obs.T  
    r_values = [rx, ry, rz]
    r_array = np.diag(np.array(r_values))
    x_est = np.dot(inv(r_array), m_obs)
    ## print m_obs.shape, r_array.shape, x_est.shape
    
    # Extract first three ROWS to obtain series of positions
    x_coords = x_est[0]
    y_coords = x_est[1]
    z_coords = x_est[2]
    ## print x_coords.shape, y_coords.shape, z_coords.shape
 
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    ## debug check
    ## x_coords2 = mT_obs[:, 0]/float(rx)
    ## y_coords2 = mT_obs[:, 1]/float(ry)
    ## z_coords2 = mT_obs[:, 2]/float(rz)
    ## ax.plot(x_coords2, y_coords2, z_coords2,
    ##        '.r', label='test Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initialize the array for s
    s_model = np.empty((6, K))
    # Initialize s at t=0 with values from problem
    s_model[:,0] = np.array([0, 0, 2, 15, 3.5, 4.0]).T
    ## print s_model[:, 0]
    ## print s_model.shape

    # Construct the array for A by filling in diagonals
    Adiag = np.array([1, 1, 1, 1-c*dt, 1-c*dt, 1-c*dt])
    Adiag3 = np.array([dt, dt, dt])
    A_prop = np.add(np.diag(Adiag), np.diag(Adiag3, k=3))
    ## print A_prop 
    
    # Initialize column array a
    a = np.array([0, 0, 0, 0, 0, g*dt]).T
    ## print a, a.shape

    # Compute the rest of s using Eq (1)
    # starting with column 1 (column 0 already initialized);
    # last column is (k-1)th
    for i in xrange(1, K, 1):
        s_model[:, i] = np.dot(A_prop, s_model[:, i-1])  # A*sk
        s_model[:, i] = np.add(s_model[:, i], a)  # +a
        ## print s_model[:, i]       

    # Extract first three ROWS to obtain series of positions
    x_coords = s_model[0]
    y_coords = s_model[1]
    z_coords = s_model[2]    

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Initialize the array for B using values from problem
    B_values = [bx, by, bz, bvx, bvy, bvz]
    B_array = np.diag(np.array(B_values))
    ## print B_array
 
    # Initialize the array for C using values from problem
    C_array = np.zeros((3, 6))
    C_array[0, 0] = rx
    C_array[1, 1] = ry
    C_array[2, 2] = rz
    ## print C_array

    # Initial conditions for s0 and Sigma0
    s0 = s_model[:,0]  # from previous part
    Sigma0 = 0.01 * np.identity(6) # A/B are 6x6 thus so is C from Eq (3)

    # Initialize global variables (used to 'cache' one column of Sigma)
    # global current_sigma_k 
    # global current_sigma  # holds Sigma at time current_sigma_k
    current_sigma_k = 0
    current_sigma = Sigma0
    ## print s0, Sigma0
    ## print current_sigma

    ####
    #
    # function definitions

    def set_sigma_k(column, k):
        '''Caches column vector and corresponding value for SigmaK)'''
        ## print 12345, column, k
        global current_sigma, current_sigma_k
        current_sigma = column
        current_sigma_k = k
        
    def predictS(sk):
        '''Calculates an intermediate guess for s at time t=k+1.

        Implements equation 2. Uses values for A and a from previous part.
        
        Args:
            sk: A 6x1 array with values of s for time t=k.

        Returns: 
            A 6x1 array representing the guess for s at time t=k+1.
        '''
        return np.add(np.dot(A_prop, sk), a)

    def predictSig(sigmak):
        '''Calculates prediction for confidence matrix.

        Implements equation 3. Uses values for A from previous part.
        Uses 'global' matrix B defined earlier.
        
        Args:
            sigmak: A 6x6 array representing a covariance matrix.

        Returns:
            A 6x6 array representing the prediction at next time step.
        ''' 
        first_term = np.dot(A_prop, sigmak)
        first_term = np.dot(first_term, A_prop.T)
        second_term = np.dot(B_array, B_array.T)
        return inv(np.add(first_term, second_term))

    def updateSig(sigma_tilde):
        '''Calculates confidence/covariance matrix at next time step.
 
        Implements equation 4. Uses 'global' matrix C defined earlier. 
        
        Args:
            sigma_tilde: A 6x6 array representing a prediction for
                         the covariance matrix.

        Returns:
            A 6x6 array representing sigma at time t=k+1.
        '''
        second_term = np.dot(C_array.T, C_array)
        return inv(np.add(sigma_tilde, second_term))
    
    def updateS(old_sk):
        '''Calculates the next column of s, for time t=k+1.
      
        Implements equation 5. Uses 'global' matrix C defined earlier.
        Uses 'm' read in from file, and earlier functions.

        Note that we update the value of 'global' variables 
        current_sigma and k before we return the column vector for s. 
        For a cleaner implementation, we could modify the set of
        parameters for this function to accept these values and
	return updated values along with the updated value for s.
      
        In this implementation, we call this function using only the 
        old value of sk. We could have defined the above functions
        within this one if we wanted to make them private to this
        function.

        Args:
            sk: A 6x1 array with values of s for time t=k.

        Returns: 
            A 6x1 array with values of s at time t=k+1.  
        '''
        s_tilde = predictS(old_sk)  # from equation 2
        bracket_term1 = np.dot(predictSig(current_sigma), s_tilde)
        bracket_term2 = np.dot(C_array.T, m_obs[:, current_sigma_k + 1])
        bracket_term = np.add(bracket_term1, bracket_term2)

        # update sigma values and time step
        print updateSig(predictSig(current_sigma))
        print current_sigma_k+1
        set_sigma_k(updateSig(predictSig(current_sigma)), 
                    current_sigma_k + 1)
        ## print "hello"
        ## print current_sigma_k, current_sigma

        return np.dot(current_sigma, bracket_term)        

    ## print updateS(s0)
    ## print "goodbye"  
    ## print current_sigma_k, current_sigma
    ###########  
    #
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # Initialize new array for s
    s_kalman = np.empty((6, K))
    # Initialize s at t=0 with same initial values
    s_kalman[:, 0] = s_model[:, 0]
    print s_kalman[:,0]

    # Compute the rest of s using Eq (1)
    # starting with column 1 (column 0 already initialized);
    # last column is (k-1)th
    for i in xrange(1, K, 1):
        print i
        s_kalman[:, i] = updateS(s_kalman[:, i-1])
        print i, current_sigma_k
        
        ## print s_model[:, i]       

    # Extract first three ROWS to obtain series of positions
    x_coords = s_kalman[0]
    y_coords = s_kalman[1]
    z_coords = s_kalman[2]    

    ax.plot(x_coords, y_coords, z_coords,
           '-r', label='Filtered trajectory')

    # Show the plot (example copy stored as P4.png)
    plt.title("Plots of real and modeled trajectories",
               horizontalalignment='right')
    ## plt.text(0.5, 1.08, "test", horizontalalignment='center', fontsize=5, transform=ax.transAxes)
    ## ax.set_title("Plots of real and modeled trajectories")
    ax.legend(loc='best')
    plt.show()
