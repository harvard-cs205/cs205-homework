import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(prop, sk, a):
    """Implements Eq. 2 in the document
    Returns s~
    prop is the propagation matrix
    sk is the current vector
    a is the "gravity vector"
    """

    return np.dot(prop, sk) + a

# Implemenets Eq. 3 in the document
# Returns Sigma~
def predictSig(prop, sigk, B):
    """Implemenets Eq. 3 in the document
    Returns Sigma~
    prop is the propagation matrix
    sigk is our curent sigma
    B is the B matrix"""

    # Compute the first term for clarity
    first_term = np.dot(np.dot(prop, sigk), np.transpose(A))
    
    # Compute the second term for clarity
    second_term = np.dot(B, np.transpose(B))

    # Combine for the parenthetical term
    paren = first_term + second_term

    # Invert and return sigma twiddle
    return np.linalg.inv(paren)


def updateSig(Sig_twid, C):
    """Implements Eq. 4 in the document
    Returns Sigma~
    Sig_twid is our current sigma twiddle
    C is the C matrix
    """
   
    # Compute the parenthetical for clarity
    paren = Sig_twid + np.dot(np.transpose(C), C)
   
    # Invert and return
    return np.linalg.inv(paren)

def updateS(sigk1, sig_twid, s_twid, C, mk1):
    """Implements eq. 5 in the document
    Returns s^(k+1)
    sigk1 is sigma^(k+1)
    sig_twid is sigma twiddle
    s_twid is s twiddle
    C is the C matrix
    mk1 is m^(k+1)
    """

    # Compute the first term in the parenthetical for clarity
    first_term = np.dot(sig_twid, s_twid)

    # Compute the second term in the parenthetical
    second_term = np.dot(np.transpose(C), mk1)

    # Combine these
    paren = first_term + second_term

    # Now return the final result
    return np.dot(sigk1, paren)


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

    # Load in the coordinates (CSV format)
    obs_data = np.loadtxt('P4_trajectory.txt', delimiter=',')

    # Separate them from the loaded array
    x_coords = obs_data[:, 0]
    y_coords = obs_data[:, 1]
    z_coords = obs_data[:, 2]

    # Plot the data
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # Alias the inverses for clarity
    rx_inv = 1. / rx
    ry_inv = 1. / ry
    rz_inv = 1. / rz

    # Get an ndarray
    inv_mat = np.array([[rx_inv, 0, 0],
                        [0, ry_inv, 0],
                        [0, 0, rz_inv]])

    # Load in the measured data
    m_data = np.loadtxt('P4_measurements.txt', delimiter=',')

    # Separate the data
    m_x = m_data[:, 0]
    m_y = m_data[:, 1]
    m_z = m_data[:, 2]

    # Get some new arrays of the same size
    m_x_fixed = np.zeros(len(m_x))
    m_y_fixed = np.zeros(len(m_x)) 
    m_z_fixed = np.zeros(len(m_x)) 

    # Fill these in with the correct values
    for kk in range(len(m_x)):

        # Get the individual values
        m_xk = m_x[kk]
        m_yk = m_y[kk]
        m_zk = m_z[kk]
        
        # Construct the current vector for application of inv_mat
        curr_vec = np.array([m_xk, m_yk, m_zk])

        # "fix" the vector by application of inv_mat
        fixed_curr_vec = np.dot(inv_mat, curr_vec)

        # Extract the data into individual lists
        m_x_fixed[kk] = fixed_curr_vec[0]
        m_y_fixed[kk] = fixed_curr_vec[1]
        m_z_fixed[kk] = fixed_curr_vec[2]

    # Plot the data
    ax.plot(m_x_fixed, m_y_fixed, m_z_fixed,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Use Eq. 1 to construct the propagation matrix A
    A = np.array([[1, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1 - c * dt, 0, 0],
                    [0, 0, 0, 0, 1 - c * dt, 0],
                    [0, 0, 0, 0, 0, 1 - c * dt]])


    # Gravity vector thing
    a = np.array([0, 0, 0, 0, 0, g * dt])

    # Large 6 x k matrix to hold all the final data
    k = len(x_coords)
    s = np.zeros((6, k))

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    s[:, 0] = np.array([0, 0, 2, 15, 3.5, 4.0])

    # Now propagate...
    # Only go to k-1 because the (k-1)th step gets us the last coords
    for ii in range(k-1):

        # Where we are now
        curr_vec = s[:, ii]

        # Where we are going
        next_vec = np.dot(A, curr_vec) + a

        # Save it
        s[:, ii+1] = next_vec
        

    ax.plot(s[0, :], s[1, :], s[2, :],
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Definition of the B matrix
    B = np.array([[bx, 0, 0, 0, 0, 0],
                    [0, by, 0, 0, 0, 0],
                    [0, 0, bz, 0, 0, 0],
                    [0, 0, 0, bvx, 0, 0],
                    [0, 0, 0, 0, bvy, 0],
                    [0, 0, 0, 0, 0, bvz]]) 

    # Definition of the C matrix
    C = np.array([[rx, 0, 0, 0, 0, 0],
                [0, ry, 0, 0, 0, 0],
                [0, 0, rz, 0, 0, 0]])

    # Initial conditions for s0 and Sigma0
    # Instantiate s_filt
    s_filt = np.zeros((6, k))

    # Give it the correct initial condition
    s_filt[:, 0] = s[:, 0]

    # Initial condition for Sigk - we will not be saving all of them
    curr_Sig = .01 * np.eye(6)

    # Get the matrix of measurements
    m = np.column_stack((m_x, m_y, m_z))

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for ii in range(k-1):

        print ii
        
        # First get the current Sigma_twiddle 
        curr_Sig_twid = predictSig(A, curr_Sig, B)

        # Now get the next Sig and s
        next_Sig = updateSig(curr_Sig_twid, C)
        next_s = updateS(next_Sig, curr_Sig_twid, predictS(A, s_filt[:, ii], a), C, m[ii, :])

        # Save the data
        s_filt[:, ii+1] = next_s

        # Now update accordingly so we don't have to save all of the Sigmas
        curr_Sig = next_Sig

    ax.plot(s_filt[0, :], s_filt[1, :], s_filt[2, :],
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
