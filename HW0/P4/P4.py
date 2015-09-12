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

    # Actual Trajectory
    # Load in ground truth trajectory
    s_true = np.loadtxt('P4_trajectory.txt',delimiter=',')
    # Initialize arrays that will hold position and velocity values
    positions = np.zeros([len(s_true),3])
    velocities = np.zeros([len(s_true),3])
    # Save the positions and velocities into the arrays
    for itn,row in enumerate(s_true):
        positions[itn] = row[0:3]
        velocities[itn] = row[3:6]
    # Split up the x, y, and z position coordinates
    x_coords = positions[:,0]
    y_coords = positions[:,1]
    z_coords = positions[:,2]
    # Plot ground truth position coordinates in blue
    ax.plot(x_coords, y_coords, z_coords,'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # Measurements
    # Initialize and create array C
    C = np.zeros([3,6])
    r_values = [rx,ry,rz]
    for index in range(3):
        C[index,index] = r_values[index]

    # Initialize and create 1/r array 
    initial_array = np.zeros([3,3])
    for itn,row in enumerate(initial_array):
        initial_array[itn,itn] = 1 / r_values[itn]
    # Convert array to matrix
    r_mat = np.asmatrix(initial_array)

    # Initialize measured array
    approx_array = np.zeros([len(positions),3])
    # Convert position array to matrix
    position_mat = np.asmatrix(positions)
    # Compute approximate position points, saving into 120x3 array
    for itn,row in enumerate(positions):
        vertical_mat =  r_mat * position_mat[itn].T
        approx_array[itn] = vertical_mat.T
    # Split up the x, y, and z measured position coordinates
    approx_x_coords = approx_array[:,0]
    approx_y_coords = approx_array[:,1]
    approx_z_coords = approx_array[:,2]
    # Plot measured positions in green
    ax.plot(approx_x_coords, approx_y_coords, approx_z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Simple Model
    # Create Propagation Array A
    A = np.array(np.vstack([np.hstack([1,0,0,dt,0,0]),np.hstack([0,1,0,0,dt,0]),
        np.hstack([0,0,1,0,0,dt]),np.hstack([0,0,0,1-c*dt,0,0]),
        np.hstack([0,0,0,0,1-c*dt,0]),np.hstack([0,0,0,0,0,1-c*dt])]))
    # Account for acceleration due to gravity
    a = np.array([0,0,0,0,0,g*dt])
    # Initial conditions for s0
    s0 = np.array([0,0,2,15,3.5,4.0]).T
    # Compute the rest of sk using Eq (1)
    # Initialize array to keep track of all positions
    sk_array = np.zeros([K,6])
    # Set position(k) initially to s0
    sk = s0
    # Compute all positions as a function of k
    for itn,row in enumerate(sk_array):
        sk_array[itn] = np.dot(A,sk)
        sk = sk_array[itn] + a
    # Split simple model's x, y, and z positions
    simple_x = sk_array[:,0]
    simple_y = sk_array[:,1]
    simple_z = sk_array[:,2]
    # Plot simple model in black
    ax.plot(simple_x,simple_y,simple_z,'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Initialize array B 
    B = np.zeros([6,6])
    b_values = [bx,by,bz,bvx,bvy,bvz]
    # Set values of B using above constants
    for itn,row in enumerate(B):
        B[itn,itn] = b_values[itn]
    # Initialize array C
    C = np.zeros([3,6])
    # Set values of C using above constants
    for itn,row in enumerate(C):
        C[itn,itn] = r_values[itn]

    # Initial conditions for s0 and Sigma0
    # s0 set above
    Sigma0 = 0.01 * np.identity(6)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    # Equation 2
    # Set position(k) to initial position
    sk_value = s0
    def predictS(A,sk_value,a):
        # Compute s(k)
        sk = np.dot(A,sk_value) + a
        # Return full position array
        return sk
    intermediate_s = predictS(A,s0,a)
    sigma_val = Sigma0
    # Equation 3
    def predictSig(sigma_val,A,B):
        return np.linalg.inv(np.dot(A,np.dot(sigma_val,A.T)) + np.dot(B,B.T))
    predictSig(sigma_val,A,B)

    # Equation 4
    # Set covariance array to initial covariance array
    sigma_val = Sigma0
    def updateSig(C,sigma_val):
        return np.linalg.inv(sigma_val + np.dot(C.T,C))
    sigma_next_k = updateSig(C,sigma_val)

    # Equation 5
    def updateS(sigma_val,positions,C,intermediate_s,index):
        inside_left = np.dot(sigma_val,intermediate_s)
        inside_right = np.dot(C.T,positions[index+1])
        s_next_k = np.dot(updateSig(C,sigma_val),inside_left + inside_right)

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
