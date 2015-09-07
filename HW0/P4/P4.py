import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

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

    trajectory = np.loadtxt("P4_trajectory.txt", delimiter=',')


    #extract coordinates
    x_coords = trajectory[:,0]
    y_coords = trajectory[:,1]
    z_coords = trajectory[:,2]

    #####################

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    #Note: there is an easier way of doing this part, but I am using the longer way to practice
    #my python =)

    observation = np.loadtxt("P4_measurements.txt", delimiter=',')

    #define the C matrix. only first 3 columns are needed
    C = np.diag([1/rx, 1/ry, 1/rz])

    #convert observation array into a matrix
    ob_matrix = np.matrix(observation)

    #create a matrix for the un-stretched values. Populated with zeros first
    fixed_matrix = np.zeros(ob_matrix.shape)

    #calculate the un-stretched values based on matrix multiplication of
    #each row with the C matrix. Transpose the row to a column for the multiplication
    #then transpose it back to a row, and save it to fixed_matrix
    for x in range (0, ob_matrix.shape[0]):
        fixed_matrix[x,:] = (C * ob_matrix[x,:].transpose()).transpose()


    #sanity check
    #print fixed_matrix

    #extract coordinates
    x_coords = fixed_matrix[:,0]
    y_coords = fixed_matrix[:,1]
    z_coords = fixed_matrix[:,2]

    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1, 0, 0, dt, 0, 0],
                   [0, 1, 0, 0, dt, 0],
                   [0, 0, 1, 0, 0, dt],
                   [0, 0, 0, 1 - c*dt, 0, 0],
                   [0, 0, 0, 0, 1 - c*dt, 0],
                   [0, 0, 0, 0, 0, 1-c * dt]
                    ])
    a = np.matrix([0, 0, 0, 0, 0, g*dt]).transpose()
    s = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()


    # Initial conditions for s0
    prediction_matrix = np.zeros((6,121))
    prediction_matrix[:,0] = s.reshape(6)

    # Compute the rest of sk using Eq (1)
    for x in range (1, K):
        prediction_matrix[:,x] = (A * prediction_matrix[:,x-1].reshape(6,1) + a).reshape(6)

    #extract coordinates
    x_coords = prediction_matrix[0,:]
    y_coords = prediction_matrix[1,:]
    z_coords = prediction_matrix[2,:]        

    ax.plot(x_coords, y_coords, z_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx,by, bz, bvx, bvy, bvz])
    C = np.matrix([[rx, 0, 0, 0, 0, 0],
                  [0, ry, 0, 0, 0, 0],
                  [0, 0, rz, 0, 0, 0]])


    observation = np.loadtxt("P4_measurements.txt", delimiter=',')

    #convert observation array into a matrix
    ob_matrix = np.matrix(observation)

    # Initial conditions for s0 and Sigma0


    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()
    sigma = 0.01 * np.identity(6)


    filtered_matrix = np.zeros((6,121))
    filtered_matrix[:,0] = s0.reshape(6)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    for x in range(1,K):
        #equation 2: predictS
        s_tilde = (A * filtered_matrix[:,x-1].reshape(6,1) + a).reshape(6).transpose()

        #equation 3: predictSig
        sigma_tilde = inv(A * sigma * A.transpose() + B*B.transpose())

        #equation 4: update sig
        sig_updated = inv(sigma_tilde + C.transpose() * C)

        #equation 5: updateS
        s_updated = sig_updated * (sigma_tilde * s_tilde + C.transpose() *  ob_matrix[x,:].transpose())

        filtered_matrix[:,x] = s_updated.reshape(6)     

        #save sigma for next iteration
        sigma = sig_updated   


    #extract coordinates
    x_coords = filtered_matrix[0,:]
    y_coords = filtered_matrix[1,:]
    z_coords = filtered_matrix[2,:]     

    ax.plot(x_coords, y_coords, z_coords,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
