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
    #####################
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=",")
    x_coords = [row[0] for row in s_true]
    y_coords = [row[1] for row in s_true]
    z_coords = [row[2] for row in s_true]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measured_matrix = np.matrix(np.loadtxt('P4_measurements.txt', delimiter=","))
    correction_matrix = np.matrix(
        [[1/rx, 0, 0],
        [0, 1/ry, 0],
        [0, 0, 1/rz]]
    )

    result = correction_matrix * measured_matrix.transpose()
    x_coords = np.array(result)[0].tolist()
    y_coords = np.array(result)[1].tolist()
    z_coords = np.array(result)[2].tolist()

    scaled_measured_matrix = result.transpose()

    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix(
        [[1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1-c*dt, 0, 0],
        [0, 0, 0, 0, 1-c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]]
    )

    a = np.matrix(
        [[0, 0, 0, 0, 0, g*dt]]
    ).transpose()
    
    s0 = np.matrix([[0, 0, 2, 15, 3.5, 4.0]]).transpose()

    # The list to store all the sk values
    ret = [s0]
    current_pos = s0
    for i in range(K-2):
        # the ith iteration computes the (i + 1)st position
        current_pos = A*current_pos + a
        ret.append(current_pos)

    x_coords =[np.array(item).tolist()[0][0] for item in ret]
    y_coords =[np.array(item).tolist()[1][0] for item in ret]
    z_coords =[np.array(item).tolist()[2][0] for item in ret]

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix(
        [[bx, 0, 0, 0, 0, 0],
        [0, by, 0, 0, 0, 0],
        [0, 0, bz, 0, 0, 0],
        [0, 0, 0, bvx, 0, 0],
        [0, 0, 0, 0, bvy, 0],
        [0, 0, 0, 0, 0, bvz]]
    )
    C = np.matrix(
        [[rx, 0, 0, 0, 0, 0],
        [0, ry, 0, 0, 0, 0],
        [0, 0, rz, 0, 0, 0]]
    )

    def predictS(sk):
        """
        Function to give the next predicted position
        ARGS sk: The kth position
        RET: the predicted next position
        """
        return A * sk + a

    def predictSig(sigk):
        """
        Function to give the next predicted sigma matrix
        ARGS sigk: the kth sigma covariance matrix
        RET: Sigma tilde
        """
        return inv(A * sigk * A.transpose() + B * B.transpose())

    def updateSig(sig_tilde):
        """
        Function to update sigma k + 1 using filtering
        ARGS sigk: the kth sigma covariance matrix
        RET: the k + 1 sigma covariance
        """
        return inv(sig_tilde + C.transpose() * C)

    def updateS(sigk1, sigtilde, stilde, mk1):
        """
        Function to give the next position using filtering
        """
        return sigk1 * (sigtilde * stilde + C.transpose()*mk1)

    current_position = s0
    current_sigma = .01 * np.identity(6)

    ret = [s0]
    for i in range(1, K-1):
        s_tilde = predictS(current_position)
        sigma_tilde = predictSig(current_sigma)
        next_sigma = updateSig(sigma_tilde)
        next_s = updateS(
            next_sigma, 
            sigma_tilde, 
            s_tilde, 
            measured_matrix[i].transpose()
        )
        ret.append(next_s)
        current_position = next_s
        current_sigma = next_sigma

    x_coords =[np.array(item).tolist()[0][0] for item in ret]
    y_coords =[np.array(item).tolist()[1][0] for item in ret]
    z_coords =[np.array(item).tolist()[2][0] for item in ret]

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
