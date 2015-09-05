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

    def get_cols(matrix):
        return [matrix[:,i] for i in range(len(matrix[0]))]

    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    #####################
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    x_coords, y_coords, z_coords = get_cols(s_true)[:3]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measured = np.loadtxt('P4_measurements.txt', delimiter=',')
    x_coords, y_coords, z_coords = [1/r * col for r, col in zip([rx, ry, rz], get_cols(measured))]

    ax.plot(x_coords, y_coords, z_coords,
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
            [0, 0, 0, 0, 0, 1-c*dt],
    ])

    a = np.matrix([0] * 5 + [g * dt]).transpose()

    def get_next_step(s_k):
        s_k = np.matrix(s_k)
        # got mixed up between column and row vectors
        return (A * s_k.transpose() + a).transpose()

    s = s_true[0]
    predictions = np.zeros((len(s_true), len(s)))
    for i in range(len(predictions)):
        predictions[i] = s
        s = get_next_step(s)

    x_coords, y_coords, z_coords = get_cols(predictions)[:3]
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')


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

    sigma = np.identity(6) * .01
    s = np.matrix(s_true[0]).transpose()

    def predictS():
        return A * s + a

    def predictSig():
        return np.linalg.inv(A * sigma * A.transpose() + B * B.transpose())

    def updateSig(sigma_tilde):
        global sigma
        sigma = np.linalg.inv(sigma_tilde + C.transpose() * C)

    def updateS(k, prediction, sigma_tilde):
        global s
        m = np.matrix(measured[k]).transpose()
        s = sigma * (sigma_tilde * prediction + C.transpose() * m)

    kalman_predictions = np.zeros((len(predictions), len(s)))
    kalman_predictions[0] = s.transpose()
    for k in range(1, len(predictions)):
        prediction = predictS()
        sigma_tilde = predictSig()
        updateSig(sigma_tilde)
        updateS(k, prediction, sigma_tilde)
        kalman_predictions[k] = s.transpose()

    x_coords, y_coords, z_coords = get_cols(kalman_predictions)[:3]




    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
