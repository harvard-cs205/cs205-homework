import numpy as np
import matplotlib as mpl
mpl.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
sns.set_context('poster', font_scale=1.25)

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
    s_true = pd.read_csv('./P4_trajectory.txt', header=None)
    x_coords = s_true.loc[:, 0]
    y_coords = s_true.loc[:, 1]
    z_coords= s_true.loc[:, 2]

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measurements = pd.read_csv('./P4_measurements.txt', header=None).values # Keep it as a ndarray
    # We now have to rescale x, y, and z appropriately. Luckily, the matrix
    # is diagonal so we can just decouple everything.
    rescaled_measurements = measurements.copy() #Copy...we don't want to overwrite the original measurements
    rescaled_measurements[:, 0] *= 1./rx
    rescaled_measurements[:, 1] *= 1./ry
    rescaled_measurements[:, 2] *= 1./rz

    rescaled_x = rescaled_measurements[:, 0]
    rescaled_y = rescaled_measurements[:, 1]
    rescaled_z = rescaled_measurements[:, 2]

    ax.plot(rescaled_x, rescaled_y, rescaled_z,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1-c*dt, 0, 0],
        [0, 0, 0, 0, 1-c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]
    ], dtype=np.double)
    a = np.array([[0, 0, 0, 0, 0, g*dt]], dtype=np.double).T

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    s0 = np.array([[0, 0, 2, 15, 3.5, 4.0]]).T

    s_history = np.zeros((6, K), dtype = np.double)
    s_history[:, 0] = s0[:, 0]
    for i in range(1, K):
        cur_s = s_history[:, [i-1]] # Keep the dimensionality
        update_position = np.dot(A, cur_s) + a
        s_history[:, i] = update_position[:, 0]

    x_predicted = s_history[0, :]
    y_predicted = s_history[1, :]
    z_predicted = s_history[2, :]

    ax.plot(x_predicted, y_predicted, z_predicted,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Redefine constants and data for sanity

    measurements = pd.read_csv('./P4_measurements.txt', header=None).values # Keep it as a ndarray

    A = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1-c*dt, 0, 0],
        [0, 0, 0, 0, 1-c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]
    ], dtype=np.double)

    a = np.array([[0, 0, 0, 0, 0, g*dt]], dtype=np.double).T

    B = np.array([
        [bx, 0, 0, 0, 0, 0],
        [0, by, 0, 0, 0, 0],
        [0, 0, bz, 0, 0, 0],
        [0, 0, 0, bvx, 0, 0],
        [0, 0, 0, 0, bvy, 0],
        [0, 0, 0, 0, 0, bvz]
    ], dtype=np.double)

    C = np.array([
        [rx, 0, 0, 0, 0, 0],
        [0, ry, 0, 0, 0, 0],
        [0, 0, rz, 0, 0, 0]
    ], dtype=np.double)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def predictS(_s_k):
        """cur_s must be oriented in the correct direction, i.e. column form."""
        return A.dot(_s_k) + a

    def predictSig(_sigma_k):
        return np.linalg.inv(A.dot(_sigma_k).dot(A.T) + B.dot(B.T))

    def updateSig(_sig_tilde):
        return np.linalg.inv(_sig_tilde + C.T.dot(C))

    def updateS(_s_tilde, _sig_tilde, _sigma_k_plus_1, _m_k_plus_1):
        return _sigma_k_plus_1.dot(_sig_tilde.dot(_s_tilde) + C.T.dot(_m_k_plus_1))

    s0 = np.array([[0, 0, 2, 15, 3.5, 4.0]]).T
    sigma_k = 0.01 * np.identity(6)

    s_history = np.zeros((6, K), dtype = np.double)
    s_history[:, 0] = s0[:, 0]

    for i in range(1, K):
        s_k = s_history[:, [i-1]] # keep dimensionality
        s_tilde = predictS(s_k)
        sig_tilde = predictSig(sigma_k)
        sigma_k_plus_1 = updateSig(sig_tilde)

        m_k_plus_1 = measurements[[i], :].T
        s_k_plus_1 = updateS(s_tilde, sig_tilde, sigma_k_plus_1, m_k_plus_1)

        s_history[:, i] = s_k_plus_1[:, 0]

        sigma_k = sigma_k_plus_1

    x_filtered = s_history[0]
    y_filtered = s_history[1]
    z_filtered = s_history[2]

    ax.plot(x_filtered, y_filtered, z_filtered,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()