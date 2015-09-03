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
    measurements = pd.read_csv('./P4_measurements.txt', header=None)
    # We now have to rescale x, y, and z appropriately. Luckily, the matrix
    # is diagonal so we can just decouple everything.
    rescaled_measurements = measurements.values
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
        [0, 0, 0, 1- c*dt, 0, 0],
        [0, 0, 0, 0, 1 - c*dt, 0],
        [0, 0, 0, 0, 0, 1-c*dt]
    ])
    a = np.array([[0, 0, 0, 0, 0, g*dt]]).T

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    s = np.array([[0, 0, 2, 15, 3.5, 4.0]]).T

    K = 130
    positions = np.zeros((6, K), dtype = np.double)
    positions[:, 0] = s[:, 0]
    for i in range(1, K):
        cur_position = positions[:, i-1]
        cur_position = np.array([cur_position]).T
        update_position = np.dot(A, cur_position) + a
        positions[:, i] = update_position[:, 0]

    x_predicted = positions[0, :]
    y_predicted = positions[1, :]
    z_predicted = positions[2, :]


    ax.plot(x_predicted, y_predicted, z_predicted,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.array([
        [bx, 0, 0, 0, 0, 0],
        [0, by, 0, 0, 0, 0],
        [0, 0, bz, 0, 0, 0],
        [0, 0, 0, bvx, 0, 0],
        [0, 0, 0, 0, bvy, 0],
        [0, 0, 0, 0, 0, bvz]
    ])
    C = np.array([
        [rx, 0, 0, 0, 0, 0],
        [0, ry, 0, 0, 0, 0],
        [0, 0, rz, 0, 0, 0]
    ])


    def predictS(s, A=A, a=a):
        """s must be oriented in the correct direction, i.e. column form."""
        return np.dot(A, s) + a

    def predictSig(s_propagated):
        print 'in the works'


    s_propagated = predictS(s)
    print s_propagated


    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    #ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    #ax.legend()
    #plt.show()