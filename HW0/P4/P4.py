import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

columns = { 'x_pos': 0,
            'y_pos': 1,
            'z_pos': 2,
            'x_vel': 3,
            'y_vel': 4,
            'z_vel': 5 }

def predictS(A, sk, a):
    '''
    Implements the function A*sk + a to predict the intermediate
    guess by our intern
    '''
    return A.dot(sk) + a

def predictSig(A, B, SigmaK):
    '''
    Implements the function (A*SigmaK*A^T + BB^T)^(-1) to predict
    our intermediate confidence.
    '''
    return np.linalg.inv(A.dot(SigmaK).dot(A.T) + B.dot(B.T))

def updateSig(Sigma, C):
    '''
    Calculates the next confidence matrix.
    '''
    return np.linalg.inv(Sigma + C.T.dot(C))

def updateS(SigmaK, Sigma, s, C, mk):
    '''
    Calculates the next state
    '''
    return SigmaK.dot(Sigma.dot(s) + C.T.dot(mk))

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
    s_true = np.loadtxt("P4_trajectory.txt", delimiter=",")
    x_coords = s_true[:,columns['x_pos']]
    y_coords = s_true[:,columns['y_pos']]
    z_coords = s_true[:,columns['z_pos']]

    ax.plot(x_coords, y_coords, z_coords,
        '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_measure = np.loadtxt("P4_measurements.txt", delimiter=",")
    Cprime = np.diag([1.0 / rx, 1.0 / ry, 1.0 / rz])
    coords = s_measure.dot(Cprime)
    x_coords = coords[:,columns['x_pos']]
    y_coords = coords[:,columns['y_pos']]
    z_coords = coords[:,columns['z_pos']]
    ax.plot(x_coords, y_coords, z_coords,
        '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([[1, 0, 0, dt, 0, 0],
                 [0, 1, 0, 0, dt, 0],
                 [0, 0, 1, 0, 0, dt],
                 [0, 0, 0, 1 - c*dt, 0, 0],
                 [0, 0, 0, 0, 1 - c*dt, 0],
                 [0, 0, 0, 0, 0, 1 - c*dt]])
    a = np.array([0, 0, 0, 0, 0, g*dt])
    s = s_true[0, :]

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    pred_measurements = np.zeros((6, K))
    pred_measurements[:, 0] = s
    for i in xrange(1, K):
        pred_measurements[:, i] = A.dot(pred_measurements[:, i-1]) + a
    x_coords = pred_measurements[columns['x_pos'], :]
    y_coords = pred_measurements[columns['y_pos'], :]
    z_coords = pred_measurements[columns['z_pos'], :]
    ax.plot(x_coords, y_coords, z_coords,
        '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.array([[rx, 0, 0, 0, 0, 0],
                  [0, ry ,0 ,0, 0, 0],
                  [0, 0, rz, 0 ,0 ,0]])

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    kalman_pred = np.zeros((6, K))
    kalman_pred[:, 0] = s_true[0, :]
    # Initial conditions for s0 and Sigma0
    Sigmak = 0.01 * np.identity(6)
    for i in xrange(1, K):
        sk = kalman_pred[:, i-1]
        s = predictS(A, sk, a)
        Sigma = predictSig(A, B, Sigmak)
        Sigmak = updateSig(Sigma, C)
        kalman_pred[:, i] = updateS(Sigmak, Sigma, s, C, s_measure[i, :])

    x_coords = kalman_pred[columns['x_pos'], :]
    y_coords = kalman_pred[columns['y_pos'], :]
    z_coords = kalman_pred[columns['z_pos'], :]
    ax.plot(x_coords, y_coords, z_coords,
        '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
