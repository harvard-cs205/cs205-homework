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
    dat_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    x_coords = dat_true[:, 0]
    y_coords = dat_true[:, 1]
    z_coords = dat_true[:, 2]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    dat_measure = np.loadtxt('P4_measurements.txt', delimiter=',')
    C = np.diag([1/rx, 1/ry, 1/rz])
    dat_scaled = np.dot(dat_measure, C)
    x_coords = dat_scaled[:, 0]
    y_coords = dat_scaled[:, 1]
    z_coords = dat_scaled[:, 2]
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
    s0 = np.array([0, 0, 2, 15, 3.5, 4.0])

    def getNextStep(s_prev):
        return np.dot(A, s_prev) + a
    
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    # Iterate over K time steps
    state = [s0]
    for step_i in range(K - 1):
        state.append(getNextStep(state[step_i]))
    state_arr = np.vstack(state)
    x_coords = state_arr[:, 0]
    y_coords = state_arr[:, 1]
    z_coords = state_arr[:, 2]


    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    Sig0 = 1e-2 * np.eye(len(s0))
    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.hstack([np.diag([rx, ry, rz]), np.diag([0, 0, 0])])
    print([Sig0, B, C])
    def predictS(s_prev):
        return getNextStep(s_prev)
    def predictSig(Sig_prev):
        return np.linalg.inv(A.dot(Sig_prev).dot(A.T) + B.dot(B.T)) 
    def updateSig(Sig_int):
        return np.linalg.inv(Sig_int + C.T.dot(C))
    def updateS(Sig_int, Sig_new, s_pred, m):
        return Sig_new.dot(Sig_int.dot(s_pred) + C.T.dot(m))

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    state_kalman = [s0]
    Sig_kalman = [Sig0] 
    for step_i in range(K - 1):
        s_prev = state_kalman[step_i] 
        Sig_prev = Sig_kalman[step_i]
        m_curr = dat_measure[step_i + 1, :] 
        # Calculate model predictions
        s_pred = predictS(s_prev) 
        Sig_pred = predictSig(Sig_prev)
        Sig_new = updateSig(Sig_pred)
        s_new = updateS(Sig_pred, Sig_new, s_pred, m_curr) 
        Sig_kalman.append(Sig_new)
        state_kalman.append(s_new)

    state_kalman_mat = np.vstack(state_kalman)
    xcoord = state_kalman_mat[:, 0]
    ycoord = state_kalman_mat[:, 1]
    zcoord = state_kalman_mat[:, 2]

    ax.plot(xcoord, ycoord, zcoord,
           '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
