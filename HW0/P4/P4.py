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
    s_true=np.loadtxt("P4_trajectory.txt", delimiter=",")
    x_coords=s_true[:,0]
    y_coords=s_true[:,1]
    z_coords=s_true[:,2]

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    m = np.loadtxt("P4_measurements.txt", delimiter=",")
    C = np.array([[1./rx, 0.0, 0.0],
                  [0.0, 1./ry, 0.0],
                  [0.0, 0.0, 1./rz]])
    s_obs=np.dot(C, m.T)
    
    x_coords=s_obs[0]
    y_coords=s_obs[1]
    z_coords=s_obs[2]

    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                  [0.0, 0.0, 0.0, 1-c*dt, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1-c*dt, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1-c*dt]])
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, g*dt]).T
    s = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0]).T

    sk = np.zeros(shape=[6,K])
    sk[:,0]=s

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    for i in range(1,K):
        s = np.dot(A, s)+a
        sk[:,i]=s

    x_coords=sk[0]
    y_coords=sk[1]
    z_coords=sk[2]
    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.array([[rx, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, ry, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, rz, 0.0, 0.0, 0.0]])

    # Initial conditions for s0 and Sigma0
    s0 = np.array([0.0, 0.0, 2.0, 15.0, 3.5, 4.0]).T
    Sigma0 = 0.01*np.identity(6)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    sk = np.zeros(shape=[6,K])
    sk[:,0]=s0
    s = s0
    Sigma = Sigma0
    for i in range(1,K):
        s_tilt = np.dot(A, s) + a
        Sigma_tilt = np.linalg.inv(np.dot(A, np.dot(Sigma, A.T)) + np.dot(B, B.T))
        Sigma = np.linalg.inv(Sigma_tilt + np.dot(C.T, C))
        s = np.dot(Sigma, (np.dot(Sigma_tilt, s_tilt)+ np.dot(C.T, m.T[:,i])))
        sk[:, i] = s

    x_coords=sk[0]
    y_coords=sk[1]
    z_coords=sk[2]
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
