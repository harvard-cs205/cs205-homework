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

    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_m = np.loadtxt('P4_measurements.txt', delimiter=',')
    x_coords = 1.0 / rx * s_m[:,0]
    y_coords = 1.0 / ry * s_m[:,1]
    z_coords = 1.0 / rz * s_m[:,2]
    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.identity(6)
    A[0:3,3:6] = np.identity(3) * dt
    A[3:6,3:6] = np.identity(3) - c * dt * np.identity(3)
    a = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, g * dt]]).transpose()

    # Initial conditions for s0
    s0 = np.matrix([[0, 0, 2, 15, 3.5, 4.0]]).transpose()


    # Compute the rest of sk using Eq (1)
    s = np.zeros((6, K))
    s[0:6,0] = s0.transpose()


    # compute 
    for i in range(1, K):
        s_k = np.matrix(s[0:6, (i-1)]).transpose()
        s_k1 = A * s_k + a
        s[0:6, i] = s_k1.reshape((6))
        #s[0:6, i] = (A * np.matrix(s[0:6, (i-1)]).transpose() + a).reshape((6)) # somehow python needs this reshaping command...

    x_coords = s[0,:]
    y_coords = s[1,:]
    z_coords = s[2,:]


    ax.plot(x_coords, y_coords, z_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix(np.diag([bx, by, bz, bvx, bvy, bvz]))
    C = np.zeros((3, 6))
    C[0:3,0:3] = np.diag([rx, ry, rz])
    C = np.matrix(C)
    # functions for the fourth part

    # returns an intermediate guess which based on the propagation matrix and gravity
    def predictS(s_k):
        return A * s_k + a

    # returns Sigma tilde
    def predictSig(Sigma_k):
        return (A * Sigma_k * A.T + B * B.T).I

    # returns Sigma_{k+1}
    def updateSig(Sigma_tilde):
        return (Sigma_tilde + C.T * C).I

    # use this function to retrieve s_{k+1} from s_k !
    def updateS(Sigma_k1, Sigma_tilde, s_tilde, m_k1):
        return Sigma_k1 * (Sigma_tilde * s_tilde + C.T * m_k1)

    # Initial conditions for s0 and Sigma0
    Sigma_0 = 0.01 * np.identity(6)
    s = np.zeros((6, K))
    s[0:6,0] = s0.transpose()
  
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    Sigma_k = Sigma_0 # init Sigma_k
    for i in range(1, K):
        s_k = np.matrix(s[0:6, (i-1)]).transpose() # of course we can also simply update s_k by s_k1 but we use this approach to skirt around redundancy
        # make first guess
        s_tilde = predictS(s_k)
        print('--')
        print(a)
        print(s_k)
        print(s_tilde)
        # predict next Sigma
        Sigma_tilde = predictSig(Sigma_k)
        Sigma_k1 = updateSig(Sigma_tilde)

        # retrieve measurement
        m_k1 = np.zeros((3, 1))
        m_k1[0:3] = np.matrix(s_m[i,:]).T
        print(m_k1)
        s_k1 = updateS(Sigma_k1, Sigma_tilde, s_tilde, m_k1)
        # store result & update Sigma_k
        s[0:6, i] = s_k1.reshape((6)) # somehow python needs this reshaping command...
        Sigma_k = Sigma_k1

    x_coords = s[0,:]
    y_coords = s[1,:]
    z_coords = s[2,:]

    ax.plot(x_coords, y_coords, z_coords,
      '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
