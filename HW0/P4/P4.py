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

    s_true = np.loadtxt("P4_trajectory.txt", delimiter = ',')

    x_coords, y_coords, z_coords = s_true[:,0], s_true[:,1], s_true[:, 2]

    #####################

    ax.plot(x_coords, y_coords, z_coords, '--b', label='True trajectory')

    #plt.show()

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    s_measure = np.loadtxt("P4_measurements.txt", delimiter = ',')

    x_measure, y_measure, z_measure = s_measure[:,0]/rx, s_measure[:,1]/ry, s_measure[:, 2]/rz
    #####################

    ax.plot(x_measure, y_measure, z_measure, '.g', label='Observed trajectory')

    #plt.show()

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a = np.array([0,0,0,0,0,g*dt]).T
    s = np.zeros((6,K))
    s0 = np.array([0,0,2,15,3.5,4]).T
    s[:,0] = s0

    for i in range(K-1):
        s[:,i+1] = np.dot(A,s[:,i]) + a

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    ax.plot(s[0,:], s[1,:], s[2,:], '-k', label='Blind Trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag(np.array([0,0,0,.25,.25,.1]))
    C = np.array([[rx,0,0,0,0,0],[0,ry,0,0,0,0],
              [0,0,rz,0,0,0]])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    sig0 = .01 * np.eye(6)
    sk1 = np.zeros((6, K))
    sk1[:,0] = s0

    def predictS(s):
        return np.dot(A,s) + a

    def predictSig(sigk):
        return np.linalg.inv(np.dot(np.dot(A,sigk),A.T)+
                         np.dot(B,B.T))

    def updateSig(sigtil):
        return np.linalg.inv(sigtil+np.dot(C.T,C))

    def updateS(sigk1, sigtil, stil, mk1):
        return np.dot(sigk1,
                  (np.dot(sigtil,stil)+np.dot(C.T,mk1)))

    file_ext = 'P4_measurements.txt'
    m = np.loadtxt(file_ext, delimiter = ',')

    for i in range(K-1):
        stil_temp = predictS(sk1[:,i])
        sigtil_temp = predictSig(sig0)
        sig0 = sigtil_temp #update sig0 
        sigk1_temp = updateSig(sigtil_temp)
        s_temp = updateS(sigk1_temp, sigtil_temp, stil_temp, m[i])
        sk1[:,i+1] = s_temp
        

    ax.plot(sk1[0,:], sk1[1,:], sk1[2,:], '-r', label='Filtered trajectory')

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
