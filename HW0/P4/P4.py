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
    x_coords,y_coords,z_coords,x_vel,y_vel,z_vel = np.loadtxt('P4_trajectory.txt', delimiter=',', unpack =True)
    ax.plot(x_coords, y_coords, z_coords,'--b', label='True trajectory')


    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    x_meas, y_meas, z_meas = np.loadtxt('P4_measurements.txt', delimiter=',', unpack=True)
    ax.plot(x_meas/rx, y_meas/ry, z_meas/rz,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    A = np.matrix([[1,0,0,dt,0,0],
         [0,1,0,0,dt,0],
         [0,0,1,0,0,dt],
         [0,0,0,(1-c*dt),0,0],
         [0,0,0,0,(1-c*dt),0],
         [0,0,0,0,0,(1-c*dt)]])
    a = np.matrix([0,0,0,0,0,(g*dt)]).transpose()
    s = np.matrix([0,0,2,15,3.5,4.0]).transpose()

    full_s = np.empty((6,0), int)
    for k in xrange(0,K):
        s = A*s+a
        full_s = np.concatenate((full_s,s),axis=1)
    x_coords1 = np.array(full_s[0,1:])[0].tolist()
    y_coords1 = np.array(full_s[1,1:])[0].tolist()
    z_coords1 = np.array(full_s[2,1:])[0].tolist()
    ax.plot(x_coords1, y_coords1, z_coords1,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    b_vals = np.array([bx,by,bz,bvx,bvy,bvz])
    B = np.diag(b_vals)
    C = np.matrix([[rx,0,0,0,0,0],
                   [0,ry,0,0,0,0],
                   [0,0,rz,0,0,0]])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    Sigma0 = .01*np.identity(6)
    m = np.matrix([x_meas,y_meas,z_meas])
    def predictS(s):
        return A*s+a

    def predictSig(Sigma):
        return np.linalg.inv(A*Sigma*A.T + B*B.T)

    def updateSig(Sigma):
        return np.linalg.inv(Sigma + C.T*C)

    def updateS(updatedSig,estS,estSig, m):
        return updatedSig*(estSig*estS + C.T*m)
    s = np.matrix([0,0,2,15,3.5,4.0]).transpose()
    for k in xrange(0,3):
        s = 

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
